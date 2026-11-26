"""
db.py — all SQLite access for BaonTrack lives here.

Keeping every query in one module means the Streamlit pages stay thin
(UI only) and the data logic is easy to test or swap out later
(e.g. moving to Postgres for the v3 multi-device sync).

Every transaction query is scoped to a user_id so classmates sharing
the same database file never see each other's data.
"""

import hashlib
import secrets
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "baontrack.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

CATEGORIES = {
    "income": ["Allowance", "Gig income", "Other income"],
    "expense": ["Food", "Transpo", "Load/Data", "School supplies", "Other expense"],
}


def get_connection() -> sqlite3.Connection:
    """Open a connection and make sure the schema exists."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    # Older baontrack.db files were created before face_only existed.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    if "face_only" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN face_only INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Auth — intentionally simple (username + PIN). This is enough to keep
# classmates' data separate on a shared prototype, not real security.
# ---------------------------------------------------------------------------

def _hash_pin(username: str, pin: str) -> str:
    # Salting with the username is enough to stop identical PINs producing
    # identical hashes across users, without needing a separate salt column.
    return hashlib.sha256(f"{username.lower()}:{pin}".encode()).hexdigest()


MIN_PIN_LENGTH = 4


def create_user(conn, username: str, pin: str) -> int | None:
    """Returns the new user's id, or None if the username is taken (compared
    case-insensitively, so 'Alice' and 'alice' collide) or the PIN is too short."""
    username = username.strip()
    if not username or not pin or len(pin) < MIN_PIN_LENGTH:
        return None
    existing = conn.execute(
        "SELECT 1 FROM users WHERE LOWER(username) = LOWER(?)", (username,)
    ).fetchone()
    if existing is not None:
        return None  # username already exists (case-insensitive)
    try:
        cur = conn.execute(
            "INSERT INTO users (username, pin_hash) VALUES (?, ?)",
            (username, _hash_pin(username, pin)),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # exact-match race with the check above


def verify_user(conn, username: str, pin: str) -> int | None:
    """Returns the user's id if username/PIN match, else None.
    Username lookup is case-insensitive so 'Alice' can log in as 'alice'."""
    username = username.strip()
    row = conn.execute(
        "SELECT id, pin_hash FROM users WHERE LOWER(username) = LOWER(?)", (username,)
    ).fetchone()
    if row is None:
        return None
    user_id, pin_hash = row
    return user_id if pin_hash == _hash_pin(username, pin) else None


# ---------------------------------------------------------------------------
# Transactions — every function below is scoped to one user_id.
# ---------------------------------------------------------------------------

def add_transaction(conn, user_id: int, txn_type: str, category: str, amount: float, txn_date: date, note: str = ""):
    conn.execute(
        "INSERT INTO transactions (user_id, type, category, amount, txn_date, note) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, txn_type, category, amount, txn_date.isoformat(), note or None),
    )
    conn.commit()


def update_transaction(conn, user_id: int, txn_id: int, txn_type: str, category: str, amount: float, txn_date: date, note: str):
    conn.execute(
        "UPDATE transactions SET type=?, category=?, amount=?, txn_date=?, note=? WHERE id=? AND user_id=?",
        (txn_type, category, amount, txn_date.isoformat(), note or None, txn_id, user_id),
    )
    conn.commit()


def delete_transaction(conn, user_id: int, txn_id: int):
    conn.execute("DELETE FROM transactions WHERE id=? AND user_id=?", (txn_id, user_id))
    conn.commit()


def get_all_transactions(conn, user_id: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT id, type, category, amount, txn_date, note FROM transactions WHERE user_id=? ORDER BY txn_date DESC, id DESC",
        conn,
        params=(user_id,),
    )
    if not df.empty:
        df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.date
    return df


def get_balance(conn, user_id: int) -> float:
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE 0 END), 0) -
            COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0)
        FROM transactions
        WHERE user_id=?
        """,
        (user_id,),
    ).fetchone()
    return row[0] or 0.0


def get_week_spend(conn, user_id: int) -> float:
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE type='expense' AND txn_date >= ? AND user_id=?",
        (week_ago, user_id),
    ).fetchone()
    return row[0] or 0.0


def get_spend_by_category(conn, user_id: int, days: int = 30) -> pd.DataFrame:
    since = (date.today() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        """
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE type='expense' AND txn_date >= ? AND user_id=?
        GROUP BY category
        ORDER BY total DESC
        """,
        conn,
        params=(since, user_id),
    )
    return df


def get_avg_daily_spend(conn, user_id: int, lookback_days: int = 14) -> float:
    """Average expense per day over the lookback window.
    Used for the 'days until broke' estimate — simple on purpose."""
    since = (date.today() - timedelta(days=lookback_days)).isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE type='expense' AND txn_date >= ? AND user_id=?",
        (since, user_id),
    ).fetchone()
    total = row[0] or 0.0
    return total / lookback_days if total else 0.0

# ---------------------------------------------------------------------------
# Face embeddings — used by face.py for the face-verification login step.
# Stores raw float32 bytes; face.py converts to/from numpy.
# ---------------------------------------------------------------------------

def add_face_embedding(conn, user_id: int, embedding_bytes: bytes):
    conn.execute(
        "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
        (user_id, embedding_bytes),
    )
    conn.commit()


def get_face_embeddings(conn, user_id: int) -> list[bytes]:
    rows = conn.execute(
        "SELECT embedding FROM face_embeddings WHERE user_id=? ORDER BY id", (user_id,)
    ).fetchall()
    return [r[0] for r in rows]


def count_face_embeddings(conn, user_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM face_embeddings WHERE user_id=?", (user_id,)
    ).fetchone()[0]


def delete_face_embeddings(conn, user_id: int):
    conn.execute("DELETE FROM face_embeddings WHERE user_id=?", (user_id,))
    conn.commit()


def username_taken(conn, username: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),)
    ).fetchone() is not None


def create_face_user(conn, username: str, embeddings: list[bytes]) -> int | None:
    """Creates an account whose only credential is the face. The PIN column is
    filled with a random secret nobody knows, so PIN login can never work for it.
    Returns the new user id, or None if the username is empty/taken."""
    username = username.strip()
    if not username or not embeddings or username_taken(conn, username):
        return None
    try:
        cur = conn.execute(
            "INSERT INTO users (username, pin_hash, face_only) VALUES (?, ?, 1)",
            (username, _hash_pin(username, secrets.token_hex(32))),
        )
        user_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
            [(user_id, e) for e in embeddings],
        )
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        conn.rollback()
        return None


def is_face_only(conn, user_id: int) -> bool:
    row = conn.execute("SELECT face_only FROM users WHERE id=?", (user_id,)).fetchone()
    return bool(row and row[0])


def get_all_face_candidates(conn) -> tuple[dict, dict]:
    """For face login: ({user_id: [embedding bytes]}, {user_id: username})."""
    rows = conn.execute(
        "SELECT f.user_id, u.username, f.embedding FROM face_embeddings f JOIN users u ON u.id = f.user_id"
    ).fetchall()
    embs, names = {}, {}
    for uid, uname, blob in rows:
        embs.setdefault(uid, []).append(blob)
        names[uid] = uname
    return embs, names