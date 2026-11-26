"""
BaonTrack — a campus financial assistant (v1 / MVP)

Pages (after login):
  1. Log Transaction      — fast entry for income/expense
  2. Dashboard            — balance, weekly spend, category chart, days-until-broke
  3. History              — editable table of everything logged
  4. Face ID              — enroll / remove your face for login verification
  5. Live Face Detection  — live webcam stream with face detection (CS 315 Activity 7)

Login is a username + PIN, scoped per user_id, so classmates sharing this
app never see each other's transactions. If the user has enrolled a face,
login adds a second step: face verification through the webcam.
"""

from datetime import date

import cv2
import pandas as pd
import streamlit as st

import ai_entry
import db
import face

st.set_page_config(page_title="BaonTrack", page_icon="💸", layout="centered")

# One shared connection per session, cached so we don't reopen it on every rerun.
if "conn" not in st.session_state:
    st.session_state.conn = db.get_connection()
conn = st.session_state.conn

# Pending-confirmation state for the three "are you sure?" flows below.
st.session_state.setdefault("pending_txn", None)
st.session_state.setdefault("pending_edit", None)
st.session_state.setdefault("pending_delete", None)
st.session_state.setdefault("user_id", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("history_errors", None)
st.session_state.setdefault("quick_entry_prefill", None)
st.session_state.setdefault("quick_entry_error", None)
st.session_state.setdefault("face_pending", None)      # {"user_id", "username"} while waiting on the face step
st.session_state.setdefault("face_attempts", 0)
st.session_state.setdefault("login_notice", None)
st.session_state.setdefault("enroll_cam_key", 0)       # bumping this resets the camera widget
st.session_state.setdefault("enroll_msg", None)

MAX_FACE_ATTEMPTS = 5
MAX_FACE_SAMPLES = 5
RECOMMENDED_FACE_SAMPLES = 3

# All category labels, used to constrain the History table's category dropdown
# to real categories (the per-type CATEGORIES dict still governs which of
# those are actually valid for a given type — see _validate_txn_fields).
ALL_CATEGORIES = db.CATEGORIES["income"] + db.CATEGORIES["expense"]


def _validate_txn_fields(txn_type: str, category: str, amount, txn_date) -> str | None:
    """Shared validation for both new entries and History edits.
    Returns an error message, or None if the fields are all valid."""
    if amount is None or pd.isna(amount) or amount <= 0:
        return "Amount has to be more than 0."
    if txn_date is None or pd.isna(txn_date):
        return "Date can't be empty."
    if category not in db.CATEGORIES.get(txn_type, []):
        return f"'{category}' isn't a valid category for {txn_type}."
    return None


def _alert_banner(headline: str, body: str, color: str = "#ff4b4b", bg: str = "#fff0f0"):
    """A bold, high-contrast banner used inside the confirmation dialogs
    so the warning is impossible to miss."""
    st.markdown(
        f"""
        <div style="background-color:{bg}; border-left:8px solid {color};
                    border-radius:8px; padding:16px 18px; margin-bottom:8px;">
            <div style="color:{color}; font-weight:700; font-size:1.05rem;">{headline}</div>
            <div style="color:#333; margin-top:4px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("⚠️ You're about to go negative")
def overdraft_dialog():
    p = st.session_state.pending_txn
    _alert_banner(
        "This expense will put you in the negative.",
        f"Logging ₱{p['amount']:,.2f} ({p['category']}) will bring your balance to "
        f"<strong>₱{p['resulting_balance']:,.2f}</strong>.",
    )
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Yes, log it anyway", type="primary", use_container_width=True):
        db.add_transaction(conn, st.session_state.user_id, p["type"], p["category"], p["amount"], p["date"], p["note"])
        st.session_state.pending_txn = None
        st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state.pending_txn = None
        st.rerun()


@st.dialog("⚠️ This edit will put you in the negative")
def edit_overdraft_dialog():
    p = st.session_state.pending_edit
    _alert_banner(
        "Saving this edit will put you in the negative.",
        f"Changing transaction #{p['txn_id']} to ₱{p['amount']:,.2f} ({p['category']}) will bring your balance to "
        f"<strong>₱{p['resulting_balance']:,.2f}</strong>.",
    )
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Yes, save it anyway", type="primary", use_container_width=True):
        db.update_transaction(
            conn, st.session_state.user_id, p["txn_id"], p["type"], p["category"], p["amount"], p["date"], p["note"]
        )
        st.session_state.pending_edit = None
        st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state.pending_edit = None
        st.rerun()


@st.dialog("🗑️ Delete this transaction?")
def delete_dialog():
    pid = st.session_state.pending_delete
    _alert_banner(
        "This can't be undone.",
        f"You're about to permanently delete transaction #{pid}.",
        color="#e02424",
        bg="#fde8e8",
    )
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Yes, delete it", type="primary", use_container_width=True):
        db.delete_transaction(conn, st.session_state.user_id, pid)
        st.session_state.pending_delete = None
        st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()


def _embedding_from_photo(photo):
    """camera_input photo -> (embedding, problem). Never raises: model download
    or decode problems come back as a message the UI can show."""
    try:
        return face.get_embedding(face.decode_image(photo.getvalue()))
    except Exception as e:
        return None, f"Face models couldn't load ({type(e).__name__}). Check your internet connection and try again."


def _stored_embeddings(user_id: int):
    return [face.embedding_from_bytes(b) for b in db.get_face_embeddings(conn, user_id)]


def _show_face_preview(photo):
    """Shows the captured photo with the detected face boxed, so the user can
    see what the detector sees before saving/verifying."""
    try:
        img = face.decode_image(photo.getvalue())
        faces = face.detect_faces(img)
        face.draw_faces(img, faces)
        st.image(img[:, :, ::-1], caption=f"{len(faces)} face(s) detected", use_container_width=True)
    except Exception:
        pass  # preview is a nicety; the real checks run on Save / Verify


def face_check_step():
    """Second login step for users who enrolled a face."""
    pending = st.session_state.face_pending
    st.title("🔐 Face verification")
    st.caption(f"Hi **{pending['username']}** — PIN accepted. One more step: look at the camera.")

    photo = st.camera_input("Take a photo", key=f"login_cam_{st.session_state.face_attempts}")
    verify_col, cancel_col = st.columns(2)
    verify = verify_col.button("Verify", type="primary", disabled=photo is None, use_container_width=True)
    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state.face_pending = None
        st.session_state.face_attempts = 0
        st.rerun()

    if verify and photo is not None:
        embedding, problem = _embedding_from_photo(photo)
        if problem:
            st.error(problem)  # no face / too small etc. — doesn't count as a failed attempt
            return
        ok, score = face.is_match(embedding, _stored_embeddings(pending["user_id"]))
        if ok:
            st.session_state.user_id = pending["user_id"]
            st.session_state.username = pending["username"]
            st.session_state.face_pending = None
            st.session_state.face_attempts = 0
            st.rerun()
        else:
            st.session_state.face_attempts += 1
            if st.session_state.face_attempts >= MAX_FACE_ATTEMPTS:
                st.session_state.face_pending = None
                st.session_state.face_attempts = 0
                st.session_state.login_notice = "Too many failed face checks. Log in again to retry."
                st.rerun()
            left = MAX_FACE_ATTEMPTS - st.session_state.face_attempts
            st.error(f"Face didn't match (similarity {score:.2f}, needs {face.MATCH_THRESHOLD}). {left} tries left — retake the photo.")


# ---------------------------------------------------------------------------
# Login gate — nothing below this runs until user_id is set.
# ---------------------------------------------------------------------------
if st.session_state.user_id is None and st.session_state.face_pending:
    face_check_step()
    st.stop()

if st.session_state.user_id is None:
    st.title("💸 BaonTrack")
    st.caption("Log in or create an account to start tracking your money.")
    if st.session_state.login_notice:
        st.warning(st.session_state.login_notice)
        st.session_state.login_notice = None

    login_tab, register_tab = st.tabs(["Log In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            pin = st.text_input("PIN", type="password")
            if st.form_submit_button("Log In", use_container_width=True):
                user_id = db.verify_user(conn, username, pin)
                if user_id and db.count_face_embeddings(conn, user_id) > 0:
                    # PIN is right, but this user turned on face verification:
                    # hold off on logging in until the face step passes.
                    st.session_state.face_pending = {"user_id": user_id, "username": username.strip()}
                    st.session_state.face_attempts = 0
                    st.rerun()
                elif user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error("Wrong username or PIN.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_pin = st.text_input("Choose a PIN", type="password")
            if st.form_submit_button("Create Account", use_container_width=True):
                if not new_username.strip() or not new_pin:
                    st.error("Username and PIN can't be empty.")
                else:
                    user_id = db.create_user(conn, new_username, new_pin)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = new_username.strip()
                        st.rerun()
                    else:
                        st.error("That username is already taken.")

    st.stop()  # don't render anything below until logged in


# Open whichever dialog is pending. Must be called at the top level of the
# script (not nested in a column/expander) for st.dialog to render correctly.
if st.session_state.pending_txn:
    overdraft_dialog()
if st.session_state.pending_edit:
    edit_overdraft_dialog()
if st.session_state.pending_delete is not None:
    delete_dialog()

user_id = st.session_state.user_id

st.sidebar.title("💸 BaonTrack")
st.sidebar.caption(f"Logged in as **{st.session_state.username}**")
if st.sidebar.button("Log Out", use_container_width=True):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()
page = st.sidebar.radio("Go to", ["Log Transaction", "Dashboard", "History", "Face ID", "Live Face Detection"])


# ---------------------------------------------------------------------------
# Page 1: Log Transaction
# ---------------------------------------------------------------------------
if page == "Log Transaction":
    st.title("Log a Transaction")

    # --- AI quick add: type it naturally, AI pre-fills the form below.
    # This never saves anything by itself — it only fills in the same
    # form a manual entry would use, so validation and the overdraft
    # dialog still apply exactly as before.
    with st.expander("✨ Quick add (type it naturally)", expanded=True):
        quick_text = st.text_input(
            "e.g. \"spent 45 on jeepney to school\" or \"received 1000 allowance\"",
            key="quick_entry_text",
            label_visibility="collapsed",
            placeholder='e.g. "spent 45 on jeepney to school"',
        )
        if st.button("Parse", use_container_width=True) and quick_text.strip():
            result = ai_entry.parse_quick_entry(quick_text, date.today())
            if result is None:
                st.session_state.quick_entry_error = (
                    "Couldn't turn that into a transaction — try rephrasing, "
                    "or just fill in the form below."
                )
                st.session_state.quick_entry_prefill = None
            else:
                st.session_state.quick_entry_error = None
                st.session_state.quick_entry_prefill = result
            st.rerun()

        if st.session_state.quick_entry_error:
            st.warning(st.session_state.quick_entry_error)
        if st.session_state.quick_entry_prefill:
            p = st.session_state.quick_entry_prefill
            note_fragment = f' · "{p["note"]}"' if p["note"] else ""
            st.caption(
                f"Parsed as **{p['type']}** · {p['category']} · ₱{p['amount']:,.2f} · "
                f"{p['date'].isoformat()}{note_fragment} "
                "— check the form below and hit Save to confirm."
            )

    prefill = st.session_state.quick_entry_prefill
    type_options = ["expense", "income"]
    default_type_index = type_options.index(prefill["type"]) if prefill else 0

    txn_type = st.radio("Type", type_options, horizontal=True, index=default_type_index)
    category_options = db.CATEGORIES[txn_type]
    default_cat_index = (
        category_options.index(prefill["category"])
        if prefill and prefill["type"] == txn_type and prefill["category"] in category_options
        else 0
    )
    category = st.selectbox("Category", category_options, index=default_cat_index)

    with st.form("log_form", clear_on_submit=True):
        # value=None + placeholder means the field starts empty, not "0.00" —
        # so typing a number doesn't require deleting anything first. A
        # quick-add prefill (matching the current type) overrides that default.
        prefill_amount = prefill["amount"] if prefill and prefill["type"] == txn_type else None
        amount = st.number_input(
            "Amount (₱)", min_value=0.0, step=5.0, format="%.2f", value=prefill_amount, placeholder="0.00"
        )
        prefill_date = prefill["date"] if prefill and prefill["type"] == txn_type else date.today()
        txn_date = st.date_input("Date", value=prefill_date, max_value=date.today())
        prefill_note = prefill["note"] if prefill and prefill["type"] == txn_type else ""
        note = st.text_input("Note (optional)", value=prefill_note)
        submitted = st.form_submit_button("Save", use_container_width=True)

        if submitted:
            err = _validate_txn_fields(txn_type, category, amount, txn_date)
            if err:
                st.error(err)
            else:
                current_balance = db.get_balance(conn, user_id)
                resulting_balance = current_balance - amount if txn_type == "expense" else current_balance + amount
                st.session_state.quick_entry_prefill = None  # consumed either way

                if txn_type == "expense" and resulting_balance < 0:
                    # Don't save yet — the dialog above asks for confirmation.
                    st.session_state.pending_txn = {
                        "type": txn_type,
                        "category": category,
                        "amount": amount,
                        "date": txn_date,
                        "note": note,
                        "resulting_balance": resulting_balance,
                    }
                    st.rerun()
                else:
                    db.add_transaction(conn, user_id, txn_type, category, amount, txn_date, note)
                    st.success(f"Logged {txn_type}: ₱{amount:,.2f} ({category})")

    st.caption("Tip: log it the moment it happens — that's the habit that makes this useful.")


# ---------------------------------------------------------------------------
# Page 2: Dashboard
# ---------------------------------------------------------------------------
elif page == "Dashboard":
    st.title("Dashboard")

    balance = db.get_balance(conn, user_id)
    week_spend = db.get_week_spend(conn, user_id)
    avg_daily = db.get_avg_daily_spend(conn, user_id)

    col1, col2 = st.columns(2)
    col1.metric("Current balance", f"₱{balance:,.2f}")
    col2.metric("Spent this week", f"₱{week_spend:,.2f}")

    st.subheader("Where it's going (last 30 days)")
    cat_df = db.get_spend_by_category(conn, user_id, days=30)
    if cat_df.empty:
        st.info("No expenses logged yet — add some on the Log Transaction page.")
    else:
        st.bar_chart(cat_df.set_index("category"))

    st.subheader("Will you make it to allowance day?")
    if avg_daily <= 0:
        st.info("Not enough spending history yet to estimate this — keep logging for a few days.")
    elif balance <= 0:
        st.error("Your balance is already at or below ₱0.")
    else:
        days_left = balance / avg_daily
        st.metric("Estimated days left", f"{days_left:,.1f} days")
        st.caption(
            f"Based on an average spend of ₱{avg_daily:,.2f}/day over the last 14 days. "
            "This is a rough estimate, not a budget plan — use it as an early warning."
        )


# ---------------------------------------------------------------------------
# Page 3: History
# ---------------------------------------------------------------------------
elif page == "History":
    st.title("Transaction History")

    df = db.get_all_transactions(conn, user_id)
    if df.empty:
        st.info("No transactions yet.")
    else:
        if st.session_state.history_errors:
            for msg in st.session_state.history_errors:
                st.error(msg)
            st.session_state.history_errors = None

        st.caption("Edit any cell directly, or check a row and use the delete box below.")
        edited_df = st.data_editor(
            df,
            key="history_editor",
            disabled=["id"],
            hide_index=True,
            use_container_width=True,
            column_config={
                "type": st.column_config.SelectboxColumn(options=["income", "expense"]),
                "category": st.column_config.SelectboxColumn(options=ALL_CATEGORIES),
                "amount": st.column_config.NumberColumn(format="₱%.2f"),
            },
        )

        # Persist any edits made in the data editor back to SQLite. Each
        # changed row is validated first (bad amount/date/category never
        # reaches the database), and an edit that would push the balance
        # negative is routed through the same confirmation dialog used for
        # new expenses, instead of being applied silently.
        if not edited_df.equals(df):
            errors = []
            overdraft_edit = None
            for _, row in edited_df.iterrows():
                match = df[df["id"] == row["id"]]
                if match.empty:
                    continue
                orig_row = match.iloc[0]
                if row.equals(orig_row):
                    continue

                err = _validate_txn_fields(row["type"], row["category"], row["amount"], row["txn_date"])
                if err:
                    errors.append(f"Row #{row['id']}: {err}")
                    continue

                current_balance = db.get_balance(conn, user_id)
                old_effect = orig_row["amount"] if orig_row["type"] == "income" else -orig_row["amount"]
                new_effect = row["amount"] if row["type"] == "income" else -row["amount"]
                resulting_balance = current_balance - old_effect + new_effect

                if row["type"] == "expense" and resulting_balance < 0 and overdraft_edit is None:
                    overdraft_edit = {
                        "txn_id": row["id"],
                        "type": row["type"],
                        "category": row["category"],
                        "amount": row["amount"],
                        "date": row["txn_date"],
                        "note": row["note"],
                        "resulting_balance": resulting_balance,
                    }
                else:
                    try:
                        db.update_transaction(
                            conn, user_id, row["id"], row["type"], row["category"], row["amount"], row["txn_date"], row["note"]
                        )
                    except Exception:
                        # Last-line-of-defense: validation above should catch bad
                        # values before they get here, but never let a DB-level
                        # rejection (e.g. a CHECK constraint) crash the page.
                        errors.append(f"Row #{row['id']}: couldn't save that edit — values rejected by the database.")

            if overdraft_edit:
                st.session_state.pending_edit = overdraft_edit
            if errors:
                st.session_state.history_errors = errors
            st.rerun()

        # Custom CSV export instead of the table's built-in download icon: Excel
        # auto-detects ISO dates and picks a column too narrow to show them,
        # which is what causes "######". A leading apostrophe forces Excel to
        # treat the date as text instead of reformatting it.
        st.divider()
        export_df = df.copy()
        export_df["txn_date"] = export_df["txn_date"].apply(lambda d: f"'{d.isoformat()}")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download CSV (Excel-friendly)",
            data=csv_bytes,
            file_name="baontrack_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Use this button rather than the table's own export icon — it keeps dates readable in Excel.")

        st.divider()
        del_id = st.selectbox("Delete a transaction by ID", [None] + df["id"].tolist())
        if del_id is not None and st.button("Delete", type="primary"):
            st.session_state.pending_delete = del_id
            st.rerun()


# ---------------------------------------------------------------------------
# Page 4: Face ID (enroll / manage)
# ---------------------------------------------------------------------------
elif page == "Face ID":
    st.title("Face ID")
    n_samples = db.count_face_embeddings(conn, user_id)

    if n_samples:
        st.success(f"Face verification is **ON** — {n_samples} sample(s) enrolled. You'll be asked for a face check after your PIN.")
    else:
        st.info("Face verification is **OFF**. Enroll a few photos of yourself to turn it on.")

    if st.session_state.enroll_msg:
        st.success(st.session_state.enroll_msg)
        st.session_state.enroll_msg = None

    st.caption(
        "Only a 128-number face *template* is saved in the database — never the photo. "
        "Biometric data is sensitive personal information (Data Privacy Act of 2012), "
        "so enrollment is opt-in and you can delete it any time."
    )

    if n_samples >= MAX_FACE_SAMPLES:
        st.info(f"You already have the maximum of {MAX_FACE_SAMPLES} samples. Remove them below if you want to re-enroll.")
    else:
        consent = st.checkbox("I agree to store my face template in BaonTrack for login verification.")
        if consent:
            st.write(
                f"Take **{RECOMMENDED_FACE_SAMPLES}** samples with slightly different angles/lighting "
                f"(sample {n_samples + 1} of {RECOMMENDED_FACE_SAMPLES}+). One face only, good light, no hat/mask."
            )
            photo = st.camera_input("Capture a sample", key=f"enroll_cam_{st.session_state.enroll_cam_key}")
            if photo is not None:
                _show_face_preview(photo)
                if st.button("Save this sample", type="primary", use_container_width=True):
                    embedding, problem = _embedding_from_photo(photo)
                    if problem:
                        st.error(problem)
                    else:
                        db.add_face_embedding(conn, user_id, face.embedding_to_bytes(embedding))
                        st.session_state.enroll_cam_key += 1  # fresh camera for the next sample
                        st.session_state.enroll_msg = f"Sample saved ({n_samples + 1} enrolled)."
                        st.rerun()

    if n_samples:
        st.divider()
        if n_samples < RECOMMENDED_FACE_SAMPLES:
            st.warning(f"Only {n_samples} sample(s) — {RECOMMENDED_FACE_SAMPLES}+ gives more reliable matching.")
        sure = st.checkbox("Yes, delete my face data and turn Face ID off")
        if st.button("Delete face data", disabled=not sure):
            db.delete_face_embeddings(conn, user_id)
            st.session_state.enroll_msg = "Face data deleted. Face verification is now off."
            st.rerun()


# ---------------------------------------------------------------------------
# Page 5: Live Face Detection (Activity 7 — video stream from the laptop camera)
# ---------------------------------------------------------------------------
elif page == "Live Face Detection":
    st.title("Live Face Detection")
    st.caption(
        "Live video from your laptop camera. Every frame goes through the YuNet face detector; "
        "if you've enrolled a face, it also labels whether the face matches *you*."
    )

    try:
        import av
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        st.error("This page needs `streamlit-webrtc` and `av`. Run: `pip install streamlit-webrtc av`")
        st.stop()

    enrolled = _stored_embeddings(user_id)
    recognise = st.toggle("Also check if the face is me", value=bool(enrolled), disabled=not enrolled)
    if not enrolled:
        st.caption("Enroll on the Face ID page to unlock recognition — detection works without it.")
    stored = enrolled if recognise else []
    my_name = st.session_state.username

    def process_frame(frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face.detect_faces(img)
        labels, colors = [], []
        for f in faces:
            if stored:
                ok, score = face.is_match(face.embed_face(img, f), stored)
                labels.append(f"{my_name} {score:.2f}" if ok else f"unknown {score:.2f}")
                colors.append((0, 200, 0) if ok else (0, 0, 230))
            else:
                labels.append(f"face {f[14]:.2f}")
                colors.append((0, 200, 0))
        face.draw_faces(img, faces, labels or None, colors or None)
        cv2.putText(img, f"faces: {len(faces)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-face",
        video_frame_callback=process_frame,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Click START, allow camera access. If the video never appears, your network may block WebRTC — use the Face ID page (photo mode) instead.")
