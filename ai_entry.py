"""
ai_entry.py — natural-language quick entry for BaonTrack.

Turns a sentence like "spent 45 on jeepney to school" into structured
transaction fields (type, category, amount, date, note). The category
list is pulled straight from db.CATEGORIES, and every field the model
returns is re-validated in Python before it's trusted — the model can
suggest a transaction, but it can never insert one directly, and it
can never invent a category that doesn't exist in the app.

Runs on Hugging Face's Inference Providers (a hosted, pay-as-you-go API
with a free tier — no GPU needed, which matters since Streamlit
Community Cloud is CPU-only). Model: Qwen/Qwen2.5-7B-Instruct, chosen
for its reliability at sticking to an exact JSON schema rather than
drifting into a category name that isn't real.

Requires an HF_TOKEN (a Hugging Face access token — read permissions
are enough), read from Streamlit secrets if present
(st.secrets["HF_TOKEN"], the normal way to store it for a Streamlit
Community Cloud deploy) and falling back to the environment variable
of the same name for local runs. Get one at
https://huggingface.co/settings/tokens.
"""

import json
import os
from datetime import date

from huggingface_hub import InferenceClient

from db import CATEGORIES

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_client = None


def _get_api_key() -> str | None:
    try:
        import streamlit as st
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass  # no secrets.toml locally — fall through to the env var
    return os.environ.get("HF_TOKEN")


def _get_client():
    global _client
    if _client is None:
        api_key = _get_api_key()
        if not api_key:
            return None
        # provider="auto" was tried first but HF's auto-router doesn't
        # currently resolve to a working backend for this model, even
        # though the model works fine when a provider is named directly.
        # Featherless is confirmed (via manual testing) to serve this
        # model, so it's pinned explicitly here rather than left to auto.
        _client = InferenceClient(model=MODEL_ID, token=api_key, provider="featherless-ai")
    return _client


def _system_prompt(today: date) -> str:
    return f"""You turn a short student expense/income note into structured JSON.

Today's date is {today.isoformat()}.

Allowed categories:
  income: {CATEGORIES["income"]}
  expense: {CATEGORIES["expense"]}

Rules:
- "type" must be exactly "income" or "expense".
- "category" must be exactly one of the allowed categories listed above for
  that type — never invent a new one. Pick the closest real match.
- "amount" is a positive number in PHP. If no amount is stated, use null.
- "note" is a short cleaned-up version of what the student typed (can be "").
- "date" is today's date unless the text clearly names another day
  ("yesterday", "last Monday") — return it as YYYY-MM-DD. Never return a
  date after today.
- If the text doesn't look like a transaction at all, return {{"type": null}}.

Respond with ONLY a JSON object, no other text, no markdown fences. Example
shape: {{"type": "expense", "category": "Food", "amount": 45.0, "date": "{today.isoformat()}", "note": "lunch"}}
"""


def parse_quick_entry(text: str, today: date) -> dict | None:
    """Calls Claude to parse free text into transaction fields.

    Returns a dict with keys type/category/amount/date/note, or None if
    the API key is missing, the call failed, or the text didn't parse
    into a valid transaction. Every field is validated against the
    app's real rules (CATEGORIES, positive amount, no future dates)
    before being returned — nothing from the model is trusted blindly.
    """
    if not text or not text.strip():
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": _system_prompt(today)},
                {"role": "user", "content": text.strip()},
            ],
            max_tokens=300,
            temperature=0,  # deterministic extraction, not creative writing
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
    except Exception:
        return None

    if parsed.get("type") not in ("income", "expense"):
        return None
    if parsed.get("category") not in CATEGORIES.get(parsed["type"], []):
        return None
    if not isinstance(parsed.get("amount"), (int, float)) or parsed["amount"] <= 0:
        return None

    try:
        parsed_date = date.fromisoformat(str(parsed.get("date", today.isoformat())))
    except (TypeError, ValueError):
        parsed_date = today
    if parsed_date > today:
        parsed_date = today  # belt-and-suspenders: never let this create a future-dated entry

    return {
        "type": parsed["type"],
        "category": parsed["category"],
        "amount": float(parsed["amount"]),
        "date": parsed_date,
        "note": (parsed.get("note") or "").strip(),
    }