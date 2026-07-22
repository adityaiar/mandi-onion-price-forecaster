"""
Optional LLM reply layer.

The LLM only rephrases the agent's numbers into natural language and answers
grounded follow-up questions. It never forecasts, computes costs, or decides
which mandi to pick - all of that stays in agent.py. If no API key is set the
app falls back to the deterministic reply, so it still runs fully offline.

Key: set ANTHROPIC_API_KEY (env var or Streamlit secret). Nothing is hardcoded.
"""
import json
import os

MODEL = "claude-opus-4-8"   # switch to "claude-haiku-4-5" for a cheaper, faster reply layer
MAX_TOKENS = 512

SYSTEM = (
    "You are a helpful assistant for onion farmers in the Nashik region of Maharashtra. "
    "Use ONLY the numbers in the recommendation and the mandi table given to you. "
    "Never invent or change prices, distances, mandi names, or dates. "
    "If a question cannot be answered from those numbers, say you do not have that information. "
    "Be concise, practical, and friendly. Prices are rupees per quintal. "
    "Write plain sentences: no markdown, no asterisks, no headings, no bullet characters. "
    "Write all numbers as ordinary digits (for example 1382), including in the Marathi text."
)

# structured output so the two languages come back as separate fields, and the UI
# never has to parse the model's formatting
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "english": {"type": "string", "description": "Two or three short sentences, plain English."},
        "marathi": {"type": "string", "description": "The same advice in Marathi (Devanagari)."},
    },
    "required": ["english", "marathi"],
    "additionalProperties": False,
}


def available() -> bool:
    """True only if a key is set and the SDK imports."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def _table_text(df):
    if df is None or len(df) == 0:
        return "(no mandis within range)"
    cols = [c for c in ["mandi", "km", "today_price", "transport",
                        "net_now", "fc_7d", "net_hold"] if c in df.columns]
    return df[cols].to_string(index=False)


def _context(reco):
    return (
        f"Model recommendation: {reco.text}\n\n"
        "Ranked mandis (net_now = today's price minus transport; "
        "fc_7d = 7-day ARIMA forecast; net_hold = forecast minus transport and storage):\n"
        f"{_table_text(reco.table)}"
    )


def phrase(reco):
    """Rephrase the recommendation. Returns (english, marathi) or None on any failure."""
    if reco.action == "none":
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        user = (_context(reco) + "\n\n"
                "Write this recommendation for the farmer. Lead with the action to take, "
                "then the price, then whether to wait. Two or three short sentences.")
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": REPLY_SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        en, mr = data.get("english", "").strip(), data.get("marathi", "").strip()
        return (en, mr) if en else None
    except Exception:
        return None


def answer(reco, question, history=None) -> str | None:
    """Answer a grounded follow-up question. Returns None on any failure."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msgs = []
        for role, text in (history or []):
            msgs.append({"role": role, "content": text})
        msgs.append({"role": "user",
                     "content": _context(reco) + f"\n\nThe farmer asks: {question}\n"
                                "Answer using only the numbers above."})
        resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                      system=SYSTEM, messages=msgs)
        return "".join(b.text for b in resp.content if b.type == "text").strip() or None
    except Exception:
        return None
