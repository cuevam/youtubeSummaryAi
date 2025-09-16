import os, json
from typing import Optional
from openai import OpenAI

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # optional: for Ollama, etc.
)

def analyze_transcript_text(text: str, topic_hint: Optional[str] = None) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        # Make the API requirement explicit in the response
        return {"error": "OPENAI_API_KEY not set on server. Add it as an env var (Space Settings → Variables & secrets)."}

    # keep request small: clip to ~15k chars
    clipped = text[:15000]

    system = (
        "You are a precise debate analyst. Extract the main thesis and core arguments "
        "from the transcript. For EACH major claim, generate the best-known counterargument "
        "(empirically supported or widely accepted) and cite notable academics/scholars who "
        "have published on that claim (by name and, if well-known, 1–2 seminal works). "
        "Be neutral, concise, and avoid hallucinations. If unsure, say so."
    )
    user = {
        "task": "Summarize & counterargue a YouTube talk.",
        "topic_hint": topic_hint,
        "instructions": [
            "Return STRICT JSON with keys: thesis, arguments[].",
            "Each arguments[] item must have: position (string), supporting_points (string[]), counterargument (string), notable_people (string[]).",
            "If you’re unsure about any notable_people, include an empty array for that item."
        ],
        "transcript_excerpt": clipped
    }

    resp = client.chat.completions.create(
        model=_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content": system},
            {"role":"user","content": json.dumps(user)}
        ]
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # If model returns near-JSON, still return raw content for debugging
        return {"raw": content}
