from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
from faster_whisper import WhisperModel
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout


import json, os, re, tempfile
import yt_dlp

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from xml.etree.ElementTree import ParseError

# near your imports
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Whisper fallback deps

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return RedirectResponse("/docs")


class TranscriptReq(BaseModel):
    url: str

def _extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)", url)
    if not m:
        raise HTTPException(status_code=400, detail="Could not parse YouTube video ID.")
    return m.group(1)

def _mmss(s: float) -> str:
    s = int(s); return f"{s//60:02d}:{s%60:02d}"

def _whisper_fallback(url: str):
    # 1) Download best audio
    with tempfile.TemporaryDirectory() as td:
        outtmpl = os.path.join(td, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 20,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        audio_mp3 = os.path.join(td, "audio.mp3")
        audio_path = audio_mp3 if os.path.exists(audio_mp3) else outtmpl.replace("%(ext)s", "webm")

        # 2) Transcribe (CPU)
        model_name = os.environ.get("WHISPER_MODEL", "tiny.en")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path)

        items = []
        for seg in segments:
            start = float(seg.start or 0.0)
            end = float(seg.end or start)
            text = (seg.text or "").strip()
            if text:
                items.append({
                    "text": text,
                    "start": start,
                    "ts": _mmss(start),
                    "duration": max(0.0, end - start),
                })
        return items

@app.post("/transcript")
def get_transcript(body: TranscriptReq):
    vid = _extract_video_id(body.url)
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)

        # 1) Prefer English (manual or auto)
        try:
            t = transcripts.find_transcript(['en', 'en-US'])
            blocks = t.fetch()
            source = "en" if not t.is_generated else "en-auto"
        except NoTranscriptFound:
            # 2) Try translating any transcript to English
            blocks = None
            source = None
            for t in transcripts:
                try:
                    if getattr(t, "is_translatable", False):
                        t_en = t.translate('en')
                        blocks = t_en.fetch()
                        source = f"translated-from-{t.language}"
                        break
                except Exception:
                    continue
            if blocks is None:
                raise NoTranscriptFound

        items = [
            {"text": b.get("text",""), "start": float(b.get("start",0.0)),
             "ts": _mmss(float(b.get("start",0.0))), "duration": float(b.get("duration",0.0))}
            for b in blocks if b.get("text")
        ]
        return {"video_id": vid, "source": source, "items": items}

    except (NoTranscriptFound, TranscriptsDisabled, ParseError, RequestsConnectionError, Timeout):
        # Fallback to Whisper
        try:
            items = _whisper_fallback(body.url)
            if not items:
                raise HTTPException(status_code=404, detail="No transcript available and Whisper produced no text.")
            return {"video_id": vid, "source": "whisper", "items": items}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper fallback error: {type(e).__name__}")


    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable or restricted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript error: {type(e).__name__}")
    

class AnalyzeReq(BaseModel):
    url: str
    title_hint: str | None = None

@app.post("/analyze")
def analyze(body: AnalyzeReq):
    # get transcript (this should already handle captions + Whisper fallback)
    try:
        tr = get_transcript(TranscriptReq(url=body.url))
    except HTTPException as e:
        # bubble up transcript errors (e.g., private/blocked video)
        raise e

    # join with timestamps to help the model anchor claims
    lines = [f"[{it['ts']}] {it['text']}" for it in tr["items"]]
    joined = "\n".join(lines)
    if len(joined) > 15000:
        joined = joined[:15000] + "\n...[truncated]"

    # require model key at runtime
    if client is None:
        raise HTTPException(status_code=501, detail="Server not configured: set OPENAI_API_KEY.")

    system = (
        "You are an expert debate analyst. Be concise, neutral, and structured. "
        "Extract the main thesis and key arguments with brief evidence (use the supplied timestamps). "
        "For each key point, add 1–2 popular counterarguments. "
        "List up to 3 notable academics tied to each topic."
    )
    user = (
        "Return a JSON object with this schema:\n"
        "{\n"
        '  "video_summary": {\n'
        '    "title": "<best guess>",\n'
        '    "main_thesis": "<1-3 sentences>",\n'
        '    "key_points": [\n'
        '      {\n'
        '        "point": "<short>",\n'
        '        "evidence_or_claims": [{"text":"<claim>","timestamp":"mm:ss"}],\n'
        '        "counterarguments": [{"position":"<popular counter>","notes":"<why>"}],\n'
        '        "notable_academics": [{"name":"<person>","why_relevant":"<1 line>"}]\n'
        '      }\n'
        '    ],\n'
        '    "open_questions": []\n'
        "  }\n"
        "}\n\n"
        "--- TRANSCRIPT (with timestamps) ---\n"
        f"{joined}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=502, detail="LLM returned non-JSON")

    data.setdefault("video_summary", {})
    data["video_summary"]["source_url"] = body.url
    if body.title_hint and "title" not in data["video_summary"]:
        data["video_summary"]["title"] = body.title_hint
    return data


    

