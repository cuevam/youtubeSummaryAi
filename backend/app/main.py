from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json, os

import re, os, tempfile

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from xml.etree.ElementTree import ParseError

# Whisper fallback deps
from faster_whisper import WhisperModel
import yt_dlp

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

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
        model_name = os.environ.get("WHISPER_MODEL", "base")
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

    except (NoTranscriptFound, TranscriptsDisabled, ParseError):
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
    # reuse our transcript function
    tr = get_transcript(TranscriptReq(url=body.url))
    lines = [f"[{it['ts']}] {it['text']}" for it in tr["items"]]
    joined = "\n".join(lines)
    if len(joined) > 10000:
        joined = joined[:10000] + "\n...[truncated]"

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("video_summary", {})
    data["video_summary"]["source_url"] = body.url
    if body.title_hint and "title" not in data["video_summary"]:
        data["video_summary"]["title"] = body.title_hint
    return data

    

