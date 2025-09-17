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
    """
    Download best-audio with yt-dlp (optionally using cookies), find the produced file
    robustly (regardless of extension/name), then transcribe it with faster-whisper.
    Verbose logs can be enabled by setting env YTDLP_VERBOSE=1.
    """
    import base64, glob

    with tempfile.TemporaryDirectory() as td:
        outtmpl = os.path.join(td, "audio.%(ext)s")
        cookies_path = None

        # Optional: pass cookies via env to handle age-restricted / member videos
        b64 = os.environ.get("YTDLP_COOKIES_B64")
        if b64:
            try:
                cookies_path = os.path.join(td, "cookies.txt")
                with open(cookies_path, "wb") as f:
                    f.write(base64.b64decode(b64))
            except Exception as e:
                print("cookies decode failed:", type(e).__name__, flush=True)
                cookies_path = None  # ignore if decoding fails

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "paths": {"home": td, "temp": td},   # ensure all files go under td
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "continuedl": True,
            "nopart": True,
            "noplaylist": True,
            "geo_bypass": True,
            "socket_timeout": 20,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "http_headers": {
                "User-Agent": os.environ.get(
                    "YTDLP_UA",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            # Prefer a pure audio file we can feed to Whisper
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }],
        }
        if os.getenv("YTDLP_VERBOSE") == "1":
            ydl_opts["quiet"] = False
            ydl_opts["verbose"] = True
            ydl_opts["no_warnings"] = False
            print("YTDLP_VERBOSE=1 → enabling yt-dlp verbose logs", flush=True)

        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path

        # --- Download ---
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"yt-dlp error: {type(e).__name__}: {e}")

        # Some visibility to help diagnose in logs
        try:
            rd = info.get("requested_downloads") or []
        except Exception:
            rd = []
        print("Temp dir:", td, flush=True)
        print("yt-dlp: filepath:", info.get("filepath"),
              " _filename:", info.get("_filename"),
              " requested_downloads:", [d.get("filepath") for d in rd],
              flush=True)
        try:
            print("Temp dir listing:", os.listdir(td), flush=True)
        except Exception as _e:
            print("Listing temp dir failed:", type(_e).__name__, flush=True)

        # --- Figure out the audio file path (robust) ---
        audio_path = None
        candidates = []

        # 1) What yt-dlp reports directly
        for item in rd:
            fp = item.get("filepath")
            if fp:
                candidates.append(fp)
        for key in ("filepath", "_filename", "filename"):
            fp = info.get(key)
            if fp:
                candidates.append(fp)

        # Also check requested_formats (sometimes present)
        rf = info.get("requested_formats") or []
        for fmt in rf:
            fp = fmt.get("filepath")
            if fp:
                candidates.append(fp)

        # 2) Common names under our temp dir
        ext_list = ("mp3","m4a","webm","mp4","opus","wav","aac","ogg","mka","mkv","m4b")
        for ext in ext_list:
            candidates.append(os.path.join(td, f"audio.{ext}"))

        # 3) Fallback: scan the temp dir for any media-looking files (recursive)
        media_exts = tuple("." + e for e in ext_list)
        for p in glob.glob(os.path.join(td, "**", "*"), recursive=True):
            if os.path.isfile(p) and p.lower().endswith(media_exts):
                candidates.append(p)

        existing = [p for p in candidates if p and os.path.exists(p)]
        if not existing:
            try:
                listing = os.listdir(td)
            except Exception:
                listing = []
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp success but audio file not found; checked {len(candidates)} paths; tmp={listing}"
            )

        # Choose the largest existing file (safest bet)
        audio_path = max(existing, key=lambda p: os.path.getsize(p))
        print("Chosen audio_path:", audio_path, flush=True)

        # --- Transcribe (CPU) ---
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


    

