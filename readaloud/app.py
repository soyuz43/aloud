#!/usr/bin/env python3
"""
ReadAloud - paste text, have it cleaned up and read back to you.

Flow:
  1. You paste text (or a photo/PDF via "Attach file").
  2. The app sends it to your OpenRouter text model (vision-capable models can
     read images/PDFs directly) with instructions to produce clean,
     speak-friendly prose.
  3. The cleaned script is synthesised by the configured speech model and
     handed to the browser as ONE audio file.

The default speech model (HeyGen Voice Turbo) only accepts up to 5,000
characters per request and only returns raw 16-bit PCM - there is no
single-file response to ask for.  Long scripts are therefore split
server-side, generated in parallel and stitched into a single WAV file before
the player is enabled, so the user never sees a half-generated audio.

Config (env vars or .env next to this file):
  OPENROUTER_API_KEY   required
  READALOUD_MODEL      default: google/gemini-2.5-flash
  READALOUD_TTS_MODEL  default: heygen/heygen-voice-1-turbo:free
  READALOUD_VOICE      default: gareth
  PORT                 default: 8765

Run:  python app.py        then open  http://localhost:8765
"""

import array
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import wave

from collections import deque
from concurrent.futures import ThreadPoolExecutor

try:
    # audioop is deprecated and removed in Python 3.13, but it is a C
    # implementation and much faster than the pure-python fallback.
    import audioop
except ImportError:  # pragma: no cover - depends on the interpreter
    audioop = None

from flask import Flask, jsonify, render_template, request, send_file

try:
    import requests
except ImportError:
    print("Missing dependency: pip install flask requests", file=sys.stderr)
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Tiny .env loader (so you don't need python-dotenv)
# --------------------------------------------------------------------------
def load_dotenv(path):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.expanduser("~/.secrets.local"))  # your existing secrets file

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("READALOUD_MODEL", "google/gemini-2.5-flash")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TTS_MODEL = os.environ.get(
    "READALOUD_TTS_MODEL", "heygen/heygen-voice-1-turbo:free")
TTS_VOICE = os.environ.get("READALOUD_VOICE", "gareth").strip().lower()
TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models?output_modalities=speech"

# Hard limit enforced by the HeyGen speech API.  Verified against the live
# endpoint: 5,000 characters is accepted, 5,001 is rejected.
TTS_MAX_CHARS_PER_REQUEST = 5000
# Upper bound for a single stitched file.  5,000 chars of speech is roughly
# four minutes of 44.1 kHz 16-bit mono audio, so this caps a download at
# roughly 160 MB / ~32 minutes of audio.
TTS_MAX_TOTAL_CHARS = 40000
# How many speech requests may be in flight at once.
TTS_CONCURRENCY = 4
# Per-request timeout for a single speech chunk.
TTS_TIMEOUT = 180
# Retries for a failed speech chunk (network error, 429, 5xx).
TTS_RETRIES = 3
# Pause inserted between stitched chunks.  Calibrated against the model's own
# inter-sentence pauses (measured 120-530 ms, typically 400-530 ms) so a seam
# reads as a paragraph break rather than a splice.
TTS_CHUNK_GAP_MS = 450
# Short fade at each chunk edge removes the click a hard cut would produce.
TTS_FADE_MS = 12
# A little of the original silence is kept so onsets do not feel clipped.
TTS_EDGE_MARGIN_MS = 20
# Per-model request size limits, in characters.
TTS_MODEL_CHAR_LIMITS = {
    "heygen/heygen-voice-1-turbo": 5000,
    "heygen/heygen-voice-1-turbo:free": 5000,
    "heygen/heygen-voice-1": 5000,
}

# Max output tokens we are willing to request from the text-cleanup model.
MODEL_MAX_OUTPUT_TOKENS = int(
    os.environ.get("READALOUD_MAX_OUTPUT_TOKENS", "65535"))
# When the cleaned script would not fit in one model response we clean it
# segment by segment instead so nothing is silently dropped.
SEGMENT_CHARS = 24000

DEFAULT_TTS_VOICES = [
    "nadine", "tabitha", "bram", "gareth", "noelle",
    "meredith", "sabrina", "lucian", "weston", "vincent",
]

SYSTEM_PROMPT = """\
You are ReadAloud, a speech preparation assistant. The user gives you text
(or an image/PDF of text). Produce ONLY the words that should be spoken aloud:

- Strip markdown symbols, URLs (say nothing or describe the link briefly),
  code fences (summarize what the code does in one sentence instead),
  page numbers, boilerplate.
- Fix obvious OCR errors.
- Convert abbreviations, numbers, dates, times and symbols into their spoken
  form ("Dr." -> "Doctor", "9/11" -> "September eleventh", "$4.50" ->
  "four dollars and fifty cents", "km/h" -> "kilometers per hour").
- Keep the meaning and order intact. Do NOT summarize unless asked.
- Never add greetings, preambles like "Sure, here it is", or commentary.
- If the user typed an instruction alongside the text (e.g. "summarize this"),
  follow it, but still output only speakable text.
- PRESERVE THE PARAGRAPH BREAKS of the original text.  Keep each paragraph
  as its own block separated by a blank line.  Paragraph breaks are where the
  reader pauses, so they must survive into the spoken script.  Do not merge
  paragraphs together and do not add new ones.
"""

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))


class TtsError(Exception):
    """Raised when a speech chunk could not be synthesised."""


# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------
def build_messages(text, files):
    """Build OpenRouter messages; attach images/PDFs as data URLs."""
    content = []

    if text.strip():
        content.append({"type": "text", "text": text})

    for _name, data in files:
        mime = sniff_mime(data)
        if mime == "application/pdf":
            content.append({
                "type": "file",
                "file": {"filename": "upload.pdf", "data": data_url(mime, data)},
            })
        elif mime.startswith("image/"):
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url(mime, data)},
            })
        else:
            # Unknown binary - try to treat as text, else skip.
            try:
                decoded = data.decode("utf-8")
                content.append({"type": "text", "text": decoded})
            except UnicodeDecodeError:
                continue

    if not content:
        return None

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def sniff_mime(data):
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "text/plain"


def data_url(mime, data):
    import base64
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# --------------------------------------------------------------------------
# Text -> speech
# --------------------------------------------------------------------------
def max_chars_per_request():
    """Character limit for the configured speech model."""
    return TTS_MODEL_CHAR_LIMITS.get(TTS_MODEL, TTS_MAX_CHARS_PER_REQUEST)


# --------------------------------------------------------------------------
# Chunking: break on paragraph boundaries so a stitch never lands mid-thought
# --------------------------------------------------------------------------
# First-priority seam: a blank line (paragraph break).  Soft single
# newlines inside a paragraph are NOT seams - they stay together so a stitch
# never lands mid-paragraph.  Line breaks are the second-priority seam, used
# only when a paragraph itself is longer than the request limit.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_LINE_SPLIT = re.compile(r"\n+")
_SENTENCE_SPLIT = re.compile(r"[.!?\u2026][\"')\]]*(?=\s|$)")


def split_for_speech(text, limit=None):
    """Split text into chunks of at most `limit` characters.

    Seams are chosen by the following hierarchy, so a stitch lands where a
    human would naturally pause and never in the middle of a passage:

    1. A blank line (paragraph break) - the most natural pause.
    2. A single newline, but only when a whole paragraph exceeds `limit`;
       soft wraps inside a paragraph stay together.
    3. A sentence end, when a single line still exceeds `limit`.
    4. A word boundary, as a last resort for a single over-long sentence.

    A token longer than `limit` is split mid-word only as a final fallback
    (see `_split_words`); this is the one case where a word can be cut.
    """
    if limit is None:
        limit = max_chars_per_request()

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = (current + " " + paragraph).strip() if current else paragraph

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # The paragraph on its own may still be too long.
        for piece in _split_paragraph(paragraph, limit):
            if current and len(current) + 1 + len(piece) <= limit:
                current = (current + " " + piece).strip()
            else:
                if current:
                    chunks.append(current)
                current = piece

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]


def _split_paragraph(paragraph, limit):
    """Break an over-long paragraph into pieces no longer than `limit`.

    The seam hierarchy is preserved inside the paragraph: keep whole lines
    together where they fit, then fall back to sentence boundaries, then to
    word boundaries (see `_split_sentences` and `_split_words`).
    """
    lines = [ln.strip() for ln in _LINE_SPLIT.split(paragraph) if ln.strip()]
    pieces = []
    current = ""
    for line in lines:
        candidate = (current + " " + line).strip() if current else line
        if len(candidate) <= limit:
            current = candidate
            continue

        # Flush what we have, then split the over-long line further.
        if current:
            pieces.append(current)
            current = ""

        if len(line) <= limit:
            current = line
            continue

        for sentence_piece in _split_sentences(line, limit):
            if current and len(current) + 1 + len(sentence_piece) <= limit:
                current = (current + " " + sentence_piece).strip()
            else:
                if current:
                    pieces.append(current)
                current = sentence_piece

    if current:
        pieces.append(current)
    return pieces


def _split_sentences(text, limit):
    """Split `text` on sentence ends, falling back to words per sentence."""
    sentences = []
    position = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        sentences.append(text[position:match.end()])
        position = match.end()
    if position < len(text):
        sentences.append(text[position:])

    pieces = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            # A single sentence longer than the limit: fall back to words.
            for word_piece in _split_words(sentence, limit):
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(word_piece)
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        pieces.append(current)
    return pieces


def _split_words(text, limit):
    """Last-resort split on word boundaries for pathologically long words."""
    words = text.split()
    pieces = []
    current = ""
    for word in words:
        while len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        if current and len(current) + 1 + len(word) > limit:
            pieces.append(current)
            current = word
        else:
            current = (current + " " + word).strip() if current else word
    if current:
        pieces.append(current)
    return pieces


# --------------------------------------------------------------------------
# Audio conditioning: trim edges, match levels, fade seams
# --------------------------------------------------------------------------
def _samples(pcm):
    """PCM bytes -> array of signed 16-bit samples."""
    usable = len(pcm) - (len(pcm) % 2)
    data = array.array("h")
    data.frombytes(pcm[:usable])
    if data.itemsize != 2:
        data = array.array("h", data)
    return data


def _trim_silence(samples, rate, threshold=400, margin_ms=TTS_EDGE_MARGIN_MS):
    """Remove leading/trailing silence, keeping a small natural margin.

    Every speech chunk comes back with roughly 300 ms of silence at the head
    and 200 ms at the tail.  Left in place, two chunks joined together get
    a dead half-second gap and an abrupt onset, which is what makes a
    stitched file sound chopped up.
    """
    total = len(samples)
    if total == 0:
        return samples

    margin = max(0, int(rate * margin_ms / 1000))

    start = 0
    while start < total and abs(samples[start]) < threshold:
        start += 1
    if start >= total:
        return samples  # all silence, leave as is

    end = total - 1
    while end > start and abs(samples[end]) < threshold:
        end -= 1
    end += 1

    start = max(0, start - margin)
    end = min(total, end + margin)
    return samples[start:end]


def _rms(samples):
    if not samples:
        return 0.0
    if audioop is not None:
        return float(audioop.rms(samples.tobytes(), 2))
    total = 0
    for value in samples:
        total += value * value
    return math.sqrt(total / len(samples))


def _apply_gain(samples, gain):
    """Scale samples by `gain`, clamping to the 16-bit range."""
    if abs(gain - 1.0) < 0.005:
        return samples
    if audioop is not None:
        # audioop saturates at the 16-bit bounds, same as the fallback below.
        out = array.array("h")
        out.frombytes(audioop.mul(samples.tobytes(), 2, gain))
        return out
    out = array.array("h", bytes(len(samples) * 2))
    limit = 32767
    for index, value in enumerate(samples):
        scaled = int(value * gain)
        if scaled > limit:
            scaled = limit
        elif scaled < -limit - 1:
            scaled = -limit - 1
        out[index] = scaled
    return out


def _peak(samples):
    """Largest absolute sample value."""
    if not samples:
        return 0
    peak = 0
    for value in samples:
        magnitude = -value if value < 0 else value
        if magnitude > peak:
            peak = magnitude
    return peak


def _safe_gain(samples, desired):
    """Loudness gain that cannot push the chunk into clipping.

    Speech chunks arrive close to full scale, so applying the raw ratio can
    hard-clip the peaks and sound harsh.  The gain is reduced until the
    loudest sample stays just below the 16-bit ceiling.
    """
    peak = _peak(samples)
    if peak <= 0:
        return 1.0
    headroom = (32700.0 / peak)
    gain = min(desired, headroom)
    # Never pull a chunk down further than 0.6, that would sound wrong.
    return max(0.6, min(1.6, gain))


def _fade(samples, rate, fade_ms=TTS_FADE_MS):
    """Fade in at the start and out at the end to avoid clicks."""
    total = len(samples)
    count = min(int(rate * fade_ms / 1000), total // 2)
    if count <= 0:
        return samples
    out = array.array("h", samples)
    for i in range(count):
        factor = i / float(count)
        out[i] = int(out[i] * factor)
        out[total - 1 - i] = int(out[total - 1 - i] * factor)
    return out


def _silence(rate, channels, milliseconds=TTS_CHUNK_GAP_MS):
    frames = int(rate * milliseconds / 1000)
    return b"\x00" * (frames * channels * 2)


def _condition_chunk(pcm, rate, channels, reference_rms):
    """Trim, level-match and fade one chunk of speech.

    `reference_rms` is the loudness of the chunks already stitched in; pass
    None for the first chunk to establish the reference.
    """
    samples = _samples(pcm)
    samples = _trim_silence(samples, rate)
    if not samples:
        return pcm, reference_rms

    current_rms = _rms(samples)
    if reference_rms is None or reference_rms <= 0 or current_rms <= 0:
        # The first chunk sets the loudness reference for the whole file.
        reference_rms = current_rms
    else:
        gain = _safe_gain(samples, reference_rms / current_rms)
        samples = _apply_gain(samples, gain)

    # Fade the edges so a hard cut cannot produce a click.
    samples = _fade(samples, rate)
    return bytes(samples.tobytes()), reference_rms


def _pcm_params(content_type):
    """Read sample rate / channel count from the speech endpoint's headers."""
    rate, channels = 44100, 1
    if content_type:
        for part in content_type.split(";")[1:]:
            key, _, value = part.partition("=")
            key = key.strip().lower()
            value = value.strip()
            try:
                if key == "rate":
                    rate = int(value)
                elif key == "channels":
                    channels = int(value)
            except ValueError:
                continue
    return rate, channels


def _request_pcm(text, voice):
    """Ask the speech model for one chunk of audio.  Returns raw PCM bytes."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ReadAloud",
    }
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
    }

    last_error = ""
    for attempt in range(TTS_RETRIES):
        try:
            response = requests.post(
                TTS_ENDPOINT, headers=headers, json=payload, timeout=TTS_TIMEOUT)
        except requests.RequestException as exc:
            last_error = f"Network error talking to OpenRouter: {exc}"
            time.sleep(2 * (attempt + 1))
            continue

        if response.status_code == 200:
            pcm = response.content
            if not pcm:
                last_error = "OpenRouter returned an empty audio chunk."
                time.sleep(1)
                continue
            rate, channels = _pcm_params(response.headers.get("content-type"))
            return pcm, rate, channels

        try:
            detail = response.json().get("error", {}).get(
                "message", response.text[:300])
        except (ValueError, AttributeError):
            detail = response.text[:300]
        last_error = f"OpenRouter {response.status_code}: {detail}"

        # 4xx other than rate limiting will not get better on a retry.
        if response.status_code < 500 and response.status_code != 429:
            break
        time.sleep(2 * (attempt + 1))

    raise TtsError(last_error or "Unknown speech synthesis error.")


# --------------------------------------------------------------------------
# Speech jobs: chunk -> parallel generate -> stitch -> single WAV
# --------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 3600


def _sweep_jobs():
    """Remove jobs (and their temp WAV files) older than JOB_TTL_SECONDS.

    Runs eagerly on each POST /api/tts submission - there is intentionally no
    background timer, so a job and its temp file persist until the next
    generation request, at which point expired entries are culled.
    """
    now = time.time()
    with JOBS_LOCK:
        expired = [job_id for job_id, job in JOBS.items()
                   if now - job["created"] > JOB_TTL_SECONDS]
        for job_id in expired:
            job = JOBS.pop(job_id)
            with job["lock"]:
                path = job.get("path")
            _remove_file(path)


def _remove_file(path):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _run_job(job):
    """Generate every chunk (in parallel) and stitch them into one WAV."""
    chunks = job["chunks"]
    handle, path = tempfile.mkstemp(prefix="readaloud-", suffix=".wav")
    os.close(handle)
    job["path"] = path

    wav = None
    bytes_written = 0
    reference_rms = None
    try:
        with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as pool:
            pending = deque()
            next_index = 0
            while next_index < len(chunks) and len(pending) < TTS_CONCURRENCY:
                pending.append((next_index, pool.submit(
                    _request_pcm, chunks[next_index], job["voice"])))
                next_index += 1

            # Chunks are generated in parallel but must be written in order,
            # so finished audio is held until every earlier chunk is ready.
            ready = {}

            while pending:
                index, future = pending.popleft()
                pcm, rate, channels = future.result()
                with job["lock"]:
                    job["completed"] = min(job["completed"] + 1, len(chunks))

                if wav is None:
                    wav = wave.open(path, "wb")
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    job["rate"] = rate
                    job["channels"] = channels
                elif channels != job["channels"] or rate != job["rate"]:
                    raise TtsError(
                        "The speech model returned inconsistent audio formats.")

                ready[index] = (pcm, rate, channels)
                if next_index < len(chunks):
                    pending.append((next_index, pool.submit(
                        _request_pcm, chunks[next_index], job["voice"])))
                    next_index += 1

                # Write out every chunk that is now contiguous and ready.
                # Guarded by the per-job lock so a progress poll cannot observe
                # job["times"] while it is being mutated (which would raise
                # "dictionary changed size during iteration").
                with job["lock"]:
                    while job["done"] in ready:
                        chunk_pcm, chunk_rate, chunk_channels = ready.pop(job["done"])
                        conditioned, reference_rms = _condition_chunk(
                            chunk_pcm, chunk_rate, chunk_channels, reference_rms)
                        del chunk_pcm

                        if job["done"] > 0:
                            # A pause in the model's own voice, so the seam reads
                            # as a paragraph break rather than a splice.
                            gap = _silence(job["rate"], job["channels"])
                            wav.writeframes(gap)
                            bytes_written += len(gap)

                        # Start time of this chunk, used by the skip controls.
                        job["times"][job["done"]] = (
                            bytes_written / float(job["rate"] * job["channels"] * 2))

                        wav.writeframes(conditioned)
                        bytes_written += len(conditioned)
                        job["done"] += 1

        if wav is None:
            raise TtsError("No audio was generated.")
        wav.close()
        wav = None

        with job["lock"]:
            job["bytes"] = bytes_written
            job["state"] = "done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the client verbatim
        if wav is not None:
            try:
                wav.close()
            except Exception:
                pass
        with job["lock"]:
            job["state"] = "error"
            job["error"] = str(exc)
            job["path"] = None
        _remove_file(path)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        model=MODEL,
        tts_model=get_model_name(TTS_MODEL, TTS_MODEL),
        has_key=bool(API_KEY),
    )


_MODELS_CACHE = {"at": 0.0, "models": []}
_MODELS_LOCK = threading.Lock()


def get_speech_models():
    """Fetch speech-capable model metadata (cached briefly)."""
    with _MODELS_LOCK:
        if time.time() - _MODELS_CACHE["at"] < 300 and _MODELS_CACHE["models"]:
            return _MODELS_CACHE["models"]

    models = []
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=10)
        response.raise_for_status()
        metadata = response.json()
        raw = metadata.get("data", []) if isinstance(metadata, dict) else []
        models = [item for item in raw if isinstance(item, dict)]
    except (requests.RequestException, ValueError):
        models = []

    with _MODELS_LOCK:
        _MODELS_CACHE["at"] = time.time()
        _MODELS_CACHE["models"] = models
    return models


def get_tts_voices():
    for model in get_speech_models():
        if model.get("id") == TTS_MODEL:
            voices = model.get("supported_voices", [])
            if isinstance(voices, list) and voices:
                return [voice for voice in voices if isinstance(voice, str)]
    return DEFAULT_TTS_VOICES


def get_model_name(model_id, fallback=None):
    for model in get_speech_models():
        if model.get("id") == model_id:
            return model.get("name") or fallback or model_id
    return fallback or model_id


@app.route("/api/tts-config")
def tts_config():
    voices = get_tts_voices()
    default_voice = TTS_VOICE if TTS_VOICE in voices else "gareth"
    if default_voice not in voices and voices:
        default_voice = voices[0]
    return jsonify(
        model=TTS_MODEL,
        model_name=get_model_name(TTS_MODEL, TTS_MODEL),
        voice=default_voice,
        voices=voices,
        max_chars_per_request=TTS_MAX_CHARS_PER_REQUEST,
        max_total_chars=TTS_MAX_TOTAL_CHARS,
    )


@app.route("/api/tts", methods=["POST"])
def start_tts():
    """Start a speech job.  Returns immediately; poll for progress."""
    if not API_KEY:
        return jsonify(error="OPENROUTER_API_KEY is not set. "
                             "Put it in .env or ~/.secrets.local."), 400

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify(error="A JSON object is required."), 400

    text = body.get("text", "")
    voice = body.get("voice", TTS_VOICE)
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Text to speak is required."), 400
    text = text.strip()
    if len(text) > TTS_MAX_TOTAL_CHARS:
        return jsonify(
            error=(f"Text is {len(text)} characters; the limit is "
                   f"{TTS_MAX_TOTAL_CHARS}. Split it into smaller parts.")), 400
    if (not isinstance(voice, str) or not voice or len(voice) > 64
            or not voice.replace("-", "").replace("_", "").isalnum()):
        return jsonify(error="Invalid voice name."), 400

    _sweep_jobs()
    chunks = split_for_speech(text)
    job = {
        "id": uuid.uuid4().hex,
        "lock": threading.Lock(),
        "state": "running",
        "done": 0,
        "completed": 0,
        "total": len(chunks),
        "error": None,
        "path": None,
        "bytes": 0,
        "rate": 44100,
        "channels": 1,
        "voice": voice,
        "chunks": chunks,
        "times": {},
        "created": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job["id"]] = job

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return jsonify(job_id=job["id"], chunks=len(chunks)), 202


@app.route("/api/tts/jobs/<job_id>")
def tts_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job."), 404
    # Snapshot the mutable job fields under its own lock so a worker update
    # (job["times"] / job["done"] / state) can never race with this read.
    with job["lock"]:
        rate = job["rate"] or 44100
        channels = job["channels"] or 1
        duration = 0.0
        if job["state"] == "done":
            duration = round(job["bytes"] / float(rate * channels * 2), 1)
        times = [job["times"][i] for i in sorted(job["times"])]
        payload = dict(
            state=job["state"],
            done=job["done"],
            completed=job["completed"],
            total=job["total"],
            error=job["error"],
            ready=job["state"] == "done",
            bytes=job["bytes"],
            duration=duration,
            rate=rate,
            channels=channels,
            times=times,
        )
    return jsonify(payload)


@app.route("/api/tts/jobs/<job_id>/audio")
def tts_job_audio(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job."), 404
    with job["lock"]:
        state = job["state"]
        path = job["path"]
        error = job["error"]
    if state == "error":
        return jsonify(error=error or "Speech generation failed."), 502
    if state != "done" or not path:
        return jsonify(error="Audio is not ready yet."), 409
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.route("/api/speak", methods=["POST"])
def speak():
    if not API_KEY:
        return jsonify(error="OPENROUTER_API_KEY is not set. "
                             "Put it in .env or ~/.secrets.local."), 400

    text = request.form.get("text", "")
    files = []
    for key in request.files:
        f = request.files[key]
        blob = f.read()
        if blob:
            files.append((f.filename, blob))

    max_bytes = 8 * 1024 * 1024
    if sum(len(b) for _, b in files) > max_bytes:
        return jsonify(error="Attached files exceed 8 MB total."), 400

    messages = build_messages(text, files)
    if messages is None:
        return jsonify(error="Nothing to read - paste text or attach a file."), 400

    script, truncated = clean_script(messages, text)

    return jsonify(
        script=script,
        model=MODEL,
        truncated=truncated,
        chars=len(script),
    )


def clean_script(messages, text):
    """Run the cleanup model, guaranteeing the whole input is covered.

    Returns (script, truncated).  If the model stops because it hit the output
    limit we clean the pasted text segment by segment so nothing is silently
    dropped.  Segmenting only applies to pasted text - attached images and
    PDFs are always handled by the model as a single unit.
    """
    budget = _output_token_budget(text)
    script, truncated = _chat_cleanup(messages, budget)
    if not truncated:
        return script, False

    source = text.strip()
    if not source:
        # The document came from an attachment; there is no text to segment.
        return script, True

    parts = []
    for segment in _split_into_segments(source, SEGMENT_CHARS):
        segment_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": segment}]},
        ]
        segment_script, segment_truncated = _chat_cleanup(
            segment_messages, _output_token_budget(segment))
        if segment_truncated:
            return segment_script, True
        parts.append(segment_script)

    return "\n\n".join(part for part in parts if part), False


def _output_token_budget(text):
    """Estimate a generous output-token budget for the cleaned script."""
    estimated_input = len(text)
    # Cleaned speech is normally a little shorter than the raw input, but
    # number expansion can grow it, so budget generously.
    budget = int(estimated_input / 2.5) + 2000
    return max(4096, min(MODEL_MAX_OUTPUT_TOKENS, budget))


def _lower_token_budget(max_tokens):
    """Next budget to try when a provider rejects our max_tokens request."""
    if max_tokens <= 8192:
        return 0
    return max(4096, max_tokens // 2)


def _split_into_segments(text, limit):
    segments = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            segments.append(remaining)
            break
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut <= limit // 2:
            cut = window.rfind("\n")
        if cut <= limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        segment = remaining[:cut].strip()
        if segment:
            segments.append(segment)
        remaining = remaining[cut:].strip()
    return segments


def _chat_cleanup(messages, max_tokens):
    """One cleanup call.  Returns (text, truncated)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ReadAloud",
    }

    try:
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=600)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error talking to OpenRouter: {exc}")

    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:300])
        except ValueError:
            detail = resp.text[:300]
        # Some providers cap max_tokens lower than we asked for.  Retry once
        # with a smaller budget before giving up.
        lowered = _lower_token_budget(max_tokens)
        if resp.status_code == 400 and lowered and "token" in detail.lower():
            return _chat_cleanup(messages, lowered)
        raise RuntimeError(f"OpenRouter {resp.status_code}: {detail}")

    body = resp.json()
    try:
        choice = body["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Unexpected response shape from OpenRouter.")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict))

    truncated = choice.get("finish_reason") in ("length", "max_tokens")
    return (content or "").strip(), truncated


@app.route("/api/health")
def health():
    return jsonify(ok=True, model=MODEL, tts_model=TTS_MODEL, has_key=bool(API_KEY))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    if not API_KEY:
        print("WARNING: OPENROUTER_API_KEY not found in environment or .env",
              file=sys.stderr)
    print(f"ReadAloud running at http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
