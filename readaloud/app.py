#!/usr/bin/env python3
"""
ReadAloud - paste text, have it cleaned up and read back to you.

Flow:
  1. You paste text (or a photo/PDF via "Attach file").
  2. The app sends it to your OpenRouter model (vision-capable models can
     read images/PDFs directly) with instructions to produce clean,
     speak-friendly prose.
  3. The reply is spoken in the browser with the Web Speech API
     (no extra install, works offline for the voice part).

Config (env vars or .env next to this file):
  OPENROUTER_API_KEY   required
  READALOUD_MODEL      default: google/gemini-2.5-flash   (free endpoint
                       example: google/gemini-2.0-flash-exp:free)
  PORT                 default: 8765

Run:  python app.py        then open  http://localhost:8765
"""

import io
import json
import os
import sys
import wave

from flask import Flask, Response, jsonify, render_template, request

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
TTS_MAX_CHARACTERS = 5000
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
"""

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))


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


@app.route("/")
def index():
    return render_template("index.html", model=MODEL, has_key=bool(API_KEY))


def get_tts_voices():
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=10)
        response.raise_for_status()
        metadata = response.json()
        models = metadata.get("data", []) if isinstance(metadata, dict) else []
        model = next((item for item in models if item.get("id") == TTS_MODEL), {})
        voices = model.get("supported_voices", [])
        if isinstance(voices, list) and voices:
            return [voice for voice in voices if isinstance(voice, str)]
    except (requests.RequestException, ValueError):
        pass
    return DEFAULT_TTS_VOICES


@app.route("/api/tts-config")
def tts_config():
    voices = get_tts_voices()
    default_voice = TTS_VOICE if TTS_VOICE in voices else "gareth"
    if default_voice not in voices and voices:
        default_voice = voices[0]
    return jsonify(model=TTS_MODEL, voice=default_voice, voices=voices)


@app.route("/api/tts", methods=["POST"])
def text_to_speech():
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
    if len(text) > TTS_MAX_CHARACTERS:
        return jsonify(
            error=f"Text must be {TTS_MAX_CHARACTERS} characters or fewer."), 400
    if (not isinstance(voice, str) or not voice or len(voice) > 64
            or not voice.replace("-", "").replace("_", "").isalnum()):
        return jsonify(error="Invalid voice name."), 400

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

    try:
        response = requests.post(
            TTS_ENDPOINT, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        return jsonify(error=f"Network error talking to OpenRouter: {exc}"), 502

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get(
                "message", response.text[:300])
        except (ValueError, AttributeError):
            detail = response.text[:300]
        return (
            jsonify(error=f"OpenRouter {response.status_code}: {detail}"),
            response.status_code,
        )

    pcm = response.content
    if not pcm or len(pcm) % 2:
        return jsonify(error="OpenRouter returned invalid PCM audio."), 502

    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(pcm)
    return Response(
        audio.getvalue(), mimetype="audio/wav",
        headers={"Cache-Control": "no-store"})


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

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ReadAloud",
    }

    try:
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        return jsonify(error=f"Network error talking to OpenRouter: {exc}"), 502

    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:300])
        except ValueError:
            detail = resp.text[:300]
        return jsonify(error=f"OpenRouter {resp.status_code}: {detail}"), 502

    body = resp.json()
    try:
        reply = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return jsonify(error="Unexpected response shape from OpenRouter."), 502

    return jsonify(script=reply, model=body.get("model", MODEL))


@app.route("/api/health")
def health():
    return jsonify(ok=True, model=MODEL, has_key=bool(API_KEY))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    if not API_KEY:
        print("WARNING: OPENROUTER_API_KEY not found in environment or .env",
              file=sys.stderr)
    print(f"ReadAloud running at http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
