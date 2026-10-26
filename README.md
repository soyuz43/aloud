# aloud
Text to Speech Web App

ReadAloud cleans pasted text or text from an image/PDF with an OpenRouter chat
model, then reads it aloud with HeyGen Voice Turbo by default. Gareth is the
default voice; browser speech remains available as a fallback or alternate.

## Setup

1. Create `readaloud/.env` from `readaloud/.env.example` and set
   `OPENROUTER_API_KEY` to your OpenRouter API key.
2. Install dependencies with `pip install -r readaloud/requirements.txt`.
3. Run `python readaloud/app.py` from the repository root.
4. Open `http://localhost:8765` in your browser.

The API key stays on the server and is used for both text cleanup and speech
generation. The speech endpoint fetches supported HeyGen voices from OpenRouter
model metadata; it has a built-in voice list if that metadata is unavailable.
Long scripts are split into smaller audio requests. If HeyGen fails, playback
automatically falls back to the selected browser voice.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `READALOUD_MODEL` | `google/gemini-2.5-flash` | Model used to clean and prepare text |
| `READALOUD_TTS_MODEL` | `heygen/heygen-voice-1-turbo:free` | OpenRouter text-to-speech model |
| `READALOUD_VOICE` | `gareth` | Default HeyGen voice |
| `PORT` | `8765` | Local web server port |

The HeyGen voice choices are Gareth, Nadine, Tabitha, Bram, Noelle, Meredith,
Sabrina, Lucian, Weston, and Vincent. The interface also includes a voice-test
button and browser speech controls.

The free HeyGen Turbo model is currently listed to expire on November 30, 2026.
If OpenRouter retires it or rate-limits requests, change `READALOUD_TTS_MODEL`
to `heygen/heygen-voice-1-turbo` (paid) in `readaloud/.env`.
