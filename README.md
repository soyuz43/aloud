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

## One audio file, not a queue of chunks

Long scripts are split, generated and stitched **server-side**, and the player
is only enabled once the whole file exists. There is no half-generated audio
and no second "generating" step when you skip ahead.

The speech model (HeyGen Voice Turbo) accepts at most 5,000 characters per
request and returns raw 16-bit PCM with no container, so a single request
cannot cover a long document. ReadAloud therefore:

- splits the script at **paragraph boundaries** first (falling back to
  sentence boundaries), so a stitch never lands in the middle of a passage;
- generates those parts in parallel;
- trims the ~300 ms of lead-in and ~200 ms of trailing silence each part
  returns, and inserts a ~450 ms pause in its place - calibrated against the
  model's own inter-sentence pauses, so a seam reads as a paragraph break;
- level-matches every part to the same RMS so the volume does not jump;
- fades the edges so a cut cannot click.

The result is a single downloadable `.wav`.

Browser speech is still available as a legacy engine: it streams sentence by
sentence and needs no server round trip.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `READALOUD_MODEL` | `google/gemini-2.5-flash` | Model used to clean and prepare text |
| `READALOUD_TTS_MODEL` | `heygen/heygen-voice-1-turbo:free` | OpenRouter text-to-speech model |
| `READALOUD_VOICE` | `gareth` | Default HeyGen voice |
| `READALOUD_MAX_OUTPUT_TOKENS` | `65535` | Upper bound on text-prep output length |
| `PORT` | `8765` | Local web server port |

The HeyGen voice choices are Gareth, Nadine, Tabitha, Bram, Noelle, Meredith,
Sabrina, Lucian, Weston, and Vincent. The interface also includes a
voice-test button and browser speech controls.

The free HeyGen Turbo model is currently listed to expire on November 30, 2026.
If OpenRouter retires it or rate-limits requests, change `READALOUD_TTS_MODEL`
to `heygen/heygen-voice-1-turbo` (paid) in `readaloud/.env`.

## Limits

- Speech: 40,000 characters per script (`TTS_MAX_TOTAL_CHARS`), split into
  5,000-character requests.
- Text preparation: the cleanup model is asked for an output budget scaled to
  the input size. If it still truncates, the pasted text is cleaned segment by
  segment instead so nothing is silently dropped; the response flags this with
  `truncated`.
- Uploads: 8 MB total across attached files.
