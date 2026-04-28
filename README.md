# JARVIS — Real-Time Voice AI Agent

A fully working real-time voice AI agent built in Python. Speak → JARVIS transcribes, reasons, and replies aloud. End-to-end voice loop with no typing required.

**Status:** ✅ Complete  
**Live Demo:** [voiceagent-gflmqk26hgjsm6szsynrsp.streamlit.app](https://voiceagent-gflmqk26hgjsm6szsynrsp.streamlit.app/)  
**Platform:** Windows 11 / Python 3.12  
**Repo:** `Aakarshkumar612/Voice_Agent` · branch `main`  
**Last updated:** 2026-04-28

---

## What It Does

- **Real-time voice conversation** — speak naturally, JARVIS listens, transcribes, reasons, and replies aloud
- **Demo booking assistant** — persona-configured to handle demo scheduling: asks for name + date, checks slot availability, confirms bookings
- **Echo-free** — after every TTS reply, mic buffer is flushed (400ms decay + drain) so JARVIS never hears itself
- **Noise-gated VAD** — only sends audio to Whisper when ≥5 consecutive frames (~320ms) exceed RMS 800, eliminating hallucinations from ambient noise
- **Full context memory** — up to 40 messages (20 turns) maintained across the session
- **Live web UI** — Streamlit dashboard with transcript, status indicator, and session stats, auto-refreshing every 0.7s

---

## Architecture

Three threads run concurrently, communicating through thread-safe primitives only:

| Thread | Role |
|--------|------|
| **Streamlit main** | Renders UI, drains `_transcript_q`, reruns every 0.7s |
| **Agent daemon** | Private `asyncio` loop — VAD → STT → LLM → TTS → echo flush |
| **PyAudio capture** | Always running — reads mic 16kHz PCM → `input_queue` |

**Thread-safe bridge** (both at module scope in `app.py`):
- `_transcript_q` — `queue.Queue` — agent pushes `(role, text)` tuples; Streamlit drains on every rerun
- `_status` — `dict + threading.Lock` — agent writes current state string; Streamlit reads for status badge

### Voice Pipeline — One Conversation Turn

```
① User speaks
   ↓
② PyAudio capture thread reads CHUNK_SIZE=1024 frames (~64ms each) at 16kHz mono Int16
   ↓
③ input_queue.put(chunk)  [runs continuously, even during TTS]
   ↓
④ VAD gate: audioop.rms(chunk, 2) > 800 for ≥5 consecutive frames (~320ms of real speech)
   ↓
⑤ Utterance ends: 25 consecutive silence frames (~1.6s below threshold)
   ↓
⑥ PCM bytes → WAV container (io.BytesIO + wave, 16kHz mono Int16, .name="audio.wav")
   ↓
⑦ groq.audio.transcriptions.create(model="whisper-large-v3", response_format="text") → plain string
   ↓
⑧ messages[].append({"role": "user", "content": text})
   ↓
⑨ groq.chat.completions.create(model="llama-3.3-70b-versatile", messages, max_tokens=256, temperature=0.7)
   ↓
⑩ reply = response.choices[0].message.content
   ↓
⑪ pyttsx3.init() → engine.setProperty(rate=165, volume=1.0) → engine.say(reply) → runAndWait() → stop()
   ↓
⑫ time.sleep(0.4) → drain input_queue (echo flush — prevents JARVIS hearing itself)
   ↓
⑬ Back to ④
```

---

## Tech Stack

| Layer | Technology | Version | Details |
|-------|-----------|---------|---------|
| STT | Groq Whisper | `whisper-large-v3` | PCM→WAV→API, `response_format="text"`, sub-200ms |
| LLM | Groq LLaMA | `llama-3.3-70b-versatile` | `max_tokens=256`, `temperature=0.7`, full history |
| TTS | pyttsx3 SAPI5 | 2.99 | Offline, Windows only, 165 wpm, volume 1.0 |
| Audio I/O | PyAudio + PortAudio | 0.2.14 | 16kHz mono Int16 capture, 1024-frame chunks |
| VAD | audioop.rms | stdlib | RMS > 800, ≥5 frames, 25-frame silence limit |
| UI | Streamlit | ≥1.32 | session_state, st.chat_message, auto-rerun |
| Async | asyncio + threading | stdlib 3.12 | Agent runs in daemon thread with private event loop |
| Thread safety | queue.Queue + Lock | stdlib | Zero cross-thread session_state violations |
| Config | python-dotenv | latest | `.env`-based, 12-factor |
| Language | Python | 3.12 | |

---

## Project Structure

```
Voice_agent/
├── app.py                  # Streamlit UI + thread-safe bridge (199 lines)
│                           #   _transcript_q (Queue), _status (dict+Lock)
│                           #   _init_state(), _drain_queue(), start/stop session
│                           #   Sidebar: API check, controls, status, guide
│                           #   Main: chat transcript, stats bar, 0.7s auto-refresh
├── config.py               # All configuration constants (20 lines)
│                           #   GROQ_API_KEY, LIVE_MODEL, INPUT_SAMPLE_RATE=16000
│                           #   OUTPUT_SAMPLE_RATE=24000, CHUNK_SIZE=1024
│                           #   SILENCE_THRESHOLD=800, MAX_HISTORY_TURNS=20
│                           #   load_system_prompt()
├── requirements.txt        # groq, pyaudio, streamlit, python-dotenv, numpy*, pyttsx3
├── .env.example            # GROQ_API_KEY placeholder
├── agent_error.log         # Runtime error log (auto-truncated each session start)
├── prompts/
│   └── system_prompt.txt   # JARVIS persona + 6 behavioral rules (10 lines)
└── core/
    ├── __init__.py         # Package marker (empty)
    ├── agent.py            # VoiceAgent — main pipeline (253 lines)
    │                       #   MIN_SPEECH_FRAMES=5, SILENCE_LIMIT=25 (module constants)
    │                       #   _collect_speech() VAD, _transcribe() STT
    │                       #   _ask_groq() LLM, _speak() TTS, _flush_mic()
    ├── audio.py            # AudioHandler — PyAudio hardware I/O (77 lines)
    │                       #   Dual streams: input 16kHz, output 24kHz (legacy)
    │                       #   Capture thread → input_queue (always running)
    ├── memory.py           # ConversationMemory — history store (30 lines)
    │                       #   Turn @dataclass (role, text, HH:MM:SS timestamp)
    │                       #   threading.Lock, auto-trim to 40 messages
    └── tools.py            # Tool registry + GROQ_TOOLS schema (62 lines)
                            #   book_demo, check_slot_availability, get_current_time
                            #   TOOL_REGISTRY, dispatch(), GROQ_TOOLS (OpenAI-format)
```

> `*` numpy is listed in requirements.txt but not imported anywhere — all PCM math uses `audioop` (stdlib). Safe to remove.

---

## Configuration Reference

**`config.py`** (from `.env`):

| Constant | Value | Notes |
|----------|-------|-------|
| `GROQ_API_KEY` | `.env` | Required |
| `LIVE_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `INPUT_SAMPLE_RATE` | 16000 Hz | Microphone capture |
| `OUTPUT_SAMPLE_RATE` | 24000 Hz | Legacy output stream only |
| `CHANNELS` | 1 | Mono |
| `CHUNK_SIZE` | 1024 frames | ~64ms per chunk |
| `SILENCE_THRESHOLD` | 800 RMS | Raised from 500 |
| `MAX_HISTORY_TURNS` | 20 turns | 40 message objects in LLM context |

**`core/agent.py`** (module-level constants):

| Constant | Value | Notes |
|----------|-------|-------|
| `MIN_SPEECH_FRAMES` | 5 | ~320ms minimum real speech |
| `SILENCE_LIMIT` | 25 frames | ~1.6s silence ends utterance |
| `max_tokens` | 256 | LLM reply cap |
| `temperature` | 0.7 | LLM sampling |
| TTS rate | 165 wpm | pyttsx3 |
| TTS volume | 1.0 | pyttsx3 |
| Flush delay | 400ms | Post-TTS mic drain |

**`app.py`**:

| Setting | Value |
|---------|-------|
| UI refresh interval | 0.7s |

---

## Setup

```bash
# 1. Create virtual environment (Python 3.12)
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# PyAudio often fails on Windows — use pipwin if so:
pip install pipwin
pipwin install pyaudio

# 3. Set your Groq API key
copy .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_here

# 4. Run
streamlit run app.py
```

Then open `http://localhost:8501` → click **Start Session** in the sidebar → wait for 🔴 Listening → speak.

---

## Tools

Three tools defined in `core/tools.py`:

| Tool | Signature | Returns |
|------|-----------|---------|
| `book_demo` | `(name, date, time="10:00 AM")` | `{status, message, booking_id: "DEMO-XXXX"}` |
| `check_slot_availability` | `(date)` | `{date, available_slots: ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"]}` |
| `get_current_time` | `()` | `{time: "HH:MM AM/PM", date: "Month DD, YYYY"}` |

> **Important:** `GROQ_TOOLS` (OpenAI-format JSON schema) is defined but **not currently passed** to `_ask_groq()`. Function calling is scaffolded but inactive — JARVIS currently handles tool queries via LLM knowledge only.  
> To activate: add `tools=GROQ_TOOLS` to the `chat.completions.create()` call in `core/agent.py:229` and add a tool-call dispatch loop.

---

## System Prompt

`prompts/system_prompt.txt` defines 6 behavioral rules for the JARVIS persona:

1. Keep every reply ≤ 3 sentences — responses are spoken aloud
2. Be friendly, natural, and conversational
3. When booking a demo, ask for name and preferred date if not provided
4. List available time slots clearly and briefly
5. No bullet points, markdown, asterisks, or special characters — plain spoken language only
6. If unclear, politely ask the user to repeat

---

## Bug Fixes (Resolved)

Three critical bugs were found and fixed during development:

**Bug 1 — Streamlit Thread Isolation Violation**
- Symptom: `AttributeError: st.session_state has no attribute "transcript"` on every transcription
- Root cause: `session_state` is bound to Streamlit's ScriptRunContext — writing from a background thread is undefined behavior
- Fix: Module-level `_transcript_q` (Queue) + `_status` (dict + Lock) bridge in `app.py`

**Bug 2 — Acoustic Echo Loop**
- Symptom: JARVIS spoke to itself indefinitely — heard its own TTS output and kept replying
- Root cause: PyAudio capture thread always running; speaker audio fed back into `input_queue`, transcribed by Whisper, and replied to
- Fix: `_flush_mic_after_speaking()` — 400ms sleep for echo decay + drain all accumulated chunks from `input_queue`

**Bug 3 — Noise Hallucination / Phantom Input**
- Symptom: JARVIS replied to silence — fan noise, AC hum, electrical noise triggered speech detection
- Root cause: `SILENCE_THRESHOLD=500` sat below Windows ambient noise floor; single noise spike set `speech_started=True`; Whisper hallucinated text from near-silence
- Fix: Raised threshold to 800 RMS + added `MIN_SPEECH_FRAMES=5` gate (≥320ms of real speech required before sending to Whisper)

---

## Known Technical Notes

| Issue | File | Detail |
|-------|------|--------|
| GROQ_TOOLS not wired | `core/agent.py:229` | `tools=GROQ_TOOLS` not passed — function calling inactive |
| `memory.get_history()` unused | `core/memory.py:23` | Agent uses its own `messages[]`; memory is a parallel log |
| `pyttsx3.init()` per call | `core/agent.py:243` | Re-creates SAPI5 engine each turn (~50–100ms overhead) |
| Legacy output stream | `core/audio.py:24,65` | 24kHz stream + playback thread — nothing writes to `output_queue` |
| numpy unused | `requirements.txt:5` | Listed but never imported; audioop handles all PCM math |
| VAD constants not in config | `core/agent.py:34,38` | `MIN_SPEECH_FRAMES` + `SILENCE_LIMIT` not in `config.py` |

---

## Future Enhancements

- [ ] Activate tool calling — add `tools=GROQ_TOOLS` + dispatch loop in `_ask_groq()`
- [ ] Persistent booking storage (SQLite / Google Calendar)
- [ ] Streaming LLM tokens → early TTS start (lower latency)
- [ ] Silero VAD for more accurate speech segmentation
- [ ] ElevenLabs / Azure TTS for more natural voice quality
- [ ] Retry/backoff for Groq API failures (`tenacity`)
- [ ] Move `MIN_SPEECH_FRAMES` + `SILENCE_LIMIT` to `config.py`
- [ ] Remove unused `numpy` from `requirements.txt`
- [ ] Persistent `pyttsx3` engine in `__init__` (avoid per-call re-init)
- [ ] pytest test suite (pytest-asyncio)
- [ ] Docker + GitHub Actions CI
- [ ] Deploy to Streamlit Community Cloud / Railway
