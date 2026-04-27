"""
core/agent.py — VoiceAgent: mic → Groq Whisper STT → Groq LLaMA → pyttsx3 TTS

Three bugs fixed vs. previous version:
  1. Echo loop  — input_queue is flushed after every TTS playback so JARVIS
                  never hears its own voice as "user" speech.
  2. Noise gate — MIN_SPEECH_FRAMES ensures at least 320 ms of real speech
                  before an utterance is sent to Whisper (kills hallucinations
                  from fan/AC noise that briefly cross the threshold).
  3. Threshold  — SILENCE_THRESHOLD raised to 800 in config.py to sit above
                  typical Windows background noise floor.
"""

import asyncio
import audioop
import io
import time
import threading
import traceback
import wave

import pyttsx3
from groq import Groq

import config
from config import load_system_prompt
from core.memory import ConversationMemory

LOG_FILE = "agent_error.log"

# Minimum consecutive speech frames before an utterance is considered real.
# 1 frame = CHUNK_SIZE / INPUT_SAMPLE_RATE = 1024 / 16000 ≈ 64 ms
# 5 frames ≈ 320 ms  — filters noise bursts and Whisper hallucinations.
MIN_SPEECH_FRAMES = 5

# How many silence frames end an utterance (25 × 64 ms ≈ 1.6 s).
SILENCE_LIMIT = 25


def _log(msg: str) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)


class VoiceAgent:
    def __init__(self, audio, memory, on_status_change=None, on_transcript=None):
        self.audio = audio
        self.memory = memory
        self.on_status_change = on_status_change or (lambda s: None)
        self.on_transcript = on_transcript or (lambda role, text: None)

        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._system_prompt = load_system_prompt() or (
            "You are JARVIS, a concise voice assistant. "
            "Keep replies under 3 sentences — they will be spoken aloud."
        )
        self._running = False
        self._loop = None
        self._thread = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        open(LOG_FILE, "w").close()
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── event loop bridge ──────────────────────────────────────────────────────

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception:
            _log(f"[AGENT ERROR]\n{traceback.format_exc()}")
            self.on_status_change("Error — check agent_error.log")
        finally:
            self._running = False
            self.on_status_change("Idle")

    # ── main conversation loop ─────────────────────────────────────────────────

    async def _main(self):
        _log(f"[INFO] JARVIS starting — STT: whisper-large-v3 | LLM: {config.LIVE_MODEL}")
        self.on_status_change("Listening")
        loop = asyncio.get_event_loop()

        # system prompt pinned at index 0; never trimmed
        messages = [{"role": "system", "content": self._system_prompt}]

        while self._running:
            self.on_status_change("Listening")

            # ① Collect real speech from mic (noise-gated)
            audio_bytes = await loop.run_in_executor(None, self._collect_speech)
            if audio_bytes is None:
                continue

            # ② Groq Whisper → text
            self.on_status_change("Thinking…")
            text = await loop.run_in_executor(None, self._transcribe, audio_bytes)
            if not text:
                _log("[INFO] Empty transcription — discarding")
                continue

            _log(f"[USER] {text}")
            self.on_transcript("user", text)
            self.memory.add_turn("user", text)
            messages.append({"role": "user", "content": text})

            # ③ Groq LLaMA → reply
            reply = await loop.run_in_executor(None, self._ask_groq, messages)
            if not reply:
                continue

            _log(f"[JARVIS] {reply}")
            self.on_transcript("agent", reply)
            self.memory.add_turn("agent", reply)
            messages.append({"role": "assistant", "content": reply})

            # Keep history bounded: system prompt + last MAX_HISTORY_TURNS pairs
            if len(messages) > config.MAX_HISTORY_TURNS * 2 + 1:
                messages = messages[:1] + messages[-(config.MAX_HISTORY_TURNS * 2):]

            # ④ Speak reply
            self.on_status_change("Speaking")
            await loop.run_in_executor(None, self._speak, reply)

            # ⑤ FIX: flush mic after speaking so JARVIS never hears itself
            await loop.run_in_executor(None, self._flush_mic_after_speaking)

    # ── audio collection ───────────────────────────────────────────────────────

    def _collect_speech(self) -> bytes | None:
        """
        Drain input_queue until we detect at least MIN_SPEECH_FRAMES of real
        speech followed by SILENCE_LIMIT frames of quiet.

        Returns raw PCM bytes, or None if what was captured isn't real speech.
        """
        chunks: list[bytes] = []
        speech_frames = 0          # frames that crossed the RMS threshold
        speech_started = False
        silence_frames = 0

        while self._running:
            try:
                chunk = self.audio.input_queue.get(timeout=0.1)
            except Exception:
                continue

            rms = audioop.rms(chunk, 2)

            if rms > config.SILENCE_THRESHOLD:
                speech_started = True
                speech_frames += 1
                silence_frames = 0
                chunks.append(chunk)
            elif speech_started:
                chunks.append(chunk)
                silence_frames += 1
                if silence_frames >= SILENCE_LIMIT:
                    break

        # Discard if not enough real speech — avoids sending noise to Whisper
        if not speech_started or speech_frames < MIN_SPEECH_FRAMES:
            return None

        return b"".join(chunks)

    # ── post-TTS mic flush ─────────────────────────────────────────────────────

    def _flush_mic_after_speaking(self) -> None:
        """
        After pyttsx3 finishes, the capture thread has been filling input_queue
        with the speaker audio (echo). Wait briefly for the speaker to go silent,
        then drain every accumulated chunk so _collect_speech starts clean.
        """
        time.sleep(0.4)   # let speaker echo decay
        flushed = 0
        while not self.audio.input_queue.empty():
            try:
                self.audio.input_queue.get_nowait()
                flushed += 1
            except Exception:
                break
        if flushed:
            _log(f"[INFO] Flushed {flushed} echo chunks after speaking")

    # ── STT — Groq Whisper ─────────────────────────────────────────────────────

    def _transcribe(self, pcm: bytes) -> str:
        """Wrap raw PCM in a WAV container and send to Groq Whisper."""
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(2)          # Int16 = 2 bytes / sample
                wf.setframerate(config.INPUT_SAMPLE_RATE)
                wf.writeframes(pcm)
            buf.seek(0)
            buf.name = "audio.wav"          # Groq SDK needs a .name attribute

            result = self._client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=buf,
                response_format="text",
            )
            return (result or "").strip()
        except Exception as e:
            _log(f"[STT ERROR] {e}")
            return ""

    # ── LLM — Groq LLaMA ──────────────────────────────────────────────────────

    def _ask_groq(self, messages: list) -> str:
        """Send full message history to Groq LLaMA and return the reply."""
        try:
            response = self._client.chat.completions.create(
                model=config.LIVE_MODEL,
                messages=messages,
                max_tokens=256,
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            _log(f"[GROQ ERROR] {e}")
            return ""

    # ── TTS — pyttsx3 ─────────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        """Speak text via pyttsx3 (Windows SAPI5 — offline, no API key needed)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            _log(f"[TTS ERROR] {e}")
