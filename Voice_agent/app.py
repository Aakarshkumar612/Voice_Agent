"""
app.py — Streamlit UI for JARVIS Voice AI Agent.

Threading model:
  - Agent runs in a daemon thread, cannot touch st.session_state directly.
  - All cross-thread communication goes through module-level thread-safe
    primitives (_transcript_q, _status). Streamlit drains them on each rerun.
"""

import queue
import threading
import time

import streamlit as st

from config import GROQ_API_KEY
from core.audio import AudioHandler
from core.agent import VoiceAgent
from core.memory import ConversationMemory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JARVIS — Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
)

# ── Module-level thread-safe state ─────────────────────────────────────────────
# These live at module scope so the background agent thread can safely write to
# them. Streamlit's main thread reads them during each rerun.
_transcript_q: queue.Queue = queue.Queue()
_status_lock = threading.Lock()
_status = {"value": "Idle"}


def _set_status(value: str) -> None:
    with _status_lock:
        _status["value"] = value


def _get_status() -> str:
    with _status_lock:
        return _status["value"]


def _push_transcript(role: str, text: str) -> None:
    _transcript_q.put((role, text))


# ── Session state bootstrap ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "running": False,
        "audio": None,
        "agent": None,
        "memory": None,
        "transcript": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# Drain the thread-safe queue into session_state on every rerun
def _drain_queue() -> None:
    while not _transcript_q.empty():
        try:
            role, text = _transcript_q.get_nowait()
            st.session_state.transcript.append((role, text))
        except queue.Empty:
            break


_drain_queue()

# ── Session control ────────────────────────────────────────────────────────────
def start_session() -> None:
    if st.session_state.running:
        return
    st.session_state.transcript = []
    _set_status("Connecting…")

    memory = ConversationMemory()
    audio = AudioHandler()
    agent = VoiceAgent(
        audio=audio,
        memory=memory,
        on_status_change=_set_status,       # writes to _status dict — safe
        on_transcript=_push_transcript,     # writes to _transcript_q — safe
    )
    audio.start()
    agent.start()

    st.session_state.memory = memory
    st.session_state.audio = audio
    st.session_state.agent = agent
    st.session_state.running = True


def stop_session() -> None:
    if not st.session_state.running:
        return
    if st.session_state.agent:
        st.session_state.agent.stop()
    if st.session_state.audio:
        st.session_state.audio.stop()
    st.session_state.running = False
    _set_status("Idle")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ JARVIS")
    st.markdown("*Real-Time Voice AI Agent*")
    st.divider()

    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set in .env")
    else:
        st.success("API key loaded ✓")

    st.divider()

    if not st.session_state.running:
        if st.button("▶ Start Session", type="primary", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Set GROQ_API_KEY in .env first.")
            else:
                start_session()
    else:
        if st.button("⏹ Stop Session", type="secondary", use_container_width=True):
            stop_session()

    st.divider()

    status = _get_status()
    if status == "Listening":
        st.markdown("🔴 **Listening…**")
    elif status == "Speaking":
        st.markdown("🔵 **Speaking…**")
    elif "Thinking" in status:
        st.markdown("🟡 **Thinking…**")
    elif "Connecting" in status:
        st.markdown("🟡 **Connecting…**")
    elif "Error" in status:
        st.markdown(f"🔴 **{status}**")
    else:
        st.markdown("⚪ **Idle**")

    st.divider()
    st.markdown("#### How to use")
    st.markdown(
        "1. Click **Start Session**\n"
        "2. Wait for 🔴 Listening\n"
        "3. Speak naturally\n"
        "4. JARVIS replies via speaker\n"
        "5. Interrupt anytime — just speak"
    )
    st.divider()
    st.markdown("#### Try saying")
    st.code("Hi, I want to book a demo")
    st.code("What slots are free on May 10th?")
    st.code("What time is it right now?")

# ── Main transcript area ───────────────────────────────────────────────────────
st.markdown("## Conversation")

transcript = st.session_state.transcript

if not st.session_state.running and not transcript:
    st.info("👈 Click **Start Session** in the sidebar to begin talking with JARVIS.")

with st.container():
    if transcript:
        for role, text in transcript:
            if role == "user":
                with st.chat_message("user"):
                    st.write(text)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(text)
    elif st.session_state.running:
        st.markdown("*Waiting for speech…*")

# ── Stats bar ──────────────────────────────────────────────────────────────────
if st.session_state.running:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Turns", len(transcript))
    col2.metric("You said", sum(1 for r, _ in transcript if r == "user"))
    col3.metric("JARVIS replied", sum(1 for r, _ in transcript if r == "agent"))

# ── Auto-refresh loop ──────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.7)
    st.rerun()
