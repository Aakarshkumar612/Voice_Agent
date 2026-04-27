"""
app.py — JARVIS Voice AI Agent (Cloud Edition)

Pipeline per turn:
  st.audio_input() → Groq Whisper STT → Groq LLaMA LLM → gTTS → st.audio()

No PyAudio, no pyttsx3, no background threads — runs on Streamlit Cloud.
"""

import io

import streamlit as st
from gtts import gTTS
from groq import Groq

from config import GROQ_API_KEY, LIVE_MODEL, load_system_prompt
from core.memory import ConversationMemory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JARVIS — Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
)

# ── Session state bootstrap ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "memory": ConversationMemory(),
        "messages": [{"role": "system", "content": load_system_prompt()}],
        "transcript": [],   # list of (role, text, audio_bytes | None)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()
_client = Groq(api_key=GROQ_API_KEY)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ JARVIS")
    st.markdown("*Voice AI Agent*")
    st.divider()

    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set in Secrets")
    else:
        st.success("API key loaded ✓")

    st.divider()

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": load_system_prompt()}]
        st.session_state.transcript = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.markdown("#### How to use")
    st.markdown(
        "1. Click the **🎤 mic** at the bottom\n"
        "2. Speak your message\n"
        "3. Click **Stop**\n"
        "4. JARVIS transcribes, thinks, and replies\n"
        "5. Audio plays in the conversation"
    )
    st.divider()
    st.markdown("#### Try saying")
    st.code("Hi, I want to book a demo")
    st.code("What slots are free on May 10th?")
    st.code("What time is it right now?")

# ── Conversation transcript ────────────────────────────────────────────────────
st.markdown("## Conversation")

if not st.session_state.transcript:
    st.info("🎤 Record your message below to start talking with JARVIS.")

for role, text, audio_bytes in st.session_state.transcript:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

# ── Stats bar ──────────────────────────────────────────────────────────────────
transcript = st.session_state.transcript
if transcript:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Turns", len(transcript))
    col2.metric("You said", sum(1 for r, *_ in transcript if r == "user"))
    col3.metric("JARVIS replied", sum(1 for r, *_ in transcript if r == "agent"))

# ── Voice input ────────────────────────────────────────────────────────────────
st.divider()
audio_value = st.audio_input("🎤 Record your message")

if audio_value and GROQ_API_KEY:

    # ① STT — Groq Whisper
    with st.spinner("Transcribing…"):
        try:
            buf = io.BytesIO(audio_value.read())
            buf.name = "audio.wav"
            user_text = (
                _client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=buf,
                    response_format="text",
                ) or ""
            ).strip()
        except Exception as e:
            st.error(f"STT error: {e}")
            st.stop()

    if not user_text:
        st.warning("No speech detected — please try again.")
        st.stop()

    st.session_state.transcript.append(("user", user_text, None))
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.memory.add_turn("user", user_text)

    # ② LLM — Groq LLaMA
    with st.spinner("Thinking…"):
        try:
            response = _client.chat.completions.create(
                model=LIVE_MODEL,
                messages=st.session_state.messages,
                max_tokens=256,
                temperature=0.7,
            )
            reply = (response.choices[0].message.content or "").strip()
        except Exception as e:
            st.error(f"LLM error: {e}")
            st.stop()

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.memory.add_turn("agent", reply)

        # ③ TTS — gTTS
        with st.spinner("Generating speech…"):
            try:
                tts_buf = io.BytesIO()
                gTTS(text=reply, lang="en").write_to_fp(tts_buf)
                tts_buf.seek(0)
                audio_bytes = tts_buf.read()
            except Exception:
                audio_bytes = None

        st.session_state.transcript.append(("agent", reply, audio_bytes))

        # Trim history: keep system prompt + last 40 messages (20 turns)
        if len(st.session_state.messages) > 41:
            st.session_state.messages = (
                st.session_state.messages[:1]
                + st.session_state.messages[-40:]
            )

    st.rerun()
