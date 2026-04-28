"""
app.py — JARVIS Voice AI Agent
Session-based conversation: Start Session → mic loop → End Session
Groq Whisper STT (en) → Groq LLaMA → gTTS → st.audio autoplay
"""

import hashlib
import io

import streamlit as st
from gtts import gTTS
from groq import Groq

from config import GROQ_API_KEY, LIVE_MODEL, load_system_prompt
from core.memory import ConversationMemory

st.set_page_config(page_title="JARVIS", page_icon="🎙️", layout="centered")


def _init_state() -> None:
    defaults = {
        "memory":          ConversationMemory(),
        "messages":        [{"role": "system", "content": load_system_prompt()}],
        "transcript":      [],
        "_processed_hash": None,
        "session_active":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Add it in App Settings → Secrets.")
    st.stop()

_client = Groq(api_key=GROQ_API_KEY)

st.title("🎙️ JARVIS")
st.caption("Your AI voice assistant")

# ── Landing screen (no active session) ────────────────────────────────────────
if not st.session_state.session_active:
    st.write("")
    st.write("Press **Start Session** to begin talking with JARVIS.")
    st.write("Once the session starts, tap the mic, say anything, tap Stop — JARVIS will reply. Keep going as long as you like.")
    st.write("")
    if st.button("▶  Start Session", type="primary", use_container_width=True):
        st.session_state.session_active = True
        st.rerun()
    st.stop()

# ── Active session header ──────────────────────────────────────────────────────
col_status, col_end = st.columns([4, 1])
with col_status:
    st.success("Session active")
with col_end:
    if st.button("⏹ End", use_container_width=True):
        st.session_state.session_active    = False
        st.session_state.messages          = [{"role": "system", "content": load_system_prompt()}]
        st.session_state.transcript        = []
        st.session_state["_processed_hash"] = None
        st.session_state.memory.clear()
        st.rerun()

# ── Mic widget — always at the top so it's one click away after every reply ───
st.write("**Tap the mic, speak, then tap Stop:**")
audio_value = st.audio_input("", label_visibility="collapsed")

# ── Process only if this is a new recording ────────────────────────────────────
if audio_value:
    audio_bytes = audio_value.getvalue()
    audio_hash  = hashlib.md5(audio_bytes).hexdigest()

    if st.session_state["_processed_hash"] != audio_hash:
        st.session_state["_processed_hash"] = audio_hash

        # ① STT — language="en" prevents Whisper from switching languages
        with st.spinner("Listening…"):
            try:
                buf      = io.BytesIO(audio_bytes)
                buf.name = "audio.wav"
                user_text = (
                    _client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=buf,
                        response_format="text",
                        language="en",
                    ) or ""
                ).strip()
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

        if not user_text:
            st.toast("Didn't catch that — please try again.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.transcript.append(("user", user_text))
            st.session_state.memory.add_turn("user", user_text)

            # ② LLM — 1024 tokens allows full 15-20 sentence answers
            with st.spinner("Thinking…"):
                try:
                    resp = _client.chat.completions.create(
                        model=LIVE_MODEL,
                        messages=st.session_state.messages,
                        max_tokens=1024,
                        temperature=0.7,
                    )
                    reply = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    st.error(f"Could not get a reply: {e}")
                    st.stop()

            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.transcript.append(("agent", reply))
                st.session_state.memory.add_turn("agent", reply)

                if len(st.session_state.messages) > 41:
                    st.session_state.messages = (
                        st.session_state.messages[:1] + st.session_state.messages[-40:]
                    )

                # ③ TTS — rendered in this pass; no st.rerun() so the file
                #    stays alive long enough for the browser to fetch and play it
                try:
                    tts_buf = io.BytesIO()
                    gTTS(text=reply, lang="en").write_to_fp(tts_buf)
                    st.audio(tts_buf.getvalue(), format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.warning(f"Audio unavailable: {e}")

# ── Conversation transcript ────────────────────────────────────────────────────
st.divider()

if not st.session_state.transcript:
    st.info("Your conversation will appear here once you start speaking.")

for role, text in st.session_state.transcript:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant", avatar="🎙️"):
            st.write(text)
