"""
app.py — JARVIS Voice AI Agent
Groq Whisper STT (English-forced) → Groq LLaMA → gTTS → st.audio autoplay

No explicit st.rerun() after processing — the natural Streamlit rerun on the
next audio_input submission handles state updates, which lets the browser fully
load and play the audio before any re-render wipes the file from the media server.
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
st.caption("Tap the mic → speak → tap Stop · JARVIS will reply in text and audio")

# ── Mic widget ─────────────────────────────────────────────────────────────────
audio_value = st.audio_input("Your message")

# ── Process only if this is a NEW recording ────────────────────────────────────
if audio_value:
    audio_bytes = audio_value.getvalue()
    audio_hash  = hashlib.md5(audio_bytes).hexdigest()

    if st.session_state["_processed_hash"] != audio_hash:
        st.session_state["_processed_hash"] = audio_hash

        # ① STT — force English so Whisper never switches languages
        with st.spinner("Transcribing…"):
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

            # ② LLM
            with st.spinner("Thinking…"):
                try:
                    resp = _client.chat.completions.create(
                        model=LIVE_MODEL,
                        messages=st.session_state.messages,
                        max_tokens=300,
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

                # Keep context window bounded
                if len(st.session_state.messages) > 41:
                    st.session_state.messages = (
                        st.session_state.messages[:1] + st.session_state.messages[-40:]
                    )

                # ③ TTS — render audio player immediately in this render pass.
                # No st.rerun() is called, so the media file stays alive long
                # enough for the browser to fetch and play it.
                try:
                    tts_buf = io.BytesIO()
                    gTTS(text=reply, lang="en").write_to_fp(tts_buf)
                    st.audio(tts_buf.getvalue(), format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.warning(f"Audio generation failed: {e}")

# ── Conversation transcript ────────────────────────────────────────────────────
st.divider()

if not st.session_state.transcript:
    st.info("Your conversation will appear here. Tap the mic above to start.")

for role, text in st.session_state.transcript:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant", avatar="🎙️"):
            st.write(text)

# ── Clear ──────────────────────────────────────────────────────────────────────
if st.session_state.transcript:
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages          = [{"role": "system", "content": load_system_prompt()}]
        st.session_state.transcript        = []
        st.session_state["_processed_hash"] = None
        st.session_state.memory.clear()
        st.rerun()
