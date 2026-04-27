"""
app.py — JARVIS Voice AI Agent
Record → Groq Whisper STT → Groq LLaMA → gTTS → autoplay

Duplicate-reply fix: MD5 hash of the recording stored in session state;
same recording on a subsequent rerun is skipped entirely.
"""

import base64
import hashlib
import io

import streamlit as st
from gtts import gTTS
from groq import Groq

from config import GROQ_API_KEY, LIVE_MODEL, load_system_prompt
from core.memory import ConversationMemory

st.set_page_config(
    page_title="JARVIS",
    page_icon="🎙️",
    layout="centered",
)


# ── Audio helper — inline base64, no media server ─────────────────────────────
def _autoplay(audio_bytes: bytes) -> None:
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay src="data:audio/mp3;base64,{b64}" style="display:none;"></audio>',
        unsafe_allow_html=True,
    )


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "memory":           ConversationMemory(),
        "messages":         [{"role": "system", "content": load_system_prompt()}],
        "transcript":       [],   # list of (role, text)
        "pending_audio":    None, # bytes | None
        "_processed_hash":  None, # MD5 of last processed recording
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
_client = Groq(api_key=GROQ_API_KEY)

# ── Play pending audio then immediately clear ──────────────────────────────────
if st.session_state.pending_audio:
    _autoplay(st.session_state.pending_audio)
    st.session_state.pending_audio = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem;">
      <div style="font-size:3.2rem;line-height:1;">🎙️</div>
      <h1 style="margin:.3rem 0 0;font-size:2rem;font-weight:800;letter-spacing:-.5px;">JARVIS</h1>
      <p style="color:#888;margin:.2rem 0 0;font-size:.9rem;">Your AI assistant — ask me anything</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not set — go to App Settings → Secrets.")
    st.stop()

# ── Transcript ─────────────────────────────────────────────────────────────────
transcript = st.session_state.transcript

if not transcript:
    st.markdown(
        "<p style='text-align:center;color:#555;padding:3rem 1rem 1rem;font-size:1rem;'>"
        "Tap the mic, say anything, then tap Stop — I'll reply.</p>",
        unsafe_allow_html=True,
    )

for role, text in transcript:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant", avatar="🎙️"):
            st.write(text)

# ── Mic input ──────────────────────────────────────────────────────────────────
st.divider()
_, col, _ = st.columns([1, 2, 1])
with col:
    audio_value = st.audio_input("", label_visibility="collapsed")

# ── Clear button ───────────────────────────────────────────────────────────────
if transcript:
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages          = [{"role": "system", "content": load_system_prompt()}]
            st.session_state.transcript        = []
            st.session_state.pending_audio     = None
            st.session_state["_processed_hash"] = None
            st.session_state.memory.clear()
            st.rerun()

# ── Early exit if nothing recorded ────────────────────────────────────────────
if not audio_value:
    st.stop()

# ── Duplicate-reply guard ──────────────────────────────────────────────────────
audio_bytes = audio_value.getvalue()
audio_hash  = hashlib.md5(audio_bytes).hexdigest()

if st.session_state["_processed_hash"] == audio_hash:
    st.stop()

st.session_state["_processed_hash"] = audio_hash   # mark BEFORE processing

# ── ① STT — Groq Whisper ──────────────────────────────────────────────────────
with st.spinner("Listening…"):
    try:
        buf      = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        user_text = (
            _client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=buf,
                response_format="text",
            ) or ""
        ).strip()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

if not user_text:
    st.toast("Didn't catch that — please try again.")
    st.stop()

st.session_state.transcript.append(("user", user_text))
st.session_state.messages.append({"role": "user", "content": user_text})
st.session_state.memory.add_turn("user", user_text)

# ── ② LLM — Groq LLaMA ────────────────────────────────────────────────────────
with st.spinner("Thinking…"):
    try:
        resp  = _client.chat.completions.create(
            model=LIVE_MODEL,
            messages=st.session_state.messages,
            max_tokens=300,
            temperature=0.7,
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"Could not get a reply: {e}")
        st.stop()

if not reply:
    st.stop()

st.session_state.messages.append({"role": "assistant", "content": reply})
st.session_state.memory.add_turn("agent", reply)
st.session_state.transcript.append(("agent", reply))

# Trim history: system prompt + last 40 messages
if len(st.session_state.messages) > 41:
    st.session_state.messages = (
        st.session_state.messages[:1] + st.session_state.messages[-40:]
    )

# ── ③ TTS — gTTS ──────────────────────────────────────────────────────────────
try:
    tts_buf = io.BytesIO()
    gTTS(text=reply, lang="en").write_to_fp(tts_buf)
    st.session_state.pending_audio = tts_buf.getvalue()
except Exception:
    pass   # TTS failure is non-fatal — reply still shows as text

st.rerun()
