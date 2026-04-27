# JARVIS — Real-Time Voice AI Agent

A real-time voice AI agent powered by **Groq API**, built with Python and Streamlit.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# Edit .env → GROQ_API_KEY=your_key_here

# 4. Run
streamlit run app.py
```

## Features
- Real-time voice conversation (speak → AI speaks back)
- Interrupt handling (speak mid-response to cut off AI)
- Conversation memory (rolling transcript)
- Function calling (book demo, check availability, get time)
- Streamlit UI with live transcript + status indicator

## Project Structure
```
voice-ai-agent/
├── app.py                 # Streamlit UI
├── config.py              # Config + env loader
├── requirements.txt
├── .env.example           # Environment variables template
├── prompts/
│   └── system_prompt.txt  # JARVIS persona
└── core/
    ├── audio.py           # Mic capture + speaker playback
    ├── agent.py           # Groq API session manager
    ├── memory.py          # Conversation history
    └── tools.py           # Function calling definitions
```

## PyAudio on Windows
If `pip install pyaudio` fails, install the wheel manually:
```
pip install pipwin
pipwin install pyaudio
```