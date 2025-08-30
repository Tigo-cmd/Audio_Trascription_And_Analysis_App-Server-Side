
---

# 📘 Backend README (Flask + Groq API)

```markdown
# 🎙️ AI Transcription Backend

This is the **backend** for the AI-powered transcription and summarization system.  
It provides REST APIs to:

- Upload audio files
- Convert audio to MP3
- Run transcription with **Groq LLM**
- Summarize transcripts
- Manage job queue and status
- Export results

---

## 🚀 Features

- Audio upload and conversion (FFmpeg / MoviePy fallback)
- Background job queue with polling
- Transcription using Groq’s API
- Text summarization endpoint
- Automatic cleanup of uploaded files
- Error-safe responses

---

## 📦 Tech Stack

- [Python 3.11+](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Groq API](https://console.groq.com/)
- [MoviePy](https://zulko.github.io/moviepy/)
- [FFmpeg](https://ffmpeg.org/)

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Tigo-cmd/Audio_Trascription_And_Analysis_App-Server-Side
cd Audio_Trascription_And_Analysis_App-Server-Side

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt
