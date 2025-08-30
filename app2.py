#!/usr/bin/env python3
"""
Integrated Flask backend for audio transcription + summary + QA using Groq.

- Audio upload + conversion preserved (moviepy / ffmpeg fallback).
- Transcription stays as a queued job (unchanged behavior).
- Summary and QA return TS-shaped objects synchronously to match frontend interfaces.
- Includes GroqChat helper for summarization and QA.
- All responses match your TS types: Job, Transcription, Summary, ChatMessage.

NOTE: For production, add persistent storage, auth, and stronger validation.
"""

from __future__ import annotations
import os
import io
import uuid
import queue
import threading
import traceback
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Optional PDF parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Groq SDK
from groq import Groq

# Load env
if "GROQ_API_KEY" not in os.environ:
    load_dotenv()


# -------------------------
# GroqChat helper (refined)
# -------------------------
@dataclass
class GroqChat:
    """Groq chat helper with context management and streaming support."""
    system_prompt: str
    api_key: Optional[str] = None
    model: str = "llama3-70b-8192"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_history_messages: int = 24
    client: Groq = field(init=False)
    context: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment or passed to constructor")
        # Groq() picks up GROQ_API_KEY from env
        self.client = Groq()
        self.reset_context(system_prompt=self.system_prompt)

    # Context utilities
    def reset_context(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.context = [{"role": "system", "content": self.system_prompt}]

    def append_user(self, text: str) -> None:
        self.context.append({"role": "user", "content": text})
        self._trim_context()

    def append_assistant(self, text: str) -> None:
        self.context.append({"role": "assistant", "content": text})
        self._trim_context()

    def _trim_context(self) -> None:
        if len(self.context) <= self.max_history_messages:
            return
        keep = self.context[:1] + self.context[-(self.max_history_messages - 1):]
        self.context = keep

    # Sync ask
    def ask(self, message: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        self.append_user(message)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.context,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stream=False,
        )
        reply = self._extract_text_from_response(resp)
        self.append_assistant(reply)
        return reply

    # Summarization & QA helpers (stateless w.r.t. stored chat)
    def summarize(self, text: str, style: str = "short", model: Optional[str] = None,
                  temperature: Optional[float] = None) -> str:
        # keep prompt short, no formatting risks
        prompt = (
            f"Summarize the following text in a {style} style. "
            f"Return concise paragraphs and bullet points where helpful.\n\n"
            f"TEXT:\n{text[:20000]}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=768,
                stream=False,
            )
            return self._extract_text_from_response(resp)
        except Exception:
            # graceful fallback
            return (text[:1200] + "...") if len(text) > 1200 else text

    def answer_with_context(self, question: str, context_text: str,
                            requirement_text: Optional[str] = None,
                            model: Optional[str] = None,
                            temperature: Optional[float] = None) -> str:
        parts = [
            "You are a concise assistant. Answer the user's question strictly using the provided context.",
            "If you cite, include timestamps if present in the context.",
            "If the answer is not in the context, say so briefly."
        ]
        if requirement_text:
            parts.append("Here are additional requirements/constraints you MUST follow:")
            parts.append(requirement_text)
        parts.append("Context:")
        parts.append(context_text[:20000])
        parts.append("Question:")
        parts.append(question)

        prompt = "\n\n".join(parts)
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=768,
                stream=False,
            )
            return self._extract_text_from_response(resp)
        except Exception:
            return "I couldn’t generate an answer due to an LLM error. Please try again."

    def _extract_text_from_response(self, resp: Any) -> str:
        try:
            if hasattr(resp, "choices"):
                parts = []
                for c in resp.choices:
                    if getattr(c, "message", None):
                        if isinstance(c.message, dict):
                            parts.append(c.message.get("content", "") or "")
                        else:
                            # pydantic object
                            try:
                                parts.append(getattr(c.message, "content", "") or "")
                            except Exception:
                                parts.append(str(c.message))
                    elif getattr(c, "delta", None):
                        parts.append(getattr(c.delta, "content", "") or "")
                    else:
                        parts.append(str(c))
                out = "".join(parts).strip()
                if out:
                    return out
            if isinstance(resp, dict):
                out = resp.get("output_text") or resp.get("text") or resp.get("output", {}).get("text")
                if out:
                    return out
            if hasattr(resp, "text"):
                return getattr(resp, "text") or str(resp)
        except Exception:
            pass
        return str(resp)


# -------------------------
# Flask app + job worker
# -------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = app.root_path
RAW_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads', 'raw')
CONVERTED_DIR = os.path.join(BASE_DIR, 'uploads', 'converted')
EXPORT_DIR = os.path.join(BASE_DIR, 'uploads', 'exports')
for d in (RAW_UPLOAD_DIR, CONVERTED_DIR, EXPORT_DIR):
    os.makedirs(d, exist_ok=True)

DIRECT_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.webm'}

# In-memory job store
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()

# Shared GroqChat for summary/QA
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant that answers questions based on provided transcript/context. "
    "Include timestamps if present. Be factual and brief."
)
try:
    groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
except Exception:
    groq_chat = None


def worker_loop():
    while True:
        job_id = job_queue.get()
        try:
            process_job(job_id)
        except Exception:
            print(f"Error processing job {job_id}:", traceback.format_exc())
        finally:
            job_queue.task_done()


worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()


# -------------------------
# Formatting helpers (TS interfaces)
# -------------------------

def format_job_for_frontend(job_entry: dict) -> dict:
    """Return Job object matching frontend `Job` interface."""
    meta = job_entry.get('meta', {})
    audio_name = os.path.basename(meta.get('orig_path', '')) or meta.get('orig_name') or 'unknown'
    return {
        "id": job_entry["job_id"],
        "audioFile": {
            "id": meta.get("audio_id") or job_entry["job_id"],
            "name": audio_name,
            "size": meta.get("orig_size") or 0,
            "type": meta.get("orig_type") or "audio/*",
            "file": None,
        },
        "status": job_entry.get("status", "queued"),
        "progress": meta.get("progress"),
        "error": job_entry.get("error"),
        "createdAt": job_entry.get("created_at"),
        "completedAt": job_entry.get("updated_at") if job_entry.get("status") == "ready" else None
    }


def format_transcription_for_frontend(job_id: str, transcription_text: str,
                                      segments_raw: Optional[list] = None,
                                      language: str = "en", duration: float = 0.0) -> dict:
    """Return a Transcription dict matching the TS interface."""
    segments = []
    if segments_raw:
        for i, s in enumerate(segments_raw):
            segments.append({
                "id": s.get("id") or f"seg_{i}",
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", s.get("start", 0.0))),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
                "confidence": float(s.get("confidence", 1.0)) if s.get("confidence") is not None else 1.0
            })
    else:
        segments = [{
            "id": f"seg_0",
            "start": 0.0,
            "end": float(duration or 0.0),
            "text": transcription_text or "",
            "speaker": None,
            "confidence": 1.0
        }]

    return {
        "id": f"trans_{job_id}",
        "jobId": job_id,
        "segments": segments,
        "fullText": transcription_text or "",
        "language": language
    }


def format_summary_for_frontend(job_id: str, transcription_id: str, style: str, summary_text: str) -> dict:
    """Return Summary object matching TS interface."""
    # simple bullet extraction: lines starting with '-', '*', or '•'
    bullets = []
    for line in (summary_text or "").splitlines():
        lt = line.strip()
        if lt.startswith(("-", "*", "•")):
            bullets.append(lt.lstrip("-*• ").strip())
    return {
        "id": f"summary_{job_id}",
        "transcriptionId": transcription_id,
        "type": style,
        "content": summary_text,
        "bullets": bullets
    }


def format_chatmessage_for_frontend(job_id: str, answer_text: str) -> dict:
    """Return ChatMessage object for QA answers."""
    return {
        "id": f"msg_{job_id}",
        "role": "assistant",
        "content": answer_text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "referencedSegments": []
    }


# -------------------------
# Core audio handling & Groq calls
# -------------------------

def save_uploaded_file(f, dest_dir, allowed_exts=None):
    fn = f.filename
    ext = os.path.splitext(fn)[1].lower()
    if allowed_exts and ext not in allowed_exts:
        raise ValueError('Unsupported file type')
    unique = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(dest_dir, unique)
    f.save(path)
    return path


def convert_to_mp3(src_path):
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out = os.path.join(CONVERTED_DIR, f"{ts}_{uuid.uuid4().hex}.mp3")
    try:
        from moviepy.editor import AudioFileClip
        audio_clip = AudioFileClip(src_path)
        duration = float(audio_clip.duration)
        audio_clip.write_audiofile(out, logger=None)
        audio_clip.close()
        return out, duration
    except Exception:
        cmd = ['ffmpeg', '-y', '-i', src_path, '-vn', '-acodec', 'libmp3lame', out]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        # duration unknown in fallback
        return out, 0.0


def transcribe_with_groq(audio_path, model='whisper-large-v3', timeout=120):
    """Transcribe audio using the Groq audio client (template call)."""
    with open(audio_path, 'rb') as f:
        data = f.read()
    client = Groq()
    transcript = client.audio.transcriptions.create(
        file=(os.path.basename(audio_path), data),
        model=model,
        timeout=timeout
    )
    # Extract text robustly depending on SDK shape
    text = None
    segments = None
    if isinstance(transcript, dict):
        text = transcript.get('text') or transcript.get('transcription')
        segments = transcript.get('segments')
    else:
        text = getattr(transcript, 'text', None)
        segments = getattr(transcript, 'segments', None)
    if text is None:
        try:
            text = str(transcript)
        except Exception:
            text = ''
    return text, segments


# Use GroqChat for summarization and QA. Lazily initialize if needed.
def summarize_text_with_groq(text, style='short', model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None:
        groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    return groq_chat.summarize(text, style=style, model=model, temperature=temperature)


def answer_question_with_groq(context_text, question, requirement_text=None, model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None:
        groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    return groq_chat.answer_with_context(
        question, context_text, requirement_text=requirement_text, model=model, temperature=temperature
    )


# -------------------------
# Job processing logic (transcription stays queued)
# -------------------------

def create_job_entry(job_type, meta=None):
    job_id = uuid.uuid4().hex
    entry = {
        'job_id': job_id,
        'type': job_type,
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'updated_at': None,
        'meta': meta or {},
        'result': None,
        'error': None
    }
    with jobs_lock:
        jobs[job_id] = entry
    return job_id


def update_job(job_id, **kwargs):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return
        j.update(kwargs)
        j['updated_at'] = datetime.utcnow().isoformat() + 'Z'


def process_job(job_id):
    """Process queued jobs. Here we only queue transcription to preserve your working flow."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job['status'] = 'processing'
        job['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    try:
        jtype = job['type']
        meta = job['meta']
        if jtype == 'transcription':
            audio_path = meta['audio_path']
            duration = meta.get('duration', 0.0)
            text, segments = transcribe_with_groq(audio_path)
            job['result'] = {'transcription': text, 'segments': segments}
            job['meta']['duration'] = duration
            job['status'] = 'ready'
        else:
            job['status'] = 'failed'
            job['error'] = f"Unknown or unsupported queued job type: {jtype}"
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        print('Job error:', traceback.format_exc())
    finally:
        job['updated_at'] = datetime.utcnow().isoformat() + 'Z'


# -------------------------
# Routes (TS-shaped JSON)
# -------------------------

@app.route('/api/v1/audio/upload', methods=['POST'])
def api_upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    try:
        raw_path = save_uploaded_file(f, RAW_UPLOAD_DIR)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    ext = os.path.splitext(raw_path)[1].lower()
    duration = 0.0
    if ext not in DIRECT_EXTS:
        try:
            send_path, duration = convert_to_mp3(raw_path)
        except Exception as e:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return jsonify({'error': f'Conversion failed: {str(e)}'}), 500
    else:
        send_path = raw_path
        # best-effort duration
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(send_path)
            duration = float(clip.duration)
            clip.close()
        except Exception:
            duration = 0.0

    meta = {
        'audio_path': send_path,
        'orig_path': raw_path,
        'orig_name': f.filename,
        'orig_size': os.path.getsize(raw_path) if os.path.exists(raw_path) else 0,
        'orig_type': getattr(f, 'content_type', None) or '',
        'duration': duration
    }

    job_id = create_job_entry('transcription', meta=meta)
    job_queue.put(job_id)

    with jobs_lock:
        job_entry = jobs[job_id]
    # Return a Job directly (TS shape)
    return jsonify(format_job_for_frontend(job_entry)), 201


@app.route('/api/v1/jobs/<job_id>/status', methods=['GET'])
def api_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(format_job_for_frontend(job)), 200


@app.route('/api/v1/jobs/<job_id>/transcription', methods=['GET'])
def api_get_transcription(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    if job['type'] != 'transcription':
        return jsonify({'error': 'Job is not a transcription job'}), 400
    if job['status'] != 'ready':
        return jsonify({'status': job['status']}), 202

    text = job['result'].get('transcription', '')
    segments_raw = job['result'].get('segments')
    duration = job['meta'].get('duration', 0.0)
    transcription_obj = format_transcription_for_frontend(
        job_id, text, segments_raw, language=job['meta'].get('language', 'en'), duration=duration
    )
    return jsonify(transcription_obj), 200


@app.route('/api/v1/jobs/<job_id>/summary', methods=['POST'])
def api_create_summary(job_id):
    """
    Synchronous summary to match TS `Summary` interface.
    Expects: JSON { style?: 'short'|'medium'|'long', text?: string }
    If text not provided, uses the transcription result of the given job.
    """
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    data = request.get_json(silent=True) or {}
    style = data.get('style', 'short')

    # Use supplied text or the transcription result
    text = data.get('text') or (job.get('result', {}) or {}).get('transcription')
    if not text:
        return jsonify({'error': 'No text available for summarization'}), 400

    # Synchronous LLM summary (keeps your flow snappy for UI)
    try:
        summary_text = summarize_text_with_groq(text, style=style)
    except Exception as e:
        return jsonify({'error': f'Summary failed: {str(e)}'}), 500

    # Create a short-lived summary job record (optional)
    summary_job_id = create_job_entry('summary', meta={'text': text, 'style': style, 'transcriptionId': job_id})
    update_job(summary_job_id, status='ready', result={'summary': summary_text})

    # Return a TS-shaped Summary directly
    return jsonify(format_summary_for_frontend(summary_job_id, job_id, style, summary_text)), 201


@app.route('/api/v1/jobs/<job_id>/qa', methods=['POST'])
def api_create_qa(job_id):
    """
    Synchronous QA to return a TS `ChatMessage` (assistant).
    Accepts JSON or multipart form:
      - question (required)
      - requirement_text or requirement_file (pdf/txt/docx)
    """
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    question = None
    requirement_text = None

    # Parse inputs (multipart OR json)
    if request.content_type and 'multipart/form-data' in request.content_type:
        question = request.form.get('question')
        # Prefer explicit context if passed; otherwise use the job transcription
        context_text = request.form.get('context')
        req_file = request.files.get('requirement_file')
        if req_file:
            r_ext = os.path.splitext(req_file.filename)[1].lower()
            if r_ext in {'.txt', '.md'}:
                requirement_text = req_file.read().decode('utf-8', errors='ignore')
            elif r_ext == '.pdf' and PyPDF2:
                buf = io.BytesIO(req_file.read())
                try:
                    reader = PyPDF2.PdfReader(buf)
                    pages = [p.extract_text() or '' for p in reader.pages]
                    requirement_text = '\n'.join(pages)
                except Exception:
                    requirement_text = None
            else:
                # store unparsed files just in case; not processed now
                save_uploaded_file(req_file, RAW_UPLOAD_DIR)
    else:
        data = request.get_json(silent=True) or {}
        question = data.get('question')
        requirement_text = data.get('requirement_text')
        context_text = data.get('context')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Get context from job transcription if not explicitly provided
    if not context_text:
        context_text = (job.get('result', {}) or {}).get('transcription') or ""

    # Synchronous LLM QA
    try:
        answer = answer_question_with_groq(context_text, question, requirement_text=requirement_text)
    except Exception as e:
        return jsonify({'error': f'QA failed: {str(e)}'}), 500

    # Create a short-lived qa job record (optional)
    qa_job_id = create_job_entry('qa', meta={
        'context_text': context_text, 'question': question, 'requirement_text': requirement_text
    })
    update_job(qa_job_id, status='ready', result={'answer': answer})

    # Return a TS-shaped ChatMessage (assistant)
    return jsonify(format_chatmessage_for_frontend(qa_job_id, answer)), 201


@app.route('/api/v1/jobs', methods=['GET'])
def api_list_jobs():
    with jobs_lock:
        all_jobs = [format_job_for_frontend(jobs[k]) for k in jobs]
    return jsonify(all_jobs), 200


@app.route('/api/v1/jobs/<job_id>/download', methods=['GET'])
def api_download(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    if job['status'] != 'ready':
        return jsonify({'status': job['status']}), 202

    if job['type'] == 'transcription':
        text = (job['result'] or {}).get('transcription', '')
        out = io.BytesIO(text.encode('utf-8'))
        out.seek(0)
        return send_file(out, as_attachment=True, download_name=f"{job_id}.txt")
    elif job['type'] == 'summary':
        text = (job['result'] or {}).get('summary', '')
        out = io.BytesIO(text.encode('utf-8'))
        out.seek(0)
        return send_file(out, as_attachment=True, download_name=f"{job_id}_summary.txt")
    elif job['type'] == 'qa':
        text = (job['result'] or {}).get('answer', '')
        out = io.BytesIO(text.encode('utf-8'))
        out.seek(0)
        return send_file(out, as_attachment=True, download_name=f"{job_id}_answer.txt")

    return jsonify({'error': 'Unsupported job type for download'}), 400


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        print('Warning: ffmpeg not found. Only direct-supported formats will work or conversions may fail.')
    app.run(host='0.0.0.0', port=port, debug=True)
