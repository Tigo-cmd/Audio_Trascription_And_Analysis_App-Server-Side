#!/usr/bin/env python3
"""
Integrated Flask backend for audio transcription + summary + QA using Groq.

This file includes:
- A refined `GroqChat` helper class (sync + streaming) for LLM interactions.
- An in-memory job queue + worker to process transcription, summary, and QA jobs.
- Audio upload and conversion (moviepy / ffmpeg fallback).
- Transcription using Groq audio.transcriptions (template call).
- Summary and QA wired to use the `GroqChat` helper.
- Helpers to format responses to match frontend TypeScript interfaces (Job, Transcription, etc.).

Notes:
- Replace model names, SDK call shapes, and any Groq-specific options to match your installed Groq SDK.
- This is a development skeleton. For production, persist jobs/results to a DB, secure uploads, and add authentication.
"""

from __future__ import annotations
import os
import io
import uuid
import time
import threading
import queue
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
        self.client = Groq()  # assumes GROQ_API_KEY available in environment
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

    def get_context(self) -> List[Dict[str, str]]:
        return list(self.context)

    def save_context(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.context, f, ensure_ascii=False, indent=2)

    def load_context(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.context = json.load(f)
        if not self.context or self.context[0].get("role") != "system":
            self.reset_context(system_prompt=self.system_prompt)
        self._trim_context()

    # Sync ask
    def ask(self, message: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        self.append_user(message)
        try:
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
        except Exception:
            raise

    # Streaming ask
    def ask_stream(self, message: str, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None):
        self.append_user(message)
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        try:
            completion_iterable = self.client.chat.completions.create(
                model=self.model,
                messages=self.context,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception:
            raise
        reply_acc = ""
        try:
            for chunk in completion_iterable:
                try:
                    delta = chunk.choices[0].delta
                    part = delta.content or ""
                except Exception:
                    part = str(chunk)
                reply_acc += part
                yield part
        finally:
            self.append_assistant(reply_acc)
            return reply_acc

    # Async ask (simple wrapper; adjust if SDK has native async)
    async def ask_async(self, message: str, temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> str:
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

    def summarize(self, text: str, style: str = None, model: str = None, temperature: float = None) -> str:
        """Summarize given text using the LLM, with optional style control."""

        if style:
            prompt = f"Summarize the following text in {style} style:\n\n{text}"
        else:
            prompt = f"Summarize the following text:\n\n{text}"

        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)

        except Exception as e:
            print(f"Summarization failed: {e}")
            return f"(Summarization failed: {e})"

    def answer_with_context(self, context: str, question: str, model: str = None, temperature: float = None) -> str:
        """Answer a question given supporting context."""
        combined = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": combined}
                ],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)
        except Exception as e:
            print(f"Context QA failed: {e}")
            return f"(QA failed: {e})"

    def _extract_text_from_response(self, resp: Any) -> str:
        try:
            # Handle SDK objects with 'choices'
            if hasattr(resp, "choices"):
                parts = []
                for c in resp.choices:
                    if getattr(c, "message", None):
                        parts.append(
                            c.message.get("content", "")
                            if isinstance(c.message, dict)
                            else getattr(c.message, "content", str(c.message))
                        )
                    elif getattr(c, "delta", None):
                        parts.append(getattr(c.delta, "content", "") or "")
                    else:
                        parts.append(str(c))
                out = "".join(parts).strip()
                if out:
                    return out

            # Handle plain dicts
            if isinstance(resp, dict):
                out = (
                        resp.get("output_text")
                        or resp.get("text")
                        or resp.get("output", {}).get("text")
                )
                if out:
                    return out

            # Handle message objects like ChatCompletionMessage
            if hasattr(resp, "content"):
                return getattr(resp, "content")

            # Fallback: objects with .text
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
jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()

# Instantiate a shared GroqChat used by summary/QA jobs
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant that answers questions based on provided transcript/context. "
    "When possible include timestamps and cite segments. Be factual and concise."
    "some questions would be asked based on the transcription, and the answers too just return the answers"
)
try:
    groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
except Exception:
    groq_chat = None


# Worker to process jobs from the queue
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
# Formatting helpers for frontend TS interfaces
# -------------------------

def iso_now():
    return datetime.utcnow().isoformat() + "Z"


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
        "progress": job_entry.get("meta", {}).get("progress", None),
        "error": job_entry.get("error"),
        "createdAt": job_entry.get("created_at"),
        "completedAt": job_entry.get("updated_at") if job_entry.get("status") == "ready" else None
    }


def format_transcription_for_frontend(job_id: str, transcription_text: str, segments_raw: Optional[list] = None,
                                      language: str = "en", duration: float = 0.0) -> dict:
    """
    Return a Transcription dict matching the TS interface.
    """
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
        # ✅ remove original raw file
        try:
            os.remove(src_path)
        except Exception:
            pass
        return out, duration
    except Exception:
        cmd = ['ffmpeg', '-y', '-i', src_path, '-vn', '-acodec', 'libmp3lame', out]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        # duration unknown in fallback
        try:
            os.remove(src_path)
        except Exception:
            pass
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
        question,
        context_text,
        # requirement_text=requirement_text,
        model=model,
        temperature=temperature)


# -------------------------
# Job processing logic
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
    """Process queued jobs. Supported types: 'transcription', 'summary', 'qa'."""
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
            # ✅ delete converted audio after successful transcription
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"Warning: could not delete {audio_path}: {e}")
        elif jtype == 'summary':
            text = meta.get('text') or ''
            style = meta.get('style', 'short')
            summary = summarize_text_with_groq(text, style=style)
            job['result'] = {'summary': summary}
            job['status'] = 'ready'
        elif jtype == 'qa':
            context = meta.get('context_text', '')
            question = meta.get('question', '')
            req_text = meta.get('requirement_text')
            answer = answer_question_with_groq(context, question, requirement_text=req_text)
            job['result'] = {'answer': answer}
            job['status'] = 'ready'
        else:
            job['status'] = 'failed'
            job['error'] = f"Unknown job type: {jtype}"
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        print('Job error:', traceback.format_exc())
    finally:
        job['updated_at'] = datetime.utcnow().isoformat() + 'Z'


# -------------------------
# Routes (returning TS-shaped JSON)
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
        # try to extract duration via moviepy (optional)
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

    # return formatted job for frontend
    with jobs_lock:
        job_entry = jobs[job_id]
    return jsonify({'job_id': job_id, 'status': 'queued', 'job': format_job_for_frontend(job_entry)}), 201


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
    transcription_obj = format_transcription_for_frontend(job_id, text, segments_raw,
                                                          language=job['meta'].get('language', 'en'), duration=duration)
    return jsonify(transcription_obj), 200


@app.route('/api/v1/jobs/<job_id>/summary', methods=['POST'])
def api_create_summary(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json() or {}
    style = data.get('style', 'short')
    text = job.get('result', {}).get('transcription') or data.get('text')
    if not text:
        return jsonify({'error': 'No text available for summarization'}), 400
    summary_job = create_job_entry('summary', meta={'text': text, 'style': style})
    job_queue.put(summary_job)
    # return the new job (formatted)
    with jobs_lock:
        sj = jobs[summary_job]
    return jsonify({'summary_job_id': summary_job, 'status': 'queued', 'job': format_job_for_frontend(sj)}), 201


@app.route('/api/v1/jobs/<job_id>/qa', methods=['POST'])
def api_create_qa(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    question = None
    requirement_text = None
    if request.content_type and 'multipart/form-data' in request.content_type:
        question = request.form.get('question')
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
                    requirement_text = ''.join(pages)
                except Exception:
                    requirement_text = None
            else:
                path = os.path.join(RAW_UPLOAD_DIR,
                                    f"req_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{r_ext}")
                req_file.save(path)
                requirement_text = None
    else:
        data = request.get_json() or {}
        question = data.get('question')
        requirement_text = data.get('requirement_text')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    context_text = job.get('result', {}).get('transcription') or request.form.get('context') or (
        request.json.get('context') if request.json else '')

    qa_job = create_job_entry('qa', meta={'context_text': context_text, 'question': question,
                                          'requirement_text': requirement_text})
    job_queue.put(qa_job)

    with jobs_lock:
        qj = jobs[qa_job]
    return jsonify({'qa_job_id': qa_job, 'status': 'queued', 'job': format_job_for_frontend(qj)}), 201


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
    fmt = request.args.get('format', 'txt')
    if job['status'] != 'ready':
        return jsonify({'status': job['status']}), 202
    if job['type'] == 'transcription':
        text = job['result'].get('transcription', '')
        out = io.BytesIO(text.encode('utf-8'))
        out.seek(0)
        return send_file(out, as_attachment=True, download_name=f"{job_id}.txt")
    elif job['type'] == 'summary':
        text = job['result'].get('summary', '')
        out = io.BytesIO(text.encode('utf-8'))
        out.seek(0)
        return send_file(out, as_attachment=True, download_name=f"{job_id}_summary.txt")
    elif job['type'] == 'qa':
        text = job['result'].get('answer', '')
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
