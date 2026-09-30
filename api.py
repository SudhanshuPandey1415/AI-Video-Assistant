import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import run_pipeline

load_dotenv()

app = FastAPI(
    title="AI Video Assistant",
    description="AI meeting transcription, summarization and RAG assistant",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

jobs: Dict[str, Dict[str, Any]] = {}
rag_chains: Dict[str, Any] = {}
jobs_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=2)


class AnalyzeRequest(BaseModel):
    source: str
    language: str = "english"


class ChatRequest(BaseModel):
    question: str


def update_job(job_id: str, progress: int, message: str, status="processing"):
    progress = max(0, min(100, int(progress)))

    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = message
            jobs[job_id]["status"] = status


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "AI Video Assistant"
    }


def process_job(job_id: str, source: str, language: str):
    try:
        update_job(job_id, 5, "Starting AI Video Assistant...")

        def progress_callback(percent, message):
            update_job(job_id, percent, message)

        result = run_pipeline(
            source=source,
            language=language,
            progress_callback=progress_callback,
        )

        rag_chain = result.pop("rag_chain", None)

        if rag_chain is not None:
            rag_chains[job_id] = rag_chain

        with jobs_lock:
            jobs[job_id]["progress"] = 100
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Analysis completed successfully."
            jobs[job_id]["result"] = result

        print(f"[{job_id}] Pipeline completed.")

    except Exception as e:
        print(f"[{job_id}] ERROR: {repr(e)}")

        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["progress"] = 0
                jobs[job_id]["message"] = str(e)
                jobs[job_id]["result"] = None


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    source = request.source.strip()
    language = request.language.strip().lower()

    if not source:
        raise HTTPException(
            status_code=400,
            detail="Source cannot be empty."
        )

    if language not in ["english", "hinglish"]:
        language = "english"

    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting to start...",
            "source": source,
            "language": language,
            "result": None,
        }

    executor.submit(
        process_job,
        job_id,
        source,
        language
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found."
            )

        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job["message"],
        }


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job["message"]
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Analysis is still running."
        )

    return {
        "success": True,
        "job_id": job_id,
        **job["result"],
    }


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Meeting analysis is not completed yet."
        )

    rag_chain = rag_chains.get(job_id)

    if rag_chain is None:
        raise HTTPException(
            status_code=410,
            detail="RAG session is no longer available."
        )

    try:
        from core.rag_engine import ask_question

        answer = await asyncio.to_thread(
            ask_question,
            rag_chain,
            question
        )

        return {
            "success": True,
            "job_id": job_id,
            "question": question,
            "answer": answer,
        }

    except Exception as e:
        print(f"[{job_id}] RAG ERROR: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG error: {str(e)}"
        )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(
                status_code=404,
                detail="Job not found."
            )

        jobs.pop(job_id, None)
        rag_chains.pop(job_id, None)

    return {
        "success": True,
        "message": "Job deleted."
    }


if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )


@app.get("/")
async def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        return {
            "message": "Frontend not found."
        }

    return FileResponse(index_file)
