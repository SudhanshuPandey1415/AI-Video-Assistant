# 🎙️ AI Video & Meeting Assistant

> An end-to-end AI meeting intelligence assistant: automatic audio extraction, multi-lingual transcription (OpenAI Whisper + Mistral Voxtral), map-reduce summarization, key decisions and action items extraction, and an interactive RAG-powered Q&A chat over meeting transcripts.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="Image/Screenshot%202026-09-30%20230422.png" alt="AI Video Assistant UI Dashboard" width="100%" />
</p>

---

## 🌟 Key Features

- **Multi-Source Ingestion**: Process YouTube video URLs directly (via `yt-dlp`) or upload/specify local audio/video files (`.wav`, `.mp3`, `.mp4`, etc.).
- **Smart Audio Processing**: Automated format conversion, mono channel normalization, 16kHz resampling, and 10-minute intelligent chunking.
- **Hybrid Speech-to-Text Routing**:
  - **English**: Local OpenAI Whisper inference (leveraging local GPU/CUDA acceleration).
  - **Hinglish / Hindi**: Cloud-based transcription using Mistral Voxtral (`voxtral-mini-latest`).
- **Comprehensive Meeting Intelligence**:
  - **Automated Title Generation**: Short, context-aware meeting header.
  - **Map-Reduce Summarization**: Recursive chunk summarization combined into a clean, bulleted executive summary.
  - **Structured Extraction**: Automatically detects action items (with owner and deadline), key decisions made, and unresolved questions.
- **RAG-Powered Chat Assistant**:
  - Semantic chunking with recursive text splitters.
  - Vector indexing via ChromaDB and `all-MiniLM-L6-v2` HuggingFace embeddings.
  - Context-grounded Q&A to answer specific questions from the meeting transcript without hallucinations.
- **Modern Glassmorphic Web UI**:
  - Responsive, dark-mode dashboard with real-time analysis progress reporting.
  - Interactive transcript viewer, structured summary tabs, and chat interface.
  - Also includes a standalone CLI runner (`main.py`).

---

## 🏗️ Architecture

```
                  ┌──────────────────────┐
                  │ YouTube URL / Local  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Audio Preprocessing │ (yt-dlp, pydub, ffmpeg)
                  │  (16kHz WAV Chunks)  │
                  └──────────┬───────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
    [English Audio]                  [Hinglish Audio]
             ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │  OpenAI Whisper │             │ Mistral Voxtral │
    │   (Local CUDA)  │             │   (Cloud API)   │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Full Meeting         │
                  │ Transcript           │
                  └──────────┬───────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Summarizer   │     │  Extractor   │     │ Vector Store     │
│ (Map-Reduce) │     │(Action Items,│     │(ChromaDB + MiniLM│
│              │     │  Decisions)  │     │   Embeddings)    │
└──────┬───────┘     └──────┬───────┘     └────────┬─────────┘
       │                    │                      │
       └────────────────────┼──────────────────────┘
                            ▼
              ┌───────────────────────────┐
              │   FastAPI Web Service     │
              │  & Glassmorphism Frontend │
              │   + RAG Meeting Chat      │
              └───────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── api.py                    # FastAPI application server and REST endpoints
├── main.py                   # Complete pipeline runner & interactive CLI interface
├── requirements.txt          # Python dependencies
├── .env.example              # Sample environment variables configuration
├── .gitignore                # Git exclusions (credentials, models, media)
├── .gitattributes            # Line ending normalization
├── LICENSE                   # MIT License
│
├── core/                     # Pipeline modules
│   ├── transcriber.py        # Speech-to-text routing (Whisper + Mistral Voxtral)
│   ├── summarizer.py         # Title generation and Map-Reduce summarization
│   ├── extractor.py          # Action items, key decisions, and questions extraction
│   ├── vector_store.py       # ChromaDB vector store builder & similarity retriever
│   └── rag_engine.py         # RAG question-answering chain
│
├── utils/                    # Utility helpers
│   └── audio_processor.py    # YouTube download, audio conversion, and chunking
│
├── frontend/                 # Web client dashboard
│   ├── index.html            # Main dashboard interface
│   ├── style.css             # Glassmorphism aesthetic styling
│   └── app.js                # Frontend logic & API interaction
│
└── downloads/                # Working directory for audio processing (git-ignored)
```

---

## 🛠️ Prerequisites

1. **Python**: Version `3.10` or higher.
2. **FFmpeg**: Required by `pydub` and `yt-dlp` for audio extraction and conversion.
   - **Windows**: Install via `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH`.
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
3. **NVIDIA GPU (Optional)**: CUDA-enabled GPU for faster local Whisper transcription. Whisper will fall back to CPU if CUDA is unavailable.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-video-assistant.git
cd ai-video-assistant
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Duplicate `.env.example` as `.env` and enter your API credentials:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in:

```env
# Required for Summaries, Key Extraction, RAG, and Voxtral
MISTRAL_API_KEY=your_mistral_api_key_here

# Whisper model size (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL=small
```

---

## 💻 Running the Application

### Option A: Web Dashboard (Recommended)

Start the FastAPI application:

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

From the dashboard you can:
1. Provide a YouTube link or local file path.
2. Select language (`English` or `Hinglish`).
3. Monitor real-time transcription and analysis stages.
4. Review generated summaries, action items, and decisions.
5. Ask questions to chat with the meeting via RAG.

---

### Option B: Command-Line Interface (CLI)

Run the standalone CLI pipeline:

```bash
python main.py
```

Follow the prompts to enter your audio source, review terminal output, and chat interactively.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check endpoint |
| `POST` | `/api/analyze` | Queue a new video analysis job |
| `GET` | `/api/status/{job_id}` | Check job progress and status |
| `GET` | `/api/result/{job_id}` | Retrieve completed transcript, summary, and action items |
| `POST` | `/api/chat/{job_id}` | Ask questions against the processed meeting transcript |
| `DELETE` | `/api/jobs/{job_id}` | Terminate and clear a job session |

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

---

## 🛡️ Security & Privacy Notice

- API keys must never be committed to source control. Ensure `.env` is always included in `.gitignore`.
- Video downloads and generated vector embeddings (`downloads/` and `vector_db/`) are strictly kept locally and excluded from git tracking.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
