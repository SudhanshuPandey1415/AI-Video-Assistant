import whisper
import os
from mistralai.client import Mistral


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_TRANSCRIBE_MODEL = "voxtral-mini-latest"

_model = None
_mistral_client = None


# ─────────────────────────────────────────────────────────────────────────────
# Whisper
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load Whisper model once and reuse it."""

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")

        _model = whisper.load_model(
            WHISPER_MODEL,
            device="cuda"
        )

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe one audio chunk using local Whisper."""

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Mistral Voxtral Transcription
# ─────────────────────────────────────────────────────────────────────────────

def get_mistral_client():
    """Create the Mistral client once and reuse it."""

    global _mistral_client

    if _mistral_client is None:

        if not MISTRAL_API_KEY:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set in environment / .env"
            )

        _mistral_client = Mistral(
            api_key=MISTRAL_API_KEY
        )

    return _mistral_client


def transcribe_chunk_mistral(
    chunk_path: str,
    language: str = "hinglish"
) -> str:
    """
    Transcribe one audio chunk using Mistral Voxtral.

    For Hinglish/Hindi, automatic language detection is used.
    """

    client = get_mistral_client()

    print(
        f"  → Transcribing {os.path.basename(chunk_path)} "
        f"using Mistral Voxtral..."
    )

    with open(chunk_path, "rb") as audio_file:

        transcription_response = client.audio.transcriptions.complete(
            model=MISTRAL_TRANSCRIBE_MODEL,
            file={
                "content": audio_file,
                "file_name": os.path.basename(chunk_path),
            },
            diarize=False
        )

    transcript = transcription_response.text

    if not transcript:
        raise RuntimeError(
            f"Mistral returned an empty transcript for {chunk_path}"
        )

    return transcript.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Single Chunk Router
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
    """
    Route one audio chunk to the appropriate transcription engine.

    english  → local Whisper on RTX 4050
    hinglish → Mistral Voxtral
    """

    if language.lower() == "hinglish":
        return transcribe_chunk_mistral(
            chunk_path,
            language=language
        )

    return transcribe_chunk_whisper(chunk_path)


# ─────────────────────────────────────────────────────────────────────────────
# Transcribe All Chunks
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_all(
    chunks: list,
    language: str = "english",
    progress_callback=None
) -> str:
    """
    Transcribe all audio chunks.

    progress_callback is optional.

    If supplied, it will be called after every completed chunk:

        progress_callback(completed, total, engine)

    Example:

        progress_callback(3, 8, "Whisper")
    """

    full_transcript = ""

    total_chunks = len(chunks)

    if total_chunks == 0:
        return ""

    engine = (
        "Mistral Voxtral"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(
        f"Using {engine} for transcription."
    )

    for i, chunk in enumerate(chunks):

        chunk_number = i + 1

        print(
            f"Transcribing chunk "
            f"{chunk_number}/{total_chunks}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

        # ─────────────────────────────────────────────────────────────────────
        # Report progress to FastAPI/frontend
        # ─────────────────────────────────────────────────────────────────────

        if progress_callback is not None:

            progress_callback(
                chunk_number,
                total_chunks,
                engine
            )

    print("Transcription complete.")

    return full_transcript.strip()