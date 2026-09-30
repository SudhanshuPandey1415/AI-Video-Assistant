from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str = None, language: str = "english", progress_callback=None) -> dict:
    """
    Run the complete AI Video Assistant pipeline.

    progress_callback(percent, message) is optional and is used by FastAPI
    to report live progress to the frontend.
    """

    def update(percent, message):
        if progress_callback:
            progress_callback(int(percent), message)
        print(f"[{percent}%] {message}")

    update(5, "Starting AI Video Assistant...")

    update(10, "Downloading / preparing audio...")
    chunks = process_input(source)

    if not chunks:
        raise RuntimeError("No audio chunks were created.")

    update(20, f"Audio ready — {len(chunks)} chunk(s) created.")

    def transcription_progress(current, total, message=None):
        # Reserve 20% -> 65% for transcription.
        percent = 20 + int((current / max(total, 1)) * 45)
        update(percent, message or f"Transcribing chunk {current}/{total}...")

    transcript = transcribe_all(
        chunks,
        language=language,
        progress_callback=transcription_progress
    )

    if not transcript.strip():
        raise RuntimeError("Transcription returned empty text.")

    update(67, "Transcription complete. Generating meeting title...")
    title = generate_title(transcript)

    update(71, "Generating full meeting summary...")
    summary = summarize(transcript)

    update(80, "Extracting action items...")
    action_item = extract_action_items(transcript)

    update(86, "Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    update(91, "Extracting open questions and follow-ups...")
    questions = extract_questions(transcript)

    update(94, "Building RAG vector store and retriever...")
    rag_chain = build_rag_chain(transcript)

    update(98, "Finalizing meeting intelligence...")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")
