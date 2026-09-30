import os
import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        }
    )


def build_vector_store(
    transcript: str
) -> Chroma:

    print("Building vector store...")

    if not transcript.strip():
        raise ValueError(
            "Cannot create vector store from empty transcript."
        )


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(
        transcript
    )


    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]


    # Unique collection for every meeting
    collection_name = (
        "meeting_"
        + uuid.uuid4().hex
    )


    embeddings = get_embeddings()


    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )


    print(
        f"Vector store created with "
        f"{len(documents)} chunks."
    )

    return vector_store


def get_retriever(
    vector_store: Chroma,
    k: int = 6
):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )