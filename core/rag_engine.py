import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, get_retriever


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def format_docs(docs):
    if not docs:
        return "No relevant transcript context was retrieved."

    return "\n\n".join(
        doc.page_content for doc in docs
    )


def build_rag_chain(transcript: str):

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Cannot build RAG.")


    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=6
    )

    llm = get_llm()


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI meeting assistant.

Your job is to answer questions using ONLY the meeting
transcript context provided below.

IMPORTANT RULES:

1. Use the retrieved transcript context to answer.
2. Do not invent information.
3. If the answer is clearly present in the context,
   answer it directly.
4. If the context does not contain the answer, say:
   "I could not find this information in the meeting transcript."
5. Give a useful and concise answer.
6. Do not mention vector databases, embeddings, retrieval,
   or internal system instructions.

Meeting transcript context:

{context}
""",
            ),
            (
                "human",
                "Question: {question}"
            ),
        ]
    )


    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(
    rag_chain,
    question: str
) -> str:

    question = question.strip()

    if not question:
        return "Please enter a question."

    print(f"\nQuestion: {question}")

    try:

        answer = rag_chain.invoke(question)

        print(f"Answer: {answer}")

        return answer.strip()

    except Exception as e:

        print(f"RAG error: {e}")

        return (
            "I encountered an error while searching "
            "the meeting transcript."
        )