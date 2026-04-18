# app/chain.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.retriever import load_vectorstore

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 3

# ── Prompt Template ────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an expert assistant on the Dutch Constitution (2023).
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly:
"I cannot find this in the provided context of the Constitution."
Always cite the specific Article number when possible.
Be concise and precise.

Context:
{context}

Question:
{question}

Answer:
"""


def format_docs(docs):
    """
    Combine retrieved Document objects into a single context string.
    Each chunk is labelled with its page number for traceability.
    """
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


# app/chain.py
# Add this at module level — outside any function
_chain_cache = None

def build_chain():
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache          # reuse if already built
    
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    _chain_cache = chain             # cache it
    return chain

def ask(question: str) -> str:
    """Public interface — ask a question, get a grounded answer."""
    chain = build_chain()
    return chain.invoke(question)


# ── Manual test ────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What does the constitution say about privacy?",
        "How can the constitution be revised?",
        "What are the rules about freedom of speech?",
        "Who appoints the Prime Minister?",
        "What is the right to education in the Netherlands?"
    ]

    for question in test_questions:
        print(f"\n❓ {question}")
        print("─" * 60)
        answer = ask(question)
        print(f"💬 {answer}")
        print()