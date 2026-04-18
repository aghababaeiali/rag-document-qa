# demo.py

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr
from app.chain import ask, build_chain

import os
from app.ingest import run_ingest

# Auto-ingest if ChromaDB doesn't exist
CHROMA_DIR = "data/chroma_db"
PDF_PATH = "data/sample_docs/grondwet-koninkrijk-ENG-V4.pdf"

if not os.path.exists(CHROMA_DIR):
    if os.path.exists(PDF_PATH):
        print("⚙️ ChromaDB not found — running ingest...")
        run_ingest()
        print("✅ Ingest complete.")
    else:
        print("❌ PDF not found — cannot build ChromaDB")

# Pre-load the chain so first query isn't slow
print("🚀 Loading RAG chain...")
build_chain()
print("✅ Chain ready.")


# ── Example questions shown in the UI ─────────────────────────
EXAMPLES = [
    "What does the constitution say about privacy?",
    "Who appoints the Prime Minister?",
    "Can capital punishment be imposed in the Netherlands?",
    "How can the constitution be revised?",
    "What are the rules about freedom of speech?",
    "How many members does the Lower House have?"
]


def answer_question(question: str) -> str:
    """
    Wrapper function that Gradio calls.
    Takes the user's question, returns the RAG answer.
    """
    if not question.strip():
        return "⚠️ Please enter a question."
    return ask(question)


# ── Build the Gradio interface ─────────────────────────────────
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Your Question",
        placeholder="Ask anything about the Dutch Constitution (2023)...",
        lines=2
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=6
    ),
    title="🇳🇱 Dutch Constitution Q&A",
    description=(
        "Ask questions about the **Dutch Constitution (2023)**. "
        "Answers are grounded in the source document using RAG "
        "(Retrieval-Augmented Generation) with Article citations.\n\n"
        "**Stack:** LangChain · ChromaDB · sentence-transformers · Llama 3.1 via Groq"
    ),
    examples=EXAMPLES,
    cache_examples=False,  # don't pre-run examples, saves API calls
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible from any network interface
        server_port=7860,         # Gradio's default port
        share=False               # set True to get a temporary public URL
    )