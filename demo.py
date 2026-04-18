# demo.py

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr
from app.chain import ask, build_chain

import os
import urllib.request
from app.ingest import run_ingest

PDF_PATH = "data/sample_docs/grondwet-koninkrijk-ENG-V4.pdf"
PDF_URL = "https://open.overheid.nl/documenten/ronl-faa96875fef77af167a9133bd3625c0e9b45fa89/pdf"
CHROMA_DIR = "data/chroma_db"

# Create directories if they don't exist
os.makedirs("data/sample_docs", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

# Download PDF if not present
if not os.path.exists(PDF_PATH):
    print("📥 Downloading Dutch Constitution PDF...")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"✅ PDF downloaded to {PDF_PATH}")
else:
    print(f"✅ PDF already exists at {PDF_PATH}")

# Build ChromaDB if not present
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("⚙️ Building ChromaDB from PDF...")
    run_ingest()
    print("✅ Ingest complete.")
else:
    print("✅ ChromaDB already exists.")

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