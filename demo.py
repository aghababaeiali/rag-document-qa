# demo.py
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
import gradio as gr
from app.chain import ask, build_chain
from app.ingest import run_ingest, CHROMA_DIR

# Build ChromaDB on first startup if not present
# run_ingest now handles PDF download internally
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("⚙️ ChromaDB not found — running ingest pipeline...")
    run_ingest()
else:
    print("✅ ChromaDB already exists.")

# ── Examples shown in the UI ──────────────────────────────────
EXAMPLES = [
    "What does the constitution say about privacy?",
    "Who appoints the Prime Minister?",
    "Can capital punishment be imposed in the Netherlands?",
    "How can the constitution be revised?",
    "What are the rules about freedom of speech?",
    "How many members does the Lower House have?"
]

def answer_question(question: str) -> str:
    if not question.strip():
        return "⚠️ Please enter a question."
    return ask(question)

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask anything about the Dutch Constitution (2023)...", lines=2),
    outputs=gr.Textbox(label="Answer", lines=6),
    title="🇳🇱 Dutch Constitution Q&A",
    description=(
        "Ask questions about the **Dutch Constitution (2023)**. "
        "Answers are grounded in the source document using RAG with Article citations.\n\n"
        "**Stack:** LangChain · ChromaDB · sentence-transformers · Llama 3.1 via Groq"
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)