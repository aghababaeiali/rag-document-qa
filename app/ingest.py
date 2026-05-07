# app/ingest.py

import os
import urllib.request
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ──────────────────────────────────────────────
PDF_PATH = "data/sample_docs/grondwet-koninkrijk-ENG-V4.pdf"
PDF_URL = "https://open.overheid.nl/documenten/ronl-faa96875fef77af167a9133bd3625c0e9b45fa89/pdf"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ensure_pdf_exists():
    """Download the PDF if it isn't already on disk."""
    os.makedirs("data/sample_docs", exist_ok=True)
    
    if os.path.exists(PDF_PATH):
        print(f"✅ PDF already exists at {PDF_PATH}")
        return
    
    print(f"📥 Downloading PDF from {PDF_URL}")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"✅ PDF downloaded.")

def load_pdf(path: str):
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    print(f" Loading PDF: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"   → {len(pages)} pages loaded")
    return pages


def split_documents(pages):
    """Split pages into smaller chunks for retrieval."""
    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]  # try splitting at paragraphs first
    )
    chunks = splitter.split_documents(pages)
    print(f"   → {len(chunks)} chunks created")
    return chunks


def embed_and_store(chunks):
    """Embed each chunk and store in ChromaDB."""
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Storing embeddings in ChromaDB at: {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print(f"   → Done! {len(chunks)} chunks embedded and stored.")
    return vectorstore


def run_ingest():
    """Full ingest pipeline: download PDF (if needed) → load → split → embed → store."""
    ensure_pdf_exists()                 # ← NEW
    pages = load_pdf(PDF_PATH)
    chunks = split_documents(pages)
    embed_and_store(chunks)
    print("\n✅ Ingest complete. ChromaDB is ready.")


if __name__ == "__main__":
    run_ingest()