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

import re
from langchain_core.documents import Document

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


def split_by_article(pages):
    """
    Structure-aware splitting: each chunk = one complete article.
    Falls back to character-based splitting for content without article markers.
    """
    print("✂️  Structure-aware splitting (by Article N)")

    # Combine all pages into one text, preserving page numbers as metadata
    full_text = ""
    page_offsets = []  # tracks where each page starts in the combined text
    
    for page in pages:
        page_offsets.append((len(full_text), page.metadata.get("page", 0)))
        full_text += page.page_content + "\n"

    def get_page_for_position(pos):
        """Find which page a character position belongs to."""
        last_page = 0
        for start, page_num in page_offsets:
            if start <= pos:
                last_page = page_num
            else:
                break
        return last_page

    # Regex matches: "Article 1", "Article 43", "Article 57a", "Article 132a"
    pattern = r'(?=\nArticle \d+[a-z]?\n)'
    article_chunks = re.split(pattern, full_text)

    chunks = []
    position = 0

    for chunk_text in article_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Identify the article number for metadata
        match = re.match(r'Article (\d+[a-z]?)', chunk_text)
        article_num = match.group(1) if match else "preamble"

        # Find which page this chunk starts on
        page_num = get_page_for_position(position)

        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "page": page_num,
                "article": article_num,
                "source": "grondwet-koninkrijk-ENG-V4.pdf"
            }
        ))
        position += len(chunk_text)

    print(f"   → {len(chunks)} article chunks created")
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
    chunks = split_by_article(pages)
    embed_and_store(chunks)
    print("\n✅ Ingest complete. ChromaDB is ready.")


if __name__ == "__main__":
    run_ingest()