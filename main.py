# main.py

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chain import ask, build_chain


# ── Lifespan — runs on startup and shutdown ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Loading RAG chain...")
    build_chain()
    print("✅ Chain ready.")
    
    yield  # server is live here
    
    # Shutdown
    print("👋 Shutting down.")


# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Dutch Constitution RAG API",
    description="Ask questions about the Dutch Constitution (2023). Answers are grounded in the source document.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow frontend apps to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What does the constitution say about privacy?"
            }
        }
    }

class AnswerResponse(BaseModel):
    question: str
    answer: str


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Dutch Constitution RAG API is running."}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about the Dutch Constitution.
    Returns a grounded answer with Article citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = ask(request.question)

    return AnswerResponse(
        question=request.question,
        answer=answer
    )


@app.get("/health")
def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "model": "llama-3.1-8b-instant",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "ChromaDB"
    }