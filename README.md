# 🇳🇱 Dutch Constitution Q&A — RAG Pipeline with Evaluation

A production-style Retrieval-Augmented Generation (RAG) system that answers questions about the **Dutch Constitution (2023)** with Article citations, grounded entirely in the source document. Includes an automated evaluation layer using RAGAS.

**[▶ Live Demo](https://huggingface.co/spaces/aliabbi/dutch-constitution-qa)** · **[API Docs](https://your-render-url/docs)**

---

## Why This Project

Most junior ML portfolios show fine-tuning. This project demonstrates a different and increasingly demanded skill set: **building a complete GenAI system** — from document ingestion to evaluation — using the tools Dutch ML teams actually use in production.

The evaluation layer (RAGAS) is the differentiator. It gives quantitative, reproducible answers to "how good is your RAG pipeline?" — something most tutorials skip entirely.

---

## Results

| Metric | Score | What It Means |
|---|---|---|
| **Faithfulness** | 0.833 | Answers stay grounded in retrieved context — low hallucination |
| **Answer Relevancy** | 0.657 | Answers address the question asked |

Evaluated on 5 held-out question-answer pairs from the Dutch Constitution using RAGAS with Llama 3.1 as the judge model.

---

## Architecture

```
PDF Document (Dutch Constitution 2023)
         ↓  [ingest.py]
Load → Chunk (500 chars, 50 overlap) → Embed (MiniLM-L6-v2) → ChromaDB
         ↓  [retriever.py]
Query → Embed → Cosine similarity search → Top-3 chunks
         ↓  [chain.py]
Chunks + Question → Prompt → Llama 3.1 (Groq) → Grounded answer
         ↓  [evaluate.py]
RAGAS → Faithfulness + Answer Relevancy scores
```

**Serving layer:**
- `main.py` — FastAPI REST API (`POST /ask`)
- `demo.py` — Gradio web UI (HuggingFace Spaces)

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain (LCEL) | Standard in NL job market; provider-agnostic abstraction |
| Vector store | ChromaDB | Zero-config local persistence, cosine similarity |
| Embeddings | `all-MiniLM-L6-v2` | Free, CPU-efficient, strong semantic retrieval baseline |
| LLM | Llama 3.1 8B via Groq | Free tier, fast inference, open-source model |
| Evaluation | RAGAS | Reference-free RAG evaluation — faithfulness + relevancy |
| API | FastAPI | Auto-generated docs, Pydantic validation, async support |
| UI | Gradio | Standard ML demo interface |
| Container | Docker | Reproducible deployment, CPU-only PyTorch for lean image |

---

## Project Structure

```
rag-document-qa/
│
├── app/
│   ├── ingest.py        # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py     # Semantic search over stored vectors
│   ├── chain.py         # RAG chain: retrieve → prompt → LLM
│   └── evaluate.py      # RAGAS evaluation pipeline
│
├── data/
│   └── sample_docs/     # Source PDFs go here
│
├── tests/
│   └── test_pipeline.py # Pytest suite
│
├── main.py              # FastAPI REST API
├── demo.py              # Gradio web UI
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11
- [Groq API key](https://console.groq.com) (free)

### Setup

```bash
git clone https://github.com/aghababaeiali/rag-document-qa
cd rag-document-qa

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY to .env

# Download the Dutch Constitution PDF:
# https://www.government.nl/documents/reports/2023/04/01/the-constitution-of-the-kingdom-of-the-netherlands-2023
# Save it to: data/sample_docs/grondwet-koninkrijk-ENG-V4.pdf
```

### Ingest the document

```bash
python -m app.ingest
```

This loads the PDF, splits it into ~100 chunks, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB. Runs once.

### Run the Gradio demo

```bash
python demo.py
# → http://localhost:7860
```

### Run the FastAPI server

```bash
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive API docs)
```

### Run evaluation

```bash
python -m app.evaluate
```

---

## Example Queries

| Question | Answer |
|---|---|
| What does the constitution say about privacy? | Article 10 guarantees the right to privacy, with restrictions only by Act of Parliament |
| Who appoints the Prime Minister? | The Prime Minister is appointed by Royal Decree (Article 43) |
| Can capital punishment be imposed? | No — Article 114 explicitly prohibits it |
| How can the constitution be revised? | Two-stage process requiring two-thirds majority in both Houses (Article 137) |

---

## Docker

```bash
# Build
docker build -t dutch-constitution-rag .

# Run
docker run -p 7860:7860 --env-file .env dutch-constitution-rag
```

Uses CPU-only PyTorch for a lean image. No GPU required — the embedding model runs efficiently on CPU.

---

## Known Limitations & Next Steps

**Current limitations (identified through evaluation):**

- **Chunk boundary bleeding** — `RecursiveCharacterTextSplitter` splits by character count, not Article boundaries. Some chunks contain content from two adjacent Articles, causing partial faithfulness failures on Q2 and Q4.
- **Retrieval misses on broad questions** — Q3 (freedom of speech) scored low on answer relevancy because the retrieved chunk covered only one sub-point of Article 7, not the full scope.

**Improvements I'd make with more time:**

1. **Structure-aware chunking** — split on "Article N" regex patterns instead of character count → cleaner, semantically complete chunks
2. **Reranker** — add a cross-encoder reranking stage (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after initial retrieval to improve precision
3. **Hybrid search** — combine dense (vector) + sparse (BM25) retrieval to catch exact Article number matches
4. **Streaming API** — add streaming response to FastAPI for better UX on long answers

---

## Evaluation Details

RAGAS runs 5 test questions through the pipeline and uses an LLM judge to score:

- **Faithfulness** — are all claims in the answer supported by retrieved context?
- **Answer Relevancy** — does the answer address what was actually asked?

```bash
python -m app.evaluate

# Output:
# Faithfulness:     0.833
# Answer Relevancy: 0.657
```

Per-question scores reveal which queries suffer from retrieval failures vs. generation failures — actionable signal for improvement.

---

## About

Built as part of a portfolio project while on orientation year visa in the Netherlands, targeting NLP/GenAI engineering roles.

**Author:** Ali Aghababaei
**Background:** MSc ICT (University of Padova) · Thesis on Explainability via Integrated Gradients
**Contact:** [LinkedIn](https://linkedin.com/in/aliaghababaeii) · [Portfolio](https://aghababaeiali.github.io)
