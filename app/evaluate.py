# app/evaluate.py

import json
import mlflow
from pathlib import Path

import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig
from app.chain import ask
from app.retriever import load_vectorstore

load_dotenv()

# ── MLflow Setup ───────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "rag-evaluation"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Configuration ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "llama-3.1-8b-instant"        # ← NEW
CHUNK_SIZE = 500                           # ← NEW
CHUNK_OVERLAP = 50                         # ← NEW

# ── Test Set ───────────────────────────────────────────────────
# These are question + ground truth pairs from the actual Constitution
# Ground truth = what the correct answer should be based on the document
TEST_SET = [
    {
        "question": "What does the constitution say about privacy?",
        "ground_truth": "Article 10 states that everyone shall have the right to respect for his privacy, without prejudice to restrictions laid down by or pursuant to Act of Parliament."
    },
    {
        "question": "Who appoints the Prime Minister?",
        "ground_truth": "The Prime Minister and other Ministers are appointed and dismissed by Royal Decree according to Article 43."
    },
    {
        "question": "What are the rules about freedom of speech?",
        "ground_truth": "Article 7 states that no one shall require prior permission to publish thoughts or opinions through the press, without prejudice to the responsibility of every person under the law."
    },
    {
        "question": "How many members does the Lower House have?",
        "ground_truth": "According to Article 51, the Lower House consists of one hundred and fifty members."
    },
    {
        "question": "Can capital punishment be imposed in the Netherlands?",
        "ground_truth": "No. Article 114 states that capital punishment may not be imposed."
    },
]


def build_eval_dataset():
    """
    Run each test question through the RAG pipeline and collect:
    - question
    - generated answer
    - retrieved contexts
    - ground truth
    RAGAS needs all four columns to compute its metrics.
    """
    print("🔄 Running test questions through RAG pipeline...")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in TEST_SET:
        question = item["question"]
        print(f"   ❓ {question}")

        # Get generated answer from our chain
        answer = ask(question)

        # Get retrieved contexts for this question
        docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)      # list of strings per question
        ground_truths.append(item["ground_truth"])

    # RAGAS expects a HuggingFace Dataset object
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation():
    """Run RAGAS evaluation and log everything to MLflow."""

    with mlflow.start_run() as run:
        print(f"\n🔬 MLflow run started: {run.info.run_id}")

        # ── Log parameters (the config you chose) ──────────────────
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("embedding_model", EMBED_MODEL)
        mlflow.log_param("llm_model", LLM_MODEL)
        mlflow.log_param("test_set_size", len(TEST_SET))

        # ── Run the evaluation (existing logic) ────────────────────
        dataset = build_eval_dataset()

        print("\n📊 Running RAGAS evaluation...")

        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0
        )
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        results = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy(strictness=1)],
            llm=groq_llm,
            embeddings=hf_embeddings,
            run_config=RunConfig(max_workers=1)
        )

        # ── Process results ────────────────────────────────────────
        df = results.to_pandas()
        faith_avg = df['faithfulness'].mean()
        relevancy_avg = df['answer_relevancy'].mean()

        # ── Log metrics ────────────────────────────────────────────
        mlflow.log_metric("faithfulness_avg", faith_avg)
        mlflow.log_metric("answer_relevancy_avg", relevancy_avg)

        # Per-question metrics for fine-grained comparison
        for i, row in df.iterrows():
            mlflow.log_metric(f"faithfulness_q{i+1}", row['faithfulness'])
            mlflow.log_metric(f"relevancy_q{i+1}", row['answer_relevancy'])

        # ── Log artifact (results JSON) ────────────────────────────
        artifact_path = Path("evaluation_outputs/results.json")
        artifact_path.parent.mkdir(exist_ok=True)

        with open(artifact_path, "w") as f:
            json.dump({
                "summary": {
                    "faithfulness_avg": float(faith_avg),
                    "answer_relevancy_avg": float(relevancy_avg),
                    "chunk_size": CHUNK_SIZE,
                    "top_k": TOP_K,
                    "embedding_model": EMBED_MODEL,
                    "llm_model": LLM_MODEL,
                },
                "per_question": [
                    {
                        "question": row['user_input'],
                        "answer": row['response'],
                        "faithfulness": float(row['faithfulness']),
                        "answer_relevancy": float(row['answer_relevancy']),
                    }
                    for _, row in df.iterrows()
                ]
            }, f, indent=2)

        mlflow.log_artifact(str(artifact_path))

        # ── Print results (your existing output) ───────────────────
        print("\n✅ Evaluation complete!")
        print("=" * 50)
        print(f"  Faithfulness:     {faith_avg:.3f}  (1.0 = perfect, no hallucination)")
        print(f"  Answer Relevancy: {relevancy_avg:.3f}  (1.0 = always on-topic)")
        print("=" * 50)

        print("\n📋 Per-question breakdown:")
        for i, row in df.iterrows():
            print(f"\n  Q{i+1}: {row['user_input'][:60]}...")
            print(f"       Faithfulness:     {row['faithfulness']:.3f}")
            print(f"       Answer Relevancy: {row['answer_relevancy']:.3f}")
            print(f"       Answer: {row['response'][:100]}...")

        print(f"\n📊 Logged to MLflow:")
        print(f"   Experiment: {EXPERIMENT_NAME}")
        print(f"   Run ID: {run.info.run_id}")


if __name__ == "__main__":
    run_evaluation()