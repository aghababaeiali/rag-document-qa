# app/evaluate.py

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

# ── Configuration ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

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
    """Run RAGAS evaluation and print results."""

    dataset = build_eval_dataset()

    print("\n📊 Running RAGAS evaluation...")

    # RAGAS needs an LLM and embeddings to compute metrics
    # We wrap our existing Groq LLM and HuggingFace embeddings
    groq_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
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

    print("\n✅ Evaluation complete!")
    print("=" * 50)

    # Convert to pandas for easy access
    df = results.to_pandas()

    # Compute averages manually from the dataframe
    faith_avg = df['faithfulness'].mean()
    relevancy_avg = df['answer_relevancy'].mean()  # ← not response_relevancy

    print(f"  Faithfulness:     {faith_avg:.3f}  (1.0 = perfect, no hallucination)")
    print(f"  Answer Relevancy: {relevancy_avg:.3f}  (1.0 = always on-topic)")
    print("=" * 50)

    # Fix per-question breakdown
    print("\n📋 Per-question breakdown:")
    for i, row in df.iterrows():
        print(f"\n  Q{i+1}: {row['user_input'][:60]}...")        # ← not 'question'
        print(f"       Faithfulness:     {row['faithfulness']:.3f}")
        print(f"       Answer Relevancy: {row['answer_relevancy']:.3f}")  # ← not response_relevancy
        print(f"       Answer: {row['response'][:100]}...")       # ← not 'answer'


if __name__ == "__main__":
    run_evaluation()