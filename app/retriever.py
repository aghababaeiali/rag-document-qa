# app/retriever.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Configuration ──────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # number of chunks to retrieve per query

_vectorstore_cache = None  # simple in-memory cache for the vectorstore instance

def load_vectorstore():
    global _vectorstore_cache
    
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    _vectorstore_cache = vectorstore
    return vectorstore


def retrieve(query: str, k: int = TOP_K):
    """
    Given a query string, return the top-k most relevant chunks.
    Returns a list of LangChain Document objects with .page_content and .metadata
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results


def retrieve_with_scores(query: str, k: int = TOP_K):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k + 2)  # fetch extra

    # Deduplicate by content
    seen = set()
    deduplicated = []
    for doc, score in results:
        content_key = doc.page_content[:100]  # first 100 chars as fingerprint
        if content_key not in seen:
            seen.add(content_key)
            deduplicated.append((doc, score))

    return deduplicated[:k]  # return only top k after deduplication


# ── Manual test — run this file directly to verify retrieval works ──
if __name__ == "__main__":
    test_query = "What does the constitution say about privacy?"

    print(f"\n🔍 Query: {test_query}")
    print("=" * 60)

    results = retrieve_with_scores(test_query)

    for i, (doc, score) in enumerate(results):
        print(f"\n📄 Chunk {i+1} | Score: {score:.4f}")
        print(f"   Page: {doc.metadata.get('page', 'N/A')}")
        print(f"   Content: {doc.page_content[:200]}...")