"""hybrid_information_retrieval_reranking.py file.

Hybrid Retrieval & Cross-Encoder Re-Ranking Module.

Handles:
- Qdrant Cloud Vector Database;
- BM25 Keyword Search;
- Flashrank Re-Ranking."""

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
import time
import logging

# -------------------------------------------------------------------------------
# LangChain and Retrieval Imports
# -------------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from rank_bm25 import BM25Okapi

# -------------------------------------------------------------------------------
# Qdrant Cloud Vector Database Utilities
# -------------------------------------------------------------------------------
from database import load_qdrant_cloud_vector_database

# -------------------------------------------------------------------------------
# Logger Instance
# -------------------------------------------------------------------------------
logger = logging.getLogger("retrieval_logger")

# -------------------------------------------------------------------------------
# BM25 Search Index Initialization
# -------------------------------------------------------------------------------
def create_bm25_search_index(documents: list[Document]):
    """Create a BM25 search index from the text chunks for a keyword-based information retrieval."""

    tokenized_documents = [document.page_content.lower().split() for document in documents]
    bm25_index = BM25Okapi(tokenized_documents)
    return bm25_index, documents

# -------------------------------------------------------------------------------
# Hybrid Retrieval Engine Initialization
# -------------------------------------------------------------------------------
def initialize_hybrid_retrieval_engine():
    """Load the Qdrant Cloud Vector Database, the BM25 Index and the Flashrank Re-Ranker."""

    qdrant_cloud_vector_database = load_qdrant_cloud_vector_database()
    
    documents_for_bm25 = []
    
    try:
        scroll_result = qdrant_cloud_vector_database.client.scroll(collection_name=qdrant_cloud_vector_database.collection_name, with_payload=True,
            with_vectors=False, limit=10000)
        
        points = scroll_result[0]

        if not points:
            logger.warning("No points found in the Qdrant Cloud Vector Database.")
            raise ValueError("Qdrant collection is empty. Please, first ingest the documents.")

        for point in points:
            page_content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})
            if page_content:
                documents_for_bm25.append(Document(page_content=page_content, metadata=metadata))

    except Exception as exception:
        logger.error(f"Failed to fetch the documents from Qdrant Cloud Vector Database: {exception}")
        raise

    bm25_index, bm25_documents = create_bm25_search_index(documents_for_bm25)
    
    flashrank_reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

    return qdrant_cloud_vector_database, flashrank_reranker, bm25_index, bm25_documents

# -------------------------------------------------------------------------------
# Pipeline Latency Measurement
# -------------------------------------------------------------------------------
def log_pipeline_phase_latency(start_time: float, phase_name: str) -> float:
    """Measure and log the latency of the pipeline phases."""

    end_time = time.perf_counter()
    latency = end_time - start_time
    logger.info(f"Phase {phase_name} completed in {latency:.3f} seconds.")
    return latency

# -------------------------------------------------------------------------------
# Hybrid Retrieval & Re-Ranking
# -------------------------------------------------------------------------------
def execute_hybrid_retrieval_and_reranking(user_query: str, qdrant_vector_database, flashrank_reranker, bm25_index, bm25_documents, top_k: int = 15):
    """Performs hybrid retrieval (Qdrant Cloud Vector Database + BM25) and cross-encoder re-ranking."""

    qdrant_vector_database_retriever = qdrant_vector_database.as_retriever(search_kwargs={"k": top_k})
    qdrant_vector_database_documents = qdrant_vector_database_retriever.invoke(user_query)

    tokenized_user_query = user_query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_user_query)
    top_bm25_indexes = bm25_scores.argsort()[-top_k:][::-1]
    top_bm25_documents = [bm25_documents[index] for index in top_bm25_indexes]

    hybrid_retrieval_documents = list(qdrant_vector_database_documents)
    seen_contents = set([document.page_content for document in hybrid_retrieval_documents])

    for document in top_bm25_documents:
        if document.page_content not in seen_contents:
            hybrid_retrieval_documents.append(document)
            seen_contents.add(document.page_content)

    re_ranking_start = time.perf_counter()
    re_ranked_documents = flashrank_reranker.compress_documents(documents=hybrid_retrieval_documents, query=user_query)
    documents = re_ranked_documents[:5] if re_ranked_documents else hybrid_retrieval_documents[:5]
    
    pages = set()
    list_documents = []
    for document in documents:
        try:
            raw_page = document.metadata.get("page", 0)
            page_val = int(raw_page) 
        except (ValueError, TypeError):
            page_val = 0
            
        if page_val not in pages:
            list_documents.append(document)
            pages.add(page_val)

    list_documents.sort(key=lambda doc: int(doc.metadata.get("page", 0)) if str(doc.metadata.get("page", "")).isdigit() else 0)

    re_ranking_latency = log_pipeline_phase_latency(re_ranking_start, "Re-Ranking")

    return list_documents, re_ranking_latency
