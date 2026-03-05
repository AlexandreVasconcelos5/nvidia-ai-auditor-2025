"""database.py file.

Persistence, observability and telemetry utilities for the RAG audit framework.

Handles SQLite audit logging, Streamlit observability and telemetry dashboard, and Qdrant Cloud Vector Database loading."""

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
import os
import logging
import sqlite3
from pathlib import Path
from typing import List

# -------------------------------------------------------------------------------
# Data Analysis and UI Imports
# -------------------------------------------------------------------------------
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Qdrant Cloud Vector Database and Embeddings Imports
# ------------------------------------------------------------------------------
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_audit_framework")

# ------------------------------------------------------------------------------
# Database Configuration
# ------------------------------------------------------------------------------
database_path = Path("rag_audit_telemetry.db")
collection_name = "ai_auditor_2025"
hugging_face_embeddings_model = "sentence-transformers/all-mpnet-base-v2"

# ------------------------------------------------------------------------------
# Audit Database Utilities
# ------------------------------------------------------------------------------
def initialize_rag_audit_database() -> None:
    """Initialize the SQLite audit telemetry database if it doesn't exist."""

    with sqlite3.connect(str(database_path), timeout=10) as connection:
        cursor = connection.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS audit_telemetry_table (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                       user_query TEXT NOT NULL,
                       consulted_pages TEXT,
                       model_response TEXT NOT NULL,
                       faithfulness_model_response_score REAL,
                       answer_relevancy_score REAL,
                       context_recall_score REAL,
                       context_precision_score REAL,
                       hybrid_retrieval_latency REAL,
                       re_ranking_latency REAL,
                       model_response_ragas_metrics_latency REAL,
                       total_pipeline_latency REAL,
                       total_token_consumption INTEGER)""")
        connection.commit()

def insert_new_interaction_record_audit_telemetry_table(
        user_query: str,
        consulted_pages: List[str],
        model_response: str,
        faithfulness_model_response_score: float,
        answer_relevancy_score: float,
        context_recall_score: float,
        context_precision_score: float,
        hybrid_retrieval_latency: float,
        re_ranking_latency: float,
        model_response_ragas_metrics_latency: float,
        total_pipeline_latency: float,
        total_token_consumption: int) -> None:
    
    """Insert a new interaction record into the audit telemetry table."""

    initialize_rag_audit_database()

    formatted_consulted_pages = ", ".join(map(str, sorted(set(consulted_pages)))) if consulted_pages else ""


    with sqlite3.connect(str(database_path), timeout=10) as connection:
        cursor = connection.cursor()
        cursor.execute("""
                       INSERT INTO audit_telemetry_table (
                       user_query,
                       consulted_pages,
                       model_response,
                       faithfulness_model_response_score,
                       answer_relevancy_score,
                       context_recall_score,
                       context_precision_score,
                       hybrid_retrieval_latency,
                       re_ranking_latency,
                       model_response_ragas_metrics_latency,
                       total_pipeline_latency,
                       total_token_consumption)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                       (user_query,
                        formatted_consulted_pages,
                        model_response,
                        faithfulness_model_response_score,
                        answer_relevancy_score,
                        context_recall_score,
                        context_precision_score,
                        hybrid_retrieval_latency,
                        re_ranking_latency,
                        model_response_ragas_metrics_latency,
                        total_pipeline_latency,
                        total_token_consumption))
        connection.commit()

# ------------------------------------------------------------------------------
# Observability & Telemetry Dashboard
# ------------------------------------------------------------------------------
def display_observability_telemetry_dashboard() -> None:
    """Render the Observability and Telemetry Dashboard."""

    initialize_rag_audit_database()
    st.markdown("---")

    try:
        with sqlite3.connect(str(database_path), timeout=10) as connection:
            audit_dataframe = pd.read_sql_query("SELECT * FROM audit_telemetry_table ORDER BY timestamp DESC", connection)
        if audit_dataframe.empty:
            st.info("The audit system is active, but no interaction records are available yet.")
            return
        
        audit_dataframe = audit_dataframe.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "user_query": "User Query",
            "consulted_pages": "Consulted Pages",
            "model_response": "Model Response",
            "faithfulness_model_response_score": "Faithfulness",
            "answer_relevancy_score": "Answer Relevancy",
            "context_recall_score": "Context Recall",
            "context_precision_score": "Context Precision",
            "hybrid_retrieval_latency": "Hybrid Retrieval Latency (s)",
            "re_ranking_latency": "Re-Ranking Latency (s)",
            "model_response_ragas_metrics_latency": "Model Response and RAGAS Metrics Latency (s)",
            "total_pipeline_latency": "Total Pipeline Latency (s)",
            "total_token_consumption": "Total Token Consumption"})

        st.header("Observability & Telemetry Dashboard")

        csv_report = audit_dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        
        st.download_button(
            label="Download the Observability & Telemetry Report",
            data=csv_report,
            file_name="rag_audit_telemetry_report.csv",
            mime="text/csv")
        st.dataframe(audit_dataframe, use_container_width=True, hide_index=True)

    except Exception as error:
        logger.exception("Error loading the observability & telemetry dashboard.")
        st.error(f"Failed to load the audit telemetry data: {error}")

# ------------------------------------------------------------
# Hugging Face Embeddings Model Loader
# ------------------------------------------------------------
@st.cache_resource
def load_hugging_face_embeddings_model():
    """Load and cache the Hugging Face Embeddings Model."""
    logger.info("Loading the Hugging Face Embeddings Model...")
    return HuggingFaceEmbeddings(model_name=hugging_face_embeddings_model)

# --------------------------------------------------------------
# Qdrant Cloud Vector Loader
# --------------------------------------------------------------
def load_qdrant_cloud_vector_database():
    """Connect to the Qdrant Cloud Vector Database."""
    
    qdrant_cloud_url = os.getenv("QDRANT_URL")
    qdrant_cloud_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_cloud_url or not qdrant_cloud_api_key:
        logger.error("QDRANT_URL or QDRANT_API_KEY not found in the environment variables.")
        raise ValueError("Missing Qdrant Cloud credentials in the .env file.")

    qdrant_client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_cloud_api_key)

    embeddings = load_hugging_face_embeddings_model()
    
    logger.info(f"Successfully connected to the Qdrant Cloud Vector collection: {collection_name}")
    
    return QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)