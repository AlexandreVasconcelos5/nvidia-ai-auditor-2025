"""app.py file.

Streamlit interface for the AI Auditor, powered by Agentic Hybrid RAG.

Handles user query, displays chat messages and presents the RAGAS metrics."""

# -------------------------------------------------------------------------------
# SQLite Version Fix
# -------------------------------------------------------------------------------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
import time
import os
import logging

# -------------------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------------
import streamlit as st

# -------------------------------------------------------------------------------
# Hybrid Retrieval Utilities
# -------------------------------------------------------------------------------
from hybrid_information_retrieval_reranking import initialize_hybrid_retrieval_engine, execute_hybrid_retrieval_and_reranking
from document_cloud_ingestion import run_document_cloud_ingestion

# -------------------------------------------------------------------------------
# Self-Correction Agent
# -------------------------------------------------------------------------------
from self_correction_agent import self_correction_agent, value_int_conversion

# -------------------------------------------------------------------------------
# Observability & Telemetry
# -------------------------------------------------------------------------------
from database import insert_new_interaction_record_audit_telemetry_table, display_observability_telemetry_dashboard

# -------------------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------------------
logger = logging.getLogger("app.logger")

# -------------------------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------------------------
st.set_page_config(page_title="AI Auditor", page_icon="🔎")
st.title("📄 NVIDIA Corporation Annual Review 2025")
st.markdown("### Agentic Hybrid RAG - Financial Audit & Analysis Assistant")

# -------------------------------------------------------------------------------
# Chat History Management
# -------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------------------------------------
# Initialize the Hybrid Retrieval Engine
# -------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing the Hybrid Retrieval Engine resources...")
def load_hybrid_engine():
    return initialize_hybrid_retrieval_engine()

try:
    engine_resources = load_hybrid_engine()
    
    qdrant_cloud_vector_database = engine_resources[0]
    flashrank_reranker = engine_resources[1]
    bm25_index = engine_resources[2]
    bm25_documents = engine_resources[3]
    logger.info("The Hybrid Retrieval Engine resources were successfully loaded.")

except Exception as error:
    logger.critical(f"Failed to load the Hybrid Retrieval Engine: {error}")
    st.error("Hybrid Retrieval Engine Initialization Error")
    st.markdown(f"The system could not start the hybrid retrieval resources: {error}")
    st.stop()

# -------------------------------------------------------------------------------
# User Query Input
# -------------------------------------------------------------------------------
user_query = st.chat_input("Enter your query regarding the NVIDIA Corporation Annual Review 2025 report:")

if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    pipeline_start = time.perf_counter()
    logger.info(f"Processing user query: {user_query}")

    retrieval_start = time.perf_counter()

    documents, re_ranking_latency = execute_hybrid_retrieval_and_reranking(
        user_query=user_query,
        qdrant_cloud_vector_database=qdrant_cloud_vector_database,
        flashrank_reranker=flashrank_reranker,
        bm25_index=bm25_index,
        bm25_documents=bm25_documents)

    retrieval_total_latency = time.perf_counter() - retrieval_start

    hybrid_retrieval_latency = retrieval_total_latency - re_ranking_latency

    initial_state_self_correction_agent = {
        "user_query": user_query,
        "retrieved_documents_context": "\n\n".join([document.page_content for document in documents]),
        "consulted_pages": [value_int_conversion(document.metadata.get("page", 0)) for document in documents],
        "model_response": "",
        "faithfulness_model_response_score": 0.0,
        "answer_relevancy_score": 0.0,
        "context_recall_score": 0.0,
        "context_precision_score": 0.0,
        "ragas_metrics_feedback": "",
        "total_token_consumption": 0,
        "iteration_number_counter": 0,
        "iteration_maximum_limit": 3}

    model_response_and_ragas_metrics_start = time.perf_counter()
    self_correction_agent_result = self_correction_agent.invoke(initial_state_self_correction_agent)
    model_response_and_ragas_metrics_latency = time.perf_counter() - model_response_and_ragas_metrics_start

    final_model_response = self_correction_agent_result.get("model_response", "The self-correction agent could not generate a valid response.")
    
    pages_list = self_correction_agent_result.get("consulted_pages", [])
    unique_pages = sorted(list(set(pages_list)))
    formatted_pages = ", ".join(map(str, unique_pages))

    with st.chat_message("assistant"):
        st.markdown(final_model_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_model_response})

        if unique_pages:
            st.markdown(f"**Consulted Pages:** {formatted_pages}")
        else:
            st.caption("No specific pages were referenced.")
    
        ragas_metrics = {
            "faithfulness": self_correction_agent_result.get("faithfulness_model_response_score", 0.0),
            "answer_relevancy": self_correction_agent_result.get("answer_relevancy_score", 0.0),
            "context_recall": self_correction_agent_result.get("context_recall_score", 0.0),
            "context_precision": self_correction_agent_result.get("context_precision_score", 0.0) }

        total_pipeline_latency = time.perf_counter() - pipeline_start

        try:
            insert_new_interaction_record_audit_telemetry_table(
                user_query=user_query,
                consulted_pages=self_correction_agent_result.get("consulted_pages", []),
                model_response=final_model_response,
                faithfulness_model_response_score=round(ragas_metrics.get("faithfulness", 0.0), 3),
                answer_relevancy_score=round(ragas_metrics.get("answer_relevancy", 0.0), 3),
                context_recall_score=round(ragas_metrics.get("context_recall", 0.0), 3),
                context_precision_score=round(ragas_metrics.get("context_precision", 0.0), 3),
                hybrid_retrieval_latency=round(hybrid_retrieval_latency, 3),
                re_ranking_latency=round(re_ranking_latency, 3),
                model_response_ragas_metrics_latency=round(model_response_and_ragas_metrics_latency, 3),
                total_pipeline_latency=round(total_pipeline_latency, 3),
                total_token_consumption=self_correction_agent_result.get("total_token_consumption", 0))
            logger.info("Interaction successfully saved to the audit database.")
        except Exception as error:
            st.error(f"Error saving interaction: {error}")

    st.info(
        f"**RAGAS Evaluation Metrics:**\n"
        f"- Faithfulness: {ragas_metrics.get('faithfulness'):.3f}\n"
        f"- Answer Relevancy: {ragas_metrics.get('answer_relevancy'):.3f}\n"
        f"- Context Recall: {ragas_metrics.get('context_recall'):.3f}\n\n"
        f"- Context Precision: {ragas_metrics.get('context_precision'):.3f}\n\n"
        f"**Agentic Self-Correction Attempts:** {self_correction_agent_result.get('iteration_number_counter', 1)}\n\n")
    
    st.info(
        f"**Pipeline Latencies:**\n"
        f"- Hybrid Retrieval Latency: {hybrid_retrieval_latency:.3f}s\n"
        f"- Re-Ranking Latency: {re_ranking_latency:.3f}s\n"
        f"- Model Response and RAGAS Metrics Latency: {model_response_and_ragas_metrics_latency:.3f}s\n"
        f"- Total Pipeline Latency: {total_pipeline_latency:.3f}s")

display_observability_telemetry_dashboard()