"""mcp_server.py file.

Standardized MCP (Model Context Protocol) Server, powered by FastMCP.

Exposes the Agentic Hybrid RAG capabilities as universal tools for LLM Orchestrators."""

# -------------------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------------------
import logging
logger = logging.getLogger("mcp_server.logger")

# -------------------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------------
# Model Context Protocol (MCP) Framework
# -------------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP

# -------------------------------------------------------------------------------
# Local Imports
# -------------------------------------------------------------------------------
from hybrid_information_retrieval_reranking import initialize_hybrid_retrieval_engine, execute_hybrid_retrieval_and_reranking
from self_correction_agent import self_correction_agent, value_int_conversion

# -------------------------------------------------------------------------------
# Initialize the MCP Server
# -------------------------------------------------------------------------------
mcp = FastMCP("AI Auditor 2025")

qdrant_cloud_vector_database = None
flashrank_reranker = None
bm25_index = None
bm25_documents = None

def ensure_resources():
    global qdrant_cloud_vector_database, flashrank_reranker, bm25_index, bm25_documents
    if qdrant_cloud_vector_database is None:
        logger.info("Loading the Hybrid Retrieval Engine resources...")
        try:
            qdrant_cloud_vector_database, flashrank_reranker, bm25_index, bm25_documents = initialize_hybrid_retrieval_engine()
            logger.info("The MCP Engine is ready.")
        except Exception as exception:
            logger.error(f"Failed to initialize: {exception}")
            raise exception

# -------------------------------------------------------------------------------
# MCP Tool: Agentic Financial Auditor
# -------------------------------------------------------------------------------
@mcp.tool()
def execute_agentic_auditor(user_query: str) -> str:
    """Agentic tool to audit NVIDIA's Corporate Annual Review 2025 Report. Handles the information hybrid retrieval, re-ranking and self-correction automatically."""
    try:
        ensure_resources()
        documents, _ = execute_hybrid_retrieval_and_reranking(
            user_query=user_query,
            qdrant_vector_store_database=qdrant_cloud_vector_database,
            flashrank_reranker=flashrank_reranker,
            bm25_index=bm25_index,
            bm25_documents=bm25_documents)
    
        initial_state = {
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

        result = self_correction_agent.invoke(initial_state)
        final_verified_model_response = result.get("model_response", "Error generating the model response.")
        consulted_pages = sorted(list(set(result.get("consulted_pages", []))))
        return f"{final_verified_model_response}\n\n[Consulted Pages: {', '.join(map(str, consulted_pages))}]"

    except Exception as exception:
        logger.error(f"Error: {exception}")
        return f"The audit tool failed to process the request: {str(exception)}"

# -------------------------------------------------------------------------------
# Execution Entry Point
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
