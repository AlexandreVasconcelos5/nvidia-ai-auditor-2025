"""self_correction_agent.py file.

Self-Correction Agent for RAG pipelines with RAGAS metrics evaluation.

Handles the generation of the model response, quality evaluation and iterative self-correction flow."""

# -------------------------------------------------------------------------------
# SQLite3 Version
# -------------------------------------------------------------------------------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
from typing import TypedDict, List
import logging

# -------------------------------------------------------------------------------
# LangGraph State Machine Imports
# -------------------------------------------------------------------------------
from langgraph.graph import StateGraph, END

# -------------------------------------------------------------------------------
# RAGAS Metrics and Context Retrieval Imports
# -------------------------------------------------------------------------------
from evaluation_engine import (evaluate_faithfulness_model_response, evaluate_answer_relevancy, evaluate_context_recall, evaluate_context_precision)
from hybrid_information_retrieval_reranking import execute_hybrid_retrieval_and_reranking, initialize_hybrid_retrieval_engine

# -------------------------------------------------------------------------------
# Prompt Template and LLM Integration
# -------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# -------------------------------------------------------------------------------
# Metadata Sanitization and Type Conversion
# -------------------------------------------------------------------------------
def value_int_conversion(value, default=0):
    """Convert a value to an integer to avoid a crash on non-numerical metadata."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# -------------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("self_correction_agent")

# -------------------------------------------------------------------------------
# LLM and Prompt Template Initialization
# -------------------------------------------------------------------------------
llama_llm_groq_response = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a rigorous financial auditor. Your goal is to extract precise data from the provided documents."
     "\n\nCritical Rules:"
     "\n1. Answer only based on the provided context."
     "\n2. If the data isn't explicitly in the text or in the tables, respond: 'Information not found in the NVIDIA Report'."
     "\n3. You are prohibited from making estimated calculations or inventing values."
     "\n4. If there are tables, prioritize the values contained within them."
     "\n\nRetrieved Documents' Context:\n{retrieved_documents_context}"
     "\n\nPrevious Attempt and Feedback:"
     "\nPrevious Answer: {previous_answer}"
     "\nRAGAS Metrics Feedback: {ragas_metrics_feedback}"
     "\n\nInstructions: If there is a previous answer and feedback, correct the errors and improve the precision."),
     ("user", "{user_query}")])

# -------------------------------------------------------------------------------
# Self-Correction Agent State Definition
# -------------------------------------------------------------------------------
class SelfCorrectionAgentStateClass(TypedDict):
    """State of the self-correction agent, maintaining user query, context, model response, RAGAS metrics scores, total token consumption and iteration info."""
    user_query: str
    retrieved_documents_context: str
    model_response: str
    consulted_pages: List[int]
    faithfulness_model_response_score: float
    answer_relevancy_score: float
    context_recall_score: float
    context_precision_score: float
    ragas_metrics_feedback: str
    total_token_consumption: int
    iteration_number_counter: int
    iteration_maximum_limit: int

# -------------------------------------------------------------------------------
# Model Response Generation
# -------------------------------------------------------------------------------
def generate_model_response(state: SelfCorrectionAgentStateClass) -> dict:
    """Generate the model response with the awareness of the previous attempt."""

    logger.info("Phase: Model Response Generation. Attempt #%d", state['iteration_number_counter'] + 1)
    
    inputs = {
        "user_query": state["user_query"],
        "retrieved_documents_context": state.get("retrieved_documents_context", ""),
        "previous_answer": state.get("model_response", "None (First attempt)"),
        "ragas_metrics_feedback": state.get("ragas_metrics_feedback", "None")}
    
    response_chain = prompt_template | llama_llm_groq_response
    model_response = response_chain.invoke(inputs)

    new_tokens = model_response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

    return {
        "model_response": model_response.content,
        "total_token_consumption": state.get("total_token_consumption", 0) + new_tokens,
        "iteration_number_counter": state['iteration_number_counter'] + 1}

# -------------------------------------------------------------------------------
# Model Response Quality Evaluation
# -------------------------------------------------------------------------------
def evaluate_model_response_quality_node(state: SelfCorrectionAgentStateClass) -> dict:
    """Evaluate the model response using RAGAS metrics: Faithfulness of the Model Response, Answer Relevancy, Context Recall and Context Precision."""

    faithfulness_model_response_result = evaluate_faithfulness_model_response(state["retrieved_documents_context"], state["model_response"])
    faithfulness_model_response_score = faithfulness_model_response_result.get("score", 0.0)

    answer_relevancy_result = evaluate_answer_relevancy(state["user_query"], state["model_response"])
    answer_relevancy_score = answer_relevancy_result.get("score", 0.0)
    
    context_recall_score = evaluate_context_recall(state["user_query"], state["retrieved_documents_context"])
    
    context_precision_score = evaluate_context_precision(state["user_query"], state["retrieved_documents_context"])

    ragas_metrics_feedback_list = []
    
    if faithfulness_model_response_score < 0.850:
        ragas_metrics_feedback_list.append(f"Faithfulness of the Model Response Feedback: {faithfulness_model_response_result.get('feedback', 'Check the retrieved documents context.')}")

    if answer_relevancy_score < 0.850:
        ragas_metrics_feedback_list.append(f"Answer Relevancy Feedback: {answer_relevancy_result.get('feedback', 'Address the query more directly.')}")

    ragas_metrics_feedback = " | ".join(ragas_metrics_feedback_list) if ragas_metrics_feedback_list else "RAGAS metrics' quality was validated."

    logger.info("Faithfulness of the Model Response Score: %.2f | Answer Relevancy Score: %.2f | Feedback: %s", faithfulness_model_response_score, answer_relevancy_score, ragas_metrics_feedback)

    return {
        "faithfulness_model_response_score": faithfulness_model_response_score,
        "answer_relevancy_score": answer_relevancy_score,
        "context_recall_score": context_recall_score,
        "context_precision_score": context_precision_score,
        "ragas_metrics_feedback": ragas_metrics_feedback,
        "consulted_pages": state.get("consulted_pages", [])}

# -------------------------------------------------------------------------------
# Decision Logic for Self-Correction
# -------------------------------------------------------------------------------
def decide_next_step_self_correction_agent_flow(state: SelfCorrectionAgentStateClass) -> str:
    """Decide which is the next step: correct or terminate, based on the RAGAS metrics and iteration limits."""

    if state["faithfulness_model_response_score"] >= 0.850 and state["answer_relevancy_score"] >= 0.850:
        logger.info("Answer quality validated. Terminating the process...")
        return "terminate"

    elif state["context_recall_score"] < 0.5:
        logger.warning("Critical failure in document context retrieval. Terminating the process...")
        return "terminate"

    elif state["iteration_number_counter"] >= state["iteration_maximum_limit"]:
        logger.warning("Iteration maximum limit reached. Interrupting the iteration cycle...")
        return "terminate"
    
    else:
        logger.info("Insufficient answer quality. Requesting self-correction...")
        return "correct"

# -------------------------------------------------------------------------------
# Self-Correction Agent State Graph Construction
# -------------------------------------------------------------------------------
state_graph_self_correction_agent = StateGraph(SelfCorrectionAgentStateClass)

state_graph_self_correction_agent.add_node("model_response_generation_node", generate_model_response)
state_graph_self_correction_agent.add_node("model_response_quality_evaluation_node", evaluate_model_response_quality_node)
state_graph_self_correction_agent.set_entry_point("model_response_generation_node")
state_graph_self_correction_agent.add_edge("model_response_generation_node", "model_response_quality_evaluation_node")
state_graph_self_correction_agent.add_conditional_edges("model_response_quality_evaluation_node", decide_next_step_self_correction_agent_flow,
                                                        {"correct": "model_response_generation_node", "terminate": END})

self_correction_agent = state_graph_self_correction_agent.compile()

# -------------------------------------------------------------------------------
# Execution Entry Point
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    test_user_query = "What was the revenue reported by NVIDIA in the 2025 fiscal year?"

    logger.info("Initializing the Hybrid Retrieval Engine...")
    qdrant_vector_store_database, re_ranker, bm25_index, bm25_documents = initialize_hybrid_retrieval_engine()

    logger.info("Retrieving the documents' context using Hybrid Search & Re-Ranking...")
    documents, _ = execute_hybrid_retrieval_and_reranking(test_user_query, qdrant_vector_store_database, re_ranker, bm25_index, bm25_documents)
    retrieved_documents_context = "\n\n".join([f"Text Chunk {index+1} (Page {document.metadata.get('page', 'N/A')})\n{document.page_content}"
                                               for index, document in enumerate(documents)])
    consulted_pages = list(set([value_int_conversion(document.metadata.get('page')) for document in documents]))

    initial_inputs_self_correction_agent = {
        "user_query": test_user_query,
        "retrieved_documents_context": retrieved_documents_context,
        "model_response": "",
        "consulted_pages": consulted_pages,
        "faithfulness_model_response_score": 0.0,
        "answer_relevancy_score": 0.0,
        "context_recall_score": 0.0,
        "context_precision_score": 0.0,
        "ragas_metrics_feedback": "",
        "total_token_consumption": 0,
        "iteration_number_counter": 0,
        "iteration_maximum_limit": 3}

    logger.info("Starting the Self-Correction Agent...")
    final_state_self_correction_agent = self_correction_agent.invoke(initial_inputs_self_correction_agent)

    logger.info("Model Final Response:\n%s", final_state_self_correction_agent["model_response"])
    logger.info("Total Token Consumption: %d", final_state_self_correction_agent["total_token_consumption"])