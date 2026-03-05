"""evaluation_engine.py file.

Utility functions for RAG retrieval and evaluation.

Uses a deterministic LLM evaluator, ChatGroq, to compute RAGAS metrics:
  - Faithfulness of the Model Response;
  - Answer Relevancy;
  - Context Recall;
  - Context Precision.

  Uses Qdrant Cloud Vector Database with Hugging Face embeddings to retrieve the documents' context."""

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
import logging
import re

# -------------------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------------
# LangChain and LLM Imports
# -------------------------------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_tools")

# -------------------------------------------------------------------------------
# LLM Initialization
# -------------------------------------------------------------------------------
chat_groq_model_evaluator = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# -------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------
def parse_numeric_score(answer_content: str) -> float:
    """Extracts the first float or integer found in the string. If none found, returns 0.0."""
    
    match = re.search(r"(\d?\.\d+|\d+)", answer_content)
    if match:
        try:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        except ValueError:
            return 0.0
    return 0.0

# -------------------------------------------------------------------------------
# Faithfulness of the Model Response Evaluation
# -------------------------------------------------------------------------------
def evaluate_faithfulness_model_response(retrieved_documents_context: str, model_response: str) -> dict:
    """Evaluates how faithfully the model response is supported by the retrieved context. 
    Returns both the numeric score and the reasoning for the agent to correct itself."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a rigorous financial auditor.
        Evaluate if the response is supported by the retrieved document's context.
        1. Provide a brief reasoning explaining any inconsistencies or missing data.
        2. Provide a final score between 0.0 and 1.0.
        
        Format:
        Reasoning: <your_analysis>
        Score: <numeric_score>"""),
        ("user", "Retrieved Document's Context: {retrieved_documents_context}\nModel Response: {model_response}")])

    evaluation_chain = prompt_template | chat_groq_model_evaluator
    model_response = evaluation_chain.invoke({"retrieved_documents_context": retrieved_documents_context, "model_response": model_response})
    content = model_response.content
    score = parse_numeric_score(content)
    reasoning_part = content.split("Score:")[0] if "Score:" in content else content
    reasoning = reasoning_part.replace("Reasoning:", "").strip()
    logger.info("Faithfulness of the Model Response Score: %.3f | Feedback: %s", score, reasoning)
    return {"score": score, "feedback": reasoning}

def faithfulness_binary_gate(faithfulness_score: float, acceptance_threshold: float = 0.85) -> float:
    """Convert a continuous faithfulness model response score into a binary decision. Returns 1.0 if accepted, 0.0 otherwise."""
    return 1.0 if faithfulness_score >= acceptance_threshold else 0.0

# -------------------------------------------------------------------------------
# Answer Relevancy Evaluation
# -------------------------------------------------------------------------------
def evaluate_answer_relevancy(user_query: str, model_response: str) -> dict:
    """Evaluates how well the model response addresses the user's query.
    Returns both the numeric score and the reasoning for the agent to correct itself."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a rigorous financial auditor.
        Evaluate how well the response addresses the user's query.
        1. Provide a brief reasoning explaining any inconsistencies or missing data.
        2. Provide a final score between 0.0 and 1.0.
        
        Format:
        Reasoning: <analysis>
        Score: <numeric_score>"""),
        ("user", "User Query: {user_query}\nModel Response: {model_response}")])

    evaluation_chain = prompt_template | chat_groq_model_evaluator
    response = evaluation_chain.invoke({"user_query": user_query, "model_response": model_response})
    content = response.content
    score = parse_numeric_score(content)
    reasoning_part = content.split("Score:")[0] if "Score:" in content else content
    reasoning = reasoning_part.replace("Reasoning:", "").strip()
    return {"score": score, "feedback": reasoning}

# -------------------------------------------------------------------------------
# Context Recall Evaluation
# -------------------------------------------------------------------------------
def evaluate_context_recall(user_query: str, retrieved_documents_context: str) -> float:
    """Evaluate whether the retrieved documents' context contains the necessary information to answer the query. Returns a continuous score between 0.0 and 1.0."""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a rigorous financial auditor. "
          "Evaluate whether the retrieved documents' context contains the necessary information to answer the user's query. "
          "Return only a numerical score between 0.0 and 1.0."),
          ("user", "User Query:\n{user_query}\n\nRetrieved Documents' Context:\n{retrieved_documents_context}")])

    evaluation_chain = prompt_template | chat_groq_model_evaluator
    model_response = evaluation_chain.invoke({"user_query": user_query, "retrieved_documents_context": retrieved_documents_context})
    score = parse_numeric_score(model_response.content)
    logger.info("Context Recall score: %.3f", score)
    return score

# -------------------------------------------------------------------------------
# Context Precision Evaluation
# -------------------------------------------------------------------------------
def evaluate_context_precision(user_query: str, retrieved_documents_context: str, top_k: int = 5) -> float:
    """Evaluate the precision of the retrieved documents' context by checking if the top-ranked text chunks contain the essential information to answer the query."""

    text_chunks = retrieved_documents_context.split("\n\n")
    top_text_chunks = "\n\n".join(text_chunks[:top_k])

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a rigorous financial auditor. "
          "Evaluate if the top ranked retrieved chunks contain the most relevant information to answer the user's query. "
          "High scores (0.85 - 1.0) should be reserved for cases where the top results provide a clear answer without needing a deeper search. " 
          "Return only a numerical score between 0.0 and 1.0."),
          ("user", "User Query:\n{user_query}\n\nTop Ranked Context:\n{top_chunks}")])
    
    evaluation_chain = prompt_template | chat_groq_model_evaluator
    model_response = evaluation_chain.invoke({"user_query": user_query, "top_chunks": top_text_chunks})
    score = parse_numeric_score(model_response.content)
    logger.info("Context Precision (Top-%d) score: %.3f", top_k, score)
    return score