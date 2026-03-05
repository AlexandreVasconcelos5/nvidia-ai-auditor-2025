"""document_ingestion_pipeline.py file.

Handles a high-fidelity ingestion of PDF documents using LlamaParse.

Converts financial reports into Markdown, splits them into semantic text chunks, and populates the Qdrant Cloud Vector Database."""

# -------------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------------
import os
import logging

# -------------------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------------
# Qdrant Cloud Vector Database and LangChain
# -------------------------------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

# -------------------------------------------------------------------------------
# Llama Cloud Parsing Engine
# -------------------------------------------------------------------------------
from llama_parse import LlamaParse

# -------------------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Document Cloud Ingestion
# -------------------------------------------------------------------------------
def run_document_cloud_ingestion():
    """Performs the full document cloud ingestion pipeline: Parsing -> Chunking -> Embeddings -> Storage."""

    pdf_report_path = "./nvidia-corporation-annual-review-2025.pdf"
    collection_name = "ai_auditor_2025"
    
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),)
    if not qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"Creating collection {collection_name}...")
        qdrant_client.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    else:
        logger.info(f"The collection '{collection_name}' already exists. Skipping the document cloud ingestion...")
        return

    logger.info("Starting the document cloud ingestion via LlamaParse...")

    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        split_by_page=True,
        parsing_instruction="This is a financial report with complex tables. Please, extract all tables in Markdown format and preserve the hierarchy of the headings.",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="openai-gpt4o")
    
    documents = parser.load_data(pdf_report_path)
    
    text_splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)

    langchain_documents = []
    for index, document in enumerate(documents):
        page = str(index + 1)
        langchain_documents.append(Document(page_content=document.text, metadata={"source": str(pdf_report_path), "page": page}))

    text_chunks = text_splitter.split_documents(langchain_documents)

    hugging_face_embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=hugging_face_embeddings_model,
        retrieval_mode=RetrievalMode.DENSE)
    
    qdrant_vector_store.add_documents(documents=text_chunks)
    
    logger.info(f"The document cloud ingestion was successfully completed. {len(text_chunks)} chunks indexed in Qdrant Cloud.")

# -------------------------------------------------------------------------------
# Execute Entry Point
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_document_cloud_ingestion()