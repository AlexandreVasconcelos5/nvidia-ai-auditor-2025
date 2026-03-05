# AI-powered Agentic RAG Auditor: Hybrid Retrieval & MCP Systems Architecture 🔎
Built an Agentic Hybrid RAG system engineered for high-fidelity auditing of complex financial reports (e.g., NVIDIA Corporation Annual Review 2025). The architecture features a LangGraph-powered self-correction loop to eliminate hallucinations and an MCP-standardized interface for integration with external AI agents and orchestrators. By integrating Hybrid Retrieval (Qdrant Cloud Vector Database + BM25) with autonomous RAGAS evaluation, the system delivers verifiable, audit-ready insights through a production-grade, interoperable framework.

🚀 Live Demonstration: https://huggingface.co/spaces/Alexandre2001/nvidia-ai-auditor-2025

🎥 Video Demonstration: https://drive.google.com/file/d/1QUMNUFK7mNn_nR8Y164274tg1G5tCzVi/view?usp=sharing


<img width="1885" height="862" alt="NVIDIA AI Auditor 2025 Print Screen 1" src="https://github.com/user-attachments/assets/6dde6279-2bf4-49b7-b84c-a7f34ac59f98" />

    User query to the AI Agentic Hybrid RAG Auditor, model response and consulted pages.


<img width="1883" height="865" alt="NVIDIA AI Auditor 2025 Print Screen 2" src="https://github.com/user-attachments/assets/1d81afb6-cbe9-45b0-8b5c-06c4702246e6" />

    RAGAS evaluation metrics, agentic self-correction attempts and per-phase latency telemetry.


<img width="1883" height="799" alt="NVIDIA AI Auditor 2025 Print Screen 3" src="https://github.com/user-attachments/assets/8a3cd0a1-25b1-4ca9-9024-d77e0da671fe" />

    Observability & Telemetry Dashboard.


### 🏗️ System Architecture:

```mermaid
graph TD
    classDef finalStyle fill:#fff,stroke:#000,stroke-width:1px,color:#000 !important,font-weight:bold;

    A[1.&nbsp;NVIDIA&nbsp;2025&nbsp;PDF&nbsp;Report]:::finalStyle
    B[2.&nbsp;LlamaParse:&nbsp;Vision-Based&nbsp;Parsing]:::finalStyle
    
    B1[3.&nbsp;Chunking]:::finalStyle
    
    C1[<span style='white-space:nowrap'>4.&nbsp;Qdrant&nbsp;Cloud&nbspVector&nbsp;Database</span>]:::finalStyle
    C2[5.&nbsp;BM25&nbsp;Lexical&nbsp;Index]:::finalStyle
    
    A --> B
    B --> B1
    B1 --> C1
    B1 --> C2

    D[6.&nbsp;User&nbsp;Query]:::finalStyle
    MCP[<span style='white-space:nowrap'>7.&nbsp;MCP&nbsp;Server:&nbsp;Universal&nbsp;Tool&nbsp;Interface</span>]:::finalStyle
    E[8.&nbsp;Hybrid&nbsp;Retrieval&nbsp;Engine]:::finalStyle
    F[<span style='white-space:nowrap'>9.&nbsp;FlashRank:&nbsp;Re-Ranking</span>]:::finalStyle
    
    D --> MCP
    MCP --> E
    C1 --> E
    C2 --> E
    E --> F

    G[<span style='white-space:nowrap'>10.&nbsp;Reasoning&nbsp;Agent:&nbsp;Llama&nbsp;3.3&nbsp;70B&nbsp;via&nbsp;Groq&nbsp;LPUs</span>]:::finalStyle
    H[<span style='white-space:nowrap'>11.&nbsp;Initial&nbsp;Model&nbsp;Response</span>]:::finalStyle
    I[<span style='white-space:nowrap'>12.&nbsp;Critic&nbsp;Agent:&nbsp;RAGAS-Based&nbsp;Self-Correction</span>]:::finalStyle
    
    F --> G
    G --> H
    H --> I
    I --&nbsp;Self-Correction&nbsp;Trigger&nbsp;--> G

    J[<span style='white-space:nowrap'>13.&nbsp;Final&nbsp;Verified&nbsp;Model&nbsp;Response&nbsp;with&nbsp;Page&nbsp;Citations</span>]:::finalStyle
    SQL[(<span style='white-space:nowrap'>14.&nbsp;SQLite:&nbsp;Observability&nbsp;&&nbsp;Telemetry&nbsp;Logging</span>)]:::finalStyle
    
    I --> J

    J -.->|User&nbsp;Query,&nbsp;Model&nbsp;Response&nbsp;&&nbsp;Page&nbsp;Citations| SQL
    I -.->|RAGAS&nbsp;Metrics&nbsp;Scores| SQL
    G -.->|Total&nbsp;Token&nbsp;Consumption| SQL
    E -.->|Hybrid&nbsp;Retrieval&nbsp;Latency| SQL
    F -.->|Re-Ranking&nbsp;Latency| SQL
    I -.->|Model&nbsp;Response&nbsp;&&nbsp;RAGAS&nbsp;Metrics&nbsp;Latency| SQL

    linkStyle default stroke:#333,stroke-width:2px;
```


### 💡 System Highlights:

- Designed and deployed a production-ready Hybrid RAG system for financial audit use-cases, engineered to be scalable to industrial documentation and complex technical reports.
- Architected a LangGraph-powered self-correction loop (ReAct) with autonomous reasoning and evaluation gating to identify and eliminate hallucinations.
- Built a Hybrid Retrieval Engine (Qdrant Cloud Vector Database + BM25) with FlashRank re-ranking to improve precision and page-level citation accuracy.
- Implemented RAGAS framework for performance quantification and SQLite telemetry for session management and interaction logging.
- Engineered modular data ingestion and an advanced OCR pipeline using LlamaParse for vision-based ingestion, structuring complex nested tables from PDFs into RAG-ready markdown.
- Designed and standardized an MCP Server to expose the RAG engine as a RESTful API toolset, decoupling retrieval logic from the interface and enabling integration with external LLM clients and orchestrators.
- Orchestrated low-latency LLM inference (Llama 3.3 on Groq LPUs) and deployed a production-ready Docker application.


### 📦 Quick Start:
- Clone & Navigate: Clone the repository and enter the project root.
- Install Dependencies: `pip install -r requirements.txt`
- Configure Credentials: Set `GROQ_API_KEY`, `LLAMA_CLOUD_API_KEY`, `QDRANT_API_KEY` and `QDRANT_URL` in a `.env` file.
- Ingest Documents: Run `python document_ingestion_pipeline.py` to populate the Qdrant Cloud Vector Database.
- Web Interface: Launch the dashboard using `streamlit run app.py`.
- MCP Server: Launch the toolset using `python mcp_server.py`.
- Docker Deployment: Build and run the containerized application:
  ```bash
  docker build -t ai-auditor .
  docker run -p 8501:8501 --env-file .env ai-auditor
  ```


### 📩 Contacts:
- Name: Alexandre Vasconcelos
- LinkedIn: https://www.linkedin.com/in/alexandre-vasconcelos-396227167/
- Email: alex.vasconcelos.2057@gmail.com
