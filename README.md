# Product Feature Comparison Assistant

A Python application that allows users to upload product specification sheets and receive a side-by-side comparison of specified features. The system leverages Retrieval-Augmented Generation (RAG) for data extraction, a Planner/Executor Multi-Agent System (MAS) for orchestration, Model Context Protocol (MCP) for tool interaction, and LangSmith for observability.

## Features

- Upload multiple product specification sheets (PDF, TXT).
- Select specific documents and features for comparison.
- Intelligent feature extraction using RAG on user-provided documents.
- Multi-agent system (Planner/Executor) orchestrates the comparison process.
- Dedicated MCP server exposes feature extraction as a tool.
- Comparison results displayed in a clear, tabular format.
- LangSmith integration for full observability of the AI pipeline.
- Interactive web interface built with Streamlit.

## Tech Stack

- **Language:** Python 3.10+
- **LLM & Embeddings:** OpenAI (via LangChain)
- **Core AI/Agent Framework:** LangChain
- **Frontend:** Streamlit
- **MCP Server:** FastAPI
- **Vector Store:** FAISS (in-memory for MCP server instance)
- **Observability:** LangSmith
- **HTTP Client (for Agent to MCP):** HTTPX

## Setup

### Prerequisites

- Python 3.10 or higher
- Git
- An active OpenAI API key
- An active LangSmith API key (optional but highly recommended for development & course requirement)

### Installation

1. **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd product-feature-comparison
    ```

2. **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On Unix/MacOS
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    * Copy the `.env.example` file to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    * Open the `.env` file and fill in your actual `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` (for LangSmith). Adjust `LANGCHAIN_PROJECT` if desired.

## Running the Application

The application consists of two main services that need to be run: the MCP Server and the Streamlit Frontend.

1. **Start the MCP Server:**
    Open a terminal, navigate to the project root, activate the virtual environment, and run:
    ```bash
    python app/mcp_server.py
    ```
    The MCP server will typically start on `http://127.0.0.1:8001` (or as configured in `.env`/`app/core/config.py`).

2. **Start the Streamlit Frontend:**
    Open a *new* terminal, navigate to the project root, activate the virtual environment, and run:
    ```bash
    streamlit run app/main.py
    ```
    The Streamlit application will typically open in your web browser at `http://localhost:8501`.

## Project Structure

```
product-feature-comparison/
├── app/                          # Main application source code
│   ├── __init__.py
│   ├── main.py                   # Streamlit frontend
│   ├── agents.py                 # Agent logic (Planner/Executor MAS)
│   ├── mcp_server.py            # FastAPI MCP server
│   ├── rag_processor.py         # RAG pipeline implementation
│   ├── services/                # Helper services
│   │   ├── __init__.py
│   │   └── document_parser.py   # PDF/TXT content extraction
│   ├── core/                    # Core configurations, Pydantic models
│   │   ├── __init__.py
│   │   ├── config.py            # Settings management
│   │   └── schemas.py           # Pydantic models for API contracts
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── logger.py            # Logging configuration
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   └── test_rag_processor.py    # RAG processor tests
├── .env                         # Local environment variables (Gitignored)
├── .env.example                 # Example environment variables
├── .gitignore                   # Files to ignore for Git
├── LICENSE                      # MIT License
├── README.md                    # This file
├── requirements.txt             # Project dependencies
└── uploads/                     # Temporary storage for uploaded files (Gitignored)
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

1. **Frontend (Streamlit)**
   - Handles file uploads and user interaction
   - Manages temporary file storage
   - Displays comparison results

2. **MCP Server (FastAPI)**
   - Exposes RAG functionality as tools
   - Manages document processing and feature extraction
   - Maintains vector stores for uploaded documents

3. **Agent System (LangChain)**
   - Orchestrates the comparison process
   - Interacts with MCP server tools
   - Provides structured responses

4. **RAG Pipeline**
   - Processes documents into vector stores
   - Extracts features using LLM
   - Manages document lifecycle

## Contributing

1. Create a feature branch (`git checkout -b feature/your-amazing-feature`)
2. Commit your changes (`git commit -m 'feat: Add some amazing feature'`)
3. Push to the branch (`git push origin feature/your-amazing-feature`)
4. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.