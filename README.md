# Product Feature Comparison Assistant

A Python application that allows users to upload product specification sheets and receive a side-by-side comparison of specified features. The system leverages RAG for data extraction, a Planner/Executor MAS for orchestration, MCP for tool interaction, and LangSmith for observability.

## Features

- Upload multiple product specification sheets (PDF, TXT)
- Extract and compare specific features across products
- Interactive web interface using Streamlit
- Intelligent feature extraction using RAG
- Multi-agent system for orchestration
- MCP server for tool interaction
- LangSmith integration for observability

## Setup

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- LangSmith API key

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd product-feature-comparison
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to `.env`

### Running the Application

1. Start the MCP server:
```bash
python mcp_server.py
```

2. Start the Streamlit frontend:
```bash
streamlit run app.py
```

## Project Structure

```
product-feature-comparison/
├── app.py                 # Streamlit frontend
├── mcp_server.py         # MCP server implementation
├── rag_processor.py      # RAG pipeline implementation
├── backend_logic.py      # Backend and agent logic
├── requirements.txt      # Project dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Architecture

The application follows a distributed architecture:

- Frontend (Streamlit) ↔ Backend (FastAPI) ↔ Vector DB (FAISS) ↔ MCP Server (FastAPI)

## Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.