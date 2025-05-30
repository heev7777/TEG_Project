# Product Feature Comparison Assistant
**TEG Final Project - Generative AI Technologies**

A sophisticated Python application that leverages **RAG**, **Multi-Agent Systems**, **Model Context Protocol**, and **LangSmith Observability** to solve the real-world business problem of comparing product specifications from documents and screenshots.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com)

## 🎯 Problem Statement

**Business Challenge**: Product managers and researchers need to efficiently compare technical specifications across multiple products from various sources (PDFs, text files, product screenshots). Manual comparison is time-consuming, error-prone, and doesn't scale.

**Solution**: An AI-powered assistant that automatically extracts and compares product features using advanced NLP techniques, providing structured comparisons and intelligent summaries.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄──►│  Multi-Agent    │◄──►│   MCP Server    │
│  (Frontend)     │    │    System       │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ File Processing │    │ Comparison      │    │  RAG Processor  │
│ (Text/PDF/IMG)  │    │ Summarization   │    │  (Vector Store) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   LangSmith     │
                    │  Observability  │
                    └─────────────────┘
```

### Component Details

#### **Frontend Layer (Streamlit)**
- **File Upload Interface**: Supports PDF, TXT, PNG, JPG, JPEG
- **Dual Processing Mode**: Text documents vs. Screenshot analysis
- **Feature Selection UI**: Dynamic feature discovery and selection
- **Results Visualization**: Structured comparison tables and AI summaries

#### **Multi-Agent System (MAS)**
- **Comparison Agent** (`agents.py`): Orchestrates feature extraction across products
- **Screenshot Agent** (`screenshot_agents.py`): Specialized AI vision analysis
- **Summary Agent**: Generates intelligent comparison summaries
- **Planner/Executor Pattern**: Streamlit acts as planner, agents execute tasks

#### **Model Context Protocol (MCP) Server**
- **Custom FastAPI MCP Server**: Exposes AI tools via HTTP endpoints
- **Document Processing Tool**: `/mcp/process_document`
- **Feature Extraction Tool**: `/mcp` with `extract_features_from_specs`
- **Screenshot Processing Tool**: `/mcp/process_screenshot`
- **Health Monitoring**: Real-time API usage and cost tracking

#### **RAG Implementation**
- **Document Loaders**: PyPDF, TextLoader for various formats
- **Text Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Store**: FAISS (per-document isolation)
- **Retrieval**: Similarity search with contextual prompting
- **LLM**: OpenAI `gpt-4o-mini` for feature extraction

#### **AI Vision System**
- **GPT-4o Vision**: Direct screenshot analysis without OCR
- **Smart Product Matching**: Handles partial product names (e.g., "M2" → "Apple M2")
- **Multi-language Support**: Recognizes Polish, English, and other languages
- **Cost Tracking**: Real-time monitoring with $1 budget limits

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key (required)
- Optional: Additional OpenAI key for screenshot processing

### Installation

1. **Clone Repository**
```bash
git clone <https://github.com/heev7777/TEG_Project.git>
cd TEG_Project
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
Create `.env` file:
```env
# Required - OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_SCREENSHOT_KEY=your_screenshot_api_key_here

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Product-Comparison-Assistant

# Server Configuration (Optional)
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### Running the Application

#### Option 1: Automated Startup (Recommended)
```bash
# On Unix/Linux/Mac
chmod +x run_servers.sh
./run_servers.sh

# On Windows
# Run MCP server: python -m app.mcp_server
# Run Streamlit: streamlit run app/main.py
```

#### Option 2: Manual Startup
```bash
# Terminal 1: Start MCP Server
python -m app.mcp_server

# Terminal 2: Start Streamlit App
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

## 📋 Usage Guide

### Text Document Comparison

1. **Upload Documents**: Upload 2-3 specification files (PDF/TXT)
2. **Select Products**: Choose documents to compare
3. **Select Features**: Pick from auto-discovered features or add custom ones
4. **View Results**: Get structured comparison table + AI summary

### Screenshot Comparison

1. **Upload Screenshots**: Upload product images (PNG/JPG)
2. **Specify Products**: 
   - **Multi-screenshot**: One product per image
   - **Single screenshot**: Multiple products in one image
3. **Product Names** (Optional): Use smart shortcuts:
   - `M2`, `M4` → Apple chip variants
   - `144Hz`, `120Hz` → Display refresh rates
   - `55`, `65` → Screen sizes
   - Brand names: `LG`, `Samsung`, `Apple`
4. **Feature Extraction**: AI automatically discovers and extracts features
5. **Analysis**: Get detailed comparison with foreign currency support

### Advanced Features

- **Cost Monitoring**: Real-time API usage tracking
- **LangSmith Tracing**: Full observability of AI operations
- **Error Handling**: Graceful handling of extraction failures
- **Multi-language**: Support for international product specs

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests with server orchestration
python run_tests.py

# Run specific test modules
python -m pytest tests/test_rag_processor.py -v
python -m pytest tests/test_product_comparison.py -v
```

### Test Coverage
- **RAG Processor**: Document loading, feature extraction, vector operations
- **End-to-End**: Full workflow from upload to comparison
- **MCP Integration**: Server health, tool routing, error handling

## 📊 TEG Course Requirements Compliance

### ✅ RAG Implementation
- **Chunking**: Recursive text splitting with optimal parameters
- **Vector Store**: FAISS with OpenAI embeddings
- **Prompt Grounding**: Contextual feature extraction with retrieved chunks
- **Multi-document**: Isolated vector stores per document

### ✅ Multi-Agent Systems
- **Planner/Executor Pattern**: 
  - Streamlit UI = Planner (user intent, task coordination)
  - Agents = Executors (comparison, summarization, screenshot analysis)
- **Agent Specialization**: Separate agents for different data types
- **Tool Orchestration**: Agents coordinate via MCP protocol

### ✅ MCP Integration
- **Custom MCP Server**: FastAPI-based tool server
- **Safe Tool Calls**: Input validation, error handling, timeout management
- **Multiple Tools**: Document processing, feature extraction, screenshot analysis
- **Protocol Compliance**: Standard MCP request/response format

### ✅ Solution Architecture
- **Frontend ↔ Backend**: Streamlit ↔ FastAPI separation
- **Scalability**: Modular design, async operations, vector store optimization
- **Security**: API key management, input sanitization, file validation
- **Database**: FAISS vector stores for efficient similarity search

### ✅ LangSmith Monitoring
- **Comprehensive Tracing**: All LLM calls logged with metadata
- **Cost Tracking**: Real-time token usage and cost monitoring
- **Error Tracking**: Failed operations logged with context
- **Performance Metrics**: Response times and success rates

### ✅ Code Quality
- **Clean Python**: Type hints, docstrings, PEP 8 compliance
- **Modular Design**: Clear separation of concerns
- **Error Handling**: Comprehensive exception management
- **Documentation**: Extensive comments and documentation

### ✅ Git Workflow
- **Clean Commits**: Descriptive commit messages
- **Project Structure**: Organized directory layout
- **`.gitignore`**: Proper exclusion of sensitive files and artifacts

## 🛠️ Tech Stack

### Core Technologies
- **Language**: Python 3.10+
- **Frontend**: Streamlit 1.31+
- **Backend**: FastAPI 0.109+
- **MCP Server**: Custom FastAPI implementation

### AI/ML Stack
- **LLM Provider**: OpenAI (gpt-4o-mini, gpt-4o)
- **Embeddings**: OpenAI text-embedding-3-small
- **Framework**: LangChain 0.1+
- **Vector Store**: FAISS (CPU)
- **Document Processing**: PyPDF, custom text parsers

### Supporting Libraries
- **HTTP Client**: httpx for async requests
- **Configuration**: Pydantic Settings
- **Observability**: LangSmith integration
- **File Processing**: Multiple format support
- **Testing**: pytest with async support

## 📁 Project Structure

```
TEG_Project/
├── app/                          # Main application package
│   ├── core/                     # Core configuration and schemas
│   │   ├── config.py            # Environment and app configuration
│   │   └── schemas.py           # Pydantic data models
│   ├── main.py                  # Streamlit UI application
│   ├── mcp_server.py           # FastAPI MCP server
│   ├── rag_processor.py        # RAG implementation
│   ├── agents.py               # Multi-agent system
│   ├── screenshot_processor.py  # AI vision system
│   └── screenshot_agents.py    # Screenshot analysis agents
├── tests/                       # Test suite
│   ├── test_rag_processor.py   # RAG component tests
│   └── test_product_comparison.py # End-to-end tests
├── uploads/                     # Document storage (gitignored)
├── requirements.txt             # Python dependencies
├── run_tests.py                # Test orchestrator
├── run_servers.sh              # Startup script
├── .env.example                # Environment template
├── .gitignore                  # Git exclusions
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM and embeddings |
| `OPENAI_SCREENSHOT_KEY` | No | Separate key for screenshot processing |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (default: true) |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key for observability |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `MCP_SERVER_HOST` | No | MCP server host (default: 127.0.0.1) |
| `MCP_SERVER_PORT` | No | MCP server port (default: 8001) |

### Model Configuration
- **Primary LLM**: `gpt-4o-mini` (cost-effective, high-quality)
- **Vision LLM**: `gpt-4o` (screenshot analysis)
- **Embeddings**: `text-embedding-3-small` (optimized for retrieval)
- **Context Window**: 1000 chars per chunk, 200 char overlap

## 🚨 Troubleshooting

### Common Issues

1. **"MCP Server not responding"**
   - Check if port 8001 is available
   - Verify OpenAI API key is valid
   - Check server logs for errors

2. **"No features extracted"**
   - Ensure document contains key-value pairs (for text files)
   - Check document format is supported
   - Verify file upload completed successfully

3. **"Screenshot processing failed"**
   - Verify `OPENAI_SCREENSHOT_KEY` is configured
   - Check image format is supported (PNG, JPG, JPEG)
   - Ensure image contains visible product specifications

4. **High API costs**
   - Monitor LangSmith traces for excessive calls
   - Use screenshot budget limits
   - Consider using smaller documents for testing

### Getting Help
- Check the logs in `mcp_server.log`
- Verify all environment variables are set
- Test with provided sample documents first
- Review LangSmith traces for debugging

## 🎯 Demo Scenarios

### Scenario 1: Text Document Comparison
Upload two product specification files (TXT/PDF) and compare features like RAM, Storage, Price across products.

### Scenario 2: Screenshot Analysis
Upload e-commerce screenshots and automatically extract product specifications using AI vision.

### Scenario 3: Mixed Analysis
Compare products from both text documents and screenshots in a single analysis.

## 🔮 Future Enhancements

- **Multi-modal RAG**: Combine text and image analysis
- **Advanced Agents**: Self-correcting extraction with confidence scores
- **Database Integration**: Persistent storage for comparison history
- **API Gateway**: Rate limiting and authentication
- **Batch Processing**: Handle multiple product catalogs simultaneously

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TEG Course**: Generative AI Technologies curriculum
- **OpenAI**: GPT models and embedding services
- **LangChain**: RAG and agent frameworks
- **Streamlit**: Rapid UI development
- **FastAPI**: High-performance API framework

---

**Author**: Enre Ertan s23372 
**Course**: TEG - Generative AI Technologies  