## 1. Project Goal & Vision (Recap)

* **Project Goal:** To develop a "Product Feature Comparison Assistant" that allows users to upload product specification documents (text files, PDFs, or images/screenshots) and receive a clear, side-by-side feature comparison and an AI-generated summary.
* **Core Technologies to Demonstrate:** Retrieval-Augmented Generation (RAG), Agentic Systems (MAS), Model Context Protocol (MCP), and LangSmith for observability.

---

## 2. Current Prototype Status

* **End-to-End Flow:**
    * **File Upload:** Users can upload multiple specification documents (TXT, PDF, PNG, JPG).
    * **OCR for Images:** Uploaded images are processed locally using Tesseract OCR to extract text.
    * **Document Processing (MCP Server):**
        * Each document (original text or OCR'd text) is sent to a FastAPI-based MCP server.
        * The server processes the document to identify a list of potential features available within it (e.g., "RAM", "Battery", "Price").
    * **Product Definition UI (Streamlit):**
        * Users can define 1-3 "products" for comparison.
        * For each product, they select one of the uploaded/processed documents as its source.
        * Crucially, they provide a "context name" for each product (e.g., "Alpha Gadget," "Phone on Left"). This name guides the RAG system.
    * **Feature Selection:** Users select from the aggregated list of features (extracted from all processed documents) they wish to compare across the defined products.
    * **Comparison Execution (Agent & MCP Tool):**
        * The Streamlit frontend (acting as a client) invokes an "agent" function.
        * This agent constructs specific queries for each defined product and feature.
        * The agent calls the `extract_features_from_specs` tool on the MCP server.
    * **RAG-Powered Feature Value Extraction (MCP Tool):**
        * The MCP server's tool uses the `RAGProcessor`.
        * For each feature and product context name, the `RAGProcessor` retrieves relevant text chunks from the specified document's vector store.
        * It then uses an LLM (OpenAI `gpt-4o-mini`) with a contextual prompt to extract the specific feature value for the given product from the retrieved chunks.
    * **Results Display:**
        * A structured comparison table is displayed in Streamlit showing the extracted values for each selected feature and defined product.
        * An LLM-generated natural language summary of the comparison is also displayed.
* **Key Technologies Implemented & Demonstrable:**
    * **Frontend:** Streamlit UI for file upload, product definition, feature selection, and results display.
    * **Backend/MCP Server:** FastAPI server exposing:
        * `/mcp/process_document` endpoint for initial document ingestion and feature name listing.
        * `/mcp` endpoint with an `extract_features_from_specs` tool for contextual RAG-based feature value extraction.
    * **RAG Implementation:**
        * Document loading (TXT, PDF, OCR for images via Pytesseract).
        * Text chunking (`RecursiveCharacterTextSplitter`).
        * Embeddings (`OpenAIEmbeddings` with `text-embedding-3-small`).
        * Vector store (FAISS) for each processed document.
        * Retrieval of relevant chunks.
        * Contextual prompting of `gpt-4o-mini` for feature value extraction, guided by `product_context_name`.
    * **Agentic System (MAS):**
        * `agents.py` module containing:
            * `run_comparison_tool_directly`: Orchestrates the call to the MCP server's `extract_features_from_specs` tool. It now handles contextual product queries.
            * `generate_comparison_summary`: Takes the structured comparison data and uses `gpt-4o-mini` to generate a natural language summary.
        * This demonstrates a basic Planner/Executor pattern: the Streamlit UI acts as the "user intent handler," the agent functions plan and execute tasks (data retrieval via tool, then summarization).
    * **LangSmith Observability:** (Show this during the demo if traces are appearing)
        * Environment variables for LangSmith are configured.
        * LLM calls (embeddings, feature extraction, summarization) made via LangChain components should be visible as traces in the LangSmith project.

---

## 3. System Architecture


* **User Interface (Streamlit - `app/main.py`):**
    * Handles file uploads (TXT, PDF, Images).
    * Performs local OCR for images using Tesseract.
    * Manages user input for defining products (associating uploaded files with product context names).
    * Allows selection of features to compare.
    * Calls agent functions to orchestrate backend tasks.
    * Displays comparison table and summary.
* **Agent Layer (`app/agents.py`):**
    * `run_comparison_tool_directly`:
        * Receives product definition queries (doc reference + product context name) and features list from UI.
        * Formats these into `ExtractFeaturesParams` for the MCP tool.
        * Makes an HTTP POST request to the MCP server's `/mcp` endpoint (tool router).
        * Parses the `ExtractFeaturesResult` from the MCP server.
    * `generate_comparison_summary`:
        * Takes structured comparison data.
        * Prompts `gpt-4o-mini` via LangChain to generate a summary.
* **MCP Server (FastAPI - `app/mcp_server.py`):**
    * Exposes an `/mcp` endpoint that routes to tools based on the `method` in the request.
    * **`extract_features_from_specs` Tool (`_tool_extract_features_from_specs` function):**
        * Receives `ExtractFeaturesParams` (list of product queries + features).
        * For each product query, calls the `RAGProcessor`.
    * Exposes an `/mcp/process_document` endpoint:
        * Receives a file path (original or OCR'd text).
        * Uses `RAGProcessor.add_document()` to process and vectorize the document.
        * Uses `extract_features_from_text()` (simple parser) to get a list of potential feature *names* from the document content.
        * Returns these feature names to the UI for populating the feature selection dropdown.
* **RAG Processor (`app/rag_processor.py`):**
    * `add_document()`:
        * Loads document content (using `DocumentParser` for PDF/TXT).
        * Chunks text.
        * Generates embeddings (OpenAI `text-embedding-3-small`).
        * Stores chunks and embeddings in a FAISS vector store (in-memory, one per `doc_reference`).
    * `extract_feature_from_doc()`:
        * Takes `doc_reference`, `feature_name`, and `product_context_name`.
        * Constructs a retrieval query (e.g., "ProductX RAM").
        * Retrieves relevant chunks from the specified document's vector store.
        * Uses a contextual prompt with `gpt-4o-mini` to extract the value for `feature_name` specifically for `product_context_name` from the retrieved chunks.
* **LLM & Embedding Models (OpenAI):**
    * `gpt-4o-mini` for feature value extraction (RAG) and summary generation.
    * `text-embedding-3-small` for creating document embeddings.
* **Observability (LangSmith):**
    * All LangChain LLM and embedding calls are (or should be) traced.

---

## 4. Live Demo

* **Scenario 1: Comparing two separate `.txt` files.**
    1.  Upload `product_a_spec.txt`.
    2.  Upload `product_b_spec.txt`.
    3.  Define Product 1: Source `product_a_spec.txt`, leave "Product Name/Context" blank (or use "Alpha Gadget").
    4.  Define Product 2: Source `product_b_spec.txt`, leave "Product Name/Context" blank (or use "Beta Widget").
    5.  Select features (e.g., RAM, Price, Battery).
    6.  Click "Compare Features".
    7.  Show the resulting table and summary.
    8.  (Optional) Briefly show corresponding traces in LangSmith if available and clear.
* **Scenario 2: Comparing two products from a single uploaded image.**
    1.  Upload a screenshot (`.png`/`.jpg`) that clearly shows specs for two distinct products (e.g., your "Screenshot 2025-05-23 010619.png").
    2.  Define Product 1: Source: the screenshot, "Product Name/Context": **enter the actual name of the first product as visible in the image** (e.g., "Alpha Gadget").
    3.  Define Product 2: Source: the screenshot, "Product Name/Context": **enter the actual name of the second product as visible in the image** (e.g., "Beta Widget").
    4.  Select features.
    5.  Click "Compare Features".
    6.  Show results. Discuss OCR quality impact if relevant.
* **Show Code Snippets (Briefly, if time allows and relevant):**
    * RAG prompt for contextual feature extraction (`rag_processor.py`).
    * Agent's call to the MCP tool (`agents.py`).
    * MCP tool definition (`mcp_server.py`).

---

## 5. Roadmap & Next Steps (Towards Final Demo)

* **UI/UX Refinements:**
    * Improve guidance for entering "Product Name/Context" for images.
    * Better handling/display of "Not found" or "Value unclear" in the comparison table.
    * More robust error handling and user feedback.
* **RAG Accuracy Improvements:**
    * Experiment with different chunking strategies for OCR'd text.
    * Further prompt engineering for both feature value extraction and summary generation, especially to handle OCR noise or ambiguous phrasing.
    * Consider adding a pre-processing step for images to improve OCR quality (e.g., using OpenCV for binarization, noise removal).
* **Agent Enhancements (Potential):**
    * **Pre-analysis Agent:** An agent that analyzes the initially extracted feature *names* from `process_document_endpoint` and perhaps suggests common features to compare or flags very unusual ones.
    * **Disambiguation:** If RAG returns "Value unclear," the agent could potentially formulate a clarifying question (though this is more complex).
* **Error Handling & Edge Cases:** More comprehensive testing and handling of edge cases (e.g., very poor OCR, documents with no clear product names).
* **Full LangSmith Integration:** Ensure all relevant LLM calls are consistently traced and perhaps add custom metadata to traces for better debugging.

---

## 6. Blockers & Challenges

* **OCR Accuracy:** The quality of OCR from images is a significant dependency. Poor OCR directly impacts RAG performance.
* **Prompt Engineering for Contextual RAG:** Getting the LLM to consistently isolate information for a specific product within a mixed-content document (especially from OCR) requires careful prompting and can be iterative.
* **Defining "Product Context":** The current reliance on user-provided "context names" for images is a manual step. Automating the identification of distinct products within a single document/image is a much harder problem.

---

## 7. Q&A

---


How to Use This for Your Demo:
Copy-Paste: Copy the Markdown content above into a new Google Doc.
Convert to Slides: Google Docs can convert to Google Slides, or you can manually create slides based on these sections.
Add Visuals:
Include screenshots of your Streamlit UI at various stages (file upload, product definition, feature selection, results table, summary).
Create a simple architecture diagram.
If you have LangSmith traces, take screenshots of those.
Practice Your Demo Flow: Walk through the scenarios you plan to demonstrate.
Prepare to Speak to Each Point: Be ready to explain the "why" behind your design choices, especially regarding RAG, MAS, and MCP.
This outline should give you a strong foundation for your half-time demo. Good luck!
