from fastapi import FastAPI, HTTPException
from typing import Dict
import uvicorn
import logging

from app.core.config import settings
from app.core.schemas import (
    ExtractFeaturesParams, ExtractFeaturesResult, MCPToolCallRequest, MCPToolCallResponse,
    ProcessDocumentRequest, ProcessDocumentResponse
)
from app.rag_processor import RAGProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- RAG Processor Singleton ---
# This instance will live as long as the MCP server process.
# For a stateless server or multi-worker setup, RAG state (vector stores)
# would need to be managed differently (e.g., external DB, shared cache, or re-loaded per request if docs are few).
# For this project, a singleton holding state for uploaded docs in a session is acceptable.
# The `doc_references` passed to the tool will map to documents processed by this singleton.
# The Streamlit app (or backend) will need to first "upload/process" docs via RAGProcessor
# before calling this MCP tool with the references. This implies the MCP server might need
# an endpoint to register/process documents if it's truly independent, or the RAGProcessor
# is shared/accessible by both the component that loads docs and the MCP tool logic.

# For simplicity in this project, let's assume the RAGProcessor is instantiated here
# and the app/main.py (or backend) will *also* instantiate it and call `add_document`.
# This is a slight cheat for a truly decoupled MCP server, but manageable for a solo project.
# A better approach for true decoupling:
# 1. Streamlit uploads to a shared temp location.
# 2. MCP tool `extract_features` receives file paths. It then loads them into its own RAGProcessor instance.
# This is more stateless for the tool but means re-processing files on each MCP call.
# Alternative for shared state: A global RAG processor or a dependency injection system for FastAPI.

# Let's go with a dependency injection approach for RAGProcessor for better testability and management.
# This RAG processor instance will be created once per server startup.
# The Streamlit app will need to interact with endpoints on *this server* to add documents
# before the agent calls the MCP tool.

# Global instance (simple approach for now, consider dependency injection for larger apps)
# This means the state (loaded documents) is held by this MCP server instance.
# Your streamlit app will need to call an endpoint on this server to load docs.
rag_processor = RAGProcessor()

app = FastAPI(
    title=settings.PROJECT_NAME + " - MCP Server",
    version=settings.VERSION,
    description=settings.DESCRIPTION + " This server exposes tools via Model Context Protocol."
)

def _tool_extract_features_from_specs(params: ExtractFeaturesParams) -> ExtractFeaturesResult:
    """
    Extracts specified features from pre-processed product documents.
    """
    logger.info(f"MCP Tool: extract_features_from_specs called with params: {params}")
    results_data: Dict[str, Dict[str, str]] = {}
    
    # Validate input parameters
    if not params.product_references:
        raise ValueError("No product references provided")
    if not params.features_list:
        raise ValueError("No features specified for comparison")
    
    for doc_ref in params.product_references:
        if doc_ref not in rag_processor.document_vector_stores:
            logger.warning(f"Document reference '{doc_ref}' not processed or not found by RAG processor.")
            results_data[doc_ref] = {feature: "Document not processed" for feature in params.features_list}
            continue
            
        product_feature_values: Dict[str, str] = {}
        for feature_name in params.features_list:
            try:
                value = rag_processor.extract_feature_from_doc(doc_ref, feature_name)
                if value is None or value.strip() == "":
                    value = "Feature not found"
                product_feature_values[feature_name] = value
            except Exception as e:
                logger.error(f"Error extracting feature '{feature_name}' from doc '{doc_ref}': {e}")
                product_feature_values[feature_name] = f"Error: {str(e)}"
                
        results_data[doc_ref] = product_feature_values
        
    logger.info(f"MCP Tool: extract_features_from_specs returning: {results_data}")
    return ExtractFeaturesResult(comparison_data=results_data)

# --- Document Processing Endpoint (NEW) ---
# The Streamlit app will call this first for each uploaded file.
@app.post("/mcp/process_document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(request: ProcessDocumentRequest):
    logger.info(f"Received request to process document: {request.doc_reference} from path: {request.file_path}")
    success = rag_processor.add_document(request.doc_reference, request.file_path)
    if success:
        return ProcessDocumentResponse(doc_reference=request.doc_reference, status="processed", message="Document processed successfully.")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {request.doc_reference}")

@app.post("/mcp/clear_documents", status_code=204)
async def clear_all_documents_endpoint():
    logger.info("Received request to clear all documents.")
    rag_processor.clear_all_documents()
    return None

# --- Main MCP Endpoint ---
@app.post("/mcp")
async def mcp_tool_router(call_request: MCPToolCallRequest) -> MCPToolCallResponse:
    try:
        if call_request.method == "extract_features_from_specs":
            try:
                tool_params = ExtractFeaturesParams(**call_request.params)
                result_data = _tool_extract_features_from_specs(tool_params)
                return MCPToolCallResponse(result={"extract_features_from_specs": result_data.model_dump()})
            except ValueError as ve:
                logger.error(f"Validation error in MCP call: {ve}")
                return MCPToolCallResponse(error={"code": 400, "message": str(ve)})
            except Exception as e:
                logger.error(f"Error processing MCP call for '{call_request.method}': {e}", exc_info=True)
                return MCPToolCallResponse(error={"code": 500, "message": f"Internal server error: {str(e)}"})
        else:
            logger.error(f"Unknown MCP method: {call_request.method}")
            return MCPToolCallResponse(error={"code": 400, "message": f"Unknown method: {call_request.method}"})
    except Exception as e:
        logger.error(f"Unexpected error in MCP router: {e}", exc_info=True)
        return MCPToolCallResponse(error={"code": 500, "message": "Internal server error"})

@app.get("/health", summary="Health Check", tags=["Management"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "rag_docs_loaded": len(rag_processor.document_vector_stores)}

if __name__ == "__main__":
    uvicorn.run(
        "app.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=True,
        log_level="info"
    )