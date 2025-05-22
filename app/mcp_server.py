from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import uvicorn
import logging
import os
from pathlib import Path
import datetime
import traceback # For detailed exception printing

from app.core.config import settings
from app.core.schemas import (
    ExtractFeaturesParams, ExtractFeaturesResult, MCPToolCallRequest, MCPToolCallResponse,
    ProcessDocumentRequest, ProcessDocumentResponse
)
from app.rag_processor import RAGProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Explicitly log to console
)
logger = logging.getLogger(__name__)

print(f"--- MCP Server Python Module Loaded (mcp_server.py top level): {datetime.datetime.now()} ---")
logger.info(f"MCP Server (mcp_server.py top level) starting up. Basic logging configured for console.")

rag_processor = RAGProcessor()
app = FastAPI(
    title=settings.PROJECT_NAME + " - MCP Server",
    version=settings.VERSION,
    description=settings.DESCRIPTION + " This server exposes tools via Model Context Protocol."
)

@app.on_event("startup")
async def startup_event():
    print(f"--- FastAPI app startup event: {datetime.datetime.now()} ---")
    logger.info("FastAPI application startup complete. RAG Processor initialized.")

def _tool_extract_features_from_specs(params: ExtractFeaturesParams) -> ExtractFeaturesResult:
    logger.info(f"MCP Tool: _tool_extract_features_from_specs called with params: {params}")
    results_data: Dict[str, Dict[str, str]] = {}
    if not params.product_references:
        logger.error("Validation error in _tool_extract_features_from_specs: No product references provided")
        raise ValueError("No product references provided")
    if not params.features_list:
        logger.error("Validation error in _tool_extract_features_from_specs: No features specified for comparison")
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
                logger.error(f"Error extracting feature '{feature_name}' from doc '{doc_ref}': {e}", exc_info=True)
                product_feature_values[feature_name] = f"Error: {str(e)}"
        results_data[doc_ref] = product_feature_values
    logger.info(f"MCP Tool: _tool_extract_features_from_specs returning: {results_data}")
    return ExtractFeaturesResult(comparison_data=results_data)

def extract_features_from_text(file_path: str) -> List[str]:
    print(f"    [PRINT DEBUG extract_features_from_text] Attempting for: {file_path} at {datetime.datetime.now()}")
    logger.info(f"[LOG extract_features_from_text] Attempting for: {file_path}")
    features = set()
    try:
        if not os.path.exists(file_path):
            print(f"    [PRINT DEBUG extract_features_from_text] File not found: {file_path}")
            logger.error(f"[LOG extract_features_from_text] File not found: {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"    [PRINT DEBUG extract_features_from_text] Successfully opened: {file_path}")
            logger.info(f"[LOG extract_features_from_text] Successfully opened: {file_path}")
            for i, line_content in enumerate(f):
                line = line_content.strip()
                if ':' in line: # Basic check for key: value pair
                    key, _ = line.split(':', 1)
                    key = key.strip()
                    if key: # Ensure key is not empty
                        features.add(key)
        sorted_features = sorted(list(features))
        print(f"    [PRINT DEBUG extract_features_from_text] Extracted from {os.path.basename(file_path)}: {sorted_features}")
        logger.info(f"[LOG extract_features_from_text] Extracted: {sorted_features}")
        return sorted_features
    except Exception as e:
        print(f"    [PRINT DEBUG extract_features_from_text] Error for {file_path}: {e}")
        logger.error(f"[LOG extract_features_from_text] Error: {e}", exc_info=True)
        return []

@app.post("/mcp/process_document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(request: ProcessDocumentRequest):
    print(f"--- [PRINT DEBUG process_document_endpoint TOP LEVEL] Entered for {request.doc_reference}, path: {request.file_path} at {datetime.datetime.now()} ---")
    
    actual_extracted_features: List[str] = []
    status = "failed" # Default status
    message = "Processing error occurred." # Default error message

    try:
        logger.info(f"[LOG process_document_endpoint] Entered for {request.doc_reference}, path: {request.file_path}")
        print(f"  [PRINT DEBUG process_document_endpoint try block] Calling rag_processor.add_document for: {request.doc_reference}")
        # Assuming RAGProcessor.add_document also handles PDF, etc.
        # For .txt files, we also want to do simple key-value extraction.
        success_rag = rag_processor.add_document(request.doc_reference, request.file_path)
        print(f"  [PRINT DEBUG process_document_endpoint try block] rag_processor.add_document success: {success_rag}")
        logger.info(f"[LOG process_document_endpoint] RAG add_document success: {success_rag}")

        if success_rag:
            status = "processed" # Base status if RAG processing is okay
            message = "Document processed by RAG."
            
            # Specifically for .txt files, attempt direct feature extraction
            if request.file_path.lower().endswith(".txt"):
                print(f"  [PRINT DEBUG process_document_endpoint try block] File is .txt, calling extract_features_from_text for: {request.file_path}")
                actual_extracted_features = extract_features_from_text(request.file_path)
                print(f"  [PRINT DEBUG process_document_endpoint try block] Features from extract_features_from_text: {actual_extracted_features}")
                logger.info(f"[LOG process_document_endpoint] Text features extracted: {actual_extracted_features}")
                if actual_extracted_features:
                     message += " Features also parsed directly from text."
                else:
                     message += " No features parsed directly from text (file might be empty or lack 'key: value' lines)."
            else:
                # For non-txt files, features might be extracted differently or not at all by this simple method
                logger.info(f"[LOG process_document_endpoint] File is not .txt ({request.file_path}), relying on RAG for any feature insights later.")
                message += " File is not .txt, direct text parsing for features skipped."
        else:
            message = "RAG document processing failed."
            logger.error(f"[LOG process_document_endpoint] RAG add_document failed for {request.doc_reference}.")
            # status remains "failed", actual_extracted_features remains []
        
        response_payload = ProcessDocumentResponse(
            doc_reference=request.doc_reference,
            status=status, # Use the determined status
            message=message,
            extracted_features=actual_extracted_features # Use features from text parsing for .txt
        )
        print(f"--- [PRINT DEBUG process_document_endpoint try block] Returning: {response_payload.model_dump_json(indent=2)} ---")
        logger.info(f"[LOG process_document_endpoint] Returning: {response_payload.status}, features count: {len(actual_extracted_features)}")
        return response_payload

    except Exception as e:
        print(f"--- [PRINT DEBUG process_document_endpoint CATCH BLOCK] EXCEPTION for {request.doc_reference}: {type(e).__name__} - {e} ---")
        traceback.print_exc() # Print full traceback to console
        logger.error(f"[LOG process_document_endpoint CATCH BLOCK] Exception: {e}", exc_info=True)
        return ProcessDocumentResponse(
            doc_reference=request.doc_reference,
            status="error",
            message=f"Unexpected server error: {type(e).__name__} - {e}",
            extracted_features=[] # Ensure an empty list on error
        )

@app.post("/mcp/clear_documents", status_code=204)
async def clear_all_documents_endpoint():
    print(f"--- [PRINT DEBUG clear_all_documents_endpoint] Entered at {datetime.datetime.now()} ---")
    logger.info("[LOG clear_all_documents_endpoint] Clearing documents.")
    rag_processor.clear_all_documents()
    return None

@app.post("/mcp")
async def mcp_tool_router(call_request: MCPToolCallRequest) -> MCPToolCallResponse:
    print(f"--- [PRINT DEBUG mcp_tool_router] Method: {call_request.method} at {datetime.datetime.now()} ---")
    logger.info(f"[LOG mcp_tool_router] Method: {call_request.method}")
    try:
        if call_request.method == "extract_features_from_specs":
            try:
                tool_params = ExtractFeaturesParams(**call_request.params)
                result_data = _tool_extract_features_from_specs(tool_params)
                response = MCPToolCallResponse(result={"extract_features_from_specs": result_data.model_dump()})
                logger.info(f"[LOG mcp_tool_router] 'extract_features_from_specs' success.")
                return response
            except ValueError as ve:
                logger.error(f"[LOG mcp_tool_router] Validation error: {ve}", exc_info=True)
                return MCPToolCallResponse(error={"code": 400, "message": str(ve)})
            except Exception as e:
                logger.error(f"[LOG mcp_tool_router] Error in 'extract_features_from_specs': {e}", exc_info=True)
                return MCPToolCallResponse(error={"code": 500, "message": f"Internal server error: {str(e)}"})
        else:
            logger.error(f"[LOG mcp_tool_router] Unknown method: {call_request.method}")
            return MCPToolCallResponse(error={"code": 400, "message": f"Unknown method: {call_request.method}"})
    except Exception as e:
        logger.error(f"[LOG mcp_tool_router] Unexpected error: {e}", exc_info=True)
        return MCPToolCallResponse(error={"code": 500, "message": "Internal server error in router"})

@app.get("/health", summary="Health Check", tags=["Management"])
async def health_check():
    return {"status": "healthy", "rag_docs_loaded": len(rag_processor.document_vector_stores)}

if __name__ == "__main__":
    print(f"--- Running {__file__} directly (if __name__ == '__main__') ---")
    uvicorn.run(
        "app.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=True,
        log_level="info"
    )
