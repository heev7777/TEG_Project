from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import uvicorn
import logging
import os
import datetime
import traceback

from app.core.config import settings
from app.core.schemas import (
    ExtractFeaturesParams, ExtractFeaturesResult, MCPToolCallRequest, MCPToolCallResponse,
    ProcessDocumentRequest, ProcessDocumentResponse,
    ProcessScreenshotRequest, ProcessScreenshotResponse,
    ExtractFeaturesFromScreenshotParams, ExtractFeaturesFromScreenshotResult
)
from app.rag_processor import RAGProcessor
from app.screenshot_processor import ScreenshotProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

rag_processor = RAGProcessor()
screenshot_processor = ScreenshotProcessor()
app = FastAPI(
    title=settings.PROJECT_NAME + " - MCP Server",
    version=settings.VERSION,
    description=settings.DESCRIPTION + " This server exposes tools via Model Context Protocol."
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup complete. RAG Processor initialized.")

def _tool_extract_features_from_specs(params: ExtractFeaturesParams) -> ExtractFeaturesResult:
    logger.info(f"MCP Tool: extract_features_from_specs called with {len(params.product_references)} products and {len(params.features_list)} features")
    results_data: Dict[str, Dict[str, str]] = {}
    
    if not params.product_references:
        logger.error("No product references provided")
        raise ValueError("No product references provided")
    if not params.features_list:
        logger.error("No features specified for comparison")
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
    
    logger.info(f"Extract features from specs completed for {len(results_data)} products")
    return ExtractFeaturesResult(comparison_data=results_data)

def extract_features_from_text(file_path: str) -> List[str]:
    logger.info(f"Extracting features from text file: {file_path}")
    features = set()
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line_content in enumerate(f):
                line = line_content.strip()
                if ':' in line:
                    key, _ = line.split(':', 1)
                    key = key.strip()
                    if key:
                        features.add(key)
        
        sorted_features = sorted(list(features))
        logger.info(f"Extracted {len(sorted_features)} features from {os.path.basename(file_path)}")
        return sorted_features
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}", exc_info=True)
        return []

@app.post("/mcp/process_document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(request: ProcessDocumentRequest):
    logger.info(f"Processing document: {request.doc_reference}, path: {request.file_path}")
    
    actual_extracted_features: List[str] = []
    status = "failed"
    message = "Processing error occurred."

    try:
        success_rag = rag_processor.add_document(request.doc_reference, request.file_path)
        logger.info(f"RAG processing success: {success_rag}")

        if success_rag:
            status = "processed"
            message = "Document processed by RAG."
            
            if request.file_path.lower().endswith(".txt"):
                actual_extracted_features = extract_features_from_text(request.file_path)
                logger.info(f"Text features extracted: {len(actual_extracted_features)}")
                if actual_extracted_features:
                    message += " Features also parsed directly from text."
                else:
                    message += " No features parsed directly from text."
            else:
                logger.info(f"File is not .txt ({request.file_path}), relying on RAG for feature insights.")
                message += " File is not .txt, direct text parsing for features skipped."
        else:
            message = "RAG document processing failed."
            logger.error(f"RAG processing failed for {request.doc_reference}")
        
        response_payload = ProcessDocumentResponse(
            doc_reference=request.doc_reference,
            status=status,
            message=message,
            extracted_features=actual_extracted_features
        )
        logger.info(f"Returning document processing response with status: {status}, features: {len(actual_extracted_features)}")
        return response_payload

    except Exception as e:
        logger.error(f"Exception processing document {request.doc_reference}: {e}", exc_info=True)
        return ProcessDocumentResponse(
            doc_reference=request.doc_reference,
            status="error",
            message=f"Unexpected server error: {type(e).__name__} - {e}",
            extracted_features=[]
        )

@app.post("/mcp/clear_documents", status_code=204)
async def clear_all_documents_endpoint():
    logger.info("Clearing all documents")
    rag_processor.clear_all_documents()
    return None

@app.post("/mcp")
async def mcp_tool_router(call_request: MCPToolCallRequest) -> MCPToolCallResponse:
    logger.info(f"MCP tool router called with method: {call_request.method}")
    try:
        if call_request.method == "extract_features_from_specs":
            try:
                tool_params = ExtractFeaturesParams(**call_request.params)
                result_data = _tool_extract_features_from_specs(tool_params)
                response = MCPToolCallResponse(result={"extract_features_from_specs": result_data.model_dump()})
                logger.info("extract_features_from_specs completed successfully")
                return response
            except ValueError as ve:
                logger.error(f"Validation error: {ve}", exc_info=True)
                return MCPToolCallResponse(error={"code": 400, "message": str(ve)})
            except Exception as e:
                logger.error(f"Error in extract_features_from_specs: {e}", exc_info=True)
                return MCPToolCallResponse(error={"code": 500, "message": f"Internal server error: {str(e)}"})
        elif call_request.method == "extract_features_from_screenshots":
            try:
                tool_params = ExtractFeaturesFromScreenshotParams(**call_request.params)
                result_data = _tool_extract_features_from_screenshots(tool_params)
                response = MCPToolCallResponse(result={"extract_features_from_screenshots": result_data.model_dump()})
                logger.info("extract_features_from_screenshots completed successfully")
                return response
            except ValueError as ve:
                logger.error(f"Validation error in screenshots: {ve}", exc_info=True)
                return MCPToolCallResponse(error={"code": 400, "message": str(ve)})
            except Exception as e:
                logger.error(f"Error in extract_features_from_screenshots: {e}", exc_info=True)
                return MCPToolCallResponse(error={"code": 500, "message": f"Internal server error: {str(e)}"})
        else:
            logger.error(f"Unknown method: {call_request.method}")
            return MCPToolCallResponse(error={"code": 400, "message": f"Unknown method: {call_request.method}"})
    except Exception as e:
        logger.error(f"Unexpected error in MCP router: {e}", exc_info=True)
        return MCPToolCallResponse(error={"code": 500, "message": "Internal server error in router"})

@app.get("/health", summary="Health Check", tags=["Management"])
async def health_check():
    return {
        "status": "healthy", 
        "rag_docs_loaded": len(rag_processor.document_vector_stores),
        "screenshots_loaded": len(screenshot_processor.processed_screenshots),
        "screenshot_api_configured": bool(settings.OPENAI_SCREENSHOT_KEY),
        "screenshot_api_usage": {
            "total_calls": screenshot_processor.api_calls,
            "total_tokens": screenshot_processor.total_tokens,
            "total_cost": round(screenshot_processor.total_cost, 4),
            "budget_limit": 1.0,
            "budget_remaining": round(max(0, 1.0 - screenshot_processor.total_cost), 4)
        }
    }

@app.post("/mcp/process_screenshot", response_model=ProcessScreenshotResponse)
async def process_screenshot_endpoint(request: ProcessScreenshotRequest):
    logger.info(f"Processing screenshot: {request.doc_reference}, image: {request.image_filename}")
    
    actual_extracted_features: List[str] = []
    status = "failed"
    message = "Screenshot processing error occurred."

    try:
        success = screenshot_processor.add_screenshot(
            request.doc_reference, 
            request.image_base64, 
            request.image_filename
        )
        
        logger.info(f"Screenshot processing success: {success}")

        if success:
            status = "processed"
            message = "Screenshot processed successfully."
            actual_extracted_features = screenshot_processor.get_available_features(request.doc_reference)
            logger.info(f"Features discovered: {len(actual_extracted_features)}")
            
            if actual_extracted_features:
                message += f" {len(actual_extracted_features)} features discovered."
            else:
                message += " No features discovered in screenshot."
        else:
            message = "Screenshot processing failed."
            logger.error(f"Screenshot processing failed for {request.doc_reference}")

        response_payload = ProcessScreenshotResponse(
            doc_reference=request.doc_reference,
            status=status,
            message=message,
            extracted_features=actual_extracted_features,
            image_filename=request.image_filename
        )
        
        logger.info(f"Returning screenshot response with status: {status}, features: {len(actual_extracted_features)}")
        return response_payload

    except Exception as e:
        logger.error(f"Exception processing screenshot {request.doc_reference}: {e}", exc_info=True)
        return ProcessScreenshotResponse(
            doc_reference=request.doc_reference,
            status="error",
            message=f"Unexpected error: {type(e).__name__} - {e}",
            extracted_features=[],
            image_filename=request.image_filename
        )

def _tool_extract_features_from_screenshots(params: ExtractFeaturesFromScreenshotParams) -> ExtractFeaturesFromScreenshotResult:
    logger.info(f"Extract features from screenshots called with {len(params.screenshot_references)} screenshots for {len(params.features_list)} features")
    
    results_data: Dict[str, Dict[str, str]] = {}

    if not params.screenshot_references:
        logger.error("No screenshot references provided")
        return ExtractFeaturesFromScreenshotResult(comparison_data={})
    
    if not params.features_list:
        logger.error("No features specified")
        return ExtractFeaturesFromScreenshotResult(comparison_data={})

    for i, screenshot_ref in enumerate(params.screenshot_references):
        product_name = None
        if params.product_names and i < len(params.product_names):
            product_name = params.product_names[i]
        
        logger.info(f"Processing screenshot '{screenshot_ref}' for product '{product_name or 'any'}'")
        
        if screenshot_ref not in screenshot_processor.processed_screenshots:
            logger.warning(f"Screenshot '{screenshot_ref}' not processed")
            results_data[screenshot_ref] = {feature: "Screenshot not processed" for feature in params.features_list}
            continue

        screenshot_feature_values: Dict[str, str] = {}
        for feature_name in params.features_list:
            try:
                value = screenshot_processor.extract_feature_from_screenshot(
                    screenshot_ref, 
                    feature_name, 
                    product_name
                )
                screenshot_feature_values[feature_name] = value
                logger.debug(f"Extracted '{feature_name}': '{value}'")
            except Exception as e:
                logger.error(f"Error extracting '{feature_name}' from '{screenshot_ref}': {e}", exc_info=True)
                screenshot_feature_values[feature_name] = f"Extraction Error: {str(e)}"

        results_data[screenshot_ref] = screenshot_feature_values

    final_result = ExtractFeaturesFromScreenshotResult(comparison_data=results_data)
    logger.info(f"Completed processing {len(results_data)} screenshots")
    return final_result

@app.post("/mcp/clear_screenshots", status_code=204)
async def clear_all_screenshots_endpoint():
    logger.info("Clearing all screenshots")
    screenshot_processor.clear_all_screenshots()
    return None

if __name__ == "__main__":
    uvicorn.run(
        "app.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=True,
        log_level="info"
    )
