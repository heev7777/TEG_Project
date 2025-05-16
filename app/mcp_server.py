from fastapi import FastAPI, HTTPException
import uvicorn
from app.core.config import get_settings
from app.core.schemas import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    HealthCheckResponse
)
from app.rag_processor import RAGProcessor
from app.utils.logger import setup_logger

# Initialize settings and logger
settings = get_settings()
logger = setup_logger(__name__)

app = FastAPI(title="Product Feature Comparison MCP Server")
rag_processor = RAGProcessor()

@app.post("/mcp/extract_features", response_model=FeatureExtractionResponse)
async def extract_features(request: FeatureExtractionRequest):
    """
    Extract specified features from product documents.
    
    Args:
        request: FeatureExtractionRequest containing product document IDs and features to extract
        
    Returns:
        FeatureExtractionResponse containing extracted feature values for each product
    """
    try:
        logger.info(f"Processing feature extraction request for products: {request.product_document_ids}")
        results = {}
        
        # Process each product document
        for product_id in request.product_document_ids:
            # TODO: Implement document loading based on product_id
            # For now, we'll assume the document is already processed
            
            # Extract features for this product
            product_features = {}
            for feature in request.features_list:
                value = rag_processor.extract_feature(feature)
                product_features[feature] = value if value else "Not found"
            
            results[product_id] = product_features
            logger.info(f"Extracted features for product {product_id}")
        
        return FeatureExtractionResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error processing feature extraction request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(status="healthy")

if __name__ == "__main__":
    port = settings.MCP_SERVER_PORT
    logger.info(f"Starting MCP server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 