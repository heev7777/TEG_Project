from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from rag_processor import RAGProcessor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Product Feature Comparison MCP Server")
rag_processor = RAGProcessor()

class FeatureExtractionRequest(BaseModel):
    product_document_ids: List[str]
    features_list: List[str]

class FeatureExtractionResponse(BaseModel):
    results: Dict[str, Dict[str, str]]

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
        
        return FeatureExtractionResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 