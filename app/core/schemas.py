from pydantic import BaseModel
from typing import List, Dict, Optional

class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction."""
    product_document_ids: List[str]
    features_list: List[str]

class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction."""
    results: Dict[str, Dict[str, str]]

class ProductComparisonRequest(BaseModel):
    """Request model for product comparison."""
    product_ids: List[str]
    features: List[str]

class ProductComparisonResponse(BaseModel):
    """Response model for product comparison."""
    comparison_results: Dict[str, Dict[str, str]]
    analysis: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str = "1.0.0" 