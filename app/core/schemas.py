# app/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class MCPToolCallRequest(BaseModel):
    method: str
    params: Dict

class MCPToolCallResponse(BaseModel):
    result: Optional[Dict] = None
    error: Optional[Dict] = None

# --- MCP Tool: extract_features_from_specs ---
class ExtractFeaturesParams(BaseModel):
    product_references: List[str] = Field(..., description="List of unique references for processed product documents.")
    features_list: List[str] = Field(..., description="List of features to extract.")

class ExtractFeaturesResult(BaseModel):
    comparison_data: Dict[str, Dict[str, str]]

# --- Streamlit Communication (if app/main.py calls a separate FastAPI backend) ---
class ComparisonRequest(BaseModel):
    uploaded_file_details: List[Dict]
    features_to_compare: List[str]
    products_to_compare: List[str]

class ComparisonResponse(BaseModel):
    comparison_table: Dict[str, Dict[str, str]]
    message: Optional[str] = None
    error: Optional[str] = None