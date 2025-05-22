# app/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ExtractFeaturesToolInputSchema(BaseModel):
    product_references_str: str = Field(description="A comma-separated string of product document references (e.g., 'doc1,doc2').")
    features_list_str: str = Field(description="A comma-separated string of features to extract (e.g., 'RAM,Price').")

class ExtractFeaturesParams(BaseModel):
    product_references: List[str] = Field(..., description="List of unique references for processed product documents.")
    features_list: List[str] = Field(..., description="List of features to extract.")

class ExtractFeaturesResult(BaseModel):
    comparison_data: Dict[str, Dict[str, str]]

class MCPToolCallRequest(BaseModel):
    method: str
    params: Dict

class MCPToolCallResponse(BaseModel):
    result: Optional[Dict] = None
    error: Optional[Dict] = None

class ProcessDocumentRequest(BaseModel):
    doc_reference: str
    file_path: str

class ProcessDocumentResponse(BaseModel):
    doc_reference: str
    status: str
    message: Optional[str] = None

# --- Tool Input Schemas ---
# Schema for the compare_product_features_via_mcp tool
class CompareProductFeaturesInput(BaseModel):
    product_references_str: str = Field(..., description="A comma-separated string of product document references (e.g., 'doc1,doc2').")
    features_list_str: str = Field(..., description="A comma-separated string of features to extract (e.g., 'RAM,Price').")

# --- Streamlit Communication (if app/main.py calls a separate FastAPI backend) ---
class ComparisonRequest(BaseModel):
    uploaded_file_details: List[Dict]
    features_to_compare: List[str]
    products_to_compare: List[str]

class ComparisonResponse(BaseModel):
    comparison_table: Dict[str, Dict[str, str]]
    message: Optional[str] = None
    error: Optional[str] = None