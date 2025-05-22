# app/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

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
    params: Dict[str, Any] = Field(default_factory=dict)

class MCPToolCallResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class ProcessDocumentRequest(BaseModel):
    doc_reference: str = Field(..., description="A unique reference ID for the document.")
    file_path: str = Field(..., description="Absolute path to the document file on the server filesystem.")

class ProcessDocumentResponse(BaseModel):
    doc_reference: str
    status: str
    message: Optional[str] = None
    extracted_features: Optional[List[str]] = None

# --- Tool Input Schemas ---
# Schema for the compare_product_features_via_mcp tool
class CompareProductFeaturesInput(BaseModel):
    product_references_str: str = Field(
        ..., description="Comma-separated string of product document references (e.g., 'doc_abc123,doc_def456')"
    )
    features_list_str: str = Field(
        ..., description="Comma-separated string of features to compare (e.g., 'RAM,Storage,Price')"
    )

# --- Streamlit Communication (if app/main.py calls a separate FastAPI backend) ---
class ComparisonRequest(BaseModel):
    uploaded_file_details: List[Dict]
    features_to_compare: List[str]
    products_to_compare: List[str]

class ComparisonResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Dict[str, str]]] = None
    text_summary: Optional[str] = None
    missing_features: Optional[Dict[str, List[str]]] = None