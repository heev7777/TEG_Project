# app/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ExtractFeaturesToolInputSchema(BaseModel):
    product_references_str: str = Field(description="A comma-separated string of product document references (e.g., 'doc1,doc2').")
    features_list_str: str = Field(description="A comma-separated string of features to extract (e.g., 'RAM,Price').")

class ExtractFeaturesParams(BaseModel):
    product_references: List[str]
    features_list: List[str]

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
    message: str
    extracted_features: List[str]

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
    comparison_data: Dict[str, Dict[str, str]]
    summary: str

# --- NEW: Screenshot Processing Schemas ---
class ProcessScreenshotRequest(BaseModel):
    doc_reference: str
    image_base64: str
    image_filename: str

class ProcessScreenshotResponse(BaseModel):
    doc_reference: str
    status: str
    message: str
    extracted_features: List[str]
    image_filename: str

class ExtractFeaturesFromScreenshotParams(BaseModel):
    screenshot_references: List[str]
    features_list: List[str]
    product_names: Optional[List[str]] = None

class ExtractFeaturesFromScreenshotResult(BaseModel):
    comparison_data: Dict[str, Dict[str, str]]
