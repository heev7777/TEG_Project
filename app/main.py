# app/main.py
import streamlit as st
import os
import logging
import httpx # For calling MCP server's /process_document and /clear_documents
import asyncio # For running async agent code in Streamlit
import uuid # For generating unique references for uploaded files
from pathlib import Path
from typing import List, Dict, Any, Set, Optional 
from queue import Queue

from app.core.config import settings
from app.agents import run_comparison_tool_directly
from app.core.schemas import ProcessDocumentRequest, ProcessDocumentResponse, ComparisonResponse, ExtractFeaturesParams

# Configure logging for the Streamlit app
# This will go to the console where Streamlit is running
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for the logger specific to this module

if 'processed_docs_info' not in st.session_state:
    st.session_state.processed_docs_info = {} 

if 'upload_queue' not in st.session_state:
    st.session_state.upload_queue = [] 

if 'processing_upload' not in st.session_state:
    st.session_state.processing_upload = False

mcp_doc_proc_client = httpx.AsyncClient(base_url=f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}")

async def process_uploaded_file_on_mcp(uploaded_file_obj, unique_doc_ref: str) -> ProcessDocumentResponse | None:
    """Saves uploaded file temporarily and asks MCP server to process it, returns response."""
    logger.info(f"[process_uploaded_file_on_mcp] Starting processing for file: {uploaded_file_obj.name}, ref: {unique_doc_ref}")
    temp_file_path = None 
    try:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_file_path = settings.UPLOAD_DIR / f"{unique_doc_ref}_{uploaded_file_obj.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        logger.info(f"[process_uploaded_file_on_mcp] Temporarily saved uploaded file to: {temp_file_path}")

        request_payload = ProcessDocumentRequest(
            doc_reference=unique_doc_ref,
            file_path=str(temp_file_path) 
        )
        target_url = f"{mcp_doc_proc_client.base_url}/mcp/process_document" 
        logger.info(f"[process_uploaded_file_on_mcp] Attempting to POST to MCP server at: {target_url} with payload: {request_payload.model_dump_json(indent=2)}")
        
        response = await mcp_doc_proc_client.post("/mcp/process_document", json=request_payload.model_dump(), timeout=60.0)
        logger.info(f"[process_uploaded_file_on_mcp] Raw response status from MCP /mcp/process_document: {response.status_code}")
        logger.info(f"[process_uploaded_file_on_mcp] Raw response text from MCP /mcp/process_document: {response.text[:500]}...") 

        response.raise_for_status() 

        response_data = ProcessDocumentResponse(**response.json())
        logger.info(f"[process_uploaded_file_on_mcp] Parsed ProcessDocumentResponse from MCP: {response_data.model_dump_json(indent=2)}")
        
        # CORRECTED: Only accept "processed" as a success status from the actual server logic
        if response_data.status == "processed":
            logger.info(f"[process_uploaded_file_on_mcp] MCP Server processed {unique_doc_ref} successfully (status: {response_data.status}). Extracted features: {response_data.extracted_features}")
            return response_data 
        else:
            logger.error(f"[process_uploaded_file_on_mcp] MCP Server failed to process {unique_doc_ref}. Status: {response_data.status}, Message: {response_data.message}")
            st.error(f"MCP Server failed to process '{uploaded_file_obj.name}': {response_data.message}")
            if temp_file_path and temp_file_path.exists(): 
                try:
                    temp_file_path.unlink()
                    logger.info(f"[process_uploaded_file_on_mcp] Deleted temp file after MCP processing failure: {temp_file_path}")
                except OSError as e_unlink:
                    logger.error(f"[process_uploaded_file_on_mcp] Error deleting temp file {temp_file_path} after MCP failure: {e_unlink}")
            return response_data # Still return response_data so the main loop can see the status
            
    except httpx.HTTPStatusError as e_http:
        st.error(f"Error contacting MCP server to process file '{uploaded_file_obj.name}': {e_http.response.status_code} - {e_http.response.text[:200]}")
        logger.error(f"[process_uploaded_file_on_mcp] HTTP error processing file {unique_doc_ref} on MCP: {e_http}", exc_info=True)
    except httpx.RequestError as e_req:
        st.error(f"Network error contacting MCP server for '{uploaded_file_obj.name}': {e_req}")
        logger.error(f"[process_uploaded_file_on_mcp] Request error processing file {unique_doc_ref} on MCP: {e_req}", exc_info=True)
    except Exception as e_gen:
        st.error(f"An unexpected error occurred while processing file '{uploaded_file_obj.name}': {e_gen}")
        logger.error(f"[process_uploaded_file_on_mcp] General error processing file {unique_doc_ref} for MCP: {e_gen}", exc_info=True)
    
    return None 

async def clear_mcp_documents():
    logger.info("[clear_mcp_documents] Attempting to clear documents on MCP server.")
    try:
        await mcp_doc_proc_client.post("/mcp/clear_documents", timeout=10.0)
        st.session_state.processed_docs_info.clear() 
        st.session_state.upload_queue.clear() 
        st.session_state.processing_upload = False 
        
        upload_dir_path = settings.UPLOAD_DIR
        if upload_dir_path.exists():
            deleted_count = 0
            error_count = 0
            for f_path in upload_dir_path.glob("*"):
                if f_path.is_file():
                    try:
                        f_path.unlink()
                        deleted_count +=1
                    except OSError as e:
                        logger.error(f"[clear_mcp_documents] Error deleting temp file {f_path}: {e}")
                        error_count +=1
            logger.info(f"[clear_mcp_documents] Deleted {deleted_count} temp files from {upload_dir_path}. {error_count} errors.")
        st.success("Cleared all processed documents from server and local temp storage.")
        logger.info("[clear_mcp_documents] MCP documents and local temp files cleared successfully.")
    except Exception as e:
        st.error(f"Failed to clear documents on MCP server: {e}")
        logger.error(f"[clear_mcp_documents] Error clearing MCP documents: {e}", exc_info=True)

async def run_comparison_logic(products_to_compare_refs: list, features: list):
    logger.info(f"[run_comparison_logic] Called with products: {products_to_compare_refs}, features: {features}")
    if not products_to_compare_refs or not features:
        st.warning("Please select products and enter features to compare.")
        logger.warning("[run_comparison_logic] Missing products or features.")
        return

    with st.spinner("Comparing features... Please wait."):
        try:
            tool_result = await run_comparison_tool_directly(
                product_doc_refs=products_to_compare_refs, 
                features_to_compare=features 
            )
            logger.info(f"[run_comparison_logic] Direct tool result: {tool_result}")

            if "error" in tool_result:
                st.error(f"Comparison failed: {tool_result['error']}")
                logger.error(f"[run_comparison_logic] Comparison tool returned error: {tool_result['error']}")
            elif tool_result:
                comparison_data = tool_result 
                st.header("Comparison Results")

                if not comparison_data:
                    st.warning("No comparison data was returned by the tool.")
                    logger.warning("[run_comparison_logic] Tool returned empty comparison data.")
                    return

                header_features = features 
                table_data = {"Feature": header_features}
                for product_ref, feature_values in comparison_data.items():
                    original_filename = "Unknown Product"
                    for original_name, doc_info in st.session_state.processed_docs_info.items():
                        if doc_info['unique_doc_ref'] == product_ref:
                            original_filename = original_name
                            break
                    table_data[original_filename] = [feature_values.get(feat, "N/A") for feat in header_features]
                
                if len(table_data) > 1 : 
                    st.dataframe(table_data)
                    logger.info("[run_comparison_logic] Displayed comparison dataframe.")
                else:
                    st.warning("Could not prepare data for display. Check product references.")
                    logger.warning("[run_comparison_logic] No product data found in comparison_data to display.")
            else:
                st.warning("Comparison tool returned no result and no error.")
                logger.warning("[run_comparison_logic] Tool returned None or empty dict without error key.")

        except Exception as e:
            st.error(f"An unexpected error occurred during comparison: {e}")
            logger.error("[run_comparison_logic] Error during comparison logic", exc_info=True)

def main_ui():
    st.set_page_config(page_title=settings.PROJECT_NAME, layout="wide")
    st.title(settings.PROJECT_NAME)
    st.markdown(settings.DESCRIPTION)

    st.sidebar.header("Controls")
    if st.sidebar.button("Clear All Processed Documents", key="clear_docs_button"):
        asyncio.run(clear_mcp_documents())
        st.rerun() 

    st.header("1. Upload Specification Sheets")
    uploaded_file_objects = st.file_uploader(
        "Choose up to 3 specification files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_file_objects:
        current_processed_names = set(st.session_state.processed_docs_info.keys())
        queue_file_names = set(item.name for item in st.session_state.upload_queue)

        for uploaded_file in uploaded_file_objects:
            if uploaded_file.name not in current_processed_names and uploaded_file.name not in queue_file_names:
                st.session_state.upload_queue.append(uploaded_file)
                logger.info(f"[File Uploader] Added '{uploaded_file.name}' to upload queue. Queue size: {len(st.session_state.upload_queue)}")

    if st.session_state.upload_queue and not st.session_state.processing_upload:
        st.session_state.processing_upload = True 
        file_to_process = st.session_state.upload_queue[0] 
        unique_doc_ref = f"doc_{uuid.uuid4().hex[:8]}"
        logger.info(f"[Upload Processor] Processing '{file_to_process.name}' from queue with ref '{unique_doc_ref}'.")

        with st.spinner(f"Processing '{file_to_process.name}' on server..."):
            response_data = asyncio.run(process_uploaded_file_on_mcp(file_to_process, unique_doc_ref))

            # CORRECTED: Only accept "processed" as a success status
            if response_data and response_data.status == "processed":
                st.session_state.processed_docs_info[file_to_process.name] = {
                    'unique_doc_ref': unique_doc_ref,
                    'temp_file_path': str(settings.UPLOAD_DIR / f"{unique_doc_ref}_{file_to_process.name}"), 
                    'extracted_features': response_data.extracted_features or [] 
                }
                logger.info(f"[Upload Processor] Stored in session state for '{file_to_process.name}': unique_doc_ref={unique_doc_ref}, extracted_features={response_data.extracted_features}")
                st.success(f"'{file_to_process.name}' processed (status: {response_data.status}) and ready for comparison.")
            elif response_data: 
                logger.warning(f"[Upload Processor] MCP processing for '{file_to_process.name}' was not successful according to expected status. Status: {response_data.status}, Message: {response_data.message}")
            else: 
                logger.error(f"[Upload Processor] Failed to process '{file_to_process.name}' on MCP server (no response data or HTTP error).")
        
        st.session_state.upload_queue.pop(0)
        st.session_state.processing_upload = False
        logger.info(f"[Upload Processor] Finished processing attempt for '{file_to_process.name}'. Rerunning Streamlit.")
        st.rerun()

    logger.info(f"[UI Update] Current st.session_state.processed_docs_info: {st.session_state.processed_docs_info}")
    all_available_features_from_processed: Set[str] = set()
    for doc_name, doc_info in st.session_state.processed_docs_info.items(): 
        features_for_doc = doc_info.get('extracted_features') 
        logger.info(f"[UI Update] Features for doc '{doc_name}': {features_for_doc}")
        if features_for_doc: 
            all_available_features_from_processed.update(features_for_doc)
    
    sorted_available_features = sorted(list(all_available_features_from_processed))
    logger.info(f"[UI Update] Collected available features for multiselect: {sorted_available_features}")

    if st.session_state.processed_docs_info:
        st.subheader("Processed Documents Ready for Comparison:")
        
        display_name_to_ref = {name: doc_info['unique_doc_ref'] for name, doc_info in st.session_state.processed_docs_info.items()}
        processed_file_display_names = list(display_name_to_ref.keys())

        selected_display_names = st.multiselect(
            "Select 2 or 3 documents to compare:",
            options=processed_file_display_names,
            max_selections=3, 
            key="product_selector"
        )
        
        products_to_compare_refs = [display_name_to_ref[name] for name in selected_display_names if name in display_name_to_ref]

        if products_to_compare_refs and len(products_to_compare_refs) >= 2:
            st.header("2. Specify Features to Compare")

            select_all_features_default = True if sorted_available_features else False 
            if 'select_all_features_checkbox_value' not in st.session_state:
                 st.session_state.select_all_features_checkbox_value = select_all_features_default

            select_all_features = st.checkbox(
                "Select All Available Features",
                key="select_all_features_checkbox",
                value=st.session_state.select_all_features_checkbox_value
            )
            st.session_state.select_all_features_checkbox_value = select_all_features


            current_multiselect_options = sorted_available_features
            
            if select_all_features:
                default_multiselect_value = current_multiselect_options
            else:
                previous_selection = st.session_state.get('features_selector_value', [])
                default_multiselect_value = [f for f in previous_selection if f in current_multiselect_options]

            selected_features = st.multiselect(
                "Select features to compare:",
                options=current_multiselect_options, 
                default=default_multiselect_value,
                key="features_selector" 
            )
            st.session_state.features_selector_value = selected_features 

            if st.button("Compare Features", key="compare_button"):
                if selected_features:
                    asyncio.run(run_comparison_logic(products_to_compare_refs, selected_features))
                else:
                    st.warning("Please select at least one feature to compare.")
        elif selected_display_names and len(selected_display_names) < 2:
            st.warning("Please select at least 2 documents to compare.")
    else:
        st.info("Upload specification sheets to begin.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Version: {settings.VERSION}")

if __name__ == "__main__":
    main_ui()
