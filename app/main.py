# app/main.py
import streamlit as st
import os
import logging
import httpx # For calling MCP server's /process_document and /clear_documents
import asyncio # For running async agent code in Streamlit
import uuid # For generating unique references for uploaded files
from pathlib import Path
from typing import List, Dict, Any, Set, Optional # Import Optional
from queue import Queue

# Assuming your project structure is app/main.py, app/core/config.py etc.
from app.core.config import settings
# from app.agents import ProductComparisonMAS # Your MAS class # REMOVE
from app.agents import run_comparison_tool_directly # Import the new direct function
from app.core.schemas import ProcessDocumentRequest, ProcessDocumentResponse, ComparisonResponse, ExtractFeaturesParams # Import ProcessDocumentResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize agent (and its own httpx client for MCP) # REMOVE AGENT INITIALIZATION
# This agent instance will be reused across Streamlit interactions for the same session.
# if 'comparison_mas' not in st.session_state:
#     st.session_state.comparison_mas = ProductComparisonMAS()

# Store processed document references, paths, and features in session state
if 'processed_docs_info' not in st.session_state:
    st.session_state.processed_docs_info = {} # Maps original filename to {unique_doc_ref, temp_file_path, extracted_features}

# Store a queue of files to be processed (using file names as identifiers in the queue)
if 'upload_queue' not in st.session_state:
    st.session_state.upload_queue = [] # List of uploaded_file_objects

# Flag to indicate if processing is currently happening
if 'processing_upload' not in st.session_state:
    st.session_state.processing_upload = False

# HTTP client for calling MCP document processing (distinct from agent's internal client)
# Best to manage this client's lifecycle if Streamlit runs things in threads.
# For simplicity, create it as needed or use a session-scoped one.
mcp_doc_proc_client = httpx.AsyncClient(base_url=f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}")

async def process_uploaded_file_on_mcp(uploaded_file_obj, unique_doc_ref: str) -> ProcessDocumentResponse | None:
    """Saves uploaded file temporarily and asks MCP server to process it, returns response."""
    try:
        # Ensure UPLOAD_DIR exists
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_file_path = settings.UPLOAD_DIR / f"{unique_doc_ref}_{uploaded_file_obj.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        logger.info(f"Temporarily saved uploaded file to: {temp_file_path}")

        request_payload = ProcessDocumentRequest(
            doc_reference=unique_doc_ref,
            file_path=str(temp_file_path) # Send absolute path
        )
        response = await mcp_doc_proc_client.post("/mcp/process_document", json=request_payload.model_dump(), timeout=60.0)
        response.raise_for_status()
        
        # Log the raw response for debugging
        logger.info(f"Raw response from MCP /mcp/process_document: {response.text}")

        # Directly parse the response into the Pydantic model
        response_data = ProcessDocumentResponse(**response.json())
        logger.info(f"Parsed response_data: {response_data.model_dump_json(indent=2)}")
        
        if response_data.status == "processed":
            logger.info(f"MCP Server processed {unique_doc_ref} successfully.")
            # Note: The server does NOT return file_path in the response. We already have the path.
            return response_data # Return the full response including features
        else:
            logger.error(f"MCP Server failed to process {unique_doc_ref}: {response_data.message}")
            # Clean up the temp file if processing failed
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    logger.info(f"Deleted temp file after processing failure: {temp_file_path}")
                except OSError as e:
                    logger.error(f"Error deleting temp file {temp_file_path} after failure: {e}")
            return None
            
    except httpx.RequestError as e:
        st.error(f"Error contacting MCP server to process file: {e}")
        logger.error(f"HTTP error processing file {unique_doc_ref} on MCP: {e}")
        # Clean up the temp file if a request error occurred
        if 'temp_file_path' in locals() and temp_file_path.exists():
             try:
                 temp_file_path.unlink()
                 logger.info(f"Deleted temp file after request error: {temp_file_path}")
             except OSError as e:
                  logger.error(f"Error deleting temp file {temp_file_path} after request error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing file {uploaded_file_obj.name}: {e}")
        logger.error(f"General error processing file {unique_doc_ref} for MCP: {e}", exc_info=True)
        # Clean up the temp file for any other exception
        if 'temp_file_path' in locals() and temp_file_path.exists():
             try:
                 temp_file_path.unlink()
                 logger.info(f"Deleted temp file after general error: {temp_file_path}")
             except OSError as e:
                  logger.error(f"Error deleting temp file {temp_file_path} after general error: {e}")
        return None

async def clear_mcp_documents():
    try:
        await mcp_doc_proc_client.post("/mcp/clear_documents", timeout=10.0)
        st.session_state.processed_docs_info.clear() # Clear processed docs info
        st.session_state.upload_queue.clear() # Clear the upload queue as well
        st.session_state.processing_upload = False # Reset processing flag
        # Clean up temp files in the upload directory
        upload_dir_path = settings.UPLOAD_DIR
        if upload_dir_path.exists():
             for f_path in upload_dir_path.glob("*"):
                 if f_path.is_file():
                     try:
                         f_path.unlink()
                         logger.info(f"Deleted temp file: {f_path}")
                     except OSError as e:
                         logger.error(f"Error deleting temp file {f_path}: {e}")
        st.success("Cleared all processed documents from server and local temp storage.")
        logger.info("MCP documents cleared successfully.")
    except Exception as e:
        st.error(f"Failed to clear documents on MCP server: {e}")
        logger.error(f"Error clearing MCP documents: {e}")

async def run_comparison_logic(products_to_compare_refs: list, features: list):
    """Handles the call to the direct tool invocation function and displays results."""
    if not products_to_compare_refs or not features:
        st.warning("Please select products and enter features to compare.")
        return

    with st.spinner("Comparing features... Please wait."):
        try:
            # Call the direct tool invocation function
            tool_result = await run_comparison_tool_directly(
                product_doc_refs=products_to_compare_refs, # Key structured input
                features_to_compare=features # Key structured input
            )
            logger.info(f"Direct tool result: {tool_result}")

            # Handle the direct tool result (which is either data or an error dict)
            if "error" in tool_result:
                 st.error(f"Comparison failed: {tool_result['error']}")
            elif tool_result:
                # If no error and result is not empty, assume it's the comparison data
                comparison_data = tool_result
                st.header("Comparison Results")

                if not comparison_data:
                    st.warning("No comparison data was returned.")
                    return

                # Create a list of features for table header
                # Use the features that were *requested* for comparison
                header_features = features

                table_data = {"Feature": header_features}
                for product_ref, feature_values in comparison_data.items():
                    # Find the original filename from session state using the unique_doc_ref
                    original_filename = "Unknown Product"
                    for original_name, doc_info in st.session_state.processed_docs_info.items():
                         if doc_info['unique_doc_ref'] == product_ref:
                             original_filename = original_name
                             break

                    table_data[original_filename] = [feature_values.get(feat, "N/A") for feat in header_features]

                st.dataframe(table_data)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error("Error during run_comparison_logic", exc_info=True)


def main_ui():
    st.set_page_config(page_title=settings.PROJECT_NAME, layout="wide")
    st.title(settings.PROJECT_NAME)
    st.markdown(settings.DESCRIPTION)

    st.sidebar.header("Controls")
    if st.sidebar.button("Clear All Processed Documents", key="clear_docs_button"):
        asyncio.run(clear_mcp_documents())
        st.rerun() # Rerun to reflect cleared state

    st.header("1. Upload Specification Sheets")
    uploaded_file_objects = st.file_uploader(
        "Choose up to 3 specification files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Add new files to the queue if they are not already processed or in queue
    if uploaded_file_objects:
        current_processed_names = set(st.session_state.processed_docs_info.keys())
        # Create a set of file names currently in the queue
        queue_file_names = set(item.name for item in st.session_state.upload_queue)

        for uploaded_file in uploaded_file_objects:
            # Only add files to the queue if they are not processed and not already in queue
            if uploaded_file.name not in current_processed_names and uploaded_file.name not in queue_file_names:
                st.session_state.upload_queue.append(uploaded_file)
                logger.info(f"Added {uploaded_file.name} to upload queue. Current queue size: {len(st.session_state.upload_queue)}")

    # Process the next file in the queue if not currently processing and queue is not empty
    if st.session_state.upload_queue and not st.session_state.processing_upload:
        st.session_state.processing_upload = True # Set processing flag
        file_to_process = st.session_state.upload_queue[0] # Get the first file (don't remove yet)
        unique_doc_ref = f"doc_{uuid.uuid4().hex[:8]}"

        with st.spinner(f"Processing {file_to_process.name} on server..."):
            # Process the file using asyncio.run
            response_data = asyncio.run(process_uploaded_file_on_mcp(file_to_process, unique_doc_ref))

            if response_data and response_data.status == "processed":
                 st.session_state.processed_docs_info[file_to_process.name] = {
                      'unique_doc_ref': unique_doc_ref,
                      'temp_file_path': str(settings.UPLOAD_DIR / f"{unique_doc_ref}_{file_to_process.name}"), # Store the locally generated path
                      'extracted_features': response_data.extracted_features or [] # Store features, default to empty list if None
                 }
                 st.success(f"'{file_to_process.name}' processed and ready for comparison.")
            else:
                # Error message already shown in process_uploaded_file_on_mcp
                pass # No need for another st.error here
        
        # After attempting to process, remove the file from the queue and reset flag
        # This happens regardless of processing success to prevent infinite loops on a failing file
        st.session_state.upload_queue.pop(0)
        st.session_state.processing_upload = False

        # Rerun to process the next in queue or update UI
        st.rerun()

    # Log the current state of processed_docs_info and available features
    logger.info(f"Current st.session_state.processed_docs_info: {st.session_state.processed_docs_info}")
    all_available_features_from_processed: Set[str] = set()
    for doc_info in st.session_state.processed_docs_info.values():
        features_for_doc = doc_info.get('extracted_features', [])
        all_available_features_from_processed.update(features_for_doc)
    logger.info(f"Collected available features for multiselect: {sorted(list(all_available_features_from_processed))}")

    # Display currently processed files and allow selection
    if st.session_state.processed_docs_info:
        st.subheader("Processed Documents Ready for Comparison:")
        
        # Create a mapping from display name (original filename) to unique_doc_ref
        display_name_to_ref = {name: doc_info['unique_doc_ref'] for name, doc_info in st.session_state.processed_docs_info.items()}
        processed_file_display_names = list(display_name_to_ref.keys())

        selected_display_names = st.multiselect(
            "Select 2 or 3 documents to compare:",
            options=processed_file_display_names,
            max_selections=3, # As per original scope
            key="product_selector"
        )

        # Collect all unique features from *all* currently processed documents for the multiselect options
        all_available_features_from_processed: Set[str] = set()
        for doc_info in st.session_state.processed_docs_info.values():
            features_for_doc = doc_info.get('extracted_features', [])
            all_available_features_from_processed.update(features_for_doc)

        sorted_available_features = sorted(list(all_available_features_from_processed))

        products_to_compare_refs = [display_name_to_ref[name] for name in selected_display_names if name in display_name_to_ref]

        if products_to_compare_refs and len(products_to_compare_refs) >= 2:
            st.header("2. Specify Features to Compare")

            # Checkbox to select all features
            select_all_features = st.checkbox("Select All Available Features", key="select_all_features_checkbox")

            # Multiselect for features
            # Set default value based on select_all_features checkbox
            default_features = sorted_available_features if select_all_features else []

            selected_features = st.multiselect(
                "Select features to compare:",
                options=sorted_available_features,
                default=default_features,
                key="features_selector"
            )

            if st.button("Compare Features", key="compare_button"):
                if selected_features:
                    # Call the updated run_comparison_logic directly
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
    # Note: To run async functions directly from top-level Streamlit script,
    # you often need asyncio.run() as shown.
    # For a more complex Streamlit app with persistent async tasks, you might explore
    # solutions like a separate thread for an asyncio event loop.
    # For this project, asyncio.run() for button clicks is fine.
    main_ui()