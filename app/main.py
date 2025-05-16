# app/main.py
import streamlit as st
import os
import logging
import httpx # For calling MCP server's /process_document and /clear_documents
import asyncio # For running async agent code in Streamlit
import uuid # For generating unique references for uploaded files
from pathlib import Path

# Assuming your project structure is app/main.py, app/core/config.py etc.
from app.core.config import settings
from app.agents import ProductComparisonMAS # Your MAS class
from app.core.schemas import ProcessDocumentRequest, ComparisonResponse # Pydantic models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize agent (and its own httpx client for MCP)
# This agent instance will be reused across Streamlit interactions for the same session.
if 'comparison_mas' not in st.session_state:
    st.session_state.comparison_mas = ProductComparisonMAS()

# Store processed document references in session state
if 'processed_doc_refs' not in st.session_state:
    st.session_state.processed_doc_refs = {} # Maps original filename to unique_doc_ref
if 'processed_doc_paths' not in st.session_state:
    st.session_state.processed_doc_paths = {} # Maps unique_doc_ref to temp file path

# HTTP client for calling MCP document processing (distinct from agent's internal client)
# Best to manage this client's lifecycle if Streamlit runs things in threads.
# For simplicity, create it as needed or use a session-scoped one.
mcp_doc_proc_client = httpx.AsyncClient(base_url=f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}")

async def process_uploaded_file_on_mcp(uploaded_file_obj, unique_doc_ref: str) -> bool:
    """Saves uploaded file temporarily and asks MCP server to process it."""
    try:
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
        response_data = response.json()
        if response_data.get("status") == "processed":
            st.session_state.processed_doc_paths[unique_doc_ref] = str(temp_file_path)
            logger.info(f"MCP Server processed {unique_doc_ref} successfully.")
            return True
        else:
            logger.error(f"MCP Server failed to process {unique_doc_ref}: {response_data.get('message')}")
            return False
    except httpx.RequestError as e:
        st.error(f"Error contacting MCP server to process file: {e}")
        logger.error(f"HTTP error processing file {unique_doc_ref} on MCP: {e}")
        return False
    except Exception as e:
        st.error(f"An error occurred while processing file {uploaded_file_obj.name}: {e}")
        logger.error(f"General error processing file {unique_doc_ref} for MCP: {e}", exc_info=True)
        return False

async def clear_mcp_documents():
    try:
        await mcp_doc_proc_client.post("/mcp/clear_documents", timeout=10.0)
        st.session_state.processed_doc_refs.clear()
        st.session_state.processed_doc_paths.clear()
        # Clean up temp files
        for f_path in (settings.UPLOAD_DIR).glob("*"):
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
    """Handles the call to the agent and displays results."""
    if not products_to_compare_refs or not features:
        st.warning("Please select products and enter features to compare.")
        return

    comparison_mas: ProductComparisonMAS = st.session_state.comparison_mas
    
    # Construct a user-friendly query for the agent, but the key parts are the lists
    user_query = (
        f"I want to compare the products referred to as '{', '.join(products_to_compare_refs)}' "
        f"based on these features: '{', '.join(features)}'. Can you get this information for me?"
    )

    with st.spinner("Comparing features... Please wait."):
        try:
            agent_response_dict = await comparison_mas.run_comparison_agent(
                user_query=user_query, # Provides natural language context
                product_doc_refs=products_to_compare_refs, # Key structured input
                features_to_compare=features # Key structured input
            )
            logger.info(f"Agent response dictionary: {agent_response_dict}")

            if agent_response_dict.get("status") == "success" and agent_response_dict.get("data"):
                comparison_data = agent_response_dict["data"] # This should be the dict from the tool
                st.header("Comparison Results")
                
                # Format for st.dataframe or st.table
                # Input: {'Product_A_Ref': {'RAM': '8GB', 'Price': 'Not found'}, ...}
                # Output for dataframe: needs to be list of dicts or dict of lists/series
                # Or, build a table manually with st.write or st.markdown
                
                if not comparison_data:
                    st.warning("No comparison data was returned by the agent.")
                    return

                # Create a list of features for table header
                all_features_in_results = set()
                for prod_data in comparison_data.values():
                    all_features_in_results.update(prod_data.keys())
                
                header_features = sorted(list(all_features_in_results))
                if not header_features: # If no features were extracted at all
                    header_features = features # Fallback to user requested features

                table_data = {"Feature": header_features}
                for product_ref, feature_values in comparison_data.items():
                    # Get original filename for display
                    original_filename = "Unknown Product"
                    for fname, u_ref in st.session_state.processed_doc_refs.items():
                        if u_ref == product_ref:
                            original_filename = fname
                            break
                    
                    table_data[original_filename] = [feature_values.get(feat, "N/A") for feat in header_features]
                
                st.dataframe(table_data)

                if agent_response_dict.get("text_summary"):
                    st.subheader("Agent's Summary:")
                    st.markdown(agent_response_dict["text_summary"])

            elif agent_response_dict.get("status") == "success_text_only":
                 st.info("Agent provided a text summary but no structured data was extracted for table display.")
                 if agent_response_dict.get("text_summary"):
                    st.subheader("Agent's Summary:")
                    st.markdown(agent_response_dict["text_summary"])
                 else:
                    st.warning("Agent did not return a usable response or data.")
            else:
                error_message = agent_response_dict.get("message", "Unknown error during comparison.")
                st.error(f"Comparison failed: {error_message}")
                if agent_response_dict.get("text_summary"): # E.g. LLM saying it can't do it
                    st.info(f"Agent's attempt: {agent_response_dict['text_summary']}")


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

    if uploaded_file_objects:
        # Only process new files or if explicitly requested
        for uploaded_file in uploaded_file_objects:
            if uploaded_file.name not in st.session_state.processed_doc_refs:
                unique_doc_ref = f"doc_{uuid.uuid4().hex[:8]}" # Create a unique ID
                st.session_state.processed_doc_refs[uploaded_file.name] = unique_doc_ref
                with st.spinner(f"Processing {uploaded_file.name} on server..."):
                    success = asyncio.run(process_uploaded_file_on_mcp(uploaded_file, unique_doc_ref))
                    if success:
                        st.success(f"'{uploaded_file.name}' processed and ready for comparison as '{unique_doc_ref}'.")
                    else:
                        st.error(f"Failed to process '{uploaded_file.name}'. Check logs.")
                        # Remove from processed if failed
                        if uploaded_file.name in st.session_state.processed_doc_refs:
                            del st.session_state.processed_doc_refs[uploaded_file.name]

    # Display currently processed files and allow selection
    if st.session_state.processed_doc_refs:
        st.subheader("Processed Documents Ready for Comparison:")
        
        # Create a mapping from display name (original filename) to unique_doc_ref
        display_name_to_ref = {name: ref for name, ref in st.session_state.processed_doc_refs.items()}
        processed_file_display_names = list(display_name_to_ref.keys())

        selected_display_names = st.multiselect(
            "Select 2 or 3 documents to compare:",
            options=processed_file_display_names,
            max_selections=3, # As per original scope
            key="product_selector"
        )
        
        # Convert selected display names back to unique_doc_refs
        products_to_compare_refs = [display_name_to_ref[name] for name in selected_display_names if name in display_name_to_ref]

        if products_to_compare_refs and len(products_to_compare_refs) >= 2 :
            st.header("2. Specify Features to Compare")
            features_input_str = st.text_input(
                "Enter features (comma-separated)",
                placeholder="e.g., RAM, Storage, Price, Screen Size, Battery Life",
                key="features_input"
            )

            if st.button("Compare Features", key="compare_button"):
                if features_input_str:
                    features_list = [f.strip() for f in features_input_str.split(",") if f.strip()]
                    if features_list:
                        asyncio.run(run_comparison_logic(products_to_compare_refs, features_list))
                    else:
                        st.warning("Please enter at least one feature to compare.")
                else:
                    st.warning("Please enter features to compare.")
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