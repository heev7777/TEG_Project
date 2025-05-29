# app/main.py
import streamlit as st
import logging
import httpx 
import asyncio 
import uuid 
import base64
from typing import List, Dict, Set

from app.core.config import settings
from app.agents import run_comparison_tool_directly, generate_comparison_summary 
from app.core.schemas import (
    ProcessDocumentRequest, ProcessDocumentResponse, ExtractFeaturesParams,
    ProcessScreenshotRequest, ProcessScreenshotResponse,
    ExtractFeaturesFromScreenshotParams, ExtractFeaturesFromScreenshotResult
)
from app.screenshot_agents import generate_screenshot_comparison_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

if 'processed_docs_info' not in st.session_state:
    st.session_state.processed_docs_info = {} 

if 'processed_screenshots_info' not in st.session_state:
    st.session_state.processed_screenshots_info = {}

if 'upload_queue' not in st.session_state:
    st.session_state.upload_queue = [] 

if 'screenshot_upload_queue' not in st.session_state:
    st.session_state.screenshot_upload_queue = []

if 'processing_upload' not in st.session_state:
    st.session_state.processing_upload = False

if 'processing_screenshot_upload' not in st.session_state:
    st.session_state.processing_screenshot_upload = False

mcp_doc_proc_client = httpx.AsyncClient(base_url=f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}")

async def process_uploaded_screenshot_on_mcp(uploaded_file_obj, unique_doc_ref: str) -> ProcessScreenshotResponse | None:
    logger.info(f"Processing screenshot: {uploaded_file_obj.name}, ref: {unique_doc_ref}")
    
    try:
        file_bytes = uploaded_file_obj.getbuffer()
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
        request_payload = ProcessScreenshotRequest(
            doc_reference=unique_doc_ref,
            image_base64=image_base64,
            image_filename=uploaded_file_obj.name
        )
        
        response = await mcp_doc_proc_client.post("/mcp/process_screenshot", json=request_payload.model_dump(), timeout=120.0)
        logger.info(f"MCP response status: {response.status_code}")
        
        response.raise_for_status()
        
        response_data = ProcessScreenshotResponse(**response.json())
        logger.info(f"MCP processed screenshot: {response_data.status}")
        
        if response_data.status == "processed":
            logger.info(f"Screenshot {unique_doc_ref} processed successfully. Features: {len(response_data.extracted_features)}")
            return response_data
        else:
            logger.error(f"MCP failed to process screenshot {unique_doc_ref}: {response_data.status}")
            st.error(f"MCP Server failed to process screenshot '{uploaded_file_obj.name}': {response_data.message}")
            return response_data
            
    except httpx.HTTPStatusError as e_http:
        st.error(f"Error contacting MCP server to process screenshot '{uploaded_file_obj.name}': {e_http.response.status_code}")
        logger.error(f"HTTP error processing screenshot {unique_doc_ref}: {e_http}", exc_info=True)
    except httpx.RequestError as e_req:
        st.error(f"Network error contacting MCP server for screenshot '{uploaded_file_obj.name}': {e_req}")
        logger.error(f"Request error processing screenshot {unique_doc_ref}: {e_req}", exc_info=True)
    except Exception as e_gen:
        st.error(f"An unexpected error occurred while processing screenshot '{uploaded_file_obj.name}': {e_gen}")
        logger.error(f"General error processing screenshot {unique_doc_ref}: {e_gen}", exc_info=True)
    
    return None

async def process_uploaded_file_on_mcp(uploaded_file_obj, unique_doc_ref: str) -> ProcessDocumentResponse | None:
    logger.info(f"Processing file: {uploaded_file_obj.name}, ref: {unique_doc_ref}")
    temp_file_path = None 
    try:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_file_path = settings.UPLOAD_DIR / f"{unique_doc_ref}_{uploaded_file_obj.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        logger.info(f"Saved temp file: {temp_file_path}")

        request_payload = ProcessDocumentRequest(
            doc_reference=unique_doc_ref,
            file_path=str(temp_file_path) 
        )
        
        response = await mcp_doc_proc_client.post("/mcp/process_document", json=request_payload.model_dump(), timeout=60.0)
        logger.info(f"MCP response status: {response.status_code}")

        response.raise_for_status() 

        response_data = ProcessDocumentResponse(**response.json())
        logger.info(f"MCP processed document: {response_data.status}")
        
        if response_data.status == "processed":
            logger.info(f"Document {unique_doc_ref} processed successfully. Features: {len(response_data.extracted_features)}")
            return response_data 
        else:
            logger.error(f"MCP failed to process {unique_doc_ref}: {response_data.status}")
            st.error(f"MCP Server failed to process '{uploaded_file_obj.name}': {response_data.message}")
            if temp_file_path and temp_file_path.exists(): 
                try:
                    temp_file_path.unlink()
                    logger.info(f"Deleted temp file after MCP failure: {temp_file_path}")
                except OSError as e_unlink:
                    logger.error(f"Error deleting temp file {temp_file_path}: {e_unlink}")
            return response_data 
            
    except httpx.HTTPStatusError as e_http:
        st.error(f"Error contacting MCP server to process file '{uploaded_file_obj.name}': {e_http.response.status_code}")
        logger.error(f"HTTP error processing file {unique_doc_ref}: {e_http}", exc_info=True)
    except httpx.RequestError as e_req:
        st.error(f"Network error contacting MCP server for '{uploaded_file_obj.name}': {e_req}")
        logger.error(f"Request error processing file {unique_doc_ref}: {e_req}", exc_info=True)
    except Exception as e_gen:
        st.error(f"An unexpected error occurred while processing file '{uploaded_file_obj.name}': {e_gen}")
        logger.error(f"General error processing file {unique_doc_ref}: {e_gen}", exc_info=True)
    
    return None 

async def clear_mcp_documents():
    logger.info("Clearing all documents and screenshots from MCP server")
    try:
        await mcp_doc_proc_client.post("/mcp/clear_documents", timeout=10.0)
        await mcp_doc_proc_client.post("/mcp/clear_screenshots", timeout=10.0)
        
        st.session_state.processed_docs_info.clear() 
        st.session_state.processed_screenshots_info.clear()
        st.session_state.upload_queue.clear() 
        st.session_state.screenshot_upload_queue.clear()
        st.session_state.processing_upload = False 
        st.session_state.processing_screenshot_upload = False
        
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
                        logger.error(f"Error deleting temp file {f_path}: {e}")
                        error_count +=1
            logger.info(f"Deleted {deleted_count} temp files. {error_count} errors.")
        st.success("Cleared all processed documents and screenshots from server and local temp storage.")
        logger.info("MCP documents, screenshots, and local temp files cleared successfully.")
    except Exception as e:
        st.error(f"Failed to clear documents/screenshots on MCP server: {e}")
        logger.error(f"Error clearing MCP documents/screenshots: {e}", exc_info=True)

async def run_screenshot_comparison_logic(screenshot_refs: list, features: list, screenshot_names_map: dict, product_names: list = None):
    logger.info(f"Running screenshot comparison for {len(screenshot_refs)} screenshots with {len(features)} features")
    
    if not screenshot_refs or not features:
        st.warning("Please select screenshots and features to compare.")
        return

    with st.spinner("Extracting features from screenshots using AI vision..."):
        try:
            if len(screenshot_refs) == 1 and product_names and len([p for p in product_names if p]) > 1:
                single_screenshot_ref = screenshot_refs[0]
                comparison_data = {}
                
                for i, product_name in enumerate(product_names):
                    if product_name and product_name.strip():
                        virtual_ref = f"{single_screenshot_ref}_product_{i+1}"
                        
                        request_payload = {
                            "method": "extract_features_from_screenshots",
                            "params": {
                                "screenshot_references": [single_screenshot_ref],
                                "features_list": features,
                                "product_names": [product_name]
                            }
                        }
                        
                        response = await mcp_doc_proc_client.post("/mcp", json=request_payload, timeout=180.0)
                        response.raise_for_status()
                        
                        result = response.json()
                        logger.info(f"MCP response for product '{product_name}': success")
                        
                        if result.get("error"):
                            st.error(f"Screenshot extraction failed for {product_name}: {result['error']}")
                            return
                            
                        product_data = result.get("result", {}).get("extract_features_from_screenshots", {}).get("comparison_data", {})
                        
                        if single_screenshot_ref in product_data:
                            comparison_data[product_name] = product_data[single_screenshot_ref]
                        
                display_names_map = {name: name for name in comparison_data.keys()}
                
            else:
                request_payload = {
                    "method": "extract_features_from_screenshots",
                    "params": {
                        "screenshot_references": screenshot_refs,
                        "features_list": features,
                        "product_names": product_names
                    }
                }
                
                response = await mcp_doc_proc_client.post("/mcp", json=request_payload, timeout=180.0)
                response.raise_for_status()
                
                result = response.json()
                logger.info("MCP screenshot tool response received")
                
                if result.get("error"):
                    st.error(f"Screenshot comparison failed: {result['error']}")
                    return
                    
                comparison_data = result.get("result", {}).get("extract_features_from_screenshots", {}).get("comparison_data", {})
                display_names_map = screenshot_names_map
            
            if comparison_data:
                st.header("Screenshot Comparison Results")
                
                table_data = {"Feature": features}
                for ref_or_product, feature_values in comparison_data.items():
                    display_name = display_names_map.get(ref_or_product, ref_or_product)
                    table_data[display_name] = [feature_values.get(feat, "N/A") for feat in features]
                
                if len(table_data) > 1:
                    st.dataframe(table_data)
                    logger.info("Displayed screenshot comparison dataframe")
                    
                    with st.spinner("Generating AI comparison summary from screenshot data..."):
                        summary_product_names = list(display_names_map.values()) if display_names_map else None
                        text_summary = generate_screenshot_comparison_summary(
                            comparison_data, 
                            features,
                            summary_product_names
                        )
                        st.subheader("AI Analysis Summary")
                        st.markdown(text_summary)
                        logger.info("Displayed AI-generated screenshot summary")
                else:
                    st.warning("Could not prepare screenshot data for display.")
            else:
                st.warning("No screenshot comparison data was returned.")
                
        except Exception as e:
            st.error(f"Error during screenshot comparison: {e}")
            logger.error(f"Screenshot comparison error: {e}", exc_info=True)

async def run_comparison_logic(products_to_compare_refs: list, features: list, selected_product_names_map: dict):
    logger.info(f"Running comparison for {len(products_to_compare_refs)} products with {len(features)} features")
    if not products_to_compare_refs or not features:
        st.warning("Please select products and enter features to compare.")
        logger.warning("Missing products or features for comparison")
        return

    comparison_data = None

    with st.spinner("Comparing features via MCP tool..."):
        tool_call_result = await run_comparison_tool_directly(
            product_doc_refs=products_to_compare_refs, 
            features_to_compare=features 
        )
        logger.info("MCP tool call completed")

        if "error" in tool_call_result:
            st.error(f"Comparison failed: {tool_call_result['error']}")
            logger.error(f"Comparison tool error: {tool_call_result['error']}")
            return 
        elif "comparison_data" in tool_call_result:
            comparison_data = tool_call_result["comparison_data"]
        else:
            st.warning("Comparison tool returned an unexpected result structure.")
            logger.warning(f"Unexpected tool result structure: {tool_call_result}")
            return

    if comparison_data:
        st.header("Comparison Results")
        if not comparison_data: 
            st.warning("No comparison data was returned by the tool.")
            logger.warning("Tool returned empty comparison data")
            return

        header_features = features 
        table_data = {"Feature": header_features}

        for product_ref_from_data, feature_values in comparison_data.items():
            original_filename = selected_product_names_map.get(product_ref_from_data, "Unknown Product")
            table_data[original_filename] = [feature_values.get(feat, "N/A") for feat in header_features]
        
        if len(table_data) > 1 : 
            st.dataframe(table_data)
            logger.info("Displayed comparison dataframe")

            with st.spinner("Generating comparison summary..."):
                text_summary = await generate_comparison_summary(
                    comparison_data, 
                    selected_product_names_map,
                    features
                )
                st.subheader("Summary")
                st.markdown(text_summary)
                logger.info("Displayed LLM-generated summary")
        else:
            st.warning("Could not prepare data for display. Check product references.")
            logger.warning("No product data found in comparison_data to display")
    else:
        st.warning("Comparison tool returned no comparison data after processing.")
        logger.warning("Tool returned no comparison_data field")


def main_ui():
    st.set_page_config(page_title=settings.PROJECT_NAME, layout="wide")
    st.title(settings.PROJECT_NAME)
    st.markdown(settings.DESCRIPTION)

    st.sidebar.header("Controls")
    if st.sidebar.button("Clear All Processed Documents & Screenshots", key="clear_docs_button"):
        asyncio.run(clear_mcp_documents())
        st.rerun() 

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📄 Upload Text Specification Files")
        uploaded_file_objects = st.file_uploader(
            "Choose up to 3 specification files (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="text_file_uploader"
        )
    
    with col2:
        st.header("📷 Upload Screenshot Images")
        uploaded_screenshot_objects = st.file_uploader(
            "Choose up to 3 screenshot images (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="screenshot_file_uploader"
        )

    if uploaded_file_objects:
        current_processed_names = set(st.session_state.processed_docs_info.keys())
        queue_file_names = set(item.name for item in st.session_state.upload_queue)

        for uploaded_file in uploaded_file_objects:
            if uploaded_file.name not in current_processed_names and uploaded_file.name not in queue_file_names:
                st.session_state.upload_queue.append(uploaded_file)
                logger.info(f"Added '{uploaded_file.name}' to upload queue")

    if uploaded_screenshot_objects:
        current_processed_screenshot_names = set(st.session_state.processed_screenshots_info.keys())
        queue_screenshot_names = set(item.name for item in st.session_state.screenshot_upload_queue)

        for uploaded_screenshot in uploaded_screenshot_objects:
            if uploaded_screenshot.name not in current_processed_screenshot_names and uploaded_screenshot.name not in queue_screenshot_names:
                st.session_state.screenshot_upload_queue.append(uploaded_screenshot)
                logger.info(f"Added '{uploaded_screenshot.name}' to screenshot upload queue")

    if st.session_state.upload_queue and not st.session_state.processing_upload:
        st.session_state.processing_upload = True 
        file_to_process = st.session_state.upload_queue[0] 
        unique_doc_ref = f"doc_{uuid.uuid4().hex[:8]}"
        logger.info(f"Processing '{file_to_process.name}' with ref '{unique_doc_ref}'")

        with st.spinner(f"Processing text file '{file_to_process.name}' on server..."):
            response_data = asyncio.run(process_uploaded_file_on_mcp(file_to_process, unique_doc_ref))

            if response_data and response_data.status == "processed":
                st.session_state.processed_docs_info[file_to_process.name] = {
                    'unique_doc_ref': unique_doc_ref,
                    'temp_file_path': str(settings.UPLOAD_DIR / f"{unique_doc_ref}_{file_to_process.name}"), 
                    'extracted_features': response_data.extracted_features or [] 
                }
                logger.info(f"Stored doc info for '{file_to_process.name}': {unique_doc_ref}")
                st.success(f"Text file '{file_to_process.name}' processed and ready for comparison.")
            elif response_data: 
                logger.warning(f"MCP processing for '{file_to_process.name}' was not successful: {response_data.status}")
            else: 
                logger.error(f"Failed to process '{file_to_process.name}' on MCP server")
        
        st.session_state.upload_queue.pop(0)
        st.session_state.processing_upload = False
        logger.info(f"Finished processing '{file_to_process.name}'. Rerunning Streamlit.")
        st.rerun()

    if st.session_state.screenshot_upload_queue and not st.session_state.processing_screenshot_upload:
        st.session_state.processing_screenshot_upload = True
        screenshot_to_process = st.session_state.screenshot_upload_queue[0]
        unique_screenshot_ref = f"screenshot_{uuid.uuid4().hex[:8]}"
        logger.info(f"Processing '{screenshot_to_process.name}' with ref '{unique_screenshot_ref}'")

        with st.spinner(f"Processing screenshot '{screenshot_to_process.name}' with AI vision..."):
            response_data = asyncio.run(process_uploaded_screenshot_on_mcp(screenshot_to_process, unique_screenshot_ref))

            if response_data and response_data.status == "processed":
                st.session_state.processed_screenshots_info[screenshot_to_process.name] = {
                    'unique_doc_ref': unique_screenshot_ref,
                    'image_filename': screenshot_to_process.name,
                    'extracted_features': response_data.extracted_features or []
                }
                logger.info(f"Stored screenshot info for '{screenshot_to_process.name}': {unique_screenshot_ref}")
                st.success(f"Screenshot '{screenshot_to_process.name}' processed and ready for comparison. Features discovered: {len(response_data.extracted_features or [])}")
            elif response_data:
                logger.warning(f"MCP processing for screenshot '{screenshot_to_process.name}' was not successful: {response_data.status}")
            else:
                logger.error(f"Failed to process screenshot '{screenshot_to_process.name}' on MCP server")

        st.session_state.screenshot_upload_queue.pop(0)
        st.session_state.processing_screenshot_upload = False
        logger.info(f"Finished processing screenshot '{screenshot_to_process.name}'. Rerunning Streamlit.")
        st.rerun()

    logger.info(f"Current state - Text docs: {len(st.session_state.processed_docs_info)}, Screenshots: {len(st.session_state.processed_screenshots_info)}")
    
    all_available_features_from_processed: Set[str] = set()
    for doc_name, doc_info in st.session_state.processed_docs_info.items(): 
        features_for_doc = doc_info.get('extracted_features') 
        if features_for_doc: 
            all_available_features_from_processed.update(features_for_doc)
    
    all_available_features_from_screenshots: Set[str] = set()
    for screenshot_name, screenshot_info in st.session_state.processed_screenshots_info.items():
        features_for_screenshot = screenshot_info.get('extracted_features')
        if features_for_screenshot:
            all_available_features_from_screenshots.update(features_for_screenshot)
    
    sorted_available_features = sorted(list(all_available_features_from_processed))
    sorted_available_screenshot_features = sorted(list(all_available_features_from_screenshots))
    logger.info(f"Available text features: {len(sorted_available_features)}, screenshot features: {len(sorted_available_screenshot_features)}")

    has_text_docs = bool(st.session_state.processed_docs_info)
    has_screenshots = bool(st.session_state.processed_screenshots_info)

    if has_text_docs or has_screenshots:
        tab1, tab2 = st.tabs(["📄 Text Document Comparison", "📷 Screenshot Comparison"])
        
        with tab1:
            if has_text_docs:
                st.subheader("Processed Text Documents Ready for Comparison:")
                
                display_name_to_ref = {name: doc_info['unique_doc_ref'] for name, doc_info in st.session_state.processed_docs_info.items()}
                processed_file_display_names = list(display_name_to_ref.keys())

                selected_display_names = st.multiselect(
                    "Select 2 or 3 text documents to compare:",
                    options=processed_file_display_names,
                    max_selections=3, 
                    key="text_product_selector"
                )
                
                products_to_compare_refs = [display_name_to_ref[name] for name in selected_display_names if name in display_name_to_ref]
                
                selected_product_name_details_map = {
                    display_name_to_ref[name]: name 
                    for name in selected_display_names 
                    if name in display_name_to_ref
                }

                if products_to_compare_refs and len(products_to_compare_refs) >= 2:
                    st.header("Specify Features to Compare (Text Documents)")

                    select_all_features_default = True if sorted_available_features else False 
                    if 'select_all_text_features_checkbox_value' not in st.session_state:
                         st.session_state.select_all_text_features_checkbox_value = select_all_features_default

                    select_all_features = st.checkbox(
                        "Select All Available Text Features",
                        key="select_all_text_features_checkbox",
                        value=st.session_state.select_all_text_features_checkbox_value
                    )
                    st.session_state.select_all_text_features_checkbox_value = select_all_features

                    current_multiselect_options = sorted_available_features
                    
                    if select_all_features:
                        default_multiselect_value = current_multiselect_options
                    else:
                        previous_selection = st.session_state.get('text_features_selector_value', [])
                        default_multiselect_value = [f for f in previous_selection if f in current_multiselect_options]

                    selected_features = st.multiselect(
                        "Select features to compare in text documents:",
                        options=current_multiselect_options, 
                        default=default_multiselect_value,
                        key="text_features_selector" 
                    )
                    st.session_state.text_features_selector_value = selected_features 

                    if st.button("Compare Text Document Features", key="compare_text_button"):
                        if selected_features:
                            asyncio.run(run_comparison_logic(products_to_compare_refs, selected_features, selected_product_name_details_map))
                        else:
                            st.warning("Please select at least one feature to compare.")
                elif selected_display_names and len(selected_display_names) < 2:
                    st.warning("Please select at least 2 text documents to compare.")
            else:
                st.info("Upload text specification files to enable text document comparison.")
        
        with tab2:
            if has_screenshots:
                st.subheader("Processed Screenshots Ready for Comparison:")
                
                screenshot_display_name_to_ref = {name: screenshot_info['unique_doc_ref'] for name, screenshot_info in st.session_state.processed_screenshots_info.items()}
                processed_screenshot_display_names = list(screenshot_display_name_to_ref.keys())

                selected_screenshot_display_names = st.multiselect(
                    "Select 2 or 3 screenshots to compare:",
                    options=processed_screenshot_display_names,
                    max_selections=3,
                    key="screenshot_product_selector"
                )
                
                screenshots_to_compare_refs = [screenshot_display_name_to_ref[name] for name in selected_screenshot_display_names if name in screenshot_display_name_to_ref]
                
                selected_screenshot_name_details_map = {
                    screenshot_display_name_to_ref[name]: name 
                    for name in selected_screenshot_display_names 
                    if name in screenshot_display_name_to_ref
                }

                if screenshots_to_compare_refs and len(screenshots_to_compare_refs) >= 1:
                    st.header("Specify Features to Compare (Screenshots)")

                    if len(screenshots_to_compare_refs) == 1:
                        st.info("🔍 **Single Screenshot Analysis**: Perfect for e-commerce pages with multiple products in one image!")
                    else:
                        st.info("📊 **Multi-Screenshot Comparison**: Comparing products across different images.")

                    with st.expander("🏷️ Product Names (Optional - for better AI extraction)"):
                        if len(screenshots_to_compare_refs) == 1:
                            st.info("For single screenshots with multiple products, specify the different product names you want to compare within the image.")
                            st.success("💡 **Smart Product Names**: Use shortcuts! Type 'M2', 'M4', '144Hz', '120Hz', '55', '65', brand names, etc.")
                        else:
                            st.info("If your screenshots contain multiple products or you want to focus on specific product names, specify them here.")
                            st.success("💡 **Smart Product Names**: Use shortcuts! Type 'M2', 'M4', '144Hz', '120Hz', '55', '65', brand names, etc.")
                        
                        product_names = []
                        
                        if len(screenshots_to_compare_refs) == 1:
                            st.write("**Products to extract from this screenshot:**")
                            num_products = st.number_input("How many products are in this screenshot?", min_value=1, max_value=5, value=2, key="num_products_single")
                            for i in range(num_products):
                                product_name = st.text_input(
                                    f"Product {i+1} name:",
                                    key=f"single_product_name_{i}",
                                    placeholder="e.g., 144Hz, 120Hz, 55, 65, LG, Samsung, M2, M4"
                                )
                                product_names.append(product_name if product_name.strip() else None)
                        else:
                            for i, screenshot_name in enumerate(selected_screenshot_display_names):
                                product_name = st.text_input(
                                    f"Product name in '{screenshot_name}' (optional):",
                                    key=f"product_name_{i}",
                                    placeholder="e.g., 144Hz, 120Hz, 55, LG, Samsung, iPhone 16, M4"
                                )
                                product_names.append(product_name if product_name.strip() else None)

                    select_all_screenshot_features_default = True if sorted_available_screenshot_features else False
                    if 'select_all_screenshot_features_checkbox_value' not in st.session_state:
                        st.session_state.select_all_screenshot_features_checkbox_value = select_all_screenshot_features_default

                    select_all_screenshot_features = st.checkbox(
                        "Select All Available Screenshot Features",
                        key="select_all_screenshot_features_checkbox",
                        value=st.session_state.select_all_screenshot_features_checkbox_value
                    )
                    st.session_state.select_all_screenshot_features_checkbox_value = select_all_screenshot_features

                    current_screenshot_options = sorted_available_screenshot_features
                    
                    if select_all_screenshot_features:
                        default_screenshot_value = current_screenshot_options
                    else:
                        previous_screenshot_selection = st.session_state.get('screenshot_features_selector_value', [])
                        default_screenshot_value = [f for f in previous_screenshot_selection if f in current_screenshot_options]

                    selected_screenshot_features = st.multiselect(
                        "Select features to extract from screenshots:",
                        options=current_screenshot_options,
                        default=default_screenshot_value,
                        key="screenshot_features_selector"
                    )
                    st.session_state.screenshot_features_selector_value = selected_screenshot_features

                    manual_features = st.text_input(
                        "Or enter additional features to extract (comma-separated):",
                        key="manual_screenshot_features",
                        placeholder="e.g., Price, RAM, Storage, Battery"
                    )
                    
                    if manual_features.strip():
                        manual_feature_list = [f.strip() for f in manual_features.split(',') if f.strip()]
                        selected_screenshot_features.extend(manual_feature_list)
                        selected_screenshot_features = list(dict.fromkeys(selected_screenshot_features))

                    if st.button("Compare Screenshot Features", key="compare_screenshot_button"):
                        if selected_screenshot_features:
                            filtered_product_names = [name for name in product_names if name] if any(product_names) else None
                            asyncio.run(run_screenshot_comparison_logic(
                                screenshots_to_compare_refs, 
                                selected_screenshot_features, 
                                selected_screenshot_name_details_map,
                                filtered_product_names
                            ))
                        else:
                            st.warning("Please select at least one feature to extract from screenshots.")
                elif selected_screenshot_display_names and len(selected_screenshot_display_names) < 1:
                    st.warning("Please select at least 1 screenshot to analyze.")
            else:
                st.info("Upload screenshot images to enable screenshot comparison.")
    else:
        st.info("Upload specification files or screenshots to begin comparison.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Version: {settings.VERSION}")

if __name__ == "__main__":
    main_ui()
