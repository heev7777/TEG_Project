import asyncio
import httpx
import logging
import os
from pathlib import Path
import pytest
import pytest_asyncio
from typing import Dict, Any

# Update import to use the new direct tool function
from app.agents import run_comparison_tool_directly
from app.core.config import settings
from app.core.schemas import ProcessDocumentRequest

# Configure logging
# logging.basicConfig(level=logging.INFO) # Pytest handles logging config, or use pytest.ini
logger = logging.getLogger(__name__)

# Test data
TEST_PRODUCTS = {
    "phone_a": {
        "content": """Product Name: Fictional Alpha Phone
RAM: 8GB DDR5
Storage: 256GB UFS 3.1
Screen: 6.5 inch OLED, 120Hz
Battery: 4500 mAh
Price: $699
Color: Midnight Blue""",
        "filename": "phone_a_spec.txt"
    },
    "phone_b": {
        "content": """Product Name: Fictional Beta Phone
RAM: 12GB DDR5
Storage: 512GB UFS 3.1
Screen: 6.7 inch AMOLED, 144Hz
Battery: 5000 mAh
Price: $899
Color: Space Gray""",
        "filename": "phone_b_spec.txt"
    }
}

# Pytest fixture for an async HTTP client, scoped per test function or module
@pytest_asyncio.fixture(scope="module")
async def mcp_client():
    # Ensure the base_url matches your running MCP server
    async with httpx.AsyncClient(base_url=f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}", timeout=30.0) as client:
        yield client

@pytest_asyncio.fixture(scope="module", autouse=True)
async def manage_test_files():
    """Creates test files before all tests in the module and cleans up after."""
    logger.info("Setting up test files for the module...")
    created_files_paths = {}
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for product_id, data in TEST_PRODUCTS.items():
        file_path = upload_dir / data["filename"]
        with open(file_path, "w") as f:
            f.write(data["content"])
        created_files_paths[product_id] = str(file_path)
    logger.info(f"Created test files: {list(created_files_paths.values())}")
    
    yield created_files_paths # This provides the paths to the tests if they need them directly

    logger.info("Cleaning up test files for the module...")
    for file_path_str in created_files_paths.values():
        try:
            os.remove(file_path_str)
            logger.info(f"Cleaned up: {file_path_str}")
        except Exception as e:
            logger.warning(f"Failed to remove test file {file_path_str}: {e}")


# Helper function to process documents for tests, using the mcp_client fixture
async def process_documents_via_mcp(mcp_client_fixture: httpx.AsyncClient, test_files_dict: Dict[str, str]) -> Dict[str, str]:
    doc_refs = {}
    for product_key, file_path_str in test_files_dict.items():
        doc_ref = f"test_{product_key}_ref" # Consistent reference
        request_data = ProcessDocumentRequest(
            doc_reference=doc_ref,
            file_path=file_path_str # MCP server will read this path
        )
        logger.info(f"Processing document for {product_key} with ref {doc_ref} via MCP: {file_path_str}")
        response = await mcp_client_fixture.post("/mcp/process_document", json=request_data.model_dump())
        response.raise_for_status() # Will raise an error for non-2xx responses
        assert response.status_code == 200, f"Failed to process document {product_key} via MCP"
        assert response.json()["status"] == "processed"
        doc_refs[product_key] = doc_ref
    return doc_refs


# Main test function, now a pytest test
@pytest.mark.asyncio
async def test_end_to_end_product_comparison(mcp_client, manage_test_files):
    # Use mcp_client and manage_test_files directly
    """
    End-to-end test for product comparison using direct tool invocation.
    Requires the MCP server to be running (orchestrated by run_tests.py).
    `manage_test_files` fixture provides the paths to the created test files.
    `mcp_client` fixture provides an HTTP client.
    """
    logger.info("Starting end-to-end product comparison test...")
    
    # 1. Process documents using the helper function and the mcp_client fixture
    # `manage_test_files` already creates the files. We just need their paths from it.
    # The keys in manage_test_files are 'phone_a', 'phone_b'
    test_file_paths_dict = manage_test_files
    
    processed_doc_refs_map = await process_documents_via_mcp(mcp_client, test_file_paths_dict)
    logger.info(f"Processed documents for test with references: {processed_doc_refs_map}")
    # processed_doc_refs_map will be like {'phone_a': 'test_phone_a_ref', 'phone_b': 'test_phone_b_ref'}

    # Test case 1: Compare specific features
    product_refs_for_agent = [processed_doc_refs_map["phone_a"], processed_doc_refs_map["phone_b"]]
    features_to_compare = ["RAM", "Storage", "Price", "Battery"]
    
    logger.info(f"Test Case 1: Comparing {product_refs_for_agent} on features {features_to_compare}")
    
    # Use the new direct tool invocation function
    result = await run_comparison_tool_directly(
        product_doc_refs=product_refs_for_agent,
        features_to_compare=features_to_compare
    )
    
    logger.info(f"Test Case 1 Tool Result: {result}")
    
    # Check for errors in the result
    assert "error" not in result, f"Test Case 1 failed: Tool returned error - {result.get('error')}"
    
    # Verify the comparison data
    comparison_data = result
    assert processed_doc_refs_map["phone_a"] in comparison_data
    assert "RAM" in comparison_data[processed_doc_refs_map["phone_a"]]
    assert "8GB DDR5" in comparison_data[processed_doc_refs_map["phone_a"]]["RAM"]
    assert "Price" in comparison_data[processed_doc_refs_map["phone_a"]]
    assert "$699" in comparison_data[processed_doc_refs_map["phone_a"]]["Price"]

    assert processed_doc_refs_map["phone_b"] in comparison_data
    assert "RAM" in comparison_data[processed_doc_refs_map["phone_b"]]
    assert "12GB DDR5" in comparison_data[processed_doc_refs_map["phone_b"]]["RAM"]
    
    logger.info("Test Case 1 assertions passed.")

    # Test case 2: Non-existent feature
    features_non_existent = ["RAM", "Touchscreen Type"] # Assuming "Touchscreen Type" is not in specs
    logger.info(f"Test Case 2: Comparing {product_refs_for_agent} on features {features_non_existent}")
    
    # Use the new direct tool invocation function
    result_non_existent = await run_comparison_tool_directly(
        product_doc_refs=product_refs_for_agent,
        features_to_compare=features_non_existent
    )
    
    logger.info(f"Test Case 2 Tool Result: {result_non_existent}")
    assert "error" not in result_non_existent
    
    # Check that "Touchscreen Type" is handled (e.g., "Not found" or similar)
    data_non_existent = result_non_existent
    for ref in product_refs_for_agent:
        assert "Touchscreen Type" in data_non_existent.get(ref, {}), f"Feature 'Touchscreen Type' missing for {ref}"
        assert data_non_existent.get(ref, {}).get("Touchscreen Type") is not None and \
               data_non_existent.get(ref, {}).get("Touchscreen Type").strip() != "", f"Feature 'Touchscreen Type' for {ref} is unexpectedly empty"

    logger.info("Test Case 2 assertions passed.")

    # (You can add more test cases here, like testing with an invalid doc_ref sent to the agent)

    # Clear documents on MCP server after tests for this module
    logger.info("Clearing documents on MCP server after tests...")
    clear_response = await mcp_client.post("/mcp/clear_documents")
    assert clear_response.status_code == 204 # No content for success
    logger.info("Documents cleared on MCP server.")


# Remove the old if __name__ == "__main__": block. Pytest will find and run test_end_to_end_product_comparison.