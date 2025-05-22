# app/agents.py
import httpx
import json # Not strictly needed here if using Pydantic for payload
import logging

from app.core.config import settings # settings will now have MCP_SERVER_URL
from app.core.schemas import ExtractFeaturesParams, MCPToolCallRequest, MCPToolCallResponse

logger = logging.getLogger(__name__)

async def run_comparison_tool_directly(
    product_doc_refs: list[str],
    features_to_compare: list[str]
) -> dict:
    """
    Directly calls the MCP server's extract_features_from_specs tool endpoint.
    """
    logger.info("Attempting direct call to MCP extract_features_from_specs tool.")
    
    # Use the MCP_SERVER_URL from settings
    if not settings.MCP_SERVER_URL:
        logger.error("MCP_SERVER_URL is not configured in settings.")
        return {"error": "MCP_SERVER_URL is not configured."}
        
    mcp_url = f"{settings.MCP_SERVER_URL}/mcp" # The /mcp path for the tool router

    tool_call_payload = MCPToolCallRequest(
        # tool_name="ProductComparisonMAS", # Not needed for the /mcp endpoint which routes by method
        method="extract_features_from_specs",
        params=ExtractFeaturesParams(
            product_references=product_doc_refs,
            features_list=features_to_compare
        ).model_dump()
    )

    logger.info(f"Sending tool call request to MCP: {mcp_url}")
    logger.info(f"Request payload: {tool_call_payload.model_dump_json(indent=2)}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                mcp_url, # Use the fully constructed URL
                json=tool_call_payload.model_dump(), # Pydantic handles JSON serialization
                timeout=60.0 # Added a timeout
            )
            logger.info(f"Received response from MCP. Status Code: {response.status_code}")
            logger.info(f"Response body: {response.text[:500]}...") # Log only first 500 chars
            response.raise_for_status() 

            mcp_response_data = response.json()
            # Validate with Pydantic model
            mcp_response = MCPToolCallResponse(**mcp_response_data)


            if mcp_response.result:
                # The actual result is nested under the method name within the 'result' dict
                # e.g. result: {"extract_features_from_specs": {"comparison_data": {...}}}
                method_result = mcp_response.result.get("extract_features_from_specs")
                if method_result and "comparison_data" in method_result:
                    comparison_data = method_result["comparison_data"]
                    logger.info("Direct tool call successful. Comparison data extracted.")
                    return comparison_data 
                else:
                    logger.warning(f"Direct tool call successful, but 'extract_features_from_specs' or 'comparison_data' key missing in result: {mcp_response.result}")
                    return {"error": "Unexpected result structure from MCP tool."}
            elif mcp_response.error:
                error_message = mcp_response.error.get('message', 'Unknown error from MCP tool')
                logger.error(f"Direct tool call returned error: {error_message}")
                return {"error": error_message}
            else:
                logger.warning(f"Direct tool call returned unexpected response format (no result or error): {mcp_response_data}")
                return {"error": "Unexpected response format from MCP server (no result or error field)."}

    except httpx.RequestError as e:
        logger.error(f"HTTP request to MCP server failed: {e}", exc_info=True)
        return {"error": f"HTTP request failed: {str(e)}"}
    except httpx.HTTPStatusError as e: # Specifically handle HTTP errors
        logger.error(f"HTTP error from MCP server: {e.response.status_code} - {e.response.text[:200]}", exc_info=True)
        return {"error": f"HTTP error from MCP: {e.response.status_code}"}
    except json.JSONDecodeError as e: # If response is not valid JSON
        logger.error(f"Failed to decode JSON response from MCP server: {e}", exc_info=True)
        return {"error": f"Invalid JSON response from MCP: {str(e)}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during direct MCP tool call: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

