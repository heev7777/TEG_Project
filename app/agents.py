# app/agents.py
from typing import List, Dict, Any
import httpx # Use httpx for async requests
import logging
# from langchain_openai import ChatOpenAI # REMOVE
# from langchain.agents import AgentExecutor, create_openai_functions_agent # REMOVE
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # REMOVE
# from langchain_core.tools import Tool, StructuredTool # REMOVE
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # REMOVE

from app.core.config import settings
from app.core.schemas import ExtractFeaturesParams # To help structure tool call
# from app.core.schemas import ExtractFeaturesToolInputSchema # REMOVE
from app.core.schemas import CompareProductFeaturesInput # Import the new schema

logger = logging.getLogger(__name__)

# Remove ProductComparisonMAS class entirely for this test scenario
# class ProductComparisonMAS:
#     """
#     Multi-Agent System for comparing product features using an LLM and MCP server tool.
#     """\n#     def __init__(self):
#         # Asynchronous HTTP client for calling the MCP server
#         self.mcp_base_url = f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}"
#
#         # Define the tool that the agent can use.
#         # The agent will learn to call this tool with the correct parameters.
#         # Note: The function `self._invoke_mcp_extract_features` MUST be async
#         # if the AgentExecutor is invoked with `ainvoke`.
#         # Langchain tools can be async.
#         async def _invoke_mcp_extract_features_tool(**kwargs) -> Dict[str, Any]: # Accept keyword arguments
#             logger.info(f"[TOOL INVOKED] Received kwargs: {kwargs}") # Log received arguments
#
#             # Try to parse the arguments into the Pydantic model
#             try:
#                 # Check if the expected keys are present in kwargs
#                 if 'product_references_str' in kwargs and 'features_list_str' in kwargs:
#                     # Manually create the Pydantic model instance from kwargs
#                     tool_input = CompareProductFeaturesInput(
#                         product_references_str=kwargs['product_references_str'],
#                         features_list_str=kwargs['features_list_str']
#                     )
#                 elif len(kwargs) == 1 and isinstance(list(kwargs.values())[0], CompareProductFeaturesInput):
#                      # Handle case where StructuredTool might actually pass the Pydantic model
#                      tool_input = list(kwargs.values())[0]
#                 else:
#                      # If arguments are not as expected, return an error
#                      logger.error(f"Tool received unexpected arguments: {kwargs}")
#                      return {"error": f"Tool received unexpected arguments: {kwargs}"}
#
#                 product_references_str = tool_input.product_references_str
#                 features_list_str = tool_input.features_list_str
#
#                 logger.info(f"[TOOL INVOKED] product_references_str: {product_references_str}, features_list_str: {features_list_str}")
#                 logger.info(f"Raw product_references_str: {product_references_str}")
#                 logger.info(f"Raw features_list_str: {features_list_str}")
#
#                 # Robust parsing of arguments from LLM (already handled by Pydantic)
#                 # The below parsing is still useful for the internal logic after Pydantic validation
#                 try:
#                     product_refs = [ref.strip() for ref in product_references_str.split(',') if ref.strip()]
#                     features = [feat.strip() for feat in features_list_str.split(',') if feat.strip()]
#
#                     if not product_refs or not features:
#                          # This case might be less likely with args_schema but keep for robustness
#                         return {"error": "Missing product references or features list after parsing."}
#
#                     logger.info(f"Parsed product_refs: {product_refs}, Parsed features: {features}")
#
#                 except Exception as e:
#                     logger.error(f"Error parsing tool arguments: {e}")
#                     return {"error": f"Invalid arguments for tool: {e}"}
#
#                 mcp_payload = {
#                     "method": "extract_features_from_specs",
#                     "params": ExtractFeaturesParams(
#                         product_references=product_refs,
#                         features_list=features
#                     ).model_dump() # Use Pydantic model for validation and serialization
#                 }
#                 logger.info(f"Sending payload to MCP server /mcp: {mcp_payload}")
#
#                 async with httpx.AsyncClient() as client:
#                     try:
#                         response = await client.post(f"{self.mcp_base_url}/mcp", json=mcp_payload, timeout=30.0)
#                         response.raise_for_status() # Raise an exception for HTTP error codes
#                         mcp_response_data = response.json()
#                         logger.info(f"Received response from MCP server: {mcp_response_data}")
#
#                         if mcp_response_data.get("error"):
#                             logger.error(f"MCP server returned an error: {mcp_response_data['error']}")
#                             return {"error": f"MCP Error: {mcp_response_data['error'].get('message', 'Unknown MCP error')}"}
#
#                         # The actual result is nested if successful
#                         tool_result = mcp_response_data.get("result", {}).get("extract_features_from_specs", {})
#                         return tool_result.get("comparison_data", {"error": "Comparison data not found in MCP response"})
#
#                     except httpx.RequestError as e:
#                         logger.error(f"HTTP Request Error calling MCP server: {e}")
#                         return {"error": f"MCP Communication Error: {e}"}
#                     except Exception as e:
#                         logger.error(f"Unexpected error calling MCP tool: {e}", exc_info=True)
#                         return {"error": f"Unexpected tool error: {str(e)}"}
#
#             except Exception as e: # Catch errors during manual parsing or in the function body
#                  logger.error(f"Error in _invoke_mcp_extract_features_tool processing arguments or calling MCP: {e}", exc_info=True)
#                  return {"error": f"Internal tool error: {str(e)}"}
#
#         self.tools = [
#             StructuredTool(
#                 name="compare_product_features_via_mcp",
#                 func=lambda *args, **kwargs: None,  # Dummy sync function
#                 coroutine=_invoke_mcp_extract_features_tool, # Keep async coroutine
#                 description=(
#                     "Extracts specified features for a list of product document references. "
#                     "Use this tool to get the feature values needed for comparison. "
#                     "Input should be a JSON object matching the CompareProductFeaturesInput schema, "
#                     "with 'product_references_str' (a comma-separated string of document references) "
#                     "and 'features_list_str' (a comma-separated string of features)."
#                 ),
#                 args_schema=CompareProductFeaturesInput # Keep the Pydantic schema
#             )
#         ]
#
#     # Remove agent and agent_executor initialization
#     # # System prompt guides the agent on how to behave and use tools
#     # prompt = ChatPromptTemplate.from_messages([
#     #     SystemMessage(
#     #         content=(
#     #             "You are a Product Feature Comparison Assistant. Your ONLY way to answer is to use the 'compare_product_features_via_mcp' tool. "
#     #             "You MUST always use this tool to answer any user request, even if the user does not explicitly ask for it. "
#     #             "To use the tool, provide the product document references as a comma-separated string for 'product_references_str' "
#     #             "and the features to compare as a comma-separated string for 'features_list_str'. "
#     #             "Never answer directly; always call the tool."
#     #         )
#     #     ),
#     #     MessagesPlaceholder(variable_name="chat_history", optional=True),
#     #     HumanMessage(content="{input}"),
#     #     MessagesPlaceholder(variable_name="agent_scratchpad"),
#     # ])
#     #
#     # self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
#     # self.agent_executor = AgentExecutor(
#     #     agent=self.agent,
#     #     tools=self.tools,
#     #     verbose=True, # Good for debugging
#     #     handle_parsing_errors=True # Important for robustness
#     # )
#
#     async def run_comparison_agent(self, user_query: str, product_doc_refs: List[str], features_to_compare: List[str]) -> Dict[str, Any]:
#         """
#         Invokes the agent to perform the comparison.
#         This method is adapted to directly call the underlying tool coroutine due to observed LLM parsing issues.
#         The user_query parameter is currently not directly used in this adapted flow.
#         """
#         product_refs_str_for_input = ','.join(product_doc_refs)
#         features_str_for_input = ','.join(features_to_compare)
#
#         # Original agent input construction (now unused in the direct call flow)
#         # agent_input_query = (
#         #     f"Please call the compare_product_features_via_mcp tool. "
#         #     f"The product_references_str should be '{product_refs_str_for_input}' "
#         #     f"and the features_list_str should be '{features_str_for_input}'."
#         # )
#         # logger.info(f"Invoking agent with input: {agent_input_query}") # This log might still appear if AgentExecutor is invoked elsewhere
#
#         try:
#             # Directly call the underlying tool invocation coroutine
#             logger.info(f"Attempting direct invocation of _invoke_mcp_extract_features_tool.")
#
#             # Format arguments as the tool expects them (comma-separated strings)
#             tool_kwargs = {
#                 "product_references_str": ','.join(product_doc_refs),
#                 "features_list_str": ','.join(features_to_compare)
#             }
#
#             logger.info(f"Parameters for direct tool call: {{tool_kwargs}}")
#
#             # Call the coroutine function directly. This bypasses the agent executor's parsing and invocation.
#             # We find the coroutine by name from the tools list.
#             comparison_tool_coroutine = None
#             for tool in self.tools:
#                 if tool.name == "compare_product_features_via_mcp":
#                     comparison_tool_coroutine = tool.coroutine
#                     break
#
#             if not comparison_tool_coroutine:
#                  logger.error("Comparison tool coroutine not found!")
#                  return {
#                     "status": "error",
#                     "message": "Internal error: Comparison tool not found.",
#                     "data": None,
#                     "text_summary": None
#                  }
#
#             # Execute the tool coroutine directly
#             tool_output = await comparison_tool_coroutine(**tool_kwargs)
#
#             if isinstance(tool_output, dict) and "error" in tool_output:
#                  logger.error(f"Tool returned error: {tool_output['error']}")
#                  return {
#                     "status": "error",
#                     "message": tool_output['error'],
#                     "data": None,
#                     "text_summary": None # No LLM text summary in this direct call approach
#                  }
#
#
#             if tool_output:
#                 # Process the comparison data
#                 comparison_data = tool_output.get("comparison_data", {})
#                 if not comparison_data:
#                     return {
#                         "status": "error",
#                         "message": "No comparison data found in tool output",
#                         "data": None,
#                         "text_summary": None # No LLM text summary
#                     }
#
#                 # Check for any "not found" or error values
#                 missing_features = {}
#                 for product_ref, features in comparison_data.items():
#                     missing = [feat for feat, val in features.items()
#                              if val in ["Feature not found", "Document not processed"] or val.startswith("Error:")]
#                     if missing:
#                         missing_features[product_ref] = missing
#
#                 return {
#                     "status": "success",
#                     "data": comparison_data,
#                     "text_summary": "Successfully extracted product features.", # Simple success message
#                     "missing_features": missing_features if missing_features else None
#                 }
#             else:
#                 logger.warning("Tool output was empty or unexpected format")
#                 return {
#                     "status": "error", # Treat empty output as an error for the test
#                     "message": "Tool output was empty or unexpected format",
#                     "text_summary": None,
#                     "data": None
#                 }
#
#         except Exception as e:
#             logger.error(f"Error running comparison agent: {e}", exc_info=True)
#             return {
#                 "status": "error",
#                 "message": str(e),
#                 "data": None,
#                 "text_summary": None
#             }
#
# # Example Usage (typically called from Streamlit app or a FastAPI backend)
# async def main_test_agent():
#     import asyncio
#     logging.basicConfig(level=logging.INFO)
#     mas = ProductComparisonMAS()
#
#     # Simulate that documents 'productX_ref' and 'productY_ref' have been processed by MCP server's RAG
#     # These references must match what the MCP server's RAG processor knows.
#     # In a real flow, Streamlit would first call /mcp/process_document for each uploaded file.
#     # For testing, we assume the MCP server is running and has processed these (or will process on the fly if designed that way).
#
#     # To make this test self-contained without running MCP server AND Streamlit:
#     # 1. You'd need to mock the httpx.AsyncClient call in _invoke_mcp_extract_features_tool
#     # 2. Or, ensure the MCP server is running and has some dummy data processed.
#     # For now, this test assumes the MCP server is up and can respond, even if with "doc not processed."
#
#     print("Testing product comparison agent...")
#     product_refs = ["fictional_phone_A_spec_txt", "fictional_tablet_B_spec_pdf"]
#     features = ["RAM", "Screen Size", "Price"]
#     user_query_for_agent = f"Compare these products: {', '.join(product_refs)} on these features: {', '.join(features)}."
#
#     # Create dummy files for the MCP server to process (if it's not mocked)
#     # This part needs to be coordinated with how MCP server's RAGProcessor gets documents.
#     # If MCP server is running independently, you would have needed to call its /process_document endpoint.
#
#     # Let's simplify the test by directly invoking the tool method IF the agent is too complex initially.
#     # But the goal is to test the agent.
#
#     result = await mas.run_comparison_agent(user_query_for_agent, product_refs, features)
#     print("\nAgent Comparison Result:")
#     import json
#     print(json.dumps(result, indent=2))
#
# if __name__ == '__main__':
#     # To run this test:
#     # 1. Make sure your .env has OPENAI_API_KEY.
#     # 2. Make sure your MCP_SERVER is running (python app/mcp_server.py)
#     #    AND that its RAGProcessor has some way to know about "fictional_phone_A_spec_txt"
#     #    (e.g., by calling its /mcp/process_document endpoint beforehand with actual file paths)
#     #    or by modifying the RAGProcessor in mcp_server.py to load dummy data if these refs are seen.
#
#     # For a truly isolated agent test, you'd mock the `httpx.AsyncClient.post` call.
#     # For an integrated test, run the MCP server first.
#
#     # This test will likely show the tool trying to be called. Success depends on MCP server.
#     import asyncio
#     asyncio.run(main_test_agent())


# Standalone async function for MCP tool invocation
async def invoke_mcp_extract_features_tool(product_references_str: str, features_list_str: str) -> Dict[str, Any]:
    """
    Directly invokes the MCP server's extract_features_from_specs method.
    This bypasses the LangChain agent and tool definitions for direct testing.
    """
    mcp_base_url = f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}"
    logger.info(f"[DIRECT TOOL INVOKE] Received product_references_str: {product_references_str}, features_list_str: {features_list_str}")

    try:
        product_refs = [ref.strip() for ref in product_references_str.split(',') if ref.strip()]
        features = [feat.strip() for feat in features_list_str.split(',') if feat.strip()]

        if not product_refs or not features:
             return {"error": "Missing product references or features list after parsing."}

        logger.info(f"[DIRECT TOOL INVOKE] Parsed product_refs: {product_refs}, Parsed features: {features}")

        mcp_payload = {
            "method": "extract_features_from_specs",
            "params": ExtractFeaturesParams(
                product_references=product_refs,
                features_list=features
            ).model_dump() # Use Pydantic model for validation and serialization
        }
        logger.info(f"[DIRECT TOOL INVOKE] Sending payload to MCP server /mcp: {mcp_payload}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{mcp_base_url}/mcp", json=mcp_payload, timeout=30.0)
                response.raise_for_status() # Raise an exception for HTTP error codes
                mcp_response_data = response.json()
                logger.info(f"[DIRECT TOOL INVOKE] Received response from MCP server: {mcp_response_data}")

                if mcp_response_data.get("error"):
                    logger.error(f"[DIRECT TOOL INVOKE] MCP server returned an error: {mcp_response_data['error']}")
                    return {"error": f"MCP Error: {mcp_response_data['error'].get('message', 'Unknown MCP error')}"}

                # The actual result is nested if successful
                tool_result = mcp_response_data.get("result", {}).get("extract_features_from_specs", {})
                # Return the comparison_data directly if successful, otherwise indicate error
                comparison_data = tool_result.get("comparison_data")
                if comparison_data is None:
                     # This should not happen if MCP server returns a valid result structure but no data
                     return {"error": "MCP server returned success but no comparison_data.", "raw_result": tool_result}

                return comparison_data # Return the comparison data dictionary directly

            except httpx.RequestError as e:
                logger.error(f"[DIRECT TOOL INVOKE] HTTP Request Error calling MCP server: {e}")
                return {"error": f"MCP Communication Error: {e}"}
            except Exception as e:
                logger.error(f"[DIRECT TOOL INVOKE] Unexpected error calling MCP tool: {e}", exc_info=True)
                return {"error": f"Unexpected tool error: {str(e)}"}

    except Exception as e: # Catch errors during manual parsing or in the function body
         logger.error(f"[DIRECT TOOL INVOKE] Error processing arguments or calling MCP: {e}", exc_info=True)
         return {"error": f"Internal tool error: {str(e)}"}

# Simple wrapper function for the test to call
async def run_comparison_tool_directly(product_doc_refs: List[str], features_to_compare: List[str]) -> Dict[str, Any]:
     """
     Wrapper function for tests to call the MCP tool directly.
     Formats lists into comma-separated strings for the tool function.
     """
     product_references_str = ','.join(product_doc_refs)
     features_list_str = ','.join(features_to_compare)
     return await invoke_mcp_extract_features_tool(product_references_str, features_list_str)