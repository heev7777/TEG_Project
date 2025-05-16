# app/agents.py
from typing import List, Dict, Any
import httpx # Use httpx for async requests
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import settings
from app.core.schemas import ExtractFeaturesParams # To help structure tool call

logger = logging.getLogger(__name__)

class ProductComparisonMAS:
    """
    Multi-Agent System for comparing product features using an LLM and MCP server tool.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0)
        # Asynchronous HTTP client for calling the MCP server
        self.mcp_base_url = f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}"
        
        # Define the tool that the agent can use.
        # The agent will learn to call this tool with the correct parameters.
        # Note: The function `self._invoke_mcp_extract_features` MUST be async
        # if the AgentExecutor is invoked with `ainvoke`.
        # Langchain tools can be async.
        async def _invoke_mcp_extract_features_tool(
            product_references_str: str, # Agent might pass as string
            features_list_str: str # Agent might pass as string
        ) -> Dict[str, Any]:
            """
            Tool to extract features from product specifications via an MCP server.
            Args:
                product_references_str: A comma-separated string of product document references (e.g., 'doc1,doc2').
                features_list_str: A comma-separated string of features to extract (e.g., 'RAM,Price').
            """
            logger.info(f"Agent tool invoked: _invoke_mcp_extract_features_tool")
            logger.info(f"Raw product_references_str: {product_references_str}")
            logger.info(f"Raw features_list_str: {features_list_str}")

            # Robust parsing of arguments from LLM
            try:
                product_refs = [ref.strip() for ref in product_references_str.split(',') if ref.strip()]
                features = [feat.strip() for feat in features_list_str.split(',') if feat.strip()]

                if not product_refs or not features:
                    return {"error": "Missing product references or features list after parsing."}

                logger.info(f"Parsed product_refs: {product_refs}, Parsed features: {features}")

            except Exception as e:
                logger.error(f"Error parsing tool arguments: {e}")
                return {"error": f"Invalid arguments for tool: {e}"}


            mcp_payload = {
                "method": "extract_features_from_specs",
                "params": ExtractFeaturesParams(
                    product_references=product_refs,
                    features_list=features
                ).model_dump() # Use Pydantic model for validation and serialization
            }
            logger.info(f"Sending payload to MCP server /mcp: {mcp_payload}")
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(f"{self.mcp_base_url}/mcp", json=mcp_payload, timeout=30.0)
                    response.raise_for_status() # Raise an exception for HTTP error codes
                    mcp_response_data = response.json()
                    logger.info(f"Received response from MCP server: {mcp_response_data}")

                    if mcp_response_data.get("error"):
                        logger.error(f"MCP server returned an error: {mcp_response_data['error']}")
                        return {"error": f"MCP Error: {mcp_response_data['error'].get('message', 'Unknown MCP error')}"}
                    
                    # The actual result is nested if successful
                    tool_result = mcp_response_data.get("result", {}).get("extract_features_from_specs", {})
                    return tool_result.get("comparison_data", {"error": "Comparison data not found in MCP response"})

                except httpx.RequestError as e:
                    logger.error(f"HTTP Request Error calling MCP server: {e}")
                    return {"error": f"MCP Communication Error: {e}"}
                except Exception as e:
                    logger.error(f"Unexpected error calling MCP tool: {e}", exc_info=True)
                    return {"error": f"Unexpected tool error: {str(e)}"}

        self.tools = [
            Tool(
                name="compare_product_features_via_mcp", # More descriptive name
                # The function MUST be async if using ainvoke
                coroutine=_invoke_mcp_extract_features_tool, # Use coroutine for async func
                description=(
                    "Extracts specified features for a list of product document references. "
                    "Use this tool to get the feature values needed for comparison. "
                    "Input should be two comma-separated strings: "
                    "the first for 'product_references_str' (e.g., 'product_doc_A,product_doc_B'), "
                    "and the second for 'features_list_str' (e.g., 'RAM,Storage,Price')."
                ),
                # If your Langchain version supports it and you need structured input for OpenAI functions:
                # args_schema=ExtractFeaturesToolInputSchema # Define a Pydantic model for tool input
            )
        ]
        
        # System prompt guides the agent on how to behave and use tools
        # This is the "Planner" part of the Planner/Executor
        # It plans by deciding to call the tool. The tool execution is the "Executor" part.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=(
                    "You are a Product Feature Comparison Assistant. Your goal is to help users compare products "
                    "based on features they specify, using information from product specification sheets they provide. "
                    "You have ONE tool: 'compare_product_features_via_mcp'. "
                    "When asked to compare products, you MUST use this tool. "
                    "Provide the product document references (as a comma-separated string for 'product_references_str') "
                    "and the features to compare (as a comma-separated string for 'features_list_str') to the tool. "
                    "After receiving the extracted features from the tool, present them clearly to the user. "
                    "If the tool returns an error or cannot find features, inform the user transparently."
                )
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True, # Good for debugging
            handle_parsing_errors=True # Important for robustness
        )

    async def run_comparison_agent(self, user_query: str, product_doc_refs: List[str], features_to_compare: List[str]) -> Dict[str, Any]:
        """
        Invokes the agent to perform the comparison.
        The user_query helps give context, but the agent should primarily use the
        tool with the provided doc_refs and features_to_compare.
        """
        # Construct a more direct input for the agent if it struggles with natural language parsing
        # for tool arguments. Or ensure the tool description is very clear.
        # For this version, the prompt asks the agent to use the tool based on the user's intent.
        # The user_query will contain this intent.
        # The actual doc_refs and features_to_compare are what the tool needs.
        # We need to make sure the LLM correctly extracts these from the user_query
        # or if the Streamlit app pre-parses them, we need to pass them to the LLM so it can
        # decide to use the tool with those specific arguments.

        # Let's refine the input to the agent to be more structured if Streamlit already parses these:
        agent_input_query = (
            f"User wants to compare products identified by references: '{','.join(product_doc_refs)}' "
            f"on the following features: '{','.join(features_to_compare)}'. "
            f"Please extract these features using the available tool and present the comparison."
        )
        logger.info(f"Invoking agent with input: {agent_input_query}")

        try:
            # Use ainvoke for asynchronous execution
            response = await self.agent_executor.ainvoke({"input": agent_input_query})
            # The 'output' key usually contains the agent's final response to the user.
            # The actual structured data might be in intermediate steps if the agent calls the tool.
            # We are interested in the tool's direct output.
            # For function calling agents, the direct result of the tool might not be in `response['output']`
            # if the LLM summarizes it. We might need to parse `response['intermediate_steps']`
            # or ensure our tool directly returns the data in a way the agent can pass through.

            # If the agent successfully calls the tool and the LLM decides the tool output is the answer,
            # it might be in 'output'. Otherwise, we might need to inspect intermediate_steps or
            # ensure the final prompt to the LLM makes it output the raw table.
            # For now, let's assume the tool's output (comparison_data) needs to be explicitly retrieved
            # if the agent's final text response is just a summary.

            # Let's simplify: the agent's primary job is to call the tool correctly.
            # The tool returns the structured data. The agent's textual response might just confirm this.
            # The Streamlit app will primarily care about the structured data from the tool call.

            # How to get the structured data from the tool call if the agent wraps it in text?
            # One way: the tool itself could be designed to return a string that includes a
            # JSON blob, which the Streamlit app then parses.
            # Or, the agent's prompt can instruct it to return the JSON directly.
            # For now, the tool returns a dict. If the AgentExecutor's final output is text,
            # we might need to adjust.
            # Let's assume the agent executor's 'output' will be the LLM's textual response,
            # and we'll try to get the structured data from an intermediate step if needed,
            # or design the final prompt to the LLM to output the table.

            # For this project, the `_invoke_mcp_extract_features_tool` already returns a dict.
            # If the LLM calls this tool, the result of that call will be available to the LLM.
            # We need to prompt the LLM to present this data clearly.
            # If the agent correctly calls the tool, `response['intermediate_steps']` would show that.
            # `response['output']` will be the LLM's final textual answer.

            # We actually want the raw structured data from the tool.
            # The `compare_product_features_via_mcp` tool returns the comparison_data dictionary.
            # If the agent uses it, this dictionary is what we need for Streamlit.
            # The agent framework should make this available.
            # For now, let's assume the Streamlit app will format it, and the agent's job
            # is to get this data. The LLM might just say "Here's the comparison: [data from tool]".
            # We need the "[data from tool]".

            # Let's reconsider: the `ProductComparisonMAS` should directly return the structured data.
            # The agent's role is to orchestrate the call to the MCP tool. The output of that tool
            # IS the desired structured data.
            
            # If the agent successfully calls the tool, the *result of the tool execution* is what we want.
            # Langchain's AgentExecutor stores tool calls and their outputs in `intermediate_steps`.
            # Example: `response.get('intermediate_steps', [])` would be a list of tuples (AgentAction, Any).
            # We'd look for the output of our 'compare_product_features_via_mcp' tool.

            # A simpler way: The agent's primary output *should be* the structured comparison data.
            # The prompt should guide it to output this directly after tool use.
            # If the final output of `ainvoke` is a string, we parse it. If it's smart, it might be structured.
            # For now, let's just return the whole agent response. Streamlit can parse `output` or look for structured data.
            # Or, the `_invoke_mcp_extract_features_tool` is what really matters, and the agent is just a wrapper to call it.

            # Let's refine the agent's responsibility.
            # The user query to the agent executor should trigger the tool.
            # The tool returns the structured data. We need to extract this specific structured data.
            # The agent's system prompt should emphasize returning this structured data.
            # Alternatively, we could bypass the final LLM formatting step if we just want the tool output.

            # Let's try to get the tool output directly if it was called.
            # This requires inspecting intermediate steps.
            if response.get("intermediate_steps"):
                for action, tool_output in response["intermediate_steps"]:
                    if action.tool == "compare_product_features_via_mcp":
                        if isinstance(tool_output, dict) and not tool_output.get("error"):
                             logger.info(f"Extracted structured data from tool call: {tool_output}")
                             return {"status": "success", "data": tool_output, "text_summary": response.get("output")}
                        else:
                            logger.error(f"Tool call resulted in error or unexpected format: {tool_output}")
                            return {"status": "error", "message": f"Tool error: {tool_output.get('error', 'Unknown')}", "text_summary": response.get("output")}
            
            # If no tool was called, or if the agent is expected to synthesize, this is the LLM's final output.
            # This might happen if the LLM tries to answer without the tool.
            logger.warning("Agent did not seem to use the tool or structured data not found in intermediate steps. Returning final output.")
            return {"status": "success_text_only", "text_summary": response.get("output"), "data": None}

        except Exception as e:
            logger.error(f"Error running comparison agent: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "data": None, "text_summary": None}

# Example Usage (typically called from Streamlit app or a FastAPI backend)
async def main_test_agent():
    import asyncio
    logging.basicConfig(level=logging.INFO)
    mas = ProductComparisonMAS()
    
    # Simulate that documents 'productX_ref' and 'productY_ref' have been processed by MCP server's RAG
    # These references must match what the MCP server's RAG processor knows.
    # In a real flow, Streamlit would first call /mcp/process_document for each uploaded file.
    # For testing, we assume the MCP server is running and has processed these (or will process on the fly if designed that way).
    
    # To make this test self-contained without running MCP server AND Streamlit:
    # 1. You'd need to mock the httpx.AsyncClient call in _invoke_mcp_extract_features_tool
    # 2. Or, ensure the MCP server is running and has some dummy data processed.
    # For now, this test assumes the MCP server is up and can respond, even if with "doc not processed."
    
    print("Testing product comparison agent...")
    product_refs = ["fictional_phone_A_spec_txt", "fictional_tablet_B_spec_pdf"]
    features = ["RAM", "Screen Size", "Price"]
    user_query_for_agent = f"Compare these products: {', '.join(product_refs)} on these features: {', '.join(features)}."

    # Create dummy files for the MCP server to process (if it's not mocked)
    # This part needs to be coordinated with how MCP server's RAGProcessor gets documents.
    # If MCP server is running independently, you would have needed to call its /process_document endpoint.
    # For this test, let's assume you manually start the MCP server and have it process some files
    # or the RAGProcessor in MCP server is designed to load them on the fly from a known path
    # based on doc_reference.

    # Let's simplify the test by directly invoking the tool method IF the agent is too complex initially.
    # But the goal is to test the agent.
    
    result = await mas.run_comparison_agent(user_query_for_agent, product_refs, features)
    print("\nAgent Comparison Result:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    # To run this test:
    # 1. Make sure your .env has OPENAI_API_KEY.
    # 2. Make sure your MCP_SERVER is running (python app/mcp_server.py)
    #    AND that its RAGProcessor has some way to know about "fictional_phone_A_spec_txt"
    #    (e.g., by calling its /mcp/process_document endpoint beforehand with actual file paths)
    #    or by modifying the RAGProcessor in mcp_server.py to load dummy data if these refs are seen.
    
    # For a truly isolated agent test, you'd mock the `httpx.AsyncClient.post` call.
    # For an integrated test, run the MCP server first.
    
    # This test will likely show the tool trying to be called. Success depends on MCP server.
    import asyncio
    asyncio.run(main_test_agent())