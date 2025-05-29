# app/agents.py
import logging
import httpx
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from app.core.config import settings
from app.core.schemas import MCPToolCallRequest, MCPToolCallResponse

logger = logging.getLogger(__name__)

async def run_comparison_tool_directly(product_doc_refs: List[str], features_to_compare: List[str]) -> Dict:
    logger.info(f"Running comparison tool directly for {len(product_doc_refs)} products and {len(features_to_compare)} features")
    
    async with httpx.AsyncClient() as client:
        try:
            request_payload = MCPToolCallRequest(
                method="extract_features_from_specs",
                params={
                    "product_references": product_doc_refs,
                    "features_list": features_to_compare
                }
            )
            
            response = await client.post(
                f"{settings.MCP_SERVER_URL}/mcp",
                json=request_payload.model_dump(),
                timeout=60.0
            )
            response.raise_for_status()
            
            mcp_response_data = response.json()
            logger.info(f"MCP response received: {mcp_response_data}")
            
            if "error" in mcp_response_data and mcp_response_data["error"]:
                logger.error(f"MCP tool error: {mcp_response_data['error']}")
                return {"error": mcp_response_data["error"]["message"]}
            
            result_data = mcp_response_data.get("result", {})
            extract_features_result = result_data.get("extract_features_from_specs", {})
            comparison_data = extract_features_result.get("comparison_data", {})
            
            if not comparison_data:
                logger.warning("No comparison data returned from MCP tool")
                return {"error": "No comparison data returned"}
            
            logger.info(f"Successfully extracted comparison data: {comparison_data}")
            return {"comparison_data": comparison_data}
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in MCP tool call: {e}")
            return {"error": f"HTTP error: {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.error(f"Request error in MCP tool call: {e}")
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in MCP tool call: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

async def generate_comparison_summary(comparison_data: Dict[str, Dict[str, str]], product_name_map: Dict[str, str], features: List[str]) -> str:
    logger.info(f"Generating comparison summary for {len(comparison_data)} products across {len(features)} features")
    
    if not comparison_data:
        return "No comparison data available."
    
    products_info = []
    for doc_ref, feature_values in comparison_data.items():
        product_name = product_name_map.get(doc_ref, f"Product {doc_ref}")
        
        features_text = []
        for feature_name in features:
            value = feature_values.get(feature_name, "Not found")
            features_text.append(f"  • {feature_name}: {value}")
        
        product_summary = f"**{product_name}**:\n" + "\n".join(features_text)
        products_info.append(product_summary)
    
    features_str = ", ".join(features)
    products_str = "\n\n".join(products_info)
    
    prompt_text = f"""
You are a product comparison expert. Based on the extracted product specifications, provide a comprehensive comparison summary.

**Features being compared:** {features_str}

**Product specifications:**

{products_str}

**Instructions:**
1. Create a detailed comparison highlighting key differences and similarities
2. Point out standout features or significant differences
3. If appropriate, suggest which product might be better for different use cases
4. Be objective and fact-based in your analysis
5. Note any features marked as "Not found" as limitations

**Format your response as a well-structured comparison summary.**
"""

    try:
        llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0.3)
        prompt_template = ChatPromptTemplate.from_template(prompt_text)
        
        chain = prompt_template | llm | StrOutputParser()
        
        handler = OpenAICallbackHandler()
        summary = await chain.ainvoke({}, callbacks=[handler])
        
        logger.info(f"Comparison summary generated - Tokens: {handler.total_tokens}, Cost: ${handler.total_cost:.4f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating comparison summary: {e}")
        return f"Error generating summary: {str(e)}"
