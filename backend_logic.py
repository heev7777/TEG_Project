from typing import List, Dict
import httpx
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductComparisonAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")
        self.mcp_client = httpx.AsyncClient(
            base_url=f"http://{os.getenv('MCP_SERVER_HOST', 'localhost')}:{os.getenv('MCP_SERVER_PORT', '8000')}"
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="extract_features",
                func=self._extract_features,
                description="Extract specified features from product documents"
            )
        ]
        
        # Create agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that compares product features. "
                      "Use the available tools to extract and compare features from product specifications."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    async def _extract_features(self, product_ids: List[str], features: List[str]) -> Dict:
        """Tool function to extract features from products using the MCP server."""
        try:
            response = await self.mcp_client.post(
                "/mcp/extract_features",
                json={
                    "product_document_ids": product_ids,
                    "features_list": features
                }
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            return {"error": str(e)}

    async def compare_products(self, product_ids: List[str], features: List[str]) -> Dict:
        """Compare specified features across products."""
        try:
            # Use the agent to handle the comparison
            result = await self.agent_executor.ainvoke({
                "input": f"Compare products {', '.join(product_ids)} on features: {', '.join(features)}"
            })
            return result
        except Exception as e:
            return {"error": str(e)}

    async def close(self):
        """Close the HTTP client."""
        await self.mcp_client.aclose() 