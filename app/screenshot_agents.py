# app/screenshot_agents.py
import logging
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from app.core.config import settings

logger = logging.getLogger(__name__)

def generate_screenshot_comparison_summary(comparison_data: Dict[str, Dict[str, str]], features: List[str], product_names: Optional[List[str]] = None) -> str:
    logger.info(f"Generating screenshot comparison summary for {len(comparison_data)} screenshots across {len(features)} features")
    
    if not comparison_data:
        return "No screenshot comparison data available."
    
    products_info = []
    for ref_or_product, feature_values in comparison_data.items():
        display_name = ref_or_product if product_names and ref_or_product in product_names else f"Screenshot {ref_or_product}"
        
        features_text = []
        for feature_name in features:
            value = feature_values.get(feature_name, "Not found")
            features_text.append(f"  • {feature_name}: {value}")
        
        product_summary = f"**{display_name}**:\n" + "\n".join(features_text)
        products_info.append(product_summary)
    
    features_str = ", ".join(features)
    products_str = "\n\n".join(products_info)
    
    prompt_text = f"""
You are a product comparison expert analyzing data extracted from product screenshots. Provide a comprehensive comparison summary.

**Features analyzed:** {features_str}

**Product specifications from screenshots:**

{products_str}

**Instructions:**
1. Create a detailed comparison highlighting key differences and similarities
2. Point out standout features or significant price/performance differences
3. If appropriate, suggest which product might be better for different use cases
4. Be objective and fact-based in your analysis
5. Note any features marked as "Not found" or "Extraction error" as limitations
6. If prices are in foreign currency, mention the currency but don't convert
7. Consider that this data was extracted from screenshots, so acknowledge potential visual analysis insights

**Format your response as a well-structured comparison summary suitable for product decision-making.**
"""

    try:
        llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0.3)
        prompt_template = ChatPromptTemplate.from_template(prompt_text)
        
        chain = prompt_template | llm | StrOutputParser()
        
        handler = OpenAICallbackHandler()
        summary = chain.invoke({}, callbacks=[handler])
        
        logger.info(f"Screenshot comparison summary generated - Tokens: {handler.total_tokens}, Cost: ${handler.total_cost:.4f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating screenshot comparison summary: {e}")
        return f"Error generating screenshot comparison summary: {str(e)}"

# Example usage for testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example comparison data that might come from screenshot processing
    test_comparison_data = {
        "screenshot_1": {
            "RAM": "8GB",
            "Storage": "256GB SSD",
            "Price": "$999"
        },
        "screenshot_2": {
            "RAM": "16GB", 
            "Storage": "512GB SSD",
            "Price": "$1299"
        }
    }
    
    test_features = ["RAM", "Storage", "Price"]
    test_product_names = ["MacBook Air", "MacBook Pro"]
    
    summary = generate_screenshot_comparison_summary(
        test_comparison_data, 
        test_features, 
        test_product_names
    )
    
    print("Generated Summary:")
    print(summary) 