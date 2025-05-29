# app/screenshot_processor.py
import base64
import logging
import json
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from app.core.config import settings

logger = logging.getLogger(__name__)

class ScreenshotProcessor:
    def __init__(self):
        logger.info(f"Initializing ScreenshotProcessor. OPENAI_SCREENSHOT_KEY configured: {bool(settings.OPENAI_SCREENSHOT_KEY)}")
        if not settings.OPENAI_SCREENSHOT_KEY:
            logger.warning("OPENAI_SCREENSHOT_KEY not configured. Screenshot processing may fail.")
            self.llm = None
        else:
            logger.info("Creating ChatOpenAI instance for screenshot processing with GPT-4o")
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=settings.OPENAI_SCREENSHOT_KEY,
                max_tokens=1500,
                temperature=0.0
            )
        
        self.processed_screenshots: Dict[str, Dict] = {}
        self.total_cost = 0.0
        self.total_tokens = 0
        self.api_calls = 0

    def add_screenshot(self, doc_reference: str, image_base64: str, filename: str) -> bool:
        """Stores a screenshot and performs initial feature detection."""
        logger.info(f"Processing screenshot: {filename} with reference: {doc_reference}")
        
        if not self.llm:
            logger.error("Screenshot LLM not initialized (missing OPENAI_SCREENSHOT_KEY)")
            return False
            
        try:
            initial_features = self._discover_features_in_screenshot(image_base64)
            
            self.processed_screenshots[doc_reference] = {
                "image_data": image_base64,
                "filename": filename,
                "extracted_features": initial_features
            }
            
            logger.info(f"Successfully processed screenshot: {doc_reference} with {len(initial_features)} features discovered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process screenshot {doc_reference} ({filename}): {e}", exc_info=True)
            return False

    def _discover_features_in_screenshot(self, image_base64: str) -> List[str]:
        """Uses GPT-4o to discover what features/specifications are visible in the screenshot."""
        if not self.llm:
            return []
            
        prompt = """
        Analyze this product specification screenshot and identify all the features/specifications that are visible.
        
        IMPORTANT: This may be from an e-commerce website, product listing, or specification sheet in any language.
        Look carefully for ALL technical specifications, even if they're in a foreign language or mixed with other UI elements.
        
        Look for ANY product specifications like:
        - Performance specs (RAM, Memory, Storage, Processor, CPU, GPU, Chipset)
        - Display specs (Screen Size, Display Type, Resolution, Refresh Rate like 60Hz/120Hz/144Hz)
        - Audio/Video specs (Speakers, Sound, HDR, Dolby, etc.)
        - Connectivity (WiFi, Bluetooth, HDMI, USB, Ports)
        - Power specs (Battery, Power consumption, Energy rating)
        - Physical specs (Weight, Dimensions, Size)
        - Software (Operating System, Smart TV OS, Apps)
        - Price/Commercial (Price, Cost, MSRP in any currency)
        - Brand/Model info (Brand, Model, Year, Version)
        - Colors and design options
        - Any other product specifications or features
        
        MULTIPLE PRODUCTS: If you see multiple products in the image, extract features that apply to products generally.
        
        Return ONLY a JSON array of the feature names you can clearly see, like:
        ["Display Size", "Refresh Rate", "Price", "Brand", "Resolution", "Smart TV"]
        
        If you can't clearly identify any features, return an empty array: []
        """

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            )
            
            with get_openai_callback() as cb:
                response = self.llm.invoke([message])
            
            self.total_cost += cb.total_cost
            self.total_tokens += cb.total_tokens
            self.api_calls += 1
            
            logger.info(f"Feature discovery - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
            logger.info(f"📊 CUMULATIVE USAGE - Total Calls: {self.api_calls}, Total Tokens: {self.total_tokens}, Total Cost: ${self.total_cost:.4f}")
            
            if self.total_cost > 1.0:
                logger.warning(f"⚠️  BUDGET WARNING: Total cost ${self.total_cost:.4f} exceeded $1.00 limit!")
            
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            features = json.loads(response_text)
            if isinstance(features, list):
                return [str(f).strip() for f in features if f and str(f).strip()]
            else:
                logger.warning(f"Unexpected response format for feature discovery: {response_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error discovering features in screenshot: {e}", exc_info=True)
            return []

    def extract_feature_from_screenshot(self, doc_reference: str, feature_name: str, product_name: Optional[str] = None) -> str:
        """Extracts a specific feature value from a processed screenshot."""
        if doc_reference not in self.processed_screenshots:
            logger.warning(f"Screenshot reference '{doc_reference}' not found")
            return "Screenshot not processed"
            
        if not self.llm:
            logger.error("Screenshot LLM not initialized")
            return "LLM not available"
            
        screenshot_data = self.processed_screenshots[doc_reference]
        image_base64 = screenshot_data["image_data"]
        
        if product_name:
            prompt = f"""
            Extract the value for "{feature_name}" specifically for the product matching "{product_name}" from this screenshot.
            
            IMPORTANT: This may be an e-commerce website in any language (English, Polish, etc.).
            Look carefully through the entire image for the specified product and feature.
            
            SMART PRODUCT MATCHING: The product name "{product_name}" may be a partial/shorthand reference. 
            Match it intelligently to products in the image:
            - "m2" or "M2" → product with Apple M2 chip
            - "m4" or "M4" → product with Apple M4 chip  
            - "2024" → product from year 2024
            - "2025" → product from year 2025
            - "air" → MacBook Air product
            - "pro" → MacBook Pro product
            - "144hz" or "144Hz" → product with 144Hz refresh rate
            - "120hz" or "120Hz" → product with 120Hz refresh rate  
            - "60hz" or "60Hz" → product with 60Hz refresh rate
            - "4k" or "4K" → product with 4K resolution
            - "hd" or "HD" → product with HD resolution
            - "55" → 55-inch display product
            - "65" → 65-inch display product
            - Brand names like "LG", "Samsung", "Hisense", "Apple", etc.
            - Any partial name should match the most relevant product
            
            Instructions:
            1. Find the product that best matches "{product_name}" (even if it's a partial match)
            2. Extract the "{feature_name}" specification for that specific product
            3. Return ONLY the value (e.g., "16GB", "3700 zł", "13.6 inches", "Apple M2")
            4. If you find a matching product but not this feature, return "Not found"
            5. If you can't find any product matching "{product_name}", return "Product not found"
            6. If the value is unclear or ambiguous, return "Value unclear"
            7. Convert foreign currency or measurements to recognizable format if needed
            
            Feature to extract: {feature_name}
            Product to match: {product_name}
            """
        else:
            prompt = f"""
            Extract the value for "{feature_name}" from this product specification screenshot.
            
            IMPORTANT: This may be an e-commerce website in any language. If there are multiple products, 
            try to extract from the most prominent one, or specify which product the value belongs to.
            
            Instructions:
            1. Look for the specification "{feature_name}" in the image
            2. Return ONLY the value (e.g., "16GB", "3700 zł", "13.6 inches", "Apple M2")
            3. If the feature is not visible, return "Not found"
            4. If the value is unclear or ambiguous, return "Value unclear"
            5. If multiple products show different values, pick the most prominent one or say "Multiple values"
            6. Convert foreign currency or measurements to recognizable format if needed
            
            Feature to extract: {feature_name}
            """

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            )
            
            with get_openai_callback() as cb:
                response = self.llm.invoke([message])
            
            self.total_cost += cb.total_cost
            self.total_tokens += cb.total_tokens
            self.api_calls += 1
            
            logger.info(f"Feature extraction '{feature_name}' (Product: {product_name or 'any'}) - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
            logger.info(f"📊 CUMULATIVE USAGE - Total Calls: {self.api_calls}, Total Tokens: {self.total_tokens}, Total Cost: ${self.total_cost:.4f}")
            
            if self.total_cost > 1.0:
                logger.warning(f"⚠️  BUDGET WARNING: Total cost ${self.total_cost:.4f} exceeded $1.00 limit!")
            
            extracted_value = response.content.strip()
            logger.info(f"Extracted '{feature_name}' for product '{product_name or 'any'}': '{extracted_value}'")
            
            return extracted_value
            
        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}' from screenshot '{doc_reference}': {e}", exc_info=True)
            return "Extraction error"

    def clear_all_screenshots(self) -> None:
        """Clears all processed screenshot data and resets cost tracking."""
        self.processed_screenshots.clear()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.api_calls = 0
        logger.info("All screenshot data and cost tracking cleared")

    def get_available_features(self, doc_reference: str) -> List[str]:
        """Returns the list of features discovered in a specific screenshot."""
        if doc_reference in self.processed_screenshots:
            return self.processed_screenshots[doc_reference].get("extracted_features", [])
        return []

# Example usage for testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with a dummy base64 image (you'd need a real screenshot for actual testing)
    processor = ScreenshotProcessor()
    
    # This would be a real base64 image in practice
    dummy_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    if processor.add_screenshot("test_screenshot", dummy_base64, "test.png"):
        print("Screenshot processed successfully")
        features = processor.get_available_features("test_screenshot")
        print(f"Available features: {features}")
        
        if features:
            value = processor.extract_feature_from_screenshot("test_screenshot", features[0])
            print(f"Extracted value: {value}")
    else:
        print("Screenshot processing failed") 