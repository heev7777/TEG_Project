import streamlit as st
from app.core.config import get_settings
from app.services.document_parser import DocumentParser
from app.utils.logger import setup_logger
import asyncio
from app.agents import ProductComparisonAgent

# Initialize settings and logger
settings = get_settings()
logger = setup_logger(__name__)

async def process_comparison(files, features):
    """Process the comparison request asynchronously."""
    agent = ProductComparisonAgent()
    try:
        # Process uploaded files
        product_ids = []
        for file in files:
            file_type = file.name.split('.')[-1].lower()
            text = DocumentParser.parse_document(file, file_type)
            product_ids.append(file.name)
            # TODO: Store processed text in vector store
        
        # Compare features
        result = await agent.compare_products(product_ids, features)
        return result
    finally:
        await agent.close()

def main():
    st.title("Product Feature Comparison Assistant")
    st.write("Upload product specification sheets and compare features across products.")

    # File upload section
    st.header("Upload Specification Sheets")
    uploaded_files = st.file_uploader(
        "Choose specification files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Feature input section
        st.header("Features to Compare")
        features_input = st.text_input(
            "Enter features to compare (comma-separated)",
            placeholder="e.g., RAM, Storage, Price"
        )
        
        if features_input:
            features = [f.strip() for f in features_input.split(",")]
            
            if st.button("Compare Features"):
                with st.spinner("Processing comparison..."):
                    try:
                        # Run async comparison
                        result = asyncio.run(process_comparison(uploaded_files, features))
                        
                        # Display results
                        st.header("Comparison Results")
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.json(result)
                    except Exception as e:
                        logger.error(f"Error during comparison: {str(e)}")
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 