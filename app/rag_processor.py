import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.llm = ChatOpenAI(
            model_name=settings.LLM_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.0
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        self.document_vector_stores: Dict[str, FAISS] = {}
        
        logger.info(f"RAGProcessor initialized with model: {settings.LLM_MODEL_NAME}, embeddings: {settings.EMBEDDING_MODEL_NAME}")

    def add_document(self, document_reference: str, file_path: str) -> bool:
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            file_extension = path_obj.suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return False
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/sections from {file_path}")
            
            if not documents:
                logger.warning(f"No content loaded from {file_path}")
                return False
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks for {document_reference}")
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            self.document_vector_stores[document_reference] = vector_store
            logger.info(f"Document {document_reference} successfully processed and stored")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {document_reference}: {e}", exc_info=True)
            return False

    def extract_feature_from_doc(self, document_reference: str, feature_name: str) -> Optional[str]:
        if document_reference not in self.document_vector_stores:
            logger.warning(f"Document {document_reference} not found in vector stores")
            return None
        
        vector_store = self.document_vector_stores[document_reference]
        
        try:
            query = f"What is the {feature_name}? {feature_name} specification value details"
            docs = vector_store.similarity_search(query, k=settings.RAG_TOP_K_RESULTS)
            
            if not docs:
                logger.info(f"No relevant documents found for feature '{feature_name}' in {document_reference}")
                return "Not found"
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt_template = ChatPromptTemplate.from_template("""
Based on the following context from a product specification document, extract the value for "{feature_name}".

Context:
{context}

Instructions:
1. Look for the specific feature "{feature_name}" in the context
2. Return ONLY the value (e.g., "16GB", "Intel i7", "5.5 inches")
3. If the feature is not found, return "Not found"
4. If the value is unclear or ambiguous, return "Unclear"
5. Do not include units unless they are part of the standard specification

Feature: {feature_name}
""")
            
            chain = prompt_template | self.llm | StrOutputParser()
            
            handler = OpenAICallbackHandler()
            result = chain.invoke({
                "feature_name": feature_name,
                "context": context
            }, callbacks=[handler])
            
            logger.info(f"Feature extraction for '{feature_name}' - Tokens: {handler.total_tokens}, Cost: ${handler.total_cost:.4f}")
            
            result = result.strip()
            logger.info(f"Extracted '{feature_name}' from {document_reference}: '{result}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}' from {document_reference}: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def clear_all_documents(self) -> None:
        self.document_vector_stores.clear()
        logger.info("All documents cleared from RAG processor")

    def get_processed_documents(self) -> Dict[str, Any]:
        return {
            doc_ref: {
                "chunks_count": vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else "Unknown"
            }
            for doc_ref, vector_store in self.document_vector_stores.items()
        }

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    processor = RAGProcessor()

    # Create a dummy spec file
    dummy_spec_path = settings.UPLOAD_DIR / "dummy_spec.txt"
    with open(dummy_spec_path, "w") as f:
        f.write("Product Alpha\nRelease Date: 2024\nRAM: 8GB LPDDR5\nStorage: 256GB NVMe SSD\nScreen Size: 6.1 inches OLED\nPrice: $799")

    doc_ref = "product_alpha_spec"
    if processor.add_document(doc_ref, str(dummy_spec_path)):
        logger.info(f"--- Extracting RAM ---")
        ram = processor.extract_feature_from_doc(doc_ref, "RAM")
        logger.info(f"Extracted RAM: {ram}") # Expected: 8GB LPDDR5 or similar

        logger.info(f"--- Extracting Price ---")
        price = processor.extract_feature_from_doc(doc_ref, "Price")
        logger.info(f"Extracted Price: {price}") # Expected: $799

        logger.info(f"--- Extracting Color (Not Present) ---")
        color = processor.extract_feature_from_doc(doc_ref, "Color")
        logger.info(f"Extracted Color: {color}") # Expected: Not found

        logger.info(f"--- Extracting Release Date ---")
        release = processor.extract_feature_from_doc(doc_ref, "Release Date")
        logger.info(f"Extracted Release Date: {release}") # Expected: 2024

    else:
        logger.error(f"Could not process dummy spec file: {dummy_spec_path}")

    processor.clear_all_documents()