# app/rag_processor.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import logging
from app.core.config import settings
from app.services.document_parser import DocumentParser

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        self.llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0)
        self.document_vector_stores: Dict[str, FAISS] = {}
        self.document_texts: Dict[str, str] = {}

    def add_document(self, doc_reference: str, file_path: str) -> bool:
        """
        Processes a single document (PDF or TXT) by its file path, chunks it, embeds it, and stores it in a FAISS vector store associated with the doc_reference.
        """
        logger.info(f"Processing document: {file_path} with reference: {doc_reference}")
        try:
            loaded_docs: List[Document] = DocumentParser.load_document(file_path)
            if not loaded_docs:
                logger.error(f"No content loaded from {file_path}")
                return False
            doc_content = loaded_docs[0].page_content
            self.document_texts[doc_reference] = doc_content
            chunks = self.text_splitter.split_text(doc_content)
            doc_chunks = [Document(page_content=chunk, metadata={"source": Path(file_path).name, "doc_ref": doc_reference}) for chunk in chunks]
            if not doc_chunks:
                logger.warning(f"No chunks created for document: {doc_reference}")
                return False
            vector_store = FAISS.from_documents(doc_chunks, self.embeddings)
            self.document_vector_stores[doc_reference] = vector_store
            logger.info(f"Successfully processed and vectorized document: {doc_reference} from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to process document {doc_reference} from {file_path}: {e}")
            return False

    def extract_feature_from_doc(self, doc_reference: str, feature_name: str) -> str:
        """
        Extracts a specific feature value from a specific processed document identified by doc_reference.
        """
        if doc_reference not in self.document_vector_stores:
            logger.warning(f"Document reference '{doc_reference}' not found in vector stores.")
            return "Document not processed"
        vector_store = self.document_vector_stores[doc_reference]
        retriever = vector_store.as_retriever(search_kwargs={"k": settings.RAG_TOP_K_RESULTS})
        prompt_template_str = """
        You are an expert assistant specialized in extracting specific information from product specification sheets.
        Your task is to find the value for the feature: '{feature_name}'.
        Use ONLY the provided "Context from Specification Sheet" to answer.
        If the feature '{feature_name}' is explicitly mentioned with a clear value in the context, provide that value.
        If the feature is mentioned but its value is ambiguous or not clearly stated, respond with "Value unclear".
        If the feature '{feature_name}' is not mentioned at all in the context, respond with "Not found".
        Do not infer or make assumptions beyond the provided text.
        Context from Specification Sheet:
        {context}
        Feature to Extract: {feature_name}
        Extracted Value:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        rag_chain = (
            RunnableParallel(
                {"context": retriever, "feature_name": RunnablePassthrough()}
            )
            | (lambda x: {"context": "\n---\n".join(doc.page_content for doc in x["context"]), "feature_name": x["feature_name"]})
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info(f"Extracting feature '{feature_name}' from document '{doc_reference}'")
        try:
            response_content = rag_chain.invoke(feature_name)
            logger.info(f"LLM response for '{feature_name}' in '{doc_reference}': {response_content}")
            if "Not found" in response_content: return "Not found"
            if "Value unclear" in response_content: return "Value unclear"
            return response_content.strip()
        except Exception as e:
            logger.error(f"Error during feature extraction for '{feature_name}' in '{doc_reference}': {e}")
            return "Extraction error"

    def clear_all_documents(self) -> None:
        """Clears all loaded document vector stores and texts."""
        self.document_vector_stores.clear()
        self.document_texts.clear()
        logger.info("All document data cleared from RAG processor.")

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