from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import get_settings
from app.utils.logger import setup_logger

# Initialize settings and logger
settings = get_settings()
logger = setup_logger(__name__)

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        self.llm = ChatOpenAI(model_name=settings.MODEL_NAME)
        self.vector_store = None
        logger.info("RAGProcessor initialized")

    def process_document(self, text: str) -> None:
        """Process a document and store it in the vector store."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(chunks, self.embeddings)
                logger.info("Created new vector store")
            else:
                self.vector_store.add_texts(chunks)
                logger.info("Updated existing vector store")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def extract_feature(self, feature_name: str) -> Optional[str]:
        """Extract a specific feature value from the processed documents."""
        if not self.vector_store:
            logger.warning("No documents in vector store")
            return None

        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that extracts specific features from product specifications. "
                          "Only use the provided context to answer. If the feature is not found, respond with 'Not found'."),
                ("human", "Based on the following specification sheet context, what is the value for the feature '{feature_name}'?\n\n"
                         "Context:\n{context}\n\n"
                         "Feature: {feature_name}\n"
                         "Value:")
            ])

            # Retrieve relevant chunks
            docs = self.vector_store.similarity_search(feature_name, k=3)
            context = "\n".join(doc.page_content for doc in docs)
            logger.info(f"Retrieved {len(docs)} chunks for feature: {feature_name}")

            # Generate response
            chain = prompt | self.llm
            response = chain.invoke({
                "feature_name": feature_name,
                "context": context
            })

            return response.content
        except Exception as e:
            logger.error(f"Error extracting feature {feature_name}: {str(e)}")
            return None

    def clear_documents(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store = None
        logger.info("Cleared vector store") 