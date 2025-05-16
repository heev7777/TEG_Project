from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.llm = ChatOpenAI(model_name="gpt-4")
        self.vector_store = None

    def process_document(self, text: str) -> None:
        """Process a document and store it in the vector store."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)

    def extract_feature(self, feature_name: str) -> Optional[str]:
        """Extract a specific feature value from the processed documents."""
        if not self.vector_store:
            return None

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

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({
            "feature_name": feature_name,
            "context": context
        })

        return response.content

    def clear_documents(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store = None 