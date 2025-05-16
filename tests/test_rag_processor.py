import pytest
from app.rag_processor import RAGProcessor

def test_rag_processor_initialization():
    """Test RAGProcessor initialization."""
    processor = RAGProcessor()
    assert processor.vector_store is None
    assert processor.llm is not None
    assert processor.embeddings is not None

def test_process_document():
    """Test document processing."""
    processor = RAGProcessor()
    test_text = "This is a test document with some content."
    processor.process_document(test_text)
    assert processor.vector_store is not None

def test_extract_feature():
    """Test feature extraction."""
    processor = RAGProcessor()
    test_text = "Product specifications: RAM: 8GB, Storage: 256GB"
    processor.process_document(test_text)
    
    # Test existing feature
    ram_value = processor.extract_feature("RAM")
    assert ram_value is not None
    
    # Test non-existing feature
    non_existing = processor.extract_feature("NonExistingFeature")
    assert non_existing is None or non_existing == "Not found"

def test_clear_documents():
    """Test clearing documents from vector store."""
    processor = RAGProcessor()
    test_text = "Test document"
    processor.process_document(test_text)
    assert processor.vector_store is not None
    
    processor.clear_documents()
    assert processor.vector_store is None 