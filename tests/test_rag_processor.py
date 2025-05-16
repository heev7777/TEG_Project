import pytest
from pathlib import Path
from app.rag_processor import RAGProcessor
from app.core.config import settings

def test_rag_processor_initialization():
    """Test RAGProcessor initialization."""
    processor = RAGProcessor()
    assert processor.embeddings is not None
    assert processor.llm is not None
    assert processor.text_splitter is not None
    assert processor.document_vector_stores == {}
    assert processor.document_texts == {}

def test_process_document():
    """Test document processing."""
    processor = RAGProcessor()
    test_doc_ref = "test_doc"
    test_file_path = settings.UPLOAD_DIR / "test_spec.txt"
    test_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("Product Test\nRAM: 8GB\nStorage: 256GB")
    success = processor.add_document(test_doc_ref, str(test_file_path))
    assert success
    assert test_doc_ref in processor.document_vector_stores
    assert test_doc_ref in processor.document_texts
    test_file_path.unlink()

def test_extract_feature():
    """Test feature extraction."""
    processor = RAGProcessor()
    test_doc_ref = "test_doc"
    test_file_path = settings.UPLOAD_DIR / "test_spec.txt"
    test_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("Product Test\nRAM: 8GB\nStorage: 256GB")
    processor.add_document(test_doc_ref, str(test_file_path))
    ram_value = processor.extract_feature_from_doc(test_doc_ref, "RAM")
    assert "8GB" in ram_value
    non_existing = processor.extract_feature_from_doc(test_doc_ref, "Color")
    assert non_existing == "Not found"
    test_file_path.unlink()

def test_clear_documents():
    """Test clearing documents from vector store."""
    processor = RAGProcessor()
    test_doc_ref = "test_doc"
    test_file_path = settings.UPLOAD_DIR / "test_spec.txt"
    test_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("Product Test\nRAM: 8GB")
    processor.add_document(test_doc_ref, str(test_file_path))
    assert test_doc_ref in processor.document_vector_stores
    processor.clear_all_documents()
    assert processor.document_vector_stores == {}
    assert processor.document_texts == {}
    test_file_path.unlink() 