# app/services/document_parser.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from typing import List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    @staticmethod
    def load_document(file_path: Union[str, Path]) -> List[Document]:
        """
        Loads a document (PDF or TXT) and returns its content as Langchain Documents.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                docs = loader.load()
                if docs:
                    combined_content = "\n\n".join(doc.page_content for doc in docs)
                    combined_metadata = {"source": str(path.name)}
                    for i, doc in enumerate(docs):
                        combined_metadata[f"page_{i+1}_char_count"] = len(doc.page_content)
                    return [Document(page_content=combined_content, metadata=combined_metadata)]
                return []
            elif path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                return loader.load()
            else:
                logger.warning(f"Unsupported file type: {path.suffix} for file {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    from app.core.config import settings # Adjust import path if necessary
    logging.basicConfig(level=logging.INFO)

    # Create dummy files for testing
    dummy_txt_path = settings.UPLOAD_DIR / "dummy.txt"
    dummy_pdf_path = settings.UPLOAD_DIR / "dummy.pdf" # You'll need a real PDF here

    with open(dummy_txt_path, "w") as f:
        f.write("This is a test TXT file.\nFeature A: Value A\nFeature B: Value B")

    logger.info(f"Testing with TXT file: {dummy_txt_path}")
    txt_docs = DocumentParser.load_document(dummy_txt_path)
    if txt_docs:
        logger.info(f"TXT Loaded content of '{txt_docs[0].metadata.get('source')}': {txt_docs[0].page_content[:100]}...")
    else:
        logger.error("Failed to load TXT.")

    # For PDF, ensure you have a test PDF in your UPLOAD_DIR or provide a correct path
    # For example, if you place "example.pdf" in the UPLOAD_DIR:
    # pdf_docs = DocumentParser.load_document(settings.UPLOAD_DIR / "example.pdf")
    # if pdf_docs:
    #     logger.info(f"PDF Loaded content of '{pdf_docs[0].metadata.get('source')}': {pdf_docs[0].page_content[:100]}...")
    # else:
    #     logger.error("Failed to load PDF. Make sure a test PDF exists.")