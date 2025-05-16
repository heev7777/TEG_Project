from typing import Union, BinaryIO
from pypdf import PdfReader
import io

class DocumentParser:
    """Service for parsing different types of documents."""
    
    @staticmethod
    def parse_pdf(file: Union[BinaryIO, bytes]) -> str:
        """
        Parse a PDF file and extract its text content.
        
        Args:
            file: PDF file object or bytes
            
        Returns:
            str: Extracted text content
        """
        try:
            if isinstance(file, bytes):
                file = io.BytesIO(file)
            
            pdf_reader = PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise ValueError(f"Error parsing PDF file: {str(e)}")
    
    @staticmethod
    def parse_txt(file: Union[BinaryIO, bytes]) -> str:
        """
        Parse a TXT file and extract its text content.
        
        Args:
            file: TXT file object or bytes
            
        Returns:
            str: Extracted text content
        """
        try:
            if isinstance(file, bytes):
                file = io.BytesIO(file)
            
            return file.read().decode('utf-8').strip()
        
        except Exception as e:
            raise ValueError(f"Error parsing TXT file: {str(e)}")
    
    @staticmethod
    def parse_document(file: Union[BinaryIO, bytes], file_type: str) -> str:
        """
        Parse a document based on its file type.
        
        Args:
            file: File object or bytes
            file_type: Type of file ('pdf' or 'txt')
            
        Returns:
            str: Extracted text content
        """
        if file_type.lower() == 'pdf':
            return DocumentParser.parse_pdf(file)
        elif file_type.lower() == 'txt':
            return DocumentParser.parse_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 