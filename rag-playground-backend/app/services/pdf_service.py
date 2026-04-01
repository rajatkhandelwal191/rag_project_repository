import os
import uuid
from typing import Optional, Dict, Any
import pdfplumber
from PyPDF2 import PdfReader


class PDFService:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return path"""
        document_id = str(uuid.uuid4())
        file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path, document_id
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or text file"""
        if file_path.endswith('.pdf'):
            return self._extract_pdf_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text_parts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
        except Exception:
            # Fallback to PyPDF2
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        
        return "\n\n".join(text_parts)
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        stat = os.stat(file_path)
        filename = os.path.basename(file_path)
        
        metadata = {
            "filename": filename,
            "size": stat.st_size,
            "extension": os.path.splitext(filename)[1].lower()
        }
        
        if file_path.endswith('.pdf'):
            try:
                with pdfplumber.open(file_path) as pdf:
                    metadata["page_count"] = len(pdf.pages)
            except:
                pass
        
        return metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        # Remove duplicate spaces
        text = ' '.join(text.split())
        
        return text
