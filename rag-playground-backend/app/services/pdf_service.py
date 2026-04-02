import os
import uuid
import logging
from typing import Optional, Dict, Any
import pdfplumber
from PyPDF2 import PdfReader
from app.config import get_settings

# Setup logger
logger = logging.getLogger(__name__)

# Conditionally import boto3 for S3 (only needed in CLOUD mode)
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class PDFService:
    def __init__(self, upload_dir: str = None):
        self.settings = get_settings()
        self.upload_dir = upload_dir or self.settings.upload_dir
        self.env = self.settings.env.upper()
        
        # Ensure local upload directory exists (for LOCAL mode)
        if self.env == "LOCAL":
            os.makedirs(self.upload_dir, exist_ok=True)
            logger.info(f"PDFService initialized in LOCAL mode. Upload dir: {self.upload_dir}")
        else:
            logger.info(f"PDFService initialized in CLOUD mode. S3 Bucket: {self.settings.supabase_s3_bucket}")
    
    def _get_s3_client(self):
        """Get S3 client for Supabase Storage"""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for cloud storage. Install with: pip install boto3")
        
        return boto3.client(
            's3',
            endpoint_url=self.settings.supabase_s3_endpoint,
            aws_access_key_id=self.settings.supabase_s3_access_key_id,
            aws_secret_access_key=self.settings.supabase_s3_secret_access_key,
            region_name='ap-northeast-2'  # Adjust based on your Supabase region
        )
    
    async def save_file(self, file_content: bytes, filename: str) -> tuple:
        """Save uploaded file - locally or to S3 based on environment"""
        document_id = str(uuid.uuid4())
        
        if self.env == "CLOUD":
            return await self._save_to_s3(file_content, filename, document_id)
        else:
            return await self._save_local(file_content, filename, document_id)
    
    async def _save_local(self, file_content: bytes, filename: str, document_id: str) -> tuple:
        """Save file to local filesystem"""
        file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"File saved locally: {file_path}")
            return file_path, document_id
        except Exception as e:
            logger.error(f"Error saving file locally: {str(e)}")
            raise
    
    async def _save_to_s3(self, file_content: bytes, filename: str, document_id: str) -> tuple:
        """Save file to Supabase S3 storage"""
        if not self.settings.supabase_s3_access_key_id or not self.settings.supabase_s3_secret_access_key:
            logger.error("S3 credentials not configured. Falling back to local storage.")
            return await self._save_local(file_content, filename, document_id)
        
        try:
            s3_client = self._get_s3_client()
            s3_key = f"uploads/{document_id}_{filename}"
            
            s3_client.put_object(
                Bucket=self.settings.supabase_s3_bucket,
                Key=s3_key,
                Body=file_content
            )
            
            file_path = f"s3://{self.settings.supabase_s3_bucket}/{s3_key}"
            logger.info(f"File uploaded to S3: {file_path}")
            return file_path, document_id
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            logger.warning("Falling back to local storage")
            return await self._save_local(file_content, filename, document_id)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or text file"""
        if file_path.startswith("s3://"):
            # For S3 files, download to temp first
            return self._extract_text_from_s3(file_path)
        
        if file_path.endswith('.pdf'):
            return self._extract_pdf_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _extract_text_from_s3(self, s3_path: str) -> str:
        """Download from S3 and extract text"""
        import tempfile
        
        try:
            s3_client = self._get_s3_client()
            bucket = self.settings.supabase_s3_bucket
            key = s3_path.replace(f"s3://{bucket}/", "")
            
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                s3_client.download_fileobj(bucket, key, tmp)
                tmp_path = tmp.name
            
            # Extract text
            text = self._extract_pdf_text(tmp_path) if tmp_path.endswith('.pdf') else open(tmp_path, 'r').read()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from S3: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text_parts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            logger.info(f"Extracted text from PDF: {len(text_parts)} pages")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}, trying PyPDF2 fallback")
            # Fallback to PyPDF2
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
                logger.info(f"Extracted text using PyPDF2 fallback: {len(text_parts)} pages")
            except Exception as e2:
                logger.error(f"Both PDF extractors failed: {str(e2)}")
                raise
        
        return "\n\n".join(text_parts)
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        metadata = {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "storage_type": "s3" if file_path.startswith("s3://") else "local"
        }
        
        # Try to get page count for PDFs
        if '.pdf' in file_path.lower():
            try:
                if file_path.startswith("s3://"):
                    # For S3 files, we'd need to download first - skip for now
                    metadata["page_count"] = None
                else:
                    with pdfplumber.open(file_path) as pdf:
                        metadata["page_count"] = len(pdf.pages)
            except Exception as e:
                logger.warning(f"Could not get page count: {str(e)}")
                metadata["page_count"] = None
        
        return metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        # Remove duplicate spaces
        text = ' '.join(text.split())
        
        logger.debug(f"Text cleaned. Original length: {len(text)}")
        
        return text.strip()
