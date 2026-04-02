from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import os
import logging
import traceback

from app.services.pdf_service import PDFService
from app.models.schemas import UploadResponse, SourceType
from app.config import get_settings

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload")

# Initialize service
pdf_service = PDFService()
settings = get_settings()

# In-memory store for development
uploaded_files: dict = {}


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, MD)"""
    logger.info(f"Upload endpoint called with file: {file.filename}")
    
    if not file:
        logger.error("No file provided in request")
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content
    try:
        content = await file.read()
        logger.info(f"File read successfully: {len(content)} bytes")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    # Check file size
    if len(content) > settings.max_file_size:
        logger.error(f"File too large: {len(content)} bytes (max: {settings.max_file_size})")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size} bytes"
        )
    
    # Determine source type
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    source_type_map = {
        '.pdf': SourceType.PDF,
        '.txt': SourceType.TEXT,
        '.md': SourceType.MARKDOWN,
        '.html': SourceType.HTML,
        '.htm': SourceType.HTML
    }
    
    source_type = source_type_map.get(ext, SourceType.TEXT)
    logger.info(f"Detected source type: {source_type} for extension: {ext}")
    
    # Save file and extract text
    try:
        logger.info(f"Saving file and extracting text...")
        file_path, document_id = await pdf_service.save_file(content, filename)
        logger.info(f"File saved: {file_path}, Document ID: {document_id}")
        
        raw_text = pdf_service.extract_text(file_path)
        logger.info(f"Text extracted: {len(raw_text)} characters")
        
        metadata = pdf_service.get_metadata(file_path)
        logger.info(f"Metadata extracted: {metadata}")
        
        # Store in memory
        uploaded_files[document_id] = {
            "document_id": document_id,
            "document_name": filename,
            "source_type": source_type,
            "file_size": len(content),
            "content_type": file.content_type or "application/octet-stream",
            "raw_text": raw_text,
            "cleaned_text": pdf_service.clean_text(raw_text),
            "metadata": metadata,
            "file_path": file_path,
            "created_at": datetime.now()
        }
        
        logger.info(f"File processed successfully. Document ID: {document_id}")
        return UploadResponse(
            document_id=document_id,
            document_name=filename,
            source_type=source_type,
            file_size=len(content),
            content_type=file.content_type or "application/octet-stream",
            raw_text=raw_text,
            metadata=metadata,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/{document_id}")
async def get_uploaded_file(document_id: str):
    """Get uploaded file info"""
    logger.info(f"Get file info called for document_id: {document_id}")
    
    if document_id not in uploaded_files:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
    logger.info(f"Returning file info for: {document_id}")
    return {
        "document_id": file_info["document_id"],
        "document_name": file_info["document_name"],
        "source_type": file_info["source_type"],
        "file_size": file_info["file_size"],
        "metadata": file_info["metadata"]
    }


@router.get("/{document_id}/text")
async def get_file_text(document_id: str, cleaned: bool = False):
    """Get raw or cleaned text of uploaded file"""
    logger.info(f"Get file text called for document_id: {document_id}, cleaned: {cleaned}")
    
    if document_id not in uploaded_files:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
    logger.info(f"Returning text for document: {document_id}")
    
    if cleaned:
        return {
            "document_id": document_id,
            "text": file_info.get("cleaned_text", file_info["raw_text"]),
            "type": "cleaned"
        }
    else:
        return {
            "document_id": document_id,
            "text": file_info["raw_text"],
            "type": "raw"
        }
