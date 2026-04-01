from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import os

from app.services.pdf_service import PDFService
from app.models.schemas import UploadResponse, SourceType
from app.config import get_settings

router = APIRouter(prefix="/upload")

# Initialize service
pdf_service = PDFService()
settings = get_settings()

# In-memory store for development
uploaded_files: dict = {}


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, MD)"""
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > settings.max_file_size:
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
    
    # Save file and extract text
    try:
        file_path, document_id = await pdf_service.save_file(content, filename)
        raw_text = pdf_service.extract_text(file_path)
        metadata = pdf_service.get_metadata(file_path)
        
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
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/{document_id}")
async def get_uploaded_file(document_id: str):
    """Get uploaded file info"""
    if document_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
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
    if document_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
    
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
