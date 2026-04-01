from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.services.chunk_service import ChunkService
from app.models.schemas import ChunkingRequest, ChunkingResponse, Chunk, ChunkingStrategy
from app.api.upload import uploaded_files

router = APIRouter(prefix="/chunking")

# Initialize service
chunk_service = ChunkService()

# Store chunks
chunk_store: dict = {}


@router.post("/", response_model=ChunkingResponse)
async def create_chunks(request: ChunkingRequest):
    """Create chunks from uploaded document"""
    
    document_id = request.document_id
    
    if document_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
    
    # Use cleaned text if available, otherwise raw text
    text = file_info.get("cleaned_text") or file_info["raw_text"]
    
    if not text:
        raise HTTPException(status_code=400, detail="Document has no text content")
    
    # Create chunks
    chunks = chunk_service.create_chunks(
        document_id=document_id,
        text=text,
        strategy=request.strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    
    # Store chunks
    chunk_store[document_id] = chunks
    
    return ChunkingResponse(
        document_id=document_id,
        chunks=chunks,
        total_chunks=len(chunks),
        strategy=request.strategy
    )


@router.get("/{document_id}", response_model=ChunkingResponse)
async def get_chunks(document_id: str):
    """Get chunks for a document"""
    
    if document_id not in chunk_store:
        raise HTTPException(status_code=404, detail="Chunks not found for this document")
    
    chunks = chunk_store[document_id]
    
    # Determine strategy from first chunk metadata
    strategy = ChunkingStrategy.FIXED
    if chunks and chunks[0].metadata:
        strategy_str = chunks[0].metadata.get("strategy", "fixed")
        try:
            strategy = ChunkingStrategy(strategy_str)
        except:
            pass
    
    return ChunkingResponse(
        document_id=document_id,
        chunks=chunks,
        total_chunks=len(chunks),
        strategy=strategy
    )


@router.get("/{document_id}/chunk/{chunk_id}", response_model=Chunk)
async def get_chunk(document_id: str, chunk_id: str):
    """Get a specific chunk"""
    
    if document_id not in chunk_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = chunk_store[document_id]
    
    for chunk in chunks:
        if chunk.id == chunk_id:
            return chunk
    
    raise HTTPException(status_code=404, detail="Chunk not found")


@router.delete("/{document_id}")
async def delete_chunks(document_id: str):
    """Delete chunks for a document"""
    
    if document_id not in chunk_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del chunk_store[document_id]
    
    return {"message": "Chunks deleted successfully"}
