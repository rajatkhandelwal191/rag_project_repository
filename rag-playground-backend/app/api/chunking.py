from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import traceback

from app.services.chunk_service import ChunkService
from app.models.schemas import ChunkingRequest, ChunkingResponse, Chunk, ChunkingStrategy
from app.api.upload import uploaded_files

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunking")

# Initialize service
chunk_service = ChunkService()

# Store chunks
chunk_store: dict = {}


@router.post("/", response_model=ChunkingResponse)
async def create_chunks(request: ChunkingRequest):
    """Create chunks from uploaded document"""
    logger.info(f"Create chunks called for document_id: {request.document_id}")
    
    document_id = request.document_id
    
    if document_id not in uploaded_files:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_info = uploaded_files[document_id]
    
    # Use cleaned text if available, otherwise raw text
    text = file_info.get("cleaned_text") or file_info["raw_text"]
    
    if not text:
        logger.error(f"Document {document_id} has no text content")
        raise HTTPException(status_code=400, detail="Document has no text content")
    
    # Create chunks
    try:
        logger.info(f"Creating chunks with strategy: {request.strategy}, size: {request.chunk_size}")
        chunks = chunk_service.create_chunks(
            document_id=document_id,
            text=text,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        logger.info(f"Created {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating chunks: {str(e)}")
    
    # Store chunks
    chunk_store[document_id] = chunks
    logger.info(f"Chunks stored for document: {document_id}")
    
    return ChunkingResponse(
        document_id=document_id,
        chunks=chunks,
        total_chunks=len(chunks),
        strategy=request.strategy
    )


@router.get("/{document_id}", response_model=ChunkingResponse)
async def get_chunks(document_id: str):
    """Get chunks for a document"""
    logger.info(f"Get chunks called for document_id: {document_id}")
    
    if document_id not in chunk_store:
        logger.warning(f"Chunks not found for document: {document_id}")
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
    logger.info(f"Get chunk called: document_id={document_id}, chunk_id={chunk_id}")
    
    if document_id not in chunk_store:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = chunk_store[document_id]
    
    for chunk in chunks:
        if chunk.id == chunk_id:
            logger.info(f"Found chunk: {chunk_id}")
            return chunk
    
    logger.warning(f"Chunk not found: {chunk_id}")
    raise HTTPException(status_code=404, detail="Chunk not found")


@router.delete("/{document_id}")
async def delete_chunks(document_id: str):
    """Delete chunks for a document"""
    logger.info(f"Delete chunks called for document_id: {document_id}")
    
    if document_id not in chunk_store:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    del chunk_store[document_id]
    logger.info(f"Chunks deleted for document: {document_id}")
    
    return {"message": "Chunks deleted successfully"}
