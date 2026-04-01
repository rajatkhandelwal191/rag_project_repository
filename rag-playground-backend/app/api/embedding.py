from fastapi import APIRouter, HTTPException
from typing import List
import uuid

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.models.schemas import (
    EmbeddingRequest, EmbeddingResponse, Embedding,
    IndexingRequest, IndexStats
)
from app.api.chunking import chunk_store

router = APIRouter(prefix="/embedding")

# Initialize services
embedding_service = EmbeddingService()
qdrant_service = QdrantService()

# Store embeddings
embedding_store: dict = {}


@router.post("/", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for chunks"""
    
    # Find chunks for the given IDs
    all_chunks = []
    chunk_texts = []
    
    for doc_id, chunks in chunk_store.items():
        for chunk in chunks:
            if chunk.id in request.chunk_ids:
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
    
    if not all_chunks:
        raise HTTPException(status_code=404, detail="No chunks found for the given IDs")
    
    # Generate embeddings
    embeddings_list, model_name, dimension = await embedding_service.generate_embeddings(
        texts=chunk_texts,
        provider=request.provider,
        model_name=request.model,
        batch_size=request.batch_size
    )
    
    # Create embedding objects
    embeddings = embedding_service.create_embedding_objects(
        chunk_ids=[c.id for c in all_chunks],
        embeddings=embeddings_list,
        model=model_name,
        document_id=all_chunks[0].document_id
    )
    
    # Store embeddings
    for emb in embeddings:
        if emb.document_id not in embedding_store:
            embedding_store[emb.document_id] = {}
        embedding_store[emb.document_id][emb.chunk_id] = emb
    
    return EmbeddingResponse(
        embeddings=embeddings,
        model=model_name,
        dimension=dimension,
        total_embedded=len(embeddings)
    )


@router.get("/{document_id}")
async def get_embeddings(document_id: str):
    """Get all embeddings for a document"""
    
    if document_id not in embedding_store:
        raise HTTPException(status_code=404, detail="No embeddings found for this document")
    
    embeddings = list(embedding_store[document_id].values())
    
    return {
        "document_id": document_id,
        "embeddings": embeddings,
        "total": len(embeddings)
    }


@router.post("/index")
async def index_embeddings(request: IndexingRequest):
    """Index embeddings into vector database"""
    
    # Create collection if it doesn't exist
    success = await qdrant_service.create_collection(
        collection_name=request.collection_name,
        distance_metric=request.distance_metric
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create collection")
    
    # Prepare vectors for indexing
    all_embeddings = []
    all_ids = []
    all_payloads = []
    
    for doc_id, chunks_dict in embedding_store.items():
        for chunk_id, embedding in chunks_dict.items():
            all_embeddings.append(embedding.values)
            all_ids.append(embedding.vector_id)
            all_payloads.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "model": embedding.model
            })
    
    if not all_embeddings:
        raise HTTPException(status_code=400, detail="No embeddings to index")
    
    # Upsert to Qdrant
    success = await qdrant_service.upsert_vectors(
        vectors=all_embeddings,
        ids=all_ids,
        payloads=all_payloads,
        collection_name=request.collection_name
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to index embeddings")
    
    return {
        "message": "Embeddings indexed successfully",
        "indexed_count": len(all_embeddings),
        "collection": request.collection_name or qdrant_service.collection_name
    }


@router.get("/index/stats")
async def get_index_stats(collection_name: str = None):
    """Get vector index statistics"""
    
    stats = await qdrant_service.get_collection_stats(collection_name)
    
    if not stats:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return stats
