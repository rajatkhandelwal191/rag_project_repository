from fastapi import APIRouter, HTTPException
from typing import List
import uuid
import logging
import traceback

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.models.schemas import (
    EmbeddingRequest, EmbeddingResponse, Embedding,
    IndexingRequest, IndexStats
)
from app.api.chunking import chunk_store

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embedding")

# Initialize services
embedding_service = EmbeddingService()
qdrant_service = QdrantService()

# Store embeddings
embedding_store: dict = {}


@router.post("/", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for chunks"""
    logger.info(f"Generate embeddings called for {len(request.chunk_ids)} chunks")
    
    # Find chunks for the given IDs
    all_chunks = []
    chunk_texts = []
    
    for doc_id, chunks in chunk_store.items():
        for chunk in chunks:
            if chunk.id in request.chunk_ids:
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
    
    if not all_chunks:
        logger.warning(f"No chunks found for IDs: {request.chunk_ids}")
        raise HTTPException(status_code=404, detail="No chunks found for the given IDs")
    
    logger.info(f"Found {len(all_chunks)} chunks to embed")
    
    # Generate embeddings
    try:
        logger.info(f"Generating embeddings with provider: {request.provider}, model: {request.model}")
        embeddings_list, model_name, dimension = await embedding_service.generate_embeddings(
            texts=chunk_texts,
            provider=request.provider,
            model_name=request.model,
            batch_size=request.batch_size
        )
        logger.info(f"Generated {len(embeddings_list)} embeddings, dimension: {dimension}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
    
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
    
    logger.info(f"Embeddings stored for {len(embeddings)} chunks")
    
    return EmbeddingResponse(
        embeddings=embeddings,
        model=model_name,
        dimension=dimension,
        total_embedded=len(embeddings)
    )


@router.get("/{document_id}")
async def get_embeddings(document_id: str):
    """Get all embeddings for a document"""
    logger.info(f"Get embeddings called for document_id: {document_id}")
    
    if document_id not in embedding_store:
        logger.warning(f"No embeddings found for document: {document_id}")
        raise HTTPException(status_code=404, detail="No embeddings found for this document")
    
    embeddings = list(embedding_store[document_id].values())
    logger.info(f"Returning {len(embeddings)} embeddings for document: {document_id}")
    
    return {
        "document_id": document_id,
        "embeddings": embeddings,
        "total": len(embeddings)
    }


@router.post("/index")
async def index_embeddings(request: IndexingRequest):
    """Index embeddings into vector database"""
    logger.info(f"Index embeddings called for collection: {request.collection_name}")
    
    # Create collection if it doesn't exist
    try:
        success = await qdrant_service.create_collection(
            collection_name=request.collection_name,
            distance_metric=request.distance_metric
        )
        if not success:
            error_msg = f"Failed to create collection: {request.collection_name}. Check Qdrant connection and dimension settings."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        logger.info(f"Collection created/verified: {request.collection_name}")
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Error creating collection: {str(e)}"
        logger.error(error_detail)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    
    # Prepare vectors for indexing
    all_embeddings = []
    all_ids = []
    all_payloads = []
    
    for doc_id, chunks_dict in embedding_store.items():
        for chunk_id, embedding in chunks_dict.items():
            # Find the chunk content from chunk_store
            chunk_content = ""
            chunk_metadata = {}
            if doc_id in chunk_store:
                for c in chunk_store[doc_id]:
                    if c.id == chunk_id:
                        chunk_content = c.content
                        chunk_metadata = {
                            "start_pos": c.start_pos,
                            "end_pos": c.end_pos,
                            "token_count": c.token_count
                        }
                        break
            
            all_embeddings.append(embedding.values)
            all_ids.append(embedding.vector_id)
            all_payloads.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "model": embedding.model,
                "content": chunk_content,  # Store content for retrieval reconstruction
                **chunk_metadata
            })
    
    if not all_embeddings:
        raise HTTPException(status_code=400, detail="No embeddings to index")
    
    # Upsert to Qdrant
    try:
        success = await qdrant_service.upsert_vectors(
            vectors=all_embeddings,
            ids=all_ids,
            payloads=all_payloads,
            collection_name=request.collection_name
        )
        if not success:
            logger.error("Failed to index embeddings to Qdrant")
            raise HTTPException(status_code=500, detail="Failed to index embeddings")
        logger.info(f"Indexed {len(all_embeddings)} vectors to Qdrant")
    except Exception as e:
        logger.error(f"Error indexing embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error indexing embeddings: {str(e)}")
    
    return {
        "message": "Embeddings indexed successfully",
        "indexed_count": len(all_embeddings),
        "collection": request.collection_name or qdrant_service.collection_name
    }


@router.get("/index/stats")
async def get_index_stats(collection_name: str = None):
    """Get vector index statistics"""
    logger.info(f"Get index stats called for collection: {collection_name}")
    
    try:
        stats = await qdrant_service.get_collection_stats(collection_name)
        if not stats:
            logger.warning(f"Collection not found: {collection_name}")
            raise HTTPException(status_code=404, detail="Collection not found")
        logger.info(f"Returning stats for collection: {collection_name}")
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting index stats: {str(e)}")


@router.get("/qdrant/health")
async def qdrant_health():
    """Check Qdrant connection health"""
    try:
        # Try to get collections list to verify connection
        collections = qdrant_service.client.get_collections()
        return {
            "status": "connected",
            "url": qdrant_service.url,
            "collections": [c.name for c in collections.collections],
            "dimension": qdrant_service.dimension
        }
    except Exception as e:
        logger.error(f"Qdrant health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Qdrant connection failed: {str(e)}")
