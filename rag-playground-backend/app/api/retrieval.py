from fastapi import APIRouter, HTTPException
import time

from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from app.models.schemas import RetrievalRequest, RetrievalResponse, RetrievedChunk, Chunk
from app.api.chunking import chunk_store

router = APIRouter(prefix="/retrieval")

# Initialize services
qdrant_service = QdrantService()
embedding_service = EmbeddingService()


@router.post("/", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """Retrieve relevant chunks for a query"""
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    start_time = time.time()
    
    # Generate embedding for query
    query_embedding, _, _ = await embedding_service.generate_embeddings(
        texts=[request.query],
        provider=embedding_service.default_model
    )
    
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    # Search in vector database
    results = await qdrant_service.search(
        query_vector=query_embedding[0],
        top_k=request.top_k,
        collection_name=request.collection_name,
        filters=request.filters
    )
    
    # Map results to chunks
    retrieved_chunks = []
    for vector_id, score, payload in results:
        chunk_id = payload.get("chunk_id")
        document_id = payload.get("document_id")
        
        # Find the chunk
        chunk = None
        if document_id in chunk_store:
            for c in chunk_store[document_id]:
                if c.id == chunk_id:
                    chunk = c
                    break
        
        if chunk:
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                embedding_id=vector_id
            ))
    
    latency = int((time.time() - start_time) * 1000)
    
    return RetrievalResponse(
        query=request.query,
        results=retrieved_chunks,
        total_found=len(retrieved_chunks),
        latency_ms=latency
    )


@router.post("/rerank")
async def rerank_results(request: dict):
    """Rerank retrieved chunks"""
    from app.services.llm_service import LLMService
    
    llm_service = LLMService()
    
    query = request.get("query", "")
    chunks = request.get("chunks", [])
    top_k = request.get("top_k", 5)
    
    if not query or not chunks:
        raise HTTPException(status_code=400, detail="Query and chunks are required")
    
    # Rerank
    reranked = await llm_service.rerank(
        query=query,
        chunks=chunks,
        top_k=top_k
    )
    
    return {
        "query": query,
        "reranked_results": [
            {
                "original_rank": rank,
                "content": content,
                "reranked_score": score
            }
            for rank, content, score in reranked
        ],
        "model": "cross-encoder"
    }


@router.get("/collections")
async def list_collections():
    """List all collections in Qdrant"""
    try:
        collections = qdrant_service.client.get_collections()
        return {
            "collections": [c.name for c in collections.collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")
