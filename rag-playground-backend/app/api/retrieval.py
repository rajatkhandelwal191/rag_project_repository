from fastapi import APIRouter, HTTPException
import time
import logging
import traceback

from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from app.models.schemas import RetrievalRequest, RetrievalResponse, RetrievedChunk, Chunk
from app.api.chunking import chunk_store

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval")

# Initialize services
qdrant_service = QdrantService()
embedding_service = EmbeddingService()


@router.post("/", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """Retrieve relevant chunks for a query"""
    logger.info(f"Retrieve called with query: '{request.query}', top_k: {request.top_k}")
    
    if not request.query:
        logger.warning("Query is empty")
        raise HTTPException(status_code=400, detail="Query is required")
    
    start_time = time.time()
    
    logger.info(f"Retrieval request: query='{request.query}', top_k={request.top_k}, collection={request.collection_name}")
    
    # Get collection info to determine the model used for indexing
    collection_name = request.collection_name or "rag_playground_chunks"
    model_name = None
    provider = None
    
    try:
        # Try to get model info from one of the indexed vectors
        collection_info = await qdrant_service.get_collection_stats(collection_name)
        if collection_info and collection_info.vector_count > 0:
            # Get a sample vector to see what model was used
            sample_results = qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True
            )
            if sample_results and sample_results[0]:
                payload = sample_results[0][0].payload
                model_name = payload.get("model")
                logger.info(f"Found indexed model: {model_name}")
    except Exception as e:
        logger.warning(f"Could not get model info from collection: {e}")
    
    # Determine provider from model name
    if model_name:
        if "gemini" in model_name.lower() or "google" in model_name.lower():
            provider = "google"
        elif "text-embedding-3" in model_name.lower() or "text-embedding-ada" in model_name.lower():
            provider = "openai"
        else:
            provider = embedding_service.default_provider
    else:
        provider = embedding_service.default_provider
        model_name = None
    
    logger.info(f"Using provider={provider}, model={model_name} for query embedding")
    
    # Generate query embedding using the same model as indexed vectors
    try:
        query_embedding, _, _ = await embedding_service.generate_embeddings(
            texts=[request.query],
            provider=provider,
            model_name=model_name
        )
        logger.info(f"Generated query embedding with dimension: {len(query_embedding[0])}")
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")
    
    # Search in vector database
    try:
        logger.info(f"Searching in collection: {request.collection_name}")
        results = await qdrant_service.search(
            query_vector=query_embedding[0],
            top_k=request.top_k,
            collection_name=request.collection_name,
            filters=request.filters
        )
        logger.info(f"Search returned {len(results)} results")
    except Exception as e:
        logger.error(f"Error searching in vector database: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")
    
    # Map results to chunks
    retrieved_chunks = []
    logger.info(f"Mapping {len(results)} results to chunks. chunk_store has {len(chunk_store)} documents")
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
        
        # If not found in chunk_store, reconstruct from payload
        if not chunk:
            logger.info(f"Reconstructing chunk {chunk_id} from Qdrant payload")
            content = payload.get("content", "")
            start_pos = payload.get("start_pos", 0)
            end_pos = payload.get("end_pos", 0)
            token_count = payload.get("token_count", 0)
            
            if content:
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    token_count=token_count,
                    metadata={"strategy": payload.get("strategy", "semantic")}
                )
                logger.info(f"Reconstructed chunk from payload: {chunk_id}")
            else:
                logger.warning(f"Cannot reconstruct chunk {chunk_id}: no content in payload")
        
        if chunk:
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                embedding_id=vector_id
            ))
    
    latency = int((time.time() - start_time) * 1000)
    logger.info(f"Retrieval complete. Found {len(retrieved_chunks)} chunks in {latency}ms")
    
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
    
    logger.info(f"Rerank called with query: '{query}', chunks: {len(chunks)}, top_k: {top_k}")
    
    if not query or not chunks:
        logger.warning("Query or chunks are empty")
        raise HTTPException(status_code=400, detail="Query and chunks are required")
    
    # Rerank
    try:
        logger.info("Starting reranking...")
        reranked = await llm_service.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k
        )
        logger.info(f"Reranking complete. Returned {len(reranked)} results")
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during reranking: {str(e)}")
    
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
    logger.info("List collections called")
    try:
        collections = qdrant_service.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        logger.info(f"Found {len(collection_names)} collections: {collection_names}")
        return {
            "collections": collection_names
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")
