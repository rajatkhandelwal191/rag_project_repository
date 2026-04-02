from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.api.upload import uploaded_files as document_store
from app.api.chunking import chunk_store
from app.api.embedding import embedding_store

router = APIRouter(prefix="/experiment")


@router.get("/summary/{document_id}")
async def get_pipeline_summary(document_id: str):
    """
    Get complete pipeline summary for a document.
    Shows all stages from ingestion to generation.
    """
    
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = document_store[document_id]
    
    # Build pipeline stages info
    stages = []
    
    # Stage 1: Ingestion
    stages.append({
        "stage": "ingestion",
        "status": "completed",
        "timestamp": document.get("created_at", datetime.now().isoformat()),
        "details": {
            "document_name": document.get("document_name"),
            "source_type": document.get("source_type"),
            "file_size": document.get("file_size"),
            "content_type": document.get("content_type"),
            "raw_text_length": len(document.get("raw_text", ""))
        }
    })
    
    # Stage 2: Preprocessing
    has_cleaned_text = "cleaned_text" in document
    stages.append({
        "stage": "preprocessing",
        "status": "completed" if has_cleaned_text else "skipped",
        "details": {
            "preprocessing_applied": document.get("preprocessing_applied", {}),
            "cleaned_text_length": len(document.get("cleaned_text", "")) if has_cleaned_text else 0,
            "text_reduction_percent": round(
                (1 - len(document.get("cleaned_text", "")) / max(len(document.get("raw_text", "")), 1)) * 100, 2
            ) if has_cleaned_text else 0
        }
    })
    
    # Stage 3: Chunking
    chunks = chunk_store.get(document_id, [])
    stages.append({
        "stage": "chunking",
        "status": "completed" if chunks else "pending",
        "details": {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.content) for c in chunks) / max(len(chunks), 1),
            "total_tokens": sum(c.token_count for c in chunks)
        }
    })
    
    # Stage 4: Embedding
    embeddings = embedding_store.get(document_id, {})
    stages.append({
        "stage": "embedding",
        "status": "completed" if embeddings else "pending",
        "details": {
            "total_embedded": len(embeddings),
            "model": list(embeddings.values())[0].model if embeddings else None,
            "dimension": list(embeddings.values())[0].dimension if embeddings else None
        }
    })
    
    # Stage 5: Indexing
    # Check if indexed (would need to query Qdrant for actual status)
    stages.append({
        "stage": "indexing",
        "status": "completed" if embeddings else "pending",
        "details": {
            "indexed_count": len(embeddings),
            "collection_name": "rag_documents"
        }
    })
    
    # Stage 6-7: Retrieval & Reranking
    stages.append({
        "stage": "retrieval_reranking",
        "status": "ready" if embeddings else "pending",
        "details": {
            "available_for_query": bool(embeddings)
        }
    })
    
    # Stage 8: Generation
    stages.append({
        "stage": "generation",
        "status": "ready" if embeddings else "pending",
        "details": {
            "llm_providers_available": ["openai", "google", "groq"]
        }
    })
    
    return {
        "document_id": document_id,
        "pipeline_summary": {
            "document_name": document.get("document_name"),
            "total_stages": len(stages),
            "completed_stages": sum(1 for s in stages if s["status"] == "completed"),
            "stages": stages
        },
        "is_pipeline_ready": all(s["status"] in ["completed", "ready", "skipped"] for s in stages),
        "can_query": bool(embeddings)
    }


@router.get("/compare")
async def compare_documents(document_ids: str):
    """
    Compare pipeline results across multiple documents.
    
    Pass document IDs as comma-separated: ?document_ids=id1,id2,id3
    """
    ids = [id.strip() for id in document_ids.split(",")]
    
    comparisons = []
    for doc_id in ids:
        if doc_id not in document_store:
            continue
            
        doc = document_store[doc_id]
        chunks = chunk_store.get(doc_id, [])
        embeddings = embedding_store.get(doc_id, {})
        
        comparisons.append({
            "document_id": doc_id,
            "document_name": doc.get("document_name"),
            "source_type": doc.get("source_type"),
            "file_size": doc.get("file_size"),
            "text_length": len(doc.get("cleaned_text", doc.get("raw_text", ""))),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "is_indexed": bool(embeddings)
        })
    
    return {
        "compared_documents": len(comparisons),
        "documents": comparisons
    }


@router.get("/stats")
async def get_system_stats():
    """Get overall system statistics across all documents"""
    
    total_docs = len(document_store)
    total_chunks = sum(len(chunks) for chunks in chunk_store.values())
    total_embeddings = sum(len(embs) for embs in embedding_store.values())
    
    # Get source type distribution
    source_types = {}
    for doc in document_store.values():
        source = doc.get("source_type", "unknown")
        source_types[source] = source_types.get(source, 0) + 1
    
    # Get average metrics
    avg_chunks_per_doc = total_chunks / max(total_docs, 1)
    avg_embeddings_per_doc = total_embeddings / max(total_docs, 1)
    
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_embeddings": total_embeddings,
        "indexed_documents": len(embedding_store),
        "source_type_distribution": source_types,
        "averages": {
            "chunks_per_document": round(avg_chunks_per_doc, 2),
            "embeddings_per_document": round(avg_embeddings_per_doc, 2)
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/models")
async def list_available_models():
    """List all available models for embedding and generation"""
    
    return {
        "embedding_models": [
            {
                "id": "gemini-embedding-001",
                "provider": "google",
                "dimensions": 768,
                "description": "Google Gemini embedding model"
            },
            {
                "id": "text-embedding-3-small",
                "provider": "openai",
                "dimensions": 1536,
                "description": "OpenAI small embedding model"
            },
            {
                "id": "text-embedding-3-large",
                "provider": "openai",
                "dimensions": 3072,
                "description": "OpenAI large embedding model"
            }
        ],
        "llm_models": [
            {
                "id": "gpt-3.5-turbo",
                "provider": "openai",
                "context_length": 4096,
                "description": "OpenAI GPT-3.5 Turbo"
            },
            {
                "id": "gpt-4",
                "provider": "openai",
                "context_length": 8192,
                "description": "OpenAI GPT-4"
            },
            {
                "id": "gemini-2.0-flash",
                "provider": "google",
                "context_length": 32768,
                "description": "Google Gemini 2.0 Flash"
            },
            {
                "id": "llama-3.1-8b-instant",
                "provider": "groq",
                "context_length": 8192,
                "description": "Meta Llama 3.1 8B via Groq"
            }
        ],
        "chunking_strategies": [
            {
                "id": "fixed",
                "name": "Fixed Size",
                "description": "Split text into fixed-size chunks with overlap"
            },
            {
                "id": "semantic",
                "name": "Semantic",
                "description": "Split at sentence boundaries"
            },
            {
                "id": "recursive",
                "name": "Recursive",
                "description": "Respect paragraph and section boundaries"
            }
        ]
    }


@router.post("/reset/{document_id}")
async def reset_pipeline(document_id: str, stage: Optional[str] = None):
    """
    Reset pipeline for a document.
    
    - **stage**: Specific stage to reset (chunking, embedding, indexing)
    - If no stage specified, resets all processing stages (keeps ingestion)
    """
    
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = document_store[document_id]
    
    if stage == "chunking" or stage is None:
        # Remove chunks
        if document_id in chunk_store:
            del chunk_store[document_id]
    
    if stage == "embedding" or stage is None:
        # Remove embeddings
        if document_id in embedding_store:
            del embedding_store[document_id]
    
    if stage == "preprocessing" or stage is None:
        # Remove cleaned text
        document.pop("cleaned_text", None)
        document.pop("preprocessing_applied", None)
    
    if stage == "all":
        # Remove everything including document
        if document_id in document_store:
            del document_store[document_id]
        if document_id in chunk_store:
            del chunk_store[document_id]
        if document_id in embedding_store:
            del embedding_store[document_id]
        return {"message": "Document and all processing data removed", "document_id": document_id}
    
    return {
        "message": f"Pipeline reset for stage: {stage or 'all processing stages'}",
        "document_id": document_id,
        "reset_stage": stage or "chunking, embedding, preprocessing"
    }
