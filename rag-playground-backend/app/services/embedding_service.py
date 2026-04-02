import os
import time
import uuid
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import openai
from app.config import get_settings
from app.models.schemas import Embedding, EmbeddingProvider


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.default_provider = EmbeddingProvider.GOOGLE
        self.model_dims = {
            "gemini-embedding-001": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # Configure Gemini
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
        
        # Configure OpenAI
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
    
    async def generate_embeddings(
        self,
        texts: List[str],
        provider: EmbeddingProvider = EmbeddingProvider.GOOGLE,
        model_name: Optional[str] = None,
        batch_size: int = 32
    ) -> tuple[List[List[float]], str, int]:
        """Generate embeddings using cloud providers"""
        
        settings = get_settings()
        
        if model_name is None:
            if provider == EmbeddingProvider.GOOGLE:
                model_name = settings.gemini_embed_model
            elif provider == EmbeddingProvider.OPENAI:
                model_name = "text-embedding-3-small"
            else:
                model_name = settings.gemini_embed_model
        
        start_time = time.time()
        
        if provider == EmbeddingProvider.GOOGLE:
            embeddings = await self._generate_gemini(texts, model_name)
        elif provider == EmbeddingProvider.OPENAI:
            embeddings = await self._generate_openai(texts, model_name)
        else:
            embeddings = await self._generate_gemini(texts, model_name)
        
        latency = int((time.time() - start_time) * 1000)
        dimension = len(embeddings[0]) if embeddings else 0
        
        return embeddings, model_name, dimension
    
    async def _generate_gemini(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using Gemini API"""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=batch,
                    task_type="retrieval_document"
                )
                
                if isinstance(result, dict) and 'embedding' in result:
                    batch_embeddings = result['embedding']
                else:
                    batch_embeddings = [result['embedding']] if 'embedding' in result else []
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Gemini embedding error: {e}")
                dim = self.model_dims.get(model, 768)
                embeddings.extend([[0.0] * dim for _ in batch])
        
        return embeddings
    
    async def _generate_openai(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"OpenAI embedding error: {e}")
                dim = self.model_dims.get(model, 1536)
                embeddings.extend([[0.0] * dim for _ in batch])
        
        return embeddings
    
    def get_dimension(self, model_name: Optional[str] = None) -> int:
        """Get embedding dimension for a model"""
        if model_name is None:
            settings = get_settings()
            model_name = settings.gemini_embed_model
        return self.model_dims.get(model_name, 768)
    
    def create_embedding_objects(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        model: str,
        document_id: str
    ) -> List[Embedding]:
        """Create Embedding objects from raw embeddings"""
        results = []
        for i, (chunk_id, vector) in enumerate(zip(chunk_ids, embeddings)):
            # Generate a proper UUID for Qdrant point ID
            vector_uuid = str(uuid.uuid4())
            results.append(Embedding(
                chunk_id=chunk_id,
                vector_id=vector_uuid,  # Qdrant requires UUID or unsigned integer
                values=vector,
                dimension=len(vector),
                model=model,
                document_id=document_id
            ))
        return results
