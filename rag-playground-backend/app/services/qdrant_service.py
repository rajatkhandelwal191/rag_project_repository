from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import get_settings
from app.models.schemas import DistanceMetric, IndexStats, RetrievedChunk
from app.models.schemas import Chunk


class QdrantService:
    def __init__(self):
        settings = get_settings()
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key or None
        self.collection_name = settings.collection_name
        self.dimension = settings.embedding_dim
        
        # Initialize client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
    
    async def create_collection(
        self,
        collection_name: Optional[str] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> bool:
        """Create a new collection"""
        name = collection_name or self.collection_name
        
        # Map distance metric
        distance_map = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT
        }
        
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Collection might already exist
            return False
    
    async def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        payloads: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> bool:
        """Upsert vectors into collection"""
        name = collection_name or self.collection_name
        
        points = []
        for i, (vector, id_, payload) in enumerate(zip(vectors, ids, payloads)):
            points.append(PointStruct(
                id=id_,
                vector=vector,
                payload=payload
            ))
        
        try:
            self.client.upsert(
                collection_name=name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search for similar vectors"""
        name = collection_name or self.collection_name
        
        try:
            results = self.client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            return [(r.id, r.score, r.payload) for r in results]
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[IndexStats]:
        """Get collection statistics"""
        name = collection_name or self.collection_name
        
        try:
            info = self.client.get_collection(name)
            return IndexStats(
                collection_name=name,
                vector_count=info.points_count,
                dimension=info.config.params.vectors.size,
                distance_metric=DistanceMetric.COSINE  # Simplified
            )
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None
    
    async def delete_collection(
        self,
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete a collection"""
        name = collection_name or self.collection_name
        
        try:
            self.client.delete_collection(name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
