from typing import List, Optional, Dict, Any
import logging
import traceback
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import get_settings
from app.models.schemas import DistanceMetric, IndexStats, RetrievedChunk
from app.models.schemas import Chunk

# Setup logger
logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        settings = get_settings()
        self.env = settings.env.upper()
        
        # Use cloud or local Qdrant based on environment
        if self.env == "CLOUD" and settings.qdrant_cloud_url:
            self.url = settings.qdrant_cloud_url
            self.api_key = settings.qdrant_cloud_api_key or None
            logger.info(f"Using Qdrant Cloud: {self.url}")
        else:
            self.url = settings.qdrant_url
            self.api_key = settings.qdrant_api_key or None
            logger.info(f"Using Local Qdrant: {self.url}")
            
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
        """Create a new collection, recreating if dimension mismatch"""
        name = collection_name or self.collection_name
        
        # Map distance metric
        distance_map = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT
        }
        
        try:
            # Check if collection exists
            try:
                existing = self.client.get_collection(name)
                existing_dim = existing.config.params.vectors.size
                
                # If dimension matches, just return True - collection is fine
                if existing_dim == self.dimension:
                    logger.info(f"Collection {name} already exists with correct dimension {self.dimension}")
                    return True
                
                # If dimension mismatch, delete and recreate
                logger.warning(f"Collection {name} exists with dim {existing_dim}, but need {self.dimension}. Recreating...")
                self.client.delete_collection(name)
            except Exception:
                # Collection doesn't exist, which is fine - we'll create it
                pass
            
            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )
            logger.info(f"Collection {name} created with dimension {self.dimension}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            logger.error(f"Qdrant URL: {self.url}")
            logger.error(f"Dimension: {self.dimension}")
            logger.error(traceback.format_exc())
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
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            query_filter = None
            if filters:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        ) for key, value in filters.items()
                    ]
                )
            
            # Use query_points (newer API) or search
            results = self.client.query_points(
                collection_name=name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter
            ).points
            
            return [(r.id, r.score, r.payload) for r in results]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            logger.error(traceback.format_exc())
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
