from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    source_type: SourceType
    file_size: int
    content_type: str
    raw_text: str
    metadata: Dict[str, Any]
    created_at: datetime


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    start_pos: int
    end_pos: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkingRequest(BaseModel):
    document_id: str
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = 1000
    chunk_overlap: int = 200


class ChunkingResponse(BaseModel):
    document_id: str
    chunks: List[Chunk]
    total_chunks: int
    strategy: ChunkingStrategy


class EmbeddingRequest(BaseModel):
    chunk_ids: List[str]
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    model: Optional[str] = None
    batch_size: int = 32


class Embedding(BaseModel):
    chunk_id: str
    vector_id: str
    values: List[float]
    dimension: int
    model: str
    document_id: str


class EmbeddingResponse(BaseModel):
    embeddings: List[Embedding]
    model: str
    dimension: int
    total_embedded: int


class IndexingRequest(BaseModel):
    collection_name: Optional[str] = None
    distance_metric: DistanceMetric = DistanceMetric.COSINE


class IndexStats(BaseModel):
    collection_name: str
    vector_count: int
    dimension: int
    distance_metric: DistanceMetric


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    embedding_id: str


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    total_found: int
    latency_ms: int


class RerankingRequest(BaseModel):
    query: str
    chunks: List[str]
    top_k: int = 5
    model: Optional[str] = None


class RerankedResult(BaseModel):
    chunk_id: str
    original_rank: int
    reranked_score: float
    content: str


class RerankingResponse(BaseModel):
    query: str
    results: List[RerankedResult]
    model: str


class GenerationRequest(BaseModel):
    query: str
    context_chunks: List[str]
    provider: LLMProvider = LLMProvider.OPENAI
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: Optional[str] = None


class GenerationResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    chunks_used: int


class EvaluationMetrics(BaseModel):
    latency: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    confidence: int
    faithfulness: int
    relevance: int
    context_utilization: int
    response_quality: int


class LogEntry(BaseModel):
    timestamp: datetime
    stage: str
    message: str
    level: Literal["info", "warning", "error"] = "info"
    details: Optional[Dict[str, Any]] = None
