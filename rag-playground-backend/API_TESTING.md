# API Testing Guide

Complete guide for testing the RAG Pipeline Backend API endpoints.

## Base URL

```
http://localhost:8080/api/v1
```

## API Documentation

Interactive docs available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

---

## 1. Upload API

### POST `/upload/`

Upload a document (PDF, TXT, MD, HTML).

#### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Document file to upload |

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/upload/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

#### Response

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_name": "document.pdf",
  "source_type": "pdf",
  "file_size": 1024567,
  "content_type": "application/pdf",
  "raw_text": "Full extracted text from document...",
  "metadata": {
    "filename": "document.pdf",
    "size": 1024567,
    "extension": ".pdf",
    "page_count": 10
  },
  "created_at": "2024-01-15T10:30:00"
}
```

#### Python Example

```python
import requests

url = "http://localhost:8080/api/v1/upload/"
files = {"file": open("document.pdf", "rb")}

response = requests.post(url, files=files)
print(response.json())
document_id = response.json()["document_id"]
```

---

## 2. Chunking API

### POST `/chunking/`

Create chunks from uploaded document.

#### Request Body

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "strategy": "fixed",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `document_id` | string | Yes | - | ID from upload response |
| `strategy` | enum | No | "fixed" | Options: "fixed", "semantic", "recursive" |
| `chunk_size` | int | No | 1000 | Characters per chunk |
| `chunk_overlap` | int | No | 200 | Overlap between chunks |

#### Strategies

- **fixed**: Fixed-size chunks with overlap
- **semantic**: Split by sentence boundaries
- **recursive**: Respects paragraphs/sections

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/chunking/" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "strategy": "fixed",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

#### Response

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks": [
    {
      "id": "chunk_550e8400-e29b-41d4-a716-446655440000_0",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "First chunk content...",
      "start_pos": 0,
      "end_pos": 1000,
      "token_count": 250,
      "metadata": {
        "strategy": "fixed"
      }
    },
    {
      "id": "chunk_550e8400-e29b-41d4-a716-446655440000_1",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Second chunk content...",
      "start_pos": 800,
      "end_pos": 1800,
      "token_count": 240,
      "metadata": {
        "strategy": "fixed"
      }
    }
  ],
  "total_chunks": 12,
  "strategy": "fixed"
}
```

#### Python Example

```python
import requests

url = "http://localhost:8080/api/v1/chunking/"
payload = {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "strategy": "semantic",
    "chunk_size": 500,
    "chunk_overlap": 100
}

response = requests.post(url, json=payload)
chunks = response.json()["chunks"]
chunk_ids = [c["id"] for c in chunks]
```

---

## 3. Embedding API

### POST `/embedding/`

Generate embeddings for chunks.

#### Request Body

```json
{
  "chunk_ids": ["chunk_xxx_0", "chunk_xxx_1", "chunk_xxx_2"],
  "provider": "google",
  "model": "gemini-embedding-001",
  "batch_size": 32
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `chunk_ids` | array | Yes | - | List of chunk IDs |
| `provider` | enum | No | "google" | Options: "google", "openai", "cohere", "local" |
| `model` | string | No | auto | Model name (depends on provider) |
| `batch_size` | int | No | 32 | Processing batch size |

#### Provider Models

| Provider | Default Model | Dimensions |
|----------|---------------|------------|
| Google | gemini-embedding-001 | 768 |
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| OpenAI | text-embedding-ada-002 | 1536 |

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/embedding/" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_ids": ["chunk_550e8400_0", "chunk_550e8400_1"],
    "provider": "google",
    "model": "gemini-embedding-001"
  }'
```

#### Response

```json
{
  "embeddings": [
    {
      "chunk_id": "chunk_550e8400_0",
      "vector_id": "vec_chunk_550e8400_0",
      "values": [0.023, -0.156, 0.892, ...],
      "dimension": 768,
      "model": "gemini-embedding-001",
      "document_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ],
  "model": "gemini-embedding-001",
  "dimension": 768,
  "total_embedded": 12
}
```

---

## 4. Indexing API

### POST `/embedding/index`

Index embeddings to Qdrant vector database.

#### Request Body

```json
{
  "collection_name": "rag_documents",
  "distance_metric": "cosine"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `collection_name` | string | No | "rag_documents" | Qdrant collection name |
| `distance_metric` | enum | No | "cosine" | Options: "cosine", "euclidean", "dot_product" |

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/embedding/index" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "distance_metric": "cosine"
  }'
```

#### Response

```json
{
  "message": "Embeddings indexed successfully",
  "indexed_count": 12,
  "collection": "my_documents"
}
```

### GET `/embedding/index/stats`

Get collection statistics.

#### cURL Example

```bash
curl "http://localhost:8080/api/v1/embedding/index/stats?collection_name=rag_documents"
```

#### Response

```json
{
  "collection_name": "rag_documents",
  "vector_count": 12,
  "dimension": 768,
  "distance_metric": "cosine"
}
```

---

## 5. Retrieval API

### POST `/retrieval/`

Search for relevant chunks using vector similarity.

#### Request Body

```json
{
  "query": "What is the main topic of this document?",
  "top_k": 5,
  "collection_name": "rag_documents",
  "filters": null
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `top_k` | int | No | 5 | Number of results |
| `collection_name` | string | No | "rag_documents" | Qdrant collection |
| `filters` | object | No | null | Metadata filters |

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/retrieval/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "top_k": 5
  }'
```

#### Response

```json
{
  "query": "What are the key findings?",
  "results": [
    {
      "chunk": {
        "id": "chunk_550e8400_3",
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "The key findings include...",
        "start_pos": 2400,
        "end_pos": 3400,
        "token_count": 245,
        "metadata": {"strategy": "fixed"}
      },
      "score": 0.8923,
      "embedding_id": "vec_chunk_550e8400_3"
    }
  ],
  "total_found": 5,
  "latency_ms": 45
}
```

---

## 6. Reranking API

### POST `/retrieval/rerank`

Rerank retrieved chunks for better relevance.

#### Request Body

```json
{
  "query": "What are the key findings?",
  "chunks": ["The first finding is...", "Another important point..."],
  "top_k": 3
}
```

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/retrieval/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "chunks": ["First chunk text...", "Second chunk text..."],
    "top_k": 3
  }'
```

#### Response

```json
{
  "query": "What are the key findings?",
  "reranked_results": [
    {
      "original_rank": 2,
      "content": "Another important point...",
      "reranked_score": 0.75
    },
    {
      "original_rank": 0,
      "content": "The first finding is...",
      "reranked_score": 0.60
    }
  ],
  "model": "cross-encoder"
}
```

---

## 7. Generation API

### POST `/generation/`

Generate LLM response using retrieved context.

#### Request Body

```json
{
  "query": "Summarize the main points",
  "context_chunks": [
    "First relevant chunk...",
    "Second relevant chunk..."
  ],
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 1024,
  "system_prompt": "You are a helpful assistant..."
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | User question |
| `context_chunks` | array | Yes | - | Retrieved context chunks |
| `provider` | enum | No | "openai" | Options: "openai", "google", "anthropic", "cohere" |
| `model` | string | No | provider default | Model name |
| `temperature` | float | No | 0.7 | Creativity (0-2) |
| `max_tokens` | int | No | 1024 | Max response length |
| `system_prompt` | string | No | default | System instructions |

#### Provider Defaults

| Provider | Default Model |
|----------|---------------|
| OpenAI | gpt-3.5-turbo |
| Google | gemini-2.0-flash |
| Groq | llama-3.1-8b-instant |

#### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/generation/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings in this document?",
    "context_chunks": [
      "The study found significant improvements...",
      "Results showed a 25% increase in..."
    ],
    "provider": "google",
    "temperature": 0.5
  }'
```

#### Response

```json
{
  "response": "Based on the context provided, the key findings are: 1) Significant improvements were observed... 2) Results showed a 25% increase...",
  "input_tokens": 450,
  "output_tokens": 128,
  "latency_ms": 1234,
  "model": "gemini-2.0-flash",
  "chunks_used": 2
}
```

### GET `/generation/models`

List available LLM models.

#### cURL Example

```bash
curl "http://localhost:8080/api/v1/generation/models"
```

#### Response

```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "models": [
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"}
      ]
    },
    {
      "id": "google",
      "name": "Google",
      "models": [
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"}
      ]
    }
  ]
}
```

---

## Complete Pipeline Example

Here's a complete example running the full RAG pipeline:

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"

# 1. Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload/", files={"file": f})
    document_id = response.json()["document_id"]
    print(f"Uploaded: {document_id}")

# 2. Create chunks
response = requests.post(f"{BASE_URL}/chunking/", json={
    "document_id": document_id,
    "strategy": "fixed",
    "chunk_size": 1000
})
chunks = response.json()["chunks"]
chunk_ids = [c["id"] for c in chunks]
print(f"Created {len(chunks)} chunks")

# 3. Generate embeddings
response = requests.post(f"{BASE_URL}/embedding/", json={
    "chunk_ids": chunk_ids,
    "provider": "google",
    "model": "gemini-embedding-001"
})
print(f"Generated embeddings")

# 4. Index to Qdrant
response = requests.post(f"{BASE_URL}/embedding/index", json={
    "collection_name": "my_collection",
    "distance_metric": "cosine"
})
print(f"Indexed: {response.json()['indexed_count']} vectors")

# 5. Query
query = "What are the main conclusions?"
response = requests.post(f"{BASE_URL}/retrieval/", json={
    "query": query,
    "top_k": 5,
    "collection_name": "my_collection"
})
results = response.json()["results"]
context_chunks = [r["chunk"]["content"] for r in results]
print(f"Retrieved {len(results)} chunks")

# 6. Generate answer
response = requests.post(f"{BASE_URL}/generation/", json={
    "query": query,
    "context_chunks": context_chunks,
    "provider": "google",
    "temperature": 0.7
})
answer = response.json()["response"]
print(f"\nAnswer: {answer}")
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | - |
| 400 | Bad Request | Missing required fields, invalid JSON |
| 404 | Not Found | Document/chunk not found |
| 413 | Payload Too Large | File exceeds MAX_FILE_SIZE |
| 500 | Server Error | Missing API keys, Qdrant not running |

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

---

## Testing with Postman

1. Import the endpoints manually or use the OpenAPI spec at `/openapi.json`
2. Set `Content-Type: application/json` for POST requests
3. For file upload, use `form-data` with a `file` field
4. Save document_id from upload response to use in subsequent requests
