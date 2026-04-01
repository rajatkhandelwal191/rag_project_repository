# RAG Pipeline Backend

A FastAPI-based backend for the RAG (Retrieval-Augmented Generation) Pipeline, supporting document upload, chunking, embedding, retrieval, and generation with multiple LLM providers.

## Features

- **Document Upload**: Support for PDF, TXT, MD, HTML files
- **Chunking Strategies**: Fixed-size, Semantic, Recursive chunking
- **Embeddings**: Cloud-based via Gemini (Google) or OpenAI
- **Vector Store**: Qdrant (local or cloud)
- **LLM Generation**: OpenAI, Gemini, Groq support
- **Reranking**: Basic keyword-based reranking

## Project Structure

```
rag-playground-backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Settings and configuration
│   ├── api/                    # API routes
│   │   ├── upload.py           # File upload endpoints
│   │   ├── chunking.py         # Text chunking endpoints
│   │   ├── embedding.py        # Embedding generation & indexing
│   │   ├── retrieval.py        # Vector search & reranking
│   │   └── generation.py       # LLM generation endpoints
│   ├── services/               # Business logic
│   │   ├── pdf_service.py      # PDF extraction
│   │   ├── chunk_service.py    # Chunking logic
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── qdrant_service.py   # Vector DB operations
│   │   └── llm_service.py      # LLM integration
│   ├── models/                 # Pydantic schemas
│   │   └── schemas.py          # Request/response models
│   └── utils/                  # Helpers
│       └── helpers.py          # Utility functions
├── uploads/                    # File storage directory
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .env.example                # Example environment file
└── .gitignore                  # Git ignore rules
```

## Quick Start

### 1. Setup Virtual Environment

```bash
cd rag-playground-backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example file and update with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Environment Mode (LOCAL or CLOUD)
ENV=LOCAL

# Qdrant (Local - for development)
QDRANT_URL=http://localhost:6333

# Or Qdrant Cloud (for production)
QDRANT_CLOUD_URL=https://your-cluster.cloud.qdrant.io
QDRANT_CLOUD_API_KEY=your_qdrant_api_key

# Required API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Optional
COHERE_API_KEY=...
SUPABASE_URL=postgresql://...
```

### 4. Start Qdrant (Local Mode)

If running in `LOCAL` mode, start Qdrant with Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or download and run the binary from [qdrant.tech](https://qdrant.tech).

### 5. Run the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```

The API will be available at `http://localhost:8080`

API documentation (Swagger UI): `http://localhost:8080/docs`

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload/` | POST | Upload PDF/text files |
| `/api/v1/chunking/` | POST | Create chunks from document |
| `/api/v1/embedding/` | POST | Generate embeddings |
| `/api/v1/embedding/index` | POST | Index to vector DB |
| `/api/v1/retrieval/` | POST | Search similar chunks |
| `/api/v1/generation/` | POST | Generate LLM response |

## Environment Modes

### LOCAL Mode
- Uses local Qdrant instance
- Requires `QDRANT_URL=http://localhost:6333`
- Good for development

### CLOUD Mode
- Uses Qdrant Cloud cluster
- Requires `QDRANT_CLOUD_URL` and `QDRANT_CLOUD_API_KEY`
- Good for production

Switch modes by setting `ENV=LOCAL` or `ENV=CLOUD` in `.env`

## Configuration Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENV` | No | LOCAL | Run mode (LOCAL/CLOUD) |
| `OPENAI_API_KEY` | Yes* | - | For OpenAI LLM/embeddings |
| `GOOGLE_API_KEY` | Yes* | - | For Gemini LLM/embeddings |
| `GROQ_API_KEY` | No | - | For Groq LLM |
| `QDRANT_URL` | LOCAL | localhost:6333 | Local Qdrant URL |
| `QDRANT_CLOUD_URL` | CLOUD | - | Cloud Qdrant URL |
| `QDRANT_CLOUD_API_KEY` | CLOUD | - | Cloud Qdrant key |
| `CHUNK_SIZE` | No | 1000 | Default chunk size |
| `CHUNK_OVERLAP` | No | 200 | Chunk overlap |

*At least one of OPENAI_API_KEY or GOOGLE_API_KEY is required

## Troubleshooting

### ModuleNotFoundError

Install missing packages:
```bash
pip install <missing-package>
```

### Qdrant Connection Error

Ensure Qdrant is running:
```bash
# Check if port 6333 is open
curl http://localhost:6333

# Or start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### API Key Errors

Verify keys are set in `.env` file and file is in the correct location.

## Development

### Adding New LLM Provider

1. Add provider config to `app/config.py`
2. Add provider enum to `app/models/schemas.py`
3. Implement in `app/services/llm_service.py`
4. Update `app/api/generation.py` if needed

### Adding New Embedding Provider

1. Add config to `app/config.py`
2. Update `app/services/embedding_service.py`
3. Add provider to `app/models/schemas.py`

## Testing

See [API_TESTING.md](API_TESTING.md) for detailed API testing examples.

## License

MIT
