from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api import upload, chunking, embedding, retrieval, generation, preprocessing, experiment

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="RAG Pipeline Backend API",
    version="1.0.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(preprocessing.router, prefix="/api/v1", tags=["preprocessing"])
app.include_router(chunking.router, prefix="/api/v1", tags=["chunking"])
app.include_router(embedding.router, prefix="/api/v1", tags=["embedding"])
app.include_router(retrieval.router, prefix="/api/v1", tags=["retrieval"])
app.include_router(generation.router, prefix="/api/v1", tags=["generation"])
app.include_router(experiment.router, prefix="/api/v1", tags=["experiment"])


@app.get("/")
async def root():
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
