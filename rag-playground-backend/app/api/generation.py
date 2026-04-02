from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging
import traceback

from app.services.llm_service import LLMService
from app.models.schemas import GenerationRequest, GenerationResponse, LLMProvider

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation")

# Initialize service
llm_service = LLMService()


@router.post("/", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate response using LLM with context"""
    logger.info(f"Generate called with query: '{request.query}', provider: {request.provider}, model: {request.model}")
    
    if not request.query:
        logger.warning("Query is empty")
        raise HTTPException(status_code=400, detail="Query is required")
    
    if not request.context_chunks:
        logger.warning("No context chunks provided")
        raise HTTPException(status_code=400, detail="Context chunks are required")
    
    try:
        logger.info(f"Generating response with {len(request.context_chunks)} context chunks")
        response = await llm_service.generate_response(
            query=request.query,
            context_chunks=request.context_chunks,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        logger.info(f"Response generated successfully. Tokens used: {response.input_tokens} input, {response.output_tokens} output")
        return response
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.post("/evaluate")
async def evaluate_response(request: dict):
    """Evaluate generation metrics"""
    from app.models.schemas import EvaluationMetrics
    
    logger.info(f"Evaluate response called with request: {request}")
    
    latency = request.get("latency", 0)
    input_tokens = request.get("input_tokens", 0)
    output_tokens = request.get("output_tokens", 0)
    
    # Calculate estimated cost (OpenAI GPT-3.5-turbo pricing)
    input_cost = (input_tokens / 1000) * 0.0015
    output_cost = (output_tokens / 1000) * 0.002
    estimated_cost = input_cost + output_cost
    
    # Mock metrics - in production, calculate based on actual results
    metrics = EvaluationMetrics(
        latency=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=round(estimated_cost, 6),
        confidence=85,
        faithfulness=78,
        relevance=82,
        context_utilization=min(100, int((input_tokens / 8192) * 100)),
        response_quality=80
    )
    
    logger.info(f"Evaluation metrics calculated: {metrics}")
    return metrics


@router.get("/models")
async def list_models():
    """List available LLM models"""
    logger.info("List models called")
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "models": [
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                    {"id": "gpt-4", "name": "GPT-4"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"}
                ]
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "models": [
                    {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet"},
                    {"id": "claude-3-opus", "name": "Claude 3 Opus"},
                    {"id": "claude-3-haiku", "name": "Claude 3 Haiku"}
                ]
            },
            {
                "id": "cohere",
                "name": "Cohere",
                "models": [
                    {"id": "command", "name": "Command"},
                    {"id": "command-light", "name": "Command Light"}
                ]
            },
            {
                "id": "google",
                "name": "Google",
                "models": [
                    {"id": "gemini-pro", "name": "Gemini Pro"},
                    {"id": "gemini-ultra", "name": "Gemini Ultra"}
                ]
            }
        ]
    }


@router.post("/templates")
async def get_prompt_templates():
    """Get available prompt templates"""
    logger.info("Get prompt templates called")
    return {
        "templates": [
            {
                "id": "default",
                "name": "Default",
                "system_prompt": "You are a helpful assistant. Answer the user's question based only on the provided context.",
                "user_prompt_template": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            },
            {
                "id": "detailed",
                "name": "Detailed Analysis",
                "system_prompt": "You are a detailed analytical assistant. Provide thorough answers with explanations.",
                "user_prompt_template": "Based on the following context:\n\n{context}\n\nProvide a detailed answer to: {query}"
            },
            {
                "id": "concise",
                "name": "Concise",
                "system_prompt": "You are a concise assistant. Provide brief, direct answers.",
                "user_prompt_template": "Context: {context}\n\nAnswer concisely: {query}"
            },
            {
                "id": "step_by_step",
                "name": "Step by Step",
                "system_prompt": "You are a methodical assistant. Break down your reasoning step by step.",
                "user_prompt_template": "Given this context:\n{context}\n\nWalk through the answer to: {query}\n\nStep 1:"
            }
        ]
    }
