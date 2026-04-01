import os
import time
from typing import List, Optional
from groq import Groq
import openai
import google.generativeai as genai
import tiktoken
from app.config import get_settings
from app.models.schemas import LLMProvider, GenerationResponse


class LLMService:
    def __init__(self):
        settings = get_settings()
        self.openai_api_key = settings.openai_api_key
        self.cohere_api_key = settings.cohere_api_key
        self.google_api_key = settings.google_api_key
        self.groq_api_key = settings.groq_api_key
        self.groq_model = settings.groq_model
        self.gemini_model = settings.gemini_chat_model
        self.default_model = settings.default_llm_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        
        # Initialize clients if keys available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
        
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            self.groq_client = None
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback: approximate tokens
            return len(text.split()) * 1.3
    
    async def generate_response(
        self,
        query: str,
        context_chunks: List[str],
        provider: LLMProvider = LLMProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> GenerationResponse:
        """Generate response using LLM"""
        
        if model is None:
            model = self.default_model
        
        # Build context
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer the user's question based only on the provided context."
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        start_time = time.time()
        
        if provider == LLMProvider.OPENAI and self.openai_api_key:
            response_text = await self._generate_openai(
                messages, model, temperature, max_tokens
            )
        elif provider == LLMProvider.GOOGLE and self.google_api_key:
            response_text = await self._generate_gemini(
                messages, model, temperature, max_tokens
            )
        else:
            # Fallback to mock response for development
            response_text = self._generate_mock(query, context_chunks)
        
        latency = int((time.time() - start_time) * 1000)
        
        # Count tokens
        input_text = "\n".join([m["content"] for m in messages])
        input_tokens = int(self.count_tokens(input_text, model))
        output_tokens = int(self.count_tokens(response_text, model))
        
        return GenerationResponse(
            response=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            model=model,
            chunks_used=len(context_chunks)
        )
    
    async def _generate_openai(
        self,
        messages: List[dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI API"""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            # Fallback to mock
            return self._generate_mock(messages[-1]["content"], [])
    
    async def _generate_gemini(
        self,
        messages: List[dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Gemini API"""
        try:
            # Convert messages to Gemini format
            system_content = ""
            user_content = ""
            
            for m in messages:
                if m["role"] == "system":
                    system_content = m["content"]
                elif m["role"] == "user":
                    user_content = m["content"]
            
            gen_model = genai.GenerativeModel(model)
            response = gen_model.generate_content(
                f"{system_content}\n\n{user_content}",
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._generate_mock(messages[-1]["content"] if messages else "", [])
    
    def _generate_mock(self, query: str, context_chunks: List[str]) -> str:
        """Generate mock response for development"""
        if context_chunks:
            summary = context_chunks[0][:200] if context_chunks[0] else "No specific information"
            return f"Based on the provided context, I found relevant information: {summary}..."
        else:
            return f"I don't have enough context to answer '{query}' accurately. Please provide more information."
    
    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 5,
        model: Optional[str] = None
    ) -> List[tuple]:
        """Rerank chunks based on relevance to query"""
        # Simplified reranking: return with scores
        # In production, use a cross-encoder or dedicated reranking model
        
        results = []
        for i, chunk in enumerate(chunks[:top_k]):
            # Simple scoring based on keyword overlap
            query_words = set(query.lower().split())
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            score = overlap / max(len(query_words), 1)
            
            results.append((i, chunk, score))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
