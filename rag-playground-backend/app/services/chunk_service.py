import uuid
import re
from typing import List, Dict, Any, Optional
from app.models.schemas import Chunk, ChunkingStrategy


class ChunkService:
    def __init__(self):
        self.chunks_store: Dict[str, List[Chunk]] = {}
    
    def create_chunks(
        self,
        document_id: str,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Chunk]:
        """Create chunks from text using specified strategy"""
        
        if strategy == ChunkingStrategy.FIXED:
            chunks = self._fixed_size_chunking(text, document_id, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._recursive_chunking(text, document_id, chunk_size, chunk_overlap)
        else:  # SEMANTIC - simplified version
            chunks = self._semantic_chunking(text, document_id, chunk_size)
        
        self.chunks_store[document_id] = chunks
        return chunks
    
    def _fixed_size_chunking(
        self,
        text: str,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Fixed-size chunking with overlap"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending punctuation
                match = re.search(r'[.!?]\s+', text[end-50:end+50])
                if match:
                    end = end - 50 + match.end()
            
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(
                    id=f"chunk_{document_id}_{chunk_id}",
                    document_id=document_id,
                    content=content,
                    start_pos=start,
                    end_pos=end,
                    token_count=len(content.split()),
                    metadata={"strategy": "fixed"}
                ))
                chunk_id += 1
            
            start = end - chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _recursive_chunking(
        self,
        text: str,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Recursive chunking that respects paragraphs and sections"""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        current_pos = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > chunk_size and current_chunk:
                content = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{document_id}_{chunk_id}",
                    document_id=document_id,
                    content=content,
                    start_pos=current_pos,
                    end_pos=current_pos + len(content),
                    token_count=len(content.split()),
                    metadata={"strategy": "recursive", "paragraphs": len(current_chunk)}
                ))
                chunk_id += 1
                current_pos += len(content) + 1
                
                # Keep overlap
                overlap_size = 0
                overlap_paras = []
                for p in reversed(current_chunk):
                    if overlap_size + len(p) < chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                current_chunk = overlap_paras
                current_size = overlap_size
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add remaining content
        if current_chunk:
            content = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{document_id}_{chunk_id}",
                document_id=document_id,
                content=content,
                start_pos=current_pos,
                end_pos=current_pos + len(content),
                token_count=len(content.split()),
                metadata={"strategy": "recursive", "paragraphs": len(current_chunk)}
            ))
        
        return chunks
    
    def _semantic_chunking(
        self,
        text: str,
        document_id: str,
        chunk_size: int
    ) -> List[Chunk]:
        """Semantic chunking based on sentence boundaries"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        current_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_size = len(sent)
            
            if current_size + sent_size > chunk_size and current_chunk:
                content = ' '.join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{document_id}_{chunk_id}",
                    document_id=document_id,
                    content=content,
                    start_pos=current_pos,
                    end_pos=current_pos + len(content),
                    token_count=len(content.split()),
                    metadata={"strategy": "semantic", "sentences": len(current_chunk)}
                ))
                chunk_id += 1
                current_pos += len(content) + 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sent)
            current_size += sent_size
        
        # Add remaining sentences
        if current_chunk:
            content = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{document_id}_{chunk_id}",
                document_id=document_id,
                content=content,
                start_pos=current_pos,
                end_pos=current_pos + len(content),
                token_count=len(content.split()),
                metadata={"strategy": "semantic", "sentences": len(current_chunk)}
            ))
        
        return chunks
    
    def get_chunks(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        return self.chunks_store.get(document_id, [])
    
    def get_chunk(self, document_id: str, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk"""
        chunks = self.chunks_store.get(document_id, [])
        for chunk in chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
