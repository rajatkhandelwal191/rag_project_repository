from fastapi import APIRouter, HTTPException
from typing import Optional
import re
import unicodedata
import logging
import traceback

from app.api.upload import uploaded_files as document_store
from app.api.chunking import chunk_store
from app.api.embedding import embedding_store

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preprocessing")


class TextPreprocessor:
    """Text preprocessing and cleaning service"""
    
    @staticmethod
    def clean_text(
        text: str,
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_phone_numbers: bool = False,
        ocr_cleanup: bool = False,
        preserve_punctuation: bool = True
    ) -> str:
        """Clean and normalize text"""
        
        # OCR cleanup (remove common OCR artifacts)
        if ocr_cleanup:
            text = TextPreprocessor.fix_ocr_artifacts(text)
        
        # Normalize unicode
        if normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        if remove_phone_numbers:
            text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Clean punctuation spacing if not preserving
        if not preserve_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def fix_ocr_artifacts(text: str) -> str:
        """Fix common OCR errors"""
        # Common OCR replacements
        fixes = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '—': '-',
            '–': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '…': '...',
        }
        
        for wrong, right in fixes.items():
            text = text.replace(wrong, right)
        
        # Fix hyphenation (word- \n word -> wordword)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Fix common character confusions
        text = re.sub(r'0(?=\D)', 'O', text)  # 0 -> O before non-digits
        text = re.sub(r'l(?=\d)', '1', text)  # l -> 1 before digits
        
        return text
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Get statistics about the text"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "line_count": text.count('\n') + 1
        }


preprocessor = TextPreprocessor()


@router.post("/{document_id}")
async def preprocess_document(
    document_id: str,
    lowercase: bool = False,
    remove_extra_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_phone_numbers: bool = False,
    ocr_cleanup: bool = False,
    preserve_punctuation: bool = True
):
    """
    Preprocess and clean document text.
    """
    logger.info(f"Preprocess document called for {document_id} with options: lowercase={lowercase}, ocr_cleanup={ocr_cleanup}")
    
    # Check if document exists
    if document_id not in document_store:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = document_store[document_id]
    raw_text = document.get("raw_text", "")
    
    if not raw_text:
        logger.error(f"Document {document_id} has no text content")
        raise HTTPException(status_code=400, detail="Document has no text content")
    
    # Get stats before cleaning
    stats_before = preprocessor.get_text_stats(raw_text)
    logger.info(f"Stats before cleaning: {stats_before}")
    
    # Clean the text
    try:
        logger.info("Starting text cleaning...")
        cleaned_text = preprocessor.clean_text(
            raw_text,
            lowercase=lowercase,
            remove_extra_whitespace=remove_extra_whitespace,
            normalize_unicode=normalize_unicode,
            remove_urls=remove_urls,
            remove_emails=remove_emails,
            remove_phone_numbers=remove_phone_numbers,
            ocr_cleanup=ocr_cleanup,
            preserve_punctuation=preserve_punctuation
        )
        logger.info(f"Text cleaned. Original: {len(raw_text)} chars, Cleaned: {len(cleaned_text)} chars")
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error cleaning text: {str(e)}")
    
    # Get stats after cleaning
    stats_after = preprocessor.get_text_stats(cleaned_text)
    
    # Store cleaned text back in document
    document["cleaned_text"] = cleaned_text
    document["preprocessing_applied"] = {
        "lowercase": lowercase,
        "remove_extra_whitespace": remove_extra_whitespace,
        "normalize_unicode": normalize_unicode,
        "remove_urls": remove_urls,
        "remove_emails": remove_emails,
        "remove_phone_numbers": remove_phone_numbers,
        "ocr_cleanup": ocr_cleanup,
        "preserve_punctuation": preserve_punctuation
    }
    
    logger.info(f"Preprocessing complete for document: {document_id}")
    
    return {
        "document_id": document_id,
        "raw_text_preview": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
        "cleaned_text_preview": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "processing_applied": document["preprocessing_applied"]
    }


@router.get("/{document_id}/text")
async def get_cleaned_text(document_id: str):
    """Get the cleaned text for a document"""
    logger.info(f"Get cleaned text called for document: {document_id}")
    
    if document_id not in document_store:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = document_store[document_id]
    cleaned_text = document.get("cleaned_text")
    
    if not cleaned_text:
        logger.warning(f"No cleaned text found for document: {document_id}")
        raise HTTPException(status_code=404, detail="No cleaned text found. Run preprocessing first.")
    
    logger.info(f"Returning cleaned text for document: {document_id}")
    return {
        "document_id": document_id,
        "cleaned_text": cleaned_text,
        "preprocessing_applied": document.get("preprocessing_applied", {})
    }


@router.post("/preview")
async def preview_preprocessing(
    text: str,
    lowercase: bool = False,
    remove_extra_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_phone_numbers: bool = False,
    ocr_cleanup: bool = False,
    preserve_punctuation: bool = True
):
    """
    Preview text preprocessing without storing results.
    Useful for testing different cleaning options.
    """
    logger.info("Preview preprocessing called")
    
    if not text:
        logger.warning("No text provided for preview")
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Get stats before
    stats_before = preprocessor.get_text_stats(text)
    
    # Clean text
    try:
        logger.info("Cleaning text for preview...")
        cleaned = preprocessor.clean_text(
            text,
            lowercase=lowercase,
            remove_extra_whitespace=remove_extra_whitespace,
            normalize_unicode=normalize_unicode,
            remove_urls=remove_urls,
            remove_emails=remove_emails,
            remove_phone_numbers=remove_phone_numbers,
            ocr_cleanup=ocr_cleanup,
            preserve_punctuation=preserve_punctuation
        )
        logger.info("Text cleaned for preview")
    except Exception as e:
        logger.error(f"Error cleaning text for preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error cleaning text: {str(e)}")
    
    # Get stats after
    stats_after = preprocessor.get_text_stats(cleaned)
    
    logger.info(f"Preview complete. Reduction: {round((1 - stats_after['char_count'] / max(stats_before['char_count'], 1)) * 100, 2)}%")
    
    return {
        "original_preview": text[:1000] + "..." if len(text) > 1000 else text,
        "cleaned_preview": cleaned[:1000] + "..." if len(cleaned) > 1000 else cleaned,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "reduction_percent": round((1 - stats_after["char_count"] / max(stats_before["char_count"], 1)) * 100, 2)
    }
