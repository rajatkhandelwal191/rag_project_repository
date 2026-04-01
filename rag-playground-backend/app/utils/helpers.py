import hashlib
import uuid
from typing import Any, Dict
from datetime import datetime


def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())


def hash_text(text: str) -> str:
    """Create SHA-256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime to ISO string"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count (1 token ≈ 4 chars for English)"""
    return len(text) // 4 + 1


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    return dictionary.get(key, default)


def chunk_list(lst: list, chunk_size: int):
    """Split list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata for JSON serialization"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = list(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_metadata(value)
        else:
            cleaned[key] = str(value)
    return cleaned
