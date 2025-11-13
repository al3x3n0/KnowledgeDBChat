"""
General helper functions.
"""

import hashlib
import uuid
from typing import Any, Dict, Optional
from uuid import UUID


def generate_uuid() -> UUID:
    """
    Generate a new UUID.
    
    Returns:
        UUID object
    """
    return uuid.uuid4()


def generate_uuid_string() -> str:
    """
    Generate a new UUID as string.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (sha256, md5, etc.)
        
    Returns:
        Hexadecimal hash string
    """
    if algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def safe_get(dictionary: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        dictionary: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at path or default
    """
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        
        if result is None:
            return default
    
    return result


def remove_none_values(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove None values from dictionary.
    
    Args:
        dictionary: Dictionary to clean
        
    Returns:
        Dictionary without None values
    """
    return {k: v for k, v in dictionary.items() if v is not None}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters for filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    return filename


def parse_bool(value: Any) -> bool:
    """
    Parse boolean value from various formats.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return False

