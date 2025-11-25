"""
Input validation utilities.
"""

import re
from typing import Optional, List
from urllib.parse import urlparse
from pathlib import Path


# Allowed file types for uploads
ALLOWED_FILE_TYPES = {
    'application/pdf': ['.pdf'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
    'application/vnd.ms-powerpoint': ['.ppt'],
    'text/plain': ['.txt'],
    'text/html': ['.html', '.htm'],
    'text/markdown': ['.md', '.markdown'],
    # Video formats
    'video/mp4': ['.mp4'],
    'video/x-msvideo': ['.avi'],
    'video/x-matroska': ['.mkv'],
    'video/quicktime': ['.mov'],
    'video/webm': ['.webm'],
    'video/x-flv': ['.flv'],
    'video/x-ms-wmv': ['.wmv'],
    # Audio formats
    'audio/mpeg': ['.mp3'],
    'audio/wav': ['.wav'],
    'audio/x-m4a': ['.m4a'],
    'audio/flac': ['.flac'],
    'audio/ogg': ['.ogg'],
    'audio/aac': ['.aac'],
}

ALLOWED_EXTENSIONS = [
    # Documents
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt', '.html', '.htm', '.md', '.markdown',
    # Video
    '.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv',
    # Audio
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'
]


def validate_file_type(filename: str, content_type: Optional[str] = None) -> bool:
    """
    Validate if a file type is allowed for upload.
    
    Args:
        filename: Name of the file
        content_type: MIME type of the file (optional)
        
    Returns:
        True if file type is allowed, False otherwise
    """
    # Check by extension first
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # If content type is provided and not generic, validate it
    if content_type:
        # Allow application/octet-stream if extension is valid (common for some uploads)
        if content_type == 'application/octet-stream':
            return True
        
        # For other content types, validate against allowed types
        if content_type not in ALLOWED_FILE_TYPES:
            return False
        if file_ext not in ALLOWED_FILE_TYPES[content_type]:
            return False
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate if a URL is well-formed.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_username(username: str) -> bool:
    """
    Validate username format.
    
    Args:
        username: Username to validate
        
    Returns:
        True if username is valid, False otherwise
    """
    # Username should be 3-50 characters, alphanumeric and underscores only
    pattern = r'^[a-zA-Z0-9_]{3,50}$'
    return bool(re.match(pattern, username))


def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    return True, None


def validate_source_type(source_type: str) -> bool:
    """
    Validate document source type.
    
    Args:
        source_type: Source type to validate
        
    Returns:
        True if source type is valid, False otherwise
    """
    valid_types = ['gitlab', 'confluence', 'web', 'file']
    return source_type.lower() in valid_types


def validate_tags(tags: List[str]) -> bool:
    """
    Validate tag list.
    
    Args:
        tags: List of tags to validate
        
    Returns:
        True if all tags are valid, False otherwise
    """
    if not isinstance(tags, list):
        return False
    
    # Each tag should be 1-50 characters, alphanumeric, spaces, hyphens, underscores
    pattern = r'^[a-zA-Z0-9\s\-_]{1,50}$'
    for tag in tags:
        if not isinstance(tag, str) or not re.match(pattern, tag):
            return False
    
    return True

