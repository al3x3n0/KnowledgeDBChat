"""
Video streaming service for efficient video playback from MinIO.
Handles range requests, chunked streaming, and video-specific optimizations.
"""

import os
from typing import Optional, Tuple
from loguru import logger

from app.services.storage_service import storage_service
from app.core.config import settings


class VideoStreamingService:
    """Service for streaming video files from MinIO with range request support."""
    
    def __init__(self):
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service and ensure MinIO connection."""
        if not self._initialized:
            await storage_service.initialize()
            self._initialized = True
    
    async def get_video_metadata(self, file_path: str) -> dict:
        """
        Get video file metadata.
        
        Args:
            file_path: Path to video file in MinIO
            
        Returns:
            Dictionary with video metadata (size, content_type, etc.)
        """
        await self.initialize()
        return await storage_service.get_file_metadata(file_path)
    
    def get_video_stream(self, file_path: str, start: Optional[int] = None, end: Optional[int] = None):
        """
        Get video file stream with optional range support.
        
        Args:
            file_path: Path to video file in MinIO
            start: Start byte position (None for beginning)
            end: End byte position (None for end)
            
        Yields:
            Video file chunks as bytes
        """
        if start is not None and end is not None:
            return storage_service.get_file_stream_range(file_path, start, end)
        else:
            return storage_service.get_file_stream(file_path)
    
    def parse_range_header(self, range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
        """
        Parse HTTP Range header.
        
        Args:
            range_header: Range header value (e.g., "bytes=0-1023")
            file_size: Total file size in bytes
            
        Returns:
            Tuple of (start, end) byte positions, or None if invalid
        """
        if not range_header or not range_header.startswith("bytes="):
            return None
        
        try:
            # Remove "bytes=" prefix
            range_spec = range_header[6:]
            parts = range_spec.split("-")
            
            if len(parts) != 2:
                return None
            
            start_str, end_str = parts
            
            # Parse start
            if start_str:
                start = int(start_str)
                if start < 0:
                    start = 0
            else:
                start = 0
            
            # Parse end
            if end_str:
                end = int(end_str)
                if end >= file_size:
                    end = file_size - 1
            else:
                end = file_size - 1
            
            # Validate range
            if start > end or start >= file_size:
                return None
            
            return (start, end)
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid Range header '{range_header}': {e}")
            return None
    
    async def get_video_info(self, file_path: str) -> dict:
        """
        Get comprehensive video information.
        
        Args:
            file_path: Path to video file in MinIO
            
        Returns:
            Dictionary with video information
        """
        metadata = await self.get_video_metadata(file_path)
        
        return {
            "size": metadata.get("size", 0),
            "content_type": metadata.get("content_type", "video/mp4"),
            "last_modified": metadata.get("last_modified"),
            "etag": metadata.get("etag"),
            "supports_range_requests": True,  # MinIO supports range requests
        }
    
    def get_content_range_header(self, start: int, end: int, file_size: int) -> str:
        """
        Generate Content-Range header value.
        
        Args:
            start: Start byte position
            end: End byte position
            file_size: Total file size
            
        Returns:
            Content-Range header value
        """
        return f"bytes {start}-{end}/{file_size}"
    
    def get_streaming_headers(
        self,
        content_type: str,
        file_size: int,
        filename: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        is_partial: bool = False
    ) -> dict:
        """
        Get HTTP headers for video streaming response.
        
        Args:
            content_type: MIME type of the video
            file_size: Total file size in bytes
            filename: Filename for Content-Disposition
            start: Start byte position (for partial content)
            end: End byte position (for partial content)
            is_partial: Whether this is a partial content response
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": content_type,
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "public, max-age=3600",
            # CORS headers for video streaming
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Authorization, Content-Type",
            "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
        }
        
        if is_partial and start is not None and end is not None:
            # Partial content response
            content_length = end - start + 1
            headers["Content-Range"] = self.get_content_range_header(start, end, file_size)
            headers["Content-Length"] = str(content_length)
        else:
            # Full content response
            headers["Content-Length"] = str(file_size)
        
        return headers


# Global video streaming service instance
video_streaming_service = VideoStreamingService()

