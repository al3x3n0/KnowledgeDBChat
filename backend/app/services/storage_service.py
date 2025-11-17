"""
Storage service for managing file uploads and downloads using MinIO.
"""

import os
import re
from datetime import timedelta
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID
from minio import Minio
from minio.error import S3Error
from loguru import logger

from app.core.config import settings


class MinIOStorageService:
    """Service for managing file storage with MinIO."""
    
    def __init__(self):
        self.client: Optional[Minio] = None
        self._initialized = False
    
    def _get_client(self) -> Minio:
        """Get or create MinIO client."""
        if not self.client:
            try:
                self.client = Minio(
                    settings.MINIO_ENDPOINT,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    secure=settings.MINIO_USE_SSL
                )
            except Exception as e:
                logger.error(f"Failed to create MinIO client: {e}")
                raise
        
        return self.client
    
    async def initialize(self):
        """Initialize MinIO connection and ensure bucket exists."""
        if self._initialized:
            return
        
        try:
            client = self._get_client()
            
            # Check if bucket exists, create if not
            if not client.bucket_exists(settings.MINIO_BUCKET_NAME):
                client.make_bucket(settings.MINIO_BUCKET_NAME)
                logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET_NAME}")
            else:
                logger.info(f"MinIO bucket already exists: {settings.MINIO_BUCKET_NAME}")
            
            self._initialized = True
        except S3Error as e:
            logger.error(f"MinIO S3Error during initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace unsafe characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        return filename
    
    def _sanitize_object_path(self, object_path: str) -> str:
        """
        Sanitize object path to ensure it's valid for MinIO.
        
        Removes '..' and '.' segments, normalizes path separators.
        Handles paths starting with './' or '../'.
        Removes redundant 'documents/' prefix since bucket name is already 'documents'.
        
        Args:
            object_path: Object path (e.g., "{id}/{filename}" or "documents/{id}/{filename}" or "./data/documents/...")
            
        Returns:
            Sanitized object path (without bucket name prefix)
        """
        if not object_path:
            raise ValueError("Object path cannot be empty")
        
        # Remove leading './' or '../' patterns
        while object_path.startswith('./'):
            object_path = object_path[2:]  # Remove './' prefix
        while object_path.startswith('../'):
            object_path = object_path[3:]  # Remove '../' prefix
        
        # Split path into segments
        segments = object_path.split('/')
        
        # Filter out empty segments, '.', and '..'
        sanitized_segments = []
        for segment in segments:
            if segment and segment != '.' and segment != '..':
                sanitized_segments.append(segment)
        
        if not sanitized_segments:
            raise ValueError("Object path must contain at least one valid segment")
        
        # Remove 'documents/' prefix if present (bucket name is already 'documents')
        if sanitized_segments[0] == 'documents' and len(sanitized_segments) > 1:
            sanitized_segments = sanitized_segments[1:]
        
        # Rejoin segments
        sanitized_path = '/'.join(sanitized_segments)
        
        # Ensure path doesn't start with '/' (MinIO doesn't like absolute paths)
        sanitized_path = sanitized_path.lstrip('/')
        
        return sanitized_path
    
    def _get_object_path(self, document_id: UUID, filename: str) -> str:
        """Generate object path for MinIO storage.
        
        Format: {document_id}/{sanitized_filename}
        Note: The bucket name is already specified separately, so we don't include it in the object path.
        """
        sanitized_filename = self._sanitize_filename(filename)
        return f"{document_id}/{sanitized_filename}"
    
    async def upload_file(
        self,
        document_id: UUID,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload file to MinIO.
        
        Args:
            document_id: Document UUID
            filename: Original filename
            content: File content as bytes
            content_type: Optional MIME type
            
        Returns:
            Object path in MinIO
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            object_path = self._get_object_path(document_id, filename)
            
            # Upload file
            client.put_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=object_path,
                data=BytesIO(content),
                length=len(content),
                content_type=content_type or "application/octet-stream"
            )
            
            logger.info(f"Uploaded file to MinIO: {object_path}")
            return object_path
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during upload: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to upload file to MinIO: {e}")
            raise
    
    def _rewrite_presigned_url_for_proxy(self, url: str) -> str:
        """
        Rewrite MinIO presigned URL to use nginx proxy.
        
        IMPORTANT: Presigned URL signatures are tied to the exact path. When nginx proxies
        /minio/documents/... to minio:9000, it strips /minio and sends /documents/... to MinIO.
        So we need to add /minio prefix to the path to match what nginx expects, and nginx
        will strip it when forwarding to MinIO, preserving the signature.
        
        Args:
            url: Original presigned URL from MinIO (e.g., "http://minio:9000/documents/...?X-Amz-...")
            
        Returns:
            Rewritten URL using proxy base URL (e.g., "http://localhost:3000/minio/documents/...?X-Amz-...")
        """
        if not settings.MINIO_PROXY_BASE_URL:
            return url
        
        try:
            # Parse the original URL
            parsed = urlparse(url)
            
            # Extract path and query (e.g., "/documents/{id}/{filename}" and "X-Amz-Algorithm=...")
            # The path already includes the bucket name: /documents/...
            path = parsed.path
            query = parsed.query
            
            # Build new URL with proxy base
            # Add /minio prefix to path so nginx can strip it when proxying
            # This preserves the signature because nginx sends /documents/... to MinIO (without /minio)
            proxy_base = settings.MINIO_PROXY_BASE_URL.rstrip('/')
            proxy_path = f"/minio{path}" if not path.startswith("/minio") else path
            
            # Construct the new URL: proxy_base + /minio + original_path + query
            # The query contains the signature which must be preserved exactly
            if query:
                final_url = f"{proxy_base}{proxy_path}?{query}"
            else:
                final_url = f"{proxy_base}{proxy_path}"
            
            logger.debug(f"Rewrote presigned URL from {url[:80]}... to {final_url[:80]}...")
            return final_url
            
        except Exception as e:
            logger.warning(f"Failed to rewrite presigned URL for proxy: {e}, using original URL")
            return url
    
    async def get_presigned_download_url(
        self,
        object_path: str,
        expiry: Optional[int] = None
    ) -> str:
        """
        Generate presigned URL for downloading a file.
        
        Args:
            object_path: Path to object in MinIO (e.g., "documents/{id}/{filename}")
            expiry: URL expiry time in seconds (defaults to MINIO_PRESIGNED_URL_EXPIRY)
            
        Returns:
            Presigned URL string (rewritten to use proxy if configured)
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize the object path to remove any invalid segments
            sanitized_path = self._sanitize_object_path(object_path)
            
            expiry_seconds = expiry or settings.MINIO_PRESIGNED_URL_EXPIRY
            
            # MinIO 7.x+ expects timedelta, not int
            expiry_timedelta = timedelta(seconds=expiry_seconds)
            
            # Generate presigned URL with internal MinIO endpoint
            # The signature is calculated based on the exact path, so we must preserve it
            url = client.presigned_get_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path,
                expires=expiry_timedelta
            )
            
            # Rewrite URL to use nginx proxy if configured
            # IMPORTANT: The signature is tied to the path, so we must preserve the exact path
            # Nginx proxy_pass with trailing slash strips the location prefix, so:
            # Request: /minio/documents/path -> Proxied to MinIO: /documents/path
            # This means we need to add /minio prefix to the path in the URL
            if settings.MINIO_PROXY_BASE_URL:
                url = self._rewrite_presigned_url_for_proxy(url)
            
            logger.debug(f"Generated presigned URL for {sanitized_path} (original: {object_path}), expires in {expiry_seconds}s")
            return url
        
        except ValueError as e:
            logger.error(f"Invalid object path: {object_path}, error: {e}")
            raise ValueError(f"Invalid object path format: {e}")
        except S3Error as e:
            logger.error(f"MinIO S3Error generating presigned URL for path '{object_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for path '{object_path}': {e}")
            raise
    
    async def delete_file(self, object_path: str) -> bool:
        """
        Delete file from MinIO.
        
        Args:
            object_path: Path to object in MinIO
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            client.remove_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=object_path
            )
            
            logger.info(f"Deleted file from MinIO: {object_path}")
            return True
        
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"File not found in MinIO: {object_path}")
                return False
            logger.error(f"MinIO S3Error during delete: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file from MinIO: {e}")
            return False
    
    async def file_exists(self, object_path: str) -> bool:
        """
        Check if file exists in MinIO.
        
        Args:
            object_path: Path to object in MinIO (may or may not include 'documents/' prefix)
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Try the path as-is first (in case it includes 'documents/' prefix from old uploads)
            try:
                client.stat_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=object_path
                )
                return True
            except S3Error as e:
                if e.code != "NoSuchKey":
                    raise
            
            # If not found and path starts with 'documents/', try without prefix
            if object_path.startswith('documents/'):
                path_without_prefix = object_path[10:]  # Remove 'documents/' prefix
                try:
                    client.stat_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=path_without_prefix
                    )
                    return True
                except S3Error as e:
                    if e.code != "NoSuchKey":
                        raise
            
            # If still not found, try sanitized version
            try:
                sanitized_path = self._sanitize_object_path(object_path)
                if sanitized_path != object_path:
                    client.stat_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_path
                    )
                    return True
            except (S3Error, ValueError):
                pass
            
            return False
        
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"File not found in MinIO: {object_path}")
                return False
            logger.error(f"MinIO S3Error checking file existence: {e}")
            return False
        except ValueError as e:
            # Invalid path format
            logger.warning(f"Invalid path format for file_exists check: {object_path} - {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}", exc_info=True)
            return False
    
    def get_file_stream(self, object_path: str):
        """
        Get file from MinIO as a stream (generator).
        
        Note: This is a synchronous generator (not async) because FastAPI's StreamingResponse
        works with regular generators. The MinIO client operations are synchronous.
        
        Args:
            object_path: Path to object in MinIO (may or may not include 'documents/' prefix)
            
        Yields:
            File chunks as bytes
        """
        try:
            # Initialize synchronously (MinIO client is thread-safe)
            if not self._initialized:
                # We need to initialize, but this is a sync method
                # So we'll initialize the client if needed
                client = self._get_client()
                if not client.bucket_exists(settings.MINIO_BUCKET_NAME):
                    client.make_bucket(settings.MINIO_BUCKET_NAME)
                    logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET_NAME}")
            
            client = self._get_client()
            
            # Try the path as-is first (in case it includes 'documents/' prefix from old uploads)
            try:
                response = client.get_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=object_path
                )
            except S3Error as e:
                if e.code == "NoSuchKey" and object_path.startswith('documents/'):
                    # Try without 'documents/' prefix
                    path_without_prefix = object_path[10:]  # Remove 'documents/' prefix
                    response = client.get_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=path_without_prefix
                    )
                else:
                    raise
            
            try:
                # Stream the file in chunks
                while True:
                    chunk = response.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk
            finally:
                # Always close the response
                response.close()
                response.release_conn()
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"File not found in MinIO: {object_path}")
            logger.error(f"MinIO S3Error getting file stream: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get file stream: {e}")
            raise
    
    async def get_file_metadata(self, object_path: str) -> dict:
        """
        Get file metadata from MinIO.
        
        Args:
            object_path: Path to object in MinIO (may or may not include 'documents/' prefix)
            
        Returns:
            Dictionary with file metadata (size, content_type, last_modified, etc.)
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Try the path as-is first (in case it includes 'documents/' prefix from old uploads)
            try:
                stat = client.stat_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=object_path
                )
            except S3Error as e:
                if e.code == "NoSuchKey" and object_path.startswith('documents/'):
                    # Try without 'documents/' prefix
                    path_without_prefix = object_path[10:]  # Remove 'documents/' prefix
                    stat = client.stat_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=path_without_prefix
                    )
                else:
                    raise
            
            return {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag
            }
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"File not found in MinIO: {object_path}")
            logger.error(f"MinIO S3Error getting file metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            raise


# Global storage service instance
storage_service = MinIOStorageService()

