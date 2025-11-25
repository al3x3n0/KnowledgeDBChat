"""
Storage service for managing file uploads and downloads using MinIO.
"""

import os
import re
import uuid
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
        self._multipart_available: Optional[bool] = None  # Cache multipart availability
    
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
            
            # Detect multipart upload availability
            self._detect_multipart_availability()
        except S3Error as e:
            logger.error(f"MinIO S3Error during initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            raise
    
    def _detect_multipart_availability(self) -> bool:
        """Prefer predictable behavior: default to manual reassembly.
        MinIO's Python client does not expose a stable public multipart API across versions,
        and probing private methods causes noisy warnings. We skip probing and use the
        manual reassembly path implemented in upload endpoint.
        """
        if self._multipart_available is not None:
            return self._multipart_available
        self._multipart_available = False
        logger.info("Multipart detection skipped; using manual reassembly strategy.")
        return False
    
    def is_multipart_available(self) -> bool:
        """
        Check if multipart upload is available.
        
        Returns:
            True if multipart upload is available, False otherwise
        """
        # Detection happens during initialize(), so if initialized, we know the result
        if not self._initialized:
            # If not initialized, assume False (will be detected on first use)
            return False
        return self._multipart_available or False
    
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
            object_path: Path to object in MinIO (may or may not include 'documents/' prefix)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present (bucket name is already 'documents')
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]  # Remove 'documents/' prefix (10 chars)
            
            logger.info(f"Deleting file from MinIO: {sanitized_path} (original: {object_path})")
            
            client.remove_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path
            )
            
            logger.info(f"Successfully deleted file from MinIO: {sanitized_path}")
            return True
        
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"File not found in MinIO: {object_path} (sanitized: {sanitized_path if 'sanitized_path' in locals() else 'N/A'})")
                # Return True if file doesn't exist - it's already deleted
                return True
            logger.error(f"MinIO S3Error during delete: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to delete file from MinIO: {e}", exc_info=True)
            return False
    
    async def download_file(self, object_path: str, local_path: str) -> bool:
        """
        Download file from MinIO to local path.
        
        Args:
            object_path: Path to object in MinIO
            local_path: Local file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]
            
            # Download file
            from pathlib import Path
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            client.fget_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path,
                file_path=str(local_file)
            )
            
            logger.info(f"Downloaded file from MinIO: {sanitized_path} to {local_path}")
            return True
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"File not found in MinIO: {object_path}")
                return False
            logger.error(f"MinIO S3Error during download: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to download file from MinIO: {e}", exc_info=True)
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
        yield from self.get_file_stream_range(object_path, None, None)
    
    def get_file_stream_range(self, object_path: str, start: Optional[int] = None, end: Optional[int] = None):
        """
        Get file from MinIO as a stream with optional range support.
        
        Args:
            object_path: Path to object in MinIO (may or may not include 'documents/' prefix)
            start: Start byte position (None for beginning)
            end: End byte position (None for end)
            
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
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Try the path as-is first (in case it includes 'documents/' prefix from old uploads)
            try:
                if start is not None and end is not None:
                    # MinIO supports range requests via get_object with offset and length
                    response = client.get_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_path,
                        offset=start,
                        length=end - start + 1
                    )
                else:
                    # Full file
                    response = client.get_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_path
                    )
            except S3Error as e:
                if e.code == "NoSuchKey" and sanitized_path.startswith('documents/'):
                    # Try without 'documents/' prefix
                    path_without_prefix = sanitized_path[10:]  # Remove 'documents/' prefix
                    if start is not None and end is not None:
                        response = client.get_object(
                            bucket_name=settings.MINIO_BUCKET_NAME,
                            object_name=path_without_prefix,
                            offset=start,
                            length=end - start + 1
                        )
                    else:
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
    
    async def create_multipart_upload(self, object_path: str, content_type: Optional[str] = None) -> str:
        """
        Create a multipart upload session in MinIO.
        
        Args:
            object_path: Path to object in MinIO
            content_type: Content type of the file
            
        Returns:
            Upload ID for the multipart upload (or placeholder ID if multipart not available)
        """
        try:
            await self.initialize()
            
            # Check if multipart is available
            if not self.is_multipart_available():
                # Return placeholder ID for manual reassembly
                upload_id = f"upload_{uuid.uuid4()}"
                logger.info(f"Multipart upload not available, using manual reassembly with ID: {upload_id}")
                return upload_id
            
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]
            
            # Use MinIO's S3-compatible multipart upload API
            try:
                # Try using the public API first (if available in newer versions)
                if hasattr(client, 'create_multipart_upload'):
                    result = client.create_multipart_upload(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_path,
                        metadata={"Content-Type": content_type or "application/octet-stream"}
                    )
                    upload_id = result.upload_id if hasattr(result, 'upload_id') else str(result)
                elif hasattr(client, '_create_multipart_upload'):
                    # Fallback to internal API (may vary by MinIO version)
                    result = client._create_multipart_upload(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_path,
                        metadata={"Content-Type": content_type or "application/octet-stream"}
                    )
                    upload_id = result.upload_id if hasattr(result, 'upload_id') else str(result)
                else:
                    # Should not happen if detection worked, but fallback anyway
                    upload_id = f"upload_{uuid.uuid4()}"
                    logger.warning(f"Multipart methods not found despite detection, using manual reassembly: {upload_id}")
                    return upload_id
                
                logger.info(f"Created multipart upload for {sanitized_path}, upload_id: {upload_id}")
                return upload_id
            except (AttributeError, S3Error) as e:
                # If multipart upload fails, mark as unavailable and use manual reassembly
                logger.warning(f"Multipart upload failed: {e}. Marking as unavailable and using manual reassembly.")
                self._multipart_available = False
                upload_id = f"upload_{uuid.uuid4()}"
                return upload_id
            
        except S3Error as e:
            logger.error(f"MinIO S3Error creating multipart upload: {e}", exc_info=True)
            # Mark as unavailable and return placeholder
            self._multipart_available = False
            upload_id = f"upload_{uuid.uuid4()}"
            return upload_id
        except Exception as e:
            logger.error(f"Failed to create multipart upload: {e}", exc_info=True)
            # Mark as unavailable and return placeholder
            self._multipart_available = False
            upload_id = f"upload_{uuid.uuid4()}"
            return upload_id
    
    async def upload_part(self, object_path: str, upload_id: str, part_number: int, data: bytes) -> str:
        """
        Upload a part of a multipart upload.
        
        Args:
            object_path: Path to object in MinIO
            upload_id: Multipart upload ID
            part_number: Part number (1-indexed)
            data: Part data
            
        Returns:
            ETag for the uploaded part
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]
            
            # Upload part using MinIO's S3-compatible multipart upload
            from io import BytesIO
            from minio.commonconfig import Part
            result = client._upload_part(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path,
                upload_id=upload_id,
                part_number=part_number,
                data=BytesIO(data),
                length=len(data)
            )
            
            # Extract ETag from response
            etag = result.etag if hasattr(result, 'etag') else result.headers.get('ETag', '').strip('"')
            logger.debug(f"Uploaded part {part_number} for {sanitized_path}, etag: {etag}")
            return etag
            
        except S3Error as e:
            logger.error(f"MinIO S3Error uploading part: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to upload part: {e}", exc_info=True)
            raise
    
    async def complete_multipart_upload(self, object_path: str, upload_id: str, parts: list) -> None:
        """
        Complete a multipart upload.
        
        Args:
            object_path: Path to object in MinIO
            upload_id: Multipart upload ID
            parts: List of (part_number, etag) tuples
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]
            
            # Convert parts to MinIO format
            from minio.commonconfig import Part
            minio_parts = [Part(part_num, etag) for part_num, etag in parts]
            
            # Complete multipart upload
            client._complete_multipart_upload(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path,
                upload_id=upload_id,
                parts=minio_parts
            )
            
            logger.info(f"Completed multipart upload for {sanitized_path}")
            
        except S3Error as e:
            logger.error(f"MinIO S3Error completing multipart upload: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to complete multipart upload: {e}", exc_info=True)
            raise
    
    async def abort_multipart_upload(self, object_path: str, upload_id: str) -> None:
        """
        Abort a multipart upload.
        
        Args:
            object_path: Path to object in MinIO
            upload_id: Multipart upload ID
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize the path
            sanitized_path = self._sanitize_object_path(object_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_path.startswith('documents/'):
                sanitized_path = sanitized_path[10:]
            
            # Abort multipart upload
            client._abort_multipart_upload(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=sanitized_path,
                upload_id=upload_id
            )
            
            logger.info(f"Aborted multipart upload for {sanitized_path}")
            
        except S3Error as e:
            logger.error(f"MinIO S3Error aborting multipart upload: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to abort multipart upload: {e}", exc_info=True)
            raise
    
    async def copy_file(self, source_path: str, dest_path: str, content_type: Optional[str] = None) -> bool:
        """
        Copy a file from source path to destination path in MinIO.
        
        Args:
            source_path: Source object path in MinIO
            dest_path: Destination object path in MinIO
            content_type: Optional content type for the destination file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.initialize()
            client = self._get_client()
            
            # Sanitize and normalize both paths
            sanitized_source = self._sanitize_object_path(source_path)
            sanitized_dest = self._sanitize_object_path(dest_path)
            
            # Remove 'documents/' prefix if present
            if sanitized_source.startswith('documents/'):
                sanitized_source = sanitized_source[10:]
            if sanitized_dest.startswith('documents/'):
                sanitized_dest = sanitized_dest[10:]
            
            # Get source file metadata if content_type not provided
            if not content_type:
                try:
                    stat = client.stat_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_source
                    )
                    content_type = stat.content_type
                except S3Error as e:
                    logger.warning(f"Could not get source file metadata: {e}, using default content type")
                    content_type = "application/octet-stream"
            
            # Use MinIO's copy_object API (server-side copy, more efficient)
            # First try copy_object, fallback to download+upload if needed
            try:
                from minio.commonconfig import CopySource
                copy_source = CopySource(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=sanitized_source
                )
                
                # Copy object (some MinIO versions may not support metadata parameter)
                try:
                    client.copy_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_dest,
                        source=copy_source,
                        metadata={"Content-Type": content_type}
                    )
                except (TypeError, AttributeError):
                    # Try without metadata parameter
                    client.copy_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_dest,
                        source=copy_source
                    )
            except (ImportError, AttributeError, S3Error) as e:
                # Fallback: use download + upload for copying
                logger.info(f"Using download+upload fallback for copy: {e}")
                import tempfile
                temp_path = tempfile.mktemp()
                try:
                    # Download source
                    client.fget_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_source,
                        file_path=temp_path
                    )
                    # Upload to destination
                    client.fput_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=sanitized_dest,
                        file_path=temp_path,
                        content_type=content_type
                    )
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            logger.info(f"Copied file from {sanitized_source} to {sanitized_dest}")
            return True
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.error(f"Source file not found in MinIO: {source_path}")
                return False
            logger.error(f"MinIO S3Error copying file: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to copy file in MinIO: {e}", exc_info=True)
            return False


storage_service = MinIOStorageService()
