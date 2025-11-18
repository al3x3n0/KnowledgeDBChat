"""
Confluence connector for ingesting pages, attachments, and comments.
"""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_connector import BaseConnector, DocumentInfo


class ConfluenceConnector(BaseConnector):
    """Connector for Atlassian Confluence content."""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.base_url = None
        self.username = None
        self.api_token = None
        self.spaces = []
        self.include_attachments = True
        self.include_comments = False
        self.page_limit = 1000
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Confluence connector with configuration."""
        try:
            self.config = config
            self.base_url = config.get("confluence_url", "").rstrip("/")
            self.username = config.get("username")
            self.api_token = config.get("api_token")
            self.spaces = config.get("spaces", [])
            self.include_attachments = config.get("include_attachments", True)
            self.include_comments = config.get("include_comments", False)
            self.page_limit = config.get("page_limit", 1000)
            
            if not self.base_url or not self.username or not self.api_token:
                raise ValueError("Confluence URL, username, and API token are required")
            
            # Initialize HTTP client with basic auth
            import base64
            credentials = base64.b64encode(f"{self.username}:{self.api_token}".encode()).decode()
            
            self.client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=30.0
            )
            
            # Test connection
            if await self.test_connection():
                self.is_initialized = True
                logger.info(f"Confluence connector initialized for {self.base_url}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Confluence connector: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Confluence."""
        try:
            response = await self.client.get(f"{self.base_url}/rest/api/user/current")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Confluence connection test failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents from configured Confluence spaces."""
        self._ensure_initialized()
        
        documents = []
        
        # If no specific spaces configured, get all accessible spaces
        if not self.spaces:
            self.spaces = await self._get_accessible_spaces()
        
        for space in self.spaces:
            space_key = space.get("key") or space.get("name")
            if not space_key:
                continue
                
            try:
                # Get pages from space
                pages = await self._get_space_pages(space_key)
                documents.extend(pages)
                
                # Get attachments if enabled
                if self.include_attachments:
                    attachments = await self._get_space_attachments(space_key)
                    documents.extend(attachments)
                    
            except Exception as e:
                logger.error(f"Error processing Confluence space {space_key}: {e}")
                continue
        
        logger.info(f"Found {len(documents)} documents in Confluence")
        return documents
    
    async def get_document_content(self, identifier: str) -> str:
        """Get content of a specific document."""
        self._ensure_initialized()
        
        try:
            doc_type, doc_id = identifier.split(":", 1)
            
            if doc_type == "page":
                return await self._get_page_content(doc_id)
            elif doc_type == "attachment":
                return await self._get_attachment_content(doc_id)
            elif doc_type == "comment":
                return await self._get_comment_content(doc_id)
            else:
                raise ValueError(f"Unknown document type: {doc_type}")
                
        except Exception as e:
            logger.error(f"Error getting Confluence document content {identifier}: {e}")
            return ""
    
    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        """Get metadata for a specific document."""
        self._ensure_initialized()
        
        try:
            doc_type, doc_id = identifier.split(":", 1)
            
            if doc_type == "page":
                return await self._get_page_metadata(doc_id)
            elif doc_type == "attachment":
                return await self._get_attachment_metadata(doc_id)
            elif doc_type == "comment":
                return await self._get_comment_metadata(doc_id)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting Confluence document metadata {identifier}: {e}")
            return {}
    
    async def _get_accessible_spaces(self) -> List[Dict[str, Any]]:
        """Get spaces accessible to the current user."""
        try:
            response = await self.client.get(
                f"{self.base_url}/rest/api/space",
                params={"limit": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                spaces = data.get("results", [])
                return [{"key": s["key"], "name": s["name"]} for s in spaces]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting Confluence spaces: {e}")
            return []
    
    async def _get_space_pages(self, space_key: str) -> List[Dict[str, Any]]:
        """Get all pages from a space."""
        pages = []
        start = 0
        limit = 50
        
        try:
            while True:
                response = await self.client.get(
                    f"{self.base_url}/rest/api/content",
                    params={
                        "spaceKey": space_key,
                        "type": "page",
                        "status": "current",
                        "expand": "version,space,ancestors",
                        "start": start,
                        "limit": limit
                    }
                )
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for page in results:
                    # Skip pages with restricted access
                    if not page.get("title"):
                        continue
                    
                    # Parse last modified date
                    last_modified = None
                    if page.get("version", {}).get("when"):
                        try:
                            last_modified = datetime.fromisoformat(
                                page["version"]["when"].replace("Z", "+00:00")
                            )
                        except:
                            pass
                    
                    pages.append({
                        "identifier": f"page:{page['id']}",
                        "title": page["title"],
                        "url": f"{self.base_url}/pages/viewpage.action?pageId={page['id']}",
                        "last_modified": last_modified,
                        "author": page.get("version", {}).get("by", {}).get("displayName"),
                        "file_type": "confluence_page",
                        "metadata": {
                            "page_id": page["id"],
                            "space_key": space_key,
                            "space_name": page.get("space", {}).get("name", ""),
                            "version": page.get("version", {}).get("number", 1),
                            "type": "page",
                            "ancestors": [a.get("title", "") for a in page.get("ancestors", [])]
                        }
                    })
                
                # Check if there are more pages
                if len(results) < limit:
                    break
                
                start += limit
                
                # Respect page limit
                if len(pages) >= self.page_limit:
                    break
            
        except Exception as e:
            logger.error(f"Error getting Confluence pages for space {space_key}: {e}")
        
        return pages
    
    async def _get_space_attachments(self, space_key: str) -> List[Dict[str, Any]]:
        """Get attachments from a space."""
        attachments = []
        
        try:
            response = await self.client.get(
                f"{self.base_url}/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "attachment",
                    "status": "current",
                    "expand": "version,container",
                    "limit": 100
                }
            )
            
            if response.status_code != 200:
                return attachments
            
            data = response.json()
            results = data.get("results", [])
            
            for attachment in results:
                # Only include text-based attachments
                media_type = attachment.get("extensions", {}).get("mediaType", "")
                if not self._is_text_attachment(media_type):
                    continue
                
                # Parse last modified date
                last_modified = None
                if attachment.get("version", {}).get("when"):
                    try:
                        last_modified = datetime.fromisoformat(
                            attachment["version"]["when"].replace("Z", "+00:00")
                        )
                    except:
                        pass
                
                attachments.append({
                    "identifier": f"attachment:{attachment['id']}",
                    "title": attachment["title"],
                    "url": f"{self.base_url}{attachment.get('_links', {}).get('download', '')}",
                    "last_modified": last_modified,
                    "author": attachment.get("version", {}).get("by", {}).get("displayName"),
                    "file_type": attachment.get("extensions", {}).get("fileSize", "attachment"),
                    "metadata": {
                        "attachment_id": attachment["id"],
                        "space_key": space_key,
                        "container_id": attachment.get("container", {}).get("id"),
                        "container_title": attachment.get("container", {}).get("title", ""),
                        "media_type": media_type,
                        "file_size": attachment.get("extensions", {}).get("fileSize", 0),
                        "type": "attachment"
                    }
                })
            
        except Exception as e:
            logger.error(f"Error getting Confluence attachments for space {space_key}: {e}")
        
        return attachments
    
    def _is_text_attachment(self, media_type: str) -> bool:
        """Check if attachment is text-based and can be processed."""
        text_types = [
            "text/plain",
            "text/markdown",
            "text/html",
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        return media_type in text_types
    
    async def _get_page_content(self, page_id: str) -> str:
        """Get content of a page."""
        try:
            response = await self.client.get(
                f"{self.base_url}/rest/api/content/{page_id}",
                params={
                    "expand": "body.storage,body.view"
                }
            )
            
            if response.status_code == 200:
                page_data = response.json()
                
                # Get storage format content (raw markup)
                storage_content = page_data.get("body", {}).get("storage", {}).get("value", "")
                
                # Get view format content (rendered HTML)
                view_content = page_data.get("body", {}).get("view", {}).get("value", "")
                
                # Prefer storage format, fall back to view format
                content = storage_content or view_content
                
                # Clean up HTML/markup
                content = self._clean_confluence_content(content)
                
                # Add page metadata as content
                title = page_data.get("title", "")
                if title:
                    content = f"Title: {title}\n\n{content}"
                
                return content
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting Confluence page content {page_id}: {e}")
            return ""
    
    async def _get_attachment_content(self, attachment_id: str) -> str:
        """Get content of an attachment."""
        try:
            # Get attachment metadata first
            response = await self.client.get(
                f"{self.base_url}/rest/api/content/{attachment_id}",
                params={"expand": "container,version"}
            )
            
            if response.status_code != 200:
                return ""
            
            attachment_data = response.json()
            download_url = attachment_data.get("_links", {}).get("download", "")
            
            if not download_url:
                return ""
            
            # Download attachment content
            download_response = await self.client.get(f"{self.base_url}{download_url}")
            
            if download_response.status_code == 200:
                # For text files, return content directly
                media_type = attachment_data.get("extensions", {}).get("mediaType", "")
                
                if media_type.startswith("text/"):
                    return download_response.text
                else:
                    # For non-text files, return metadata as content
                    return f"Attachment: {attachment_data.get('title', 'Unknown')}\n" \
                           f"Type: {media_type}\n" \
                           f"Container: {attachment_data.get('container', {}).get('title', 'Unknown')}"
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting Confluence attachment content {attachment_id}: {e}")
            return ""
    
    async def _get_comment_content(self, comment_id: str) -> str:
        """Get content of a comment."""
        try:
            response = await self.client.get(
                f"{self.base_url}/rest/api/content/{comment_id}",
                params={"expand": "body.view,container,version"}
            )
            
            if response.status_code == 200:
                comment_data = response.json()
                
                content = comment_data.get("body", {}).get("view", {}).get("value", "")
                content = self._clean_confluence_content(content)
                
                # Add comment metadata
                author = comment_data.get("version", {}).get("by", {}).get("displayName", "Unknown")
                container_title = comment_data.get("container", {}).get("title", "Unknown Page")
                
                return f"Comment by {author} on '{container_title}':\n\n{content}"
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting Confluence comment content {comment_id}: {e}")
            return ""
    
    def _clean_confluence_content(self, content: str) -> str:
        """Clean Confluence HTML/markup content."""
        if not content:
            return ""
        
        # Remove HTML tags but keep content
        import re
        
        # Remove common Confluence macros
        content = re.sub(r'<ac:.*?</ac:.*?>', '', content, flags=re.DOTALL)
        content = re.sub(r'<ac:.*?/>', '', content)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    async def _get_page_metadata(self, page_id: str) -> Dict[str, Any]:
        """Get metadata for a page."""
        return {
            "page_id": page_id,
            "type": "page"
        }
    
    async def _get_attachment_metadata(self, attachment_id: str) -> Dict[str, Any]:
        """Get metadata for an attachment."""
        return {
            "attachment_id": attachment_id,
            "type": "attachment"
        }
    
    async def _get_comment_metadata(self, comment_id: str) -> Dict[str, Any]:
        """Get metadata for a comment."""
        return {
            "comment_id": comment_id,
            "type": "comment"
        }
    
    async def cleanup(self):
        """Clean up the HTTP client."""
        if self.client:
            await self.client.aclose()







