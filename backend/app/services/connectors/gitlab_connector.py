"""
GitLab connector for ingesting repository content, wikis, and issues.
"""

import base64
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_connector import BaseConnector, DocumentInfo


class GitLabConnector(BaseConnector):
    """Connector for GitLab repositories and content."""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.base_url = None
        self.token = None
        self.projects = []
        self.include_wikis = True
        self.include_issues = True
        self.include_merge_requests = False
        self.file_extensions = ['.md', '.txt', '.rst', '.py', '.js', '.ts', '.java', '.cpp', '.h']
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize GitLab connector with configuration."""
        try:
            self.config = config
            self.base_url = config.get("gitlab_url", "").rstrip("/")
            self.token = config.get("token")
            self.projects = config.get("projects", [])
            self.include_wikis = config.get("include_wikis", True)
            self.include_issues = config.get("include_issues", True)
            self.include_merge_requests = config.get("include_merge_requests", False)
            self.file_extensions = config.get("file_extensions", self.file_extensions)
            
            if not self.base_url or not self.token:
                raise ValueError("GitLab URL and token are required")
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            # Test connection
            if await self.test_connection():
                self.is_initialized = True
                logger.info(f"GitLab connector initialized for {self.base_url}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize GitLab connector: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to GitLab."""
        try:
            response = await self.client.get(f"{self.base_url}/api/v4/user")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GitLab connection test failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents from configured GitLab projects."""
        self._ensure_initialized()
        
        documents = []
        
        # If no specific projects configured, get user's projects
        if not self.projects:
            self.projects = await self._get_user_projects()
        
        for project in self.projects:
            project_id = project.get("id") or project.get("name")
            if not project_id:
                continue
                
            try:
                # Get project info
                project_info = await self._get_project_info(project_id)
                if not project_info:
                    continue
                
                # Get repository files
                if project.get("include_files", True):
                    files = await self._get_repository_files(project_info["id"])
                    documents.extend(files)
                
                # Get wiki pages
                if self.include_wikis and project.get("include_wikis", True):
                    wiki_pages = await self._get_wiki_pages(project_info["id"])
                    documents.extend(wiki_pages)
                
                # Get issues
                if self.include_issues and project.get("include_issues", True):
                    issues = await self._get_issues(project_info["id"])
                    documents.extend(issues)
                
                # Get merge requests
                if self.include_merge_requests and project.get("include_merge_requests", False):
                    merge_requests = await self._get_merge_requests(project_info["id"])
                    documents.extend(merge_requests)
                    
            except Exception as e:
                logger.error(f"Error processing GitLab project {project_id}: {e}")
                continue
        
        logger.info(f"Found {len(documents)} documents in GitLab")
        return documents
    
    async def get_document_content(self, identifier: str) -> str:
        """Get content of a specific document."""
        self._ensure_initialized()
        
        try:
            doc_type, project_id, doc_id = identifier.split(":", 2)
            
            if doc_type == "file":
                return await self._get_file_content(project_id, doc_id)
            elif doc_type == "wiki":
                return await self._get_wiki_content(project_id, doc_id)
            elif doc_type == "issue":
                return await self._get_issue_content(project_id, doc_id)
            elif doc_type == "merge_request":
                return await self._get_merge_request_content(project_id, doc_id)
            else:
                raise ValueError(f"Unknown document type: {doc_type}")
                
        except Exception as e:
            logger.error(f"Error getting GitLab document content {identifier}: {e}")
            return ""
    
    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        """Get metadata for a specific document."""
        self._ensure_initialized()
        
        try:
            doc_type, project_id, doc_id = identifier.split(":", 2)
            
            if doc_type == "file":
                return await self._get_file_metadata(project_id, doc_id)
            elif doc_type == "wiki":
                return await self._get_wiki_metadata(project_id, doc_id)
            elif doc_type == "issue":
                return await self._get_issue_metadata(project_id, doc_id)
            elif doc_type == "merge_request":
                return await self._get_merge_request_metadata(project_id, doc_id)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting GitLab document metadata {identifier}: {e}")
            return {}
    
    async def _get_user_projects(self) -> List[Dict[str, Any]]:
        """Get projects accessible to the current user."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects",
                params={"membership": True, "per_page": 100}
            )
            
            if response.status_code == 200:
                projects = response.json()
                return [{"id": p["id"], "name": p["name"], "path": p["path_with_namespace"]} 
                       for p in projects]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting GitLab projects: {e}")
            return []
    
    async def _get_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project information."""
        try:
            response = await self.client.get(f"{self.base_url}/api/v4/projects/{project_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting GitLab project info {project_id}: {e}")
            return None
    
    async def _get_repository_files(self, project_id: int) -> List[Dict[str, Any]]:
        """Get repository files from a project."""
        files = []
        
        try:
            # Get repository tree
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/repository/tree",
                params={"recursive": True, "per_page": 100}
            )
            
            if response.status_code != 200:
                return files
            
            tree_items = response.json()
            
            for item in tree_items:
                if item["type"] == "blob":  # File
                    file_path = item["path"]
                    file_ext = "." + file_path.split(".")[-1] if "." in file_path else ""
                    
                    # Filter by file extensions
                    if self.file_extensions and file_ext not in self.file_extensions:
                        continue
                    
                    files.append({
                        "identifier": f"file:{project_id}:{file_path}",
                        "title": file_path,
                        "url": f"{self.base_url}/{item.get('path', '')}",
                        "file_type": file_ext.lstrip('.'),
                        "metadata": {
                            "project_id": project_id,
                            "file_path": file_path,
                            "type": "repository_file"
                        }
                    })
            
        except Exception as e:
            logger.error(f"Error getting GitLab repository files for project {project_id}: {e}")
        
        return files
    
    async def _get_wiki_pages(self, project_id: int) -> List[Dict[str, Any]]:
        """Get wiki pages from a project."""
        pages = []
        
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/wikis"
            )
            
            if response.status_code != 200:
                return pages
            
            wiki_pages = response.json()
            
            for page in wiki_pages:
                pages.append({
                    "identifier": f"wiki:{project_id}:{page['slug']}",
                    "title": page["title"],
                    "url": f"{self.base_url}/projects/{project_id}/wikis/{page['slug']}",
                    "file_type": "wiki",
                    "metadata": {
                        "project_id": project_id,
                        "wiki_slug": page["slug"],
                        "type": "wiki_page"
                    }
                })
            
        except Exception as e:
            logger.error(f"Error getting GitLab wiki pages for project {project_id}: {e}")
        
        return pages
    
    async def _get_issues(self, project_id: int) -> List[Dict[str, Any]]:
        """Get issues from a project."""
        issues = []
        
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/issues",
                params={"state": "all", "per_page": 100}
            )
            
            if response.status_code != 200:
                return issues
            
            issue_list = response.json()
            
            for issue in issue_list:
                issues.append({
                    "identifier": f"issue:{project_id}:{issue['iid']}",
                    "title": f"Issue #{issue['iid']}: {issue['title']}",
                    "url": issue["web_url"],
                    "author": issue.get("author", {}).get("name"),
                    "last_modified": datetime.fromisoformat(
                        issue["updated_at"].replace("Z", "+00:00")
                    ),
                    "file_type": "issue",
                    "metadata": {
                        "project_id": project_id,
                        "issue_iid": issue["iid"],
                        "state": issue["state"],
                        "labels": issue.get("labels", []),
                        "type": "issue"
                    }
                })
            
        except Exception as e:
            logger.error(f"Error getting GitLab issues for project {project_id}: {e}")
        
        return issues
    
    async def _get_merge_requests(self, project_id: int) -> List[Dict[str, Any]]:
        """Get merge requests from a project."""
        merge_requests = []
        
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/merge_requests",
                params={"state": "all", "per_page": 100}
            )
            
            if response.status_code != 200:
                return merge_requests
            
            mr_list = response.json()
            
            for mr in mr_list:
                merge_requests.append({
                    "identifier": f"merge_request:{project_id}:{mr['iid']}",
                    "title": f"MR #{mr['iid']}: {mr['title']}",
                    "url": mr["web_url"],
                    "author": mr.get("author", {}).get("name"),
                    "last_modified": datetime.fromisoformat(
                        mr["updated_at"].replace("Z", "+00:00")
                    ),
                    "file_type": "merge_request",
                    "metadata": {
                        "project_id": project_id,
                        "merge_request_iid": mr["iid"],
                        "state": mr["state"],
                        "source_branch": mr.get("source_branch"),
                        "target_branch": mr.get("target_branch"),
                        "type": "merge_request"
                    }
                })
            
        except Exception as e:
            logger.error(f"Error getting GitLab merge requests for project {project_id}: {e}")
        
        return merge_requests
    
    async def _get_file_content(self, project_id: str, file_path: str) -> str:
        """Get content of a repository file."""
        try:
            # URL encode file path
            import urllib.parse
            encoded_path = urllib.parse.quote(file_path, safe='')
            
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/repository/files/{encoded_path}",
                params={"ref": "main"}  # or "master"
            )
            
            if response.status_code == 404:
                # Try with master branch
                response = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{project_id}/repository/files/{encoded_path}",
                    params={"ref": "master"}
                )
            
            if response.status_code == 200:
                file_data = response.json()
                content = file_data.get("content", "")
                
                # Decode base64 content
                if file_data.get("encoding") == "base64":
                    content = base64.b64decode(content).decode("utf-8")
                
                return content
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting GitLab file content {file_path}: {e}")
            return ""
    
    async def _get_wiki_content(self, project_id: str, wiki_slug: str) -> str:
        """Get content of a wiki page."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/wikis/{wiki_slug}"
            )
            
            if response.status_code == 200:
                wiki_data = response.json()
                return wiki_data.get("content", "")
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting GitLab wiki content {wiki_slug}: {e}")
            return ""
    
    async def _get_issue_content(self, project_id: str, issue_iid: str) -> str:
        """Get content of an issue."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/issues/{issue_iid}"
            )
            
            if response.status_code == 200:
                issue_data = response.json()
                
                content_parts = [
                    f"Title: {issue_data.get('title', '')}",
                    f"Description: {issue_data.get('description', '')}",
                    f"State: {issue_data.get('state', '')}",
                    f"Labels: {', '.join(issue_data.get('labels', []))}",
                    f"Author: {issue_data.get('author', {}).get('name', '')}",
                ]
                
                return "\n\n".join(filter(None, content_parts))
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting GitLab issue content {issue_iid}: {e}")
            return ""
    
    async def _get_merge_request_content(self, project_id: str, mr_iid: str) -> str:
        """Get content of a merge request."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}"
            )
            
            if response.status_code == 200:
                mr_data = response.json()
                
                content_parts = [
                    f"Title: {mr_data.get('title', '')}",
                    f"Description: {mr_data.get('description', '')}",
                    f"State: {mr_data.get('state', '')}",
                    f"Source Branch: {mr_data.get('source_branch', '')}",
                    f"Target Branch: {mr_data.get('target_branch', '')}",
                    f"Author: {mr_data.get('author', {}).get('name', '')}",
                ]
                
                return "\n\n".join(filter(None, content_parts))
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error getting GitLab merge request content {mr_iid}: {e}")
            return ""
    
    async def _get_file_metadata(self, project_id: str, file_path: str) -> Dict[str, Any]:
        """Get metadata for a repository file."""
        # For now, return basic metadata
        return {
            "project_id": project_id,
            "file_path": file_path,
            "type": "repository_file"
        }
    
    async def _get_wiki_metadata(self, project_id: str, wiki_slug: str) -> Dict[str, Any]:
        """Get metadata for a wiki page."""
        return {
            "project_id": project_id,
            "wiki_slug": wiki_slug,
            "type": "wiki_page"
        }
    
    async def _get_issue_metadata(self, project_id: str, issue_iid: str) -> Dict[str, Any]:
        """Get metadata for an issue."""
        return {
            "project_id": project_id,
            "issue_iid": issue_iid,
            "type": "issue"
        }
    
    async def _get_merge_request_metadata(self, project_id: str, mr_iid: str) -> Dict[str, Any]:
        """Get metadata for a merge request."""
        return {
            "project_id": project_id,
            "merge_request_iid": mr_iid,
            "type": "merge_request"
        }
    
    async def cleanup(self):
        """Clean up the HTTP client."""
        if self.client:
            await self.client.aclose()








