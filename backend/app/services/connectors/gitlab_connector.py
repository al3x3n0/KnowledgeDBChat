"""
GitLab connector for ingesting repository content, wikis, and issues.
"""

import base64
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import urllib.parse

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
        self.ignore_globs = []
        self.use_gitignore = False
    
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
            self.ignore_globs = config.get("ignore_globs", []) or []
            self.use_gitignore = bool(config.get("use_gitignore", False))
            self.ignore_globs = config.get("ignore_globs", []) or []
            
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
                    # Optionally merge .gitignore patterns
                    if self.use_gitignore:
                        try:
                            await self._merge_gitignore(project_info["id"], project_info.get("default_branch"))
                            await self._merge_nested_gitignores(project_info["id"], project_info.get("default_branch"))
                        except Exception:
                            pass
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

    async def list_changed_documents(self, since: datetime) -> List[Dict[str, Any]]:
        """List documents changed since a timestamp."""
        self._ensure_initialized()
        changed: List[Dict[str, Any]] = []
        try:
            # If no specific projects configured, get user's projects
            if not self.projects:
                self.projects = await self._get_user_projects()
            for project in self.projects:
                project_id = project.get("id") or project.get("name")
                if not project_id:
                    continue
                # Changed repository files via commit diffs
                try:
                    files = await self._list_changed_files(project_id, since)
                    for file_path in files:
                        file_ext = "." + file_path.split(".")[-1] if "." in file_path else ""
                        if self.file_extensions and file_ext not in self.file_extensions:
                            continue
                        if self._should_ignore(file_path):
                            continue
                        changed.append({
                            "identifier": f"file:{project_id}:{file_path}",
                            "title": file_path,
                            "url": f"{self.base_url}/{file_path}",
                            "file_type": file_ext.lstrip('.'),
                            "metadata": {"project_id": project_id, "file_path": file_path, "type": "repository_file"}
                        })
                except Exception:
                    pass
                # Changed issues
                if self.include_issues and project.get("include_issues", True):
                    changed.extend(await self._list_issues(project_id, updated_after=since))
                # Changed merge requests
                if self.include_merge_requests and project.get("include_merge_requests", False):
                    changed.extend(await self._list_merge_requests(project_id, updated_after=since))
        except Exception as e:
            logger.warning(f"Failed to list GitLab changed documents: {e}")
        return changed
    
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
            # Get repository tree (paginate)
            page = 1
            while True:
                response = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{project_id}/repository/tree",
                    params={"recursive": True, "per_page": 100, "page": page}
                )
                if response.status_code == 429:
                    import asyncio
                    await asyncio.sleep(2)
                    continue
                if response.status_code != 200:
                    break
                tree_items = response.json()
                if not tree_items:
                    break
                for item in tree_items:
                    if item["type"] == "blob":  # File
                        file_path = item["path"]
                        file_ext = "." + file_path.split(".")[-1] if "." in file_path else ""
                        
                        # Filter by file extensions
                        if self.file_extensions and file_ext not in self.file_extensions:
                            continue
                        # Ignore patterns
                        if self._should_ignore(file_path):
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
                page += 1
        except Exception as e:
            logger.error(f"Error getting GitLab repository files for project {project_id}: {e}")
        
        return files

    def _should_ignore(self, path: str) -> bool:
        try:
            from fnmatch import fnmatch
            for pat in self.ignore_globs:
                if fnmatch(path, pat):
                    return True
        except Exception:
            return False
        return False

    async def _merge_gitignore(self, project_id: int, default_branch: Optional[str] = None) -> None:
        """Fetch and merge root .gitignore into ignore_globs."""
        try:
            if not default_branch:
                info = await self._get_project_info(project_id)
                default_branch = (info or {}).get("default_branch")
            if not default_branch:
                return
            # GitLab API expects encoded file path
            file_path_enc = "%2Egitignore"  # ".gitignore"
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/repository/files/{file_path_enc}",
                params={"ref": default_branch}
            )
            if resp.status_code != 200:
                return
            data = resp.json()
            content_b64 = data.get("content", "")
            import base64
            raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore") if content_b64 else ""
            patterns = []
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('/'):
                    s = s[1:]
                if s.endswith('/'):
                    s = s + '**'
                if not s.startswith('**/') and not s.startswith('*') and '/' in s:
                    s = '**/' + s
                patterns.append(s)
            merged = set(self.ignore_globs or [])
            for p in patterns:
                merged.add(p)
            self.ignore_globs = list(merged)
        except Exception as e:
            from loguru import logger
            logger.debug(f"Failed to merge .gitignore for project {project_id}: {e}")

    async def _merge_nested_gitignores(self, project_id: int, default_branch: Optional[str] = None) -> None:
        """Fetch nested .gitignore files and merge directory-anchored patterns."""
        try:
            if not default_branch:
                info = await self._get_project_info(project_id)
                default_branch = (info or {}).get("default_branch")
            if not default_branch:
                return
            # List repo tree for .gitignore files
            page = 1
            gitignores = []
            while True:
                resp = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{project_id}/repository/tree",
                    params={"recursive": True, "per_page": 100, "page": page, "ref": default_branch}
                )
                if resp.status_code != 200:
                    break
                items = resp.json()
                if not items:
                    break
                for it in items:
                    p = it.get("path", "")
                    if it.get("type") == "blob" and p.endswith('.gitignore') and p != '.gitignore':
                        gitignores.append(p)
                page += 1
            if not gitignores:
                return
            import base64, os
            merged = set(self.ignore_globs or [])
            for path in gitignores:
                dir_prefix = os.path.dirname(path)
                if not dir_prefix:
                    continue
                enc_path = path.replace('/', '%2F')
                resp = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{project_id}/repository/files/{enc_path}",
                    params={"ref": default_branch}
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                content_b64 = data.get('content', '')
                raw = base64.b64decode(content_b64).decode('utf-8', errors='ignore') if content_b64 else ''
                for line in raw.splitlines():
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if s.startswith('/'):
                        s = s[1:]
                    if s.endswith('/'):
                        s = s + '**'
                    anchored = f"{dir_prefix}/{s}"
                    if not anchored.startswith('**/') and '/' in anchored and not anchored.startswith(dir_prefix + '/**'):
                        anchored = '**/' + anchored
                    merged.add(anchored)
            self.ignore_globs = list(merged)
        except Exception as e:
            from loguru import logger
            logger.debug(f"Failed nested .gitignore merge for project {project_id}: {e}")
    
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

    async def _list_issues(self, project_id: int, updated_after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        try:
            params = {"state": "all", "per_page": 100}
            if updated_after:
                params["updated_after"] = updated_after.isoformat()
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/issues",
                params=params
            )
            if response.status_code != 200:
                return issues
            issue_list = response.json()
            for issue in issue_list:
                issues.append({
                    "identifier": f"issue:{project_id}:{issue['iid']}",
                    "title": f"Issue #{issue['iid']}: {issue['title']}",
                    "url": issue.get("web_url"),
                    "author": (issue.get("author") or {}).get("name"),
                    "last_modified": datetime.fromisoformat(issue["updated_at"].replace("Z", "+00:00")) if issue.get("updated_at") else None,
                    "file_type": "issue",
                    "metadata": {"project_id": project_id, "issue_iid": issue["iid"], "state": issue.get("state"), "labels": issue.get("labels", []), "type": "issue"}
                })
        except Exception as e:
            logger.warning(f"Failed to list GitLab issues: {e}")
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

    async def _list_merge_requests(self, project_id: int, updated_after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        mrs: List[Dict[str, Any]] = []
        try:
            params = {"state": "all", "per_page": 100}
            if updated_after:
                params["updated_after"] = updated_after.isoformat()
            response = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/merge_requests",
                params=params
            )
            if response.status_code != 200:
                return mrs
            mr_list = response.json()
            for mr in mr_list:
                mrs.append({
                    "identifier": f"merge_request:{project_id}:{mr['iid']}",
                    "title": f"MR !{mr['iid']}: {mr.get('title','')}",
                    "url": mr.get("web_url"),
                    "author": (mr.get("author") or {}).get("name"),
                    "last_modified": datetime.fromisoformat(mr["updated_at"].replace("Z", "+00:00")) if mr.get("updated_at") else None,
                    "file_type": "merge_request",
                    "metadata": {"project_id": project_id, "merge_request_iid": mr["iid"], "state": mr.get("state"), "type": "merge_request"}
                })
        except Exception as e:
            logger.warning(f"Failed to list GitLab MRs: {e}")
        return mrs

    async def _list_changed_files(self, project_id: int, since: datetime) -> List[str]:
        paths: List[str] = []
        seen = set()
        try:
            # List commits since
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{project_id}/repository/commits",
                params={"since": since.isoformat(), "per_page": 50}
            )
            if resp.status_code != 200:
                return paths
            commits = resp.json()
            for c in commits:
                sha = c.get("id") or c.get("short_id")
                if not sha:
                    continue
                diffs = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{project_id}/repository/commits/{sha}/diff"
                )
                if diffs.status_code != 200:
                    continue
                for d in diffs.json():
                    p = d.get("new_path") or d.get("old_path")
                    if p and p not in seen and not self._should_ignore(p):
                        seen.add(p)
                        paths.append(p)
        except Exception as e:
            logger.warning(f"Failed to list changed files for project {project_id}: {e}")
        return paths
    
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

    async def list_branches(self, project_id: str) -> List[Dict[str, Any]]:
        """Return repository branches for a project."""
        self._ensure_initialized()
        branches: List[Dict[str, Any]] = []
        encoded_id = urllib.parse.quote_plus(str(project_id))
        page = 1
        try:
            while True:
                resp = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{encoded_id}/repository/branches",
                    params={"per_page": 100, "page": page},
                )
                if resp.status_code != 200:
                    logger.warning(f"GitLab branch list failed for {project_id}: {resp.status_code} {resp.text}")
                    break
                data = resp.json()
                if not data:
                    break
                for item in data:
                    commit = item.get("commit") or {}
                    branches.append(
                        {
                            "name": item.get("name"),
                            "commit_sha": commit.get("id") or commit.get("short_id"),
                            "commit_message": commit.get("title"),
                            "commit_author": commit.get("author_name"),
                            "commit_date": commit.get("committed_date"),
                            "protected": item.get("protected"),
                        }
                    )
                if len(data) < 100:
                    break
                page += 1
        except Exception as exc:
            logger.error(f"Error listing branches for GitLab project {project_id}: {exc}")
        return branches

    async def compare_branches(self, project_id: str, base_branch: str, compare_branch: str) -> Dict[str, Any]:
        """Use GitLab compare API to diff two branches."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{encoded_id}/repository/compare",
                params={"from": base_branch, "to": compare_branch},
            )
            if resp.status_code != 200:
                raise ValueError(f"GitLab compare API failed: {resp.status_code} {resp.text}")
            return resp.json()
        except Exception as exc:
            logger.error(f"Error comparing GitLab branches for {project_id}: {exc}")
            raise

    # =========================================================================
    # Repository Metadata Methods (for report generation)
    # =========================================================================

    async def get_repository_info(self, project_id: str) -> Dict[str, Any]:
        """Get repository metadata including stars, forks, description, etc."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        try:
            resp = await self.client.get(f"{self.base_url}/api/v4/projects/{encoded_id}")
            if resp.status_code != 200:
                logger.error(f"Failed to get repo info for {project_id}: {resp.status_code}")
                return {}
            data = resp.json()
            return {
                "name": data.get("name", ""),
                "full_name": data.get("path_with_namespace", str(project_id)),
                "description": data.get("description"),
                "url": data.get("web_url", f"{self.base_url}/{project_id}"),
                "default_branch": data.get("default_branch", "main"),
                "stars": data.get("star_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "visibility": data.get("visibility", "private"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("last_activity_at"),
                "topics": data.get("topics", []) or data.get("tag_list", []),
                "archived": data.get("archived", False),
                "avatar_url": data.get("avatar_url"),
                "namespace": data.get("namespace", {}).get("name"),
            }
        except Exception as e:
            logger.error(f"Error getting repo info for {project_id}: {e}")
            return {}

    async def get_readme(self, project_id: str) -> Dict[str, Any]:
        """Get README content."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        try:
            # Get project info first to find default branch
            info = await self.get_repository_info(project_id)
            default_branch = info.get("default_branch", "main")

            # Try common README filenames
            readme_files = ["README.md", "README.rst", "README.txt", "README", "readme.md"]
            for readme_name in readme_files:
                encoded_path = urllib.parse.quote(readme_name, safe='')
                resp = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{encoded_id}/repository/files/{encoded_path}",
                    params={"ref": default_branch}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = None
                    if data.get("encoding") == "base64" and data.get("content"):
                        try:
                            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                    return {
                        "content": content,
                        "html": None,  # GitLab doesn't provide rendered HTML easily
                        "name": data.get("file_name", readme_name),
                        "path": data.get("file_path"),
                    }

            return {"content": None, "html": None, "name": None}
        except Exception as e:
            logger.error(f"Error getting README for {project_id}: {e}")
            return {"content": None, "html": None, "name": None}

    async def get_recent_commits(
        self,
        project_id: str,
        limit: int = 20,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent commits for a project."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        commits: List[Dict[str, Any]] = []
        try:
            params = {"per_page": min(limit, 100)}
            if branch:
                params["ref_name"] = branch

            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{encoded_id}/repository/commits",
                params=params
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get commits for {project_id}: {resp.status_code}")
                return commits

            for item in resp.json()[:limit]:
                commits.append({
                    "sha": item.get("short_id", "")[:8],
                    "full_sha": item.get("id", ""),
                    "message": item.get("title", ""),
                    "full_message": item.get("message", ""),
                    "author": item.get("author_name", "Unknown"),
                    "author_email": item.get("author_email"),
                    "date": item.get("committed_date"),
                    "url": item.get("web_url"),
                })
        except Exception as e:
            logger.error(f"Error getting commits for {project_id}: {e}")
        return commits

    async def get_contributors(self, project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top contributors for a project."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        contributors: List[Dict[str, Any]] = []
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{encoded_id}/repository/contributors",
                params={"per_page": min(limit, 100)}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get contributors for {project_id}: {resp.status_code}")
                return contributors

            for item in resp.json()[:limit]:
                contributors.append({
                    "username": item.get("name", ""),
                    "name": item.get("name"),
                    "contributions": item.get("commits", 0),
                    "email": item.get("email"),
                    "avatar_url": None,  # GitLab doesn't provide this in contributors endpoint
                })
        except Exception as e:
            logger.error(f"Error getting contributors for {project_id}: {e}")
        return contributors

    async def get_languages(self, project_id: str) -> Dict[str, Any]:
        """Get language statistics for a project."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        try:
            resp = await self.client.get(f"{self.base_url}/api/v4/projects/{encoded_id}/languages")
            if resp.status_code != 200:
                logger.warning(f"Failed to get languages for {project_id}: {resp.status_code}")
                return {"languages": {}, "total_bytes": 0, "percentages": {}}

            data = resp.json()
            # GitLab returns percentages directly, not bytes
            # Convert to our format
            return {
                "languages": {},  # GitLab doesn't give byte counts
                "total_bytes": 0,
                "percentages": data,
            }
        except Exception as e:
            logger.error(f"Error getting languages for {project_id}: {e}")
            return {"languages": {}, "total_bytes": 0, "percentages": {}}

    async def get_file_tree(self, project_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get repository file tree structure."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        try:
            # Get project info for name
            info = await self.get_repository_info(project_id)
            repo_name = info.get("name", str(project_id))
            default_branch = info.get("default_branch", "main")

            # Get tree
            all_items = []
            page = 1
            while True:
                resp = await self.client.get(
                    f"{self.base_url}/api/v4/projects/{encoded_id}/repository/tree",
                    params={"recursive": True, "per_page": 100, "page": page, "ref": default_branch}
                )
                if resp.status_code != 200:
                    break
                items = resp.json()
                if not items:
                    break
                all_items.extend(items)
                if len(items) < 100:
                    break
                page += 1

            # Build tree structure
            root = {"name": repo_name, "type": "directory", "path": "", "children": []}
            nodes = {"/": root}

            for item in all_items:
                path = item.get("path", "")
                if not path:
                    continue

                # Skip deeply nested files
                depth = path.count("/")
                if depth >= max_depth:
                    continue

                parts = path.split("/")
                parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"

                # Ensure parent exists
                if parent_path not in nodes and len(parts) > 1:
                    current_path = ""
                    for i, part in enumerate(parts[:-1]):
                        current_path = "/" + "/".join(parts[:i+1])
                        if current_path not in nodes:
                            parent_parent = "/" + "/".join(parts[:i]) if i > 0 else "/"
                            new_node = {"name": part, "type": "directory", "path": current_path.lstrip("/"), "children": []}
                            nodes[current_path] = new_node
                            if parent_parent in nodes:
                                nodes[parent_parent]["children"].append(new_node)

                # Create node
                node = {
                    "name": parts[-1],
                    "type": "file" if item.get("type") == "blob" else "directory",
                    "path": path,
                }
                if item.get("type") == "tree":
                    node["children"] = []

                node_path = "/" + path
                nodes[node_path] = node

                if parent_path in nodes:
                    nodes[parent_path]["children"].append(node)

            # Generate text representation
            text_lines = [f"{repo_name}/"]
            self._tree_to_text(root, "", text_lines)

            return {
                "tree": root,
                "text": "\n".join(text_lines),
                "total_files": len([t for t in all_items if t.get("type") == "blob"]),
                "total_dirs": len([t for t in all_items if t.get("type") == "tree"]),
            }
        except Exception as e:
            logger.error(f"Error getting file tree for {project_id}: {e}")
            return {"tree": None, "text": ""}

    def _tree_to_text(self, node: Dict, prefix: str, lines: List[str]) -> None:
        """Convert tree node to text lines recursively."""
        children = node.get("children", [])
        children = sorted(children, key=lambda x: (x.get("type") != "directory", x.get("name", "").lower()))

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            name = child.get("name", "")
            if child.get("type") == "directory":
                name += "/"
            lines.append(f"{prefix}{connector}{name}")

            if child.get("type") == "directory" and child.get("children"):
                extension = "    " if is_last else "│   "
                self._tree_to_text(child, prefix + extension, lines)

    async def get_open_issues(self, project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get open issues for a project."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        issues: List[Dict[str, Any]] = []
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{encoded_id}/issues",
                params={"state": "opened", "per_page": min(limit, 100), "order_by": "updated_at", "sort": "desc"}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get issues for {project_id}: {resp.status_code}")
                return issues

            for item in resp.json()[:limit]:
                issues.append({
                    "number": item.get("iid"),
                    "title": item.get("title", ""),
                    "state": item.get("state", "opened"),
                    "author": (item.get("author") or {}).get("name", "Unknown"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": item.get("labels", []),
                    "url": item.get("web_url"),
                    "comments": item.get("user_notes_count", 0),
                })
        except Exception as e:
            logger.error(f"Error getting issues for {project_id}: {e}")
        return issues

    async def get_open_merge_requests(self, project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get open merge requests for a project."""
        self._ensure_initialized()
        encoded_id = urllib.parse.quote_plus(str(project_id))
        mrs: List[Dict[str, Any]] = []
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/v4/projects/{encoded_id}/merge_requests",
                params={"state": "opened", "per_page": min(limit, 100), "order_by": "updated_at", "sort": "desc"}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get MRs for {project_id}: {resp.status_code}")
                return mrs

            for item in resp.json()[:limit]:
                mrs.append({
                    "number": item.get("iid"),
                    "title": item.get("title", ""),
                    "state": item.get("state", "opened"),
                    "author": (item.get("author") or {}).get("name", "Unknown"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": item.get("labels", []),
                    "source_branch": item.get("source_branch", ""),
                    "target_branch": item.get("target_branch", ""),
                    "url": item.get("web_url"),
                    "draft": item.get("work_in_progress", False) or item.get("draft", False),
                })
        except Exception as e:
            logger.error(f"Error getting MRs for {project_id}: {e}")
        return mrs

    async def cleanup(self):
        """Clean up the HTTP client."""
        if self.client:
            await self.client.aclose()



