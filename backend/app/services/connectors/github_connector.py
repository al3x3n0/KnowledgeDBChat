"""
GitHub connector for ingesting repository content and issues.
"""

import base64
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote_plus
from loguru import logger

from .base_connector import BaseConnector


class GitHubConnector(BaseConnector):
    """Connector for GitHub repositories and content."""

    def __init__(self):
        super().__init__()
        self.client: Optional[httpx.AsyncClient] = None
        self.api_base: str = "https://api.github.com"
        self.token: Optional[str] = None
        # Repositories to index. Items can be {"owner": "org", "repo": "name"} or "owner/repo" strings
        self.repos: List[Dict[str, str]] = []
        self.include_issues: bool = True
        self.include_files: bool = True
        self.include_pull_requests: bool = False
        self.include_wiki: bool = False
        self.file_extensions = ['.md', '.txt', '.rst', '.py', '.js', '.ts', '.java', '.cpp', '.h']
        self.ignore_globs: List[str] = []
        self.max_pages: int = 10
        self.incremental_files: bool = True
        self.use_gitignore: bool = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            self.config = config
            self.api_base = (config.get("github_api_base") or config.get("api_base") or self.api_base).rstrip("/")
            self.token = config.get("token") or config.get("github_token")
            self.include_issues = config.get("include_issues", True)
            self.include_files = config.get("include_files", True)
            self.include_pull_requests = config.get("include_pull_requests", False)
            self.include_wiki = config.get("include_wiki", False)
            self.file_extensions = config.get("file_extensions", self.file_extensions)
            self.ignore_globs = config.get("ignore_globs", []) or []
            self.max_pages = int(config.get("max_pages", self.max_pages))
            self.incremental_files = bool(config.get("incremental_files", True))
            self.use_gitignore = bool(config.get("use_gitignore", False))

            repos_cfg = config.get("repos", [])
            self.repos = []
            for r in repos_cfg:
                if isinstance(r, str) and "/" in r:
                    owner, repo = r.split("/", 1)
                    self.repos.append({"owner": owner, "repo": repo})
                elif isinstance(r, dict) and r.get("owner") and r.get("repo"):
                    self.repos.append({"owner": r["owner"], "repo": r["repo"]})

            if not self.repos:
                raise ValueError("At least one repo must be specified: repos=[\"owner/repo\", ...]")

            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "KnowledgeDBChat-GitHubConnector"
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self.client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
            )

            if await self.test_connection():
                self.is_initialized = True
                logger.info("GitHub connector initialized")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize GitHub connector: {e}")
            return False

    async def test_connection(self) -> bool:
        try:
            if self.token:
                resp = await self.client.get(f"{self.api_base}/user")
                if resp.status_code == 200:
                    return True
                logger.warning(f"GitHub auth check failed: {resp.status_code} - {resp.text[:200]}")
                return False
            if not self.repos:
                return True
            owner = self.repos[0]["owner"]
            name = self.repos[0]["repo"]
            resp = await self.client.get(f"{self.api_base}/repos/{owner}/{name}")
            if resp.status_code == 200:
                return True
            # For public repos, 403 might mean rate limiting - log but allow to proceed
            if resp.status_code == 403:
                remaining = resp.headers.get("x-ratelimit-remaining", "unknown")
                logger.warning(f"GitHub rate limit hit (remaining: {remaining}). Proceeding anyway for public repo.")
                # Still mark as initialized - individual calls may work
                return True
            logger.warning(f"GitHub repo check failed for {owner}/{name}: {resp.status_code} - {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return False

    async def list_documents(self) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        docs: List[Dict[str, Any]] = []
        try:
            for repo in self.repos:
                owner = repo["owner"]
                name = repo["repo"]
                full = f"{owner}/{name}"

                if self.include_files:
                    # Optionally merge .gitignore patterns
                    if self.use_gitignore:
                        await self._merge_gitignore(owner, name)
                        await self._merge_nested_gitignores(owner, name)
                    files = await self._list_repo_files(owner, name)
                    for path in files:
                        ext = "." + path.split(".")[-1] if "." in path else ""
                        if self.file_extensions and ext not in self.file_extensions:
                            continue
                        if self._should_ignore(path):
                            continue
                        docs.append({
                            "identifier": f"file:{full}:{path}",
                            "title": path,
                            "url": f"https://github.com/{owner}/{name}/blob/HEAD/{path}",
                            "file_type": ext.lstrip('.'),
                            "metadata": {"owner": owner, "repo": name, "path": path, "type": "repository_file"},
                        })

                if self.include_issues:
                    issues = await self._list_repo_issues(owner, name)
                    docs.extend(issues)

                if self.include_pull_requests:
                    pulls = await self._list_repo_pulls(owner, name)
                    docs.extend(pulls)

                if self.include_wiki:
                    wiki_docs = await self._list_repo_wiki(owner, name)
                    docs.extend(wiki_docs)
        except Exception as e:
            logger.error(f"Error listing GitHub documents: {e}")
        return docs

    async def list_changed_documents(self, since: datetime) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        changed: List[Dict[str, Any]] = []
        try:
            iso = since.isoformat()
        except Exception:
            iso = None
        try:
            for repo in self.repos:
                owner = repo["owner"]
                name = repo["repo"]
                full = f"{owner}/{name}"
                # Issues since
                if self.include_issues and iso:
                    issues = await self._list_repo_issues(owner, name, since_iso=iso)
                    changed.extend(issues)
                # Pull requests since
                if self.include_pull_requests and iso:
                    pulls = await self._list_repo_pulls(owner, name, since_iso=iso)
                    changed.extend(pulls)
                # Files changed since
                if self.include_files and self.incremental_files and iso:
                    if self.use_gitignore:
                        await self._merge_gitignore(owner, name)
                        await self._merge_nested_gitignores(owner, name)
                    paths = await self._list_changed_files(owner, name, since_iso=iso)
                    for path in paths:
                        ext = "." + path.split(".")[-1] if "." in path else ""
                        if self.file_extensions and ext not in self.file_extensions:
                            continue
                        if self._should_ignore(path):
                            continue
                        changed.append({
                            "identifier": f"file:{full}:{path}",
                            "title": path,
                            "url": f"https://github.com/{owner}/{name}/blob/HEAD/{path}",
                            "file_type": ext.lstrip('.'),
                            "metadata": {"owner": owner, "repo": name, "path": path, "type": "repository_file"},
                        })
                elif self.include_files and not self.incremental_files:
                    # Fallback to full listing when incremental disabled
                    files = await self._list_repo_files(owner, name)
                    for path in files:
                        ext = "." + path.split(".")[-1] if "." in path else ""
                        if self.file_extensions and ext not in self.file_extensions:
                            continue
                        if self._should_ignore(path):
                            continue
                        changed.append({
                            "identifier": f"file:{full}:{path}",
                            "title": path,
                            "url": f"https://github.com/{owner}/{name}/blob/HEAD/{path}",
                            "file_type": ext.lstrip('.'),
                            "metadata": {"owner": owner, "repo": name, "path": path, "type": "repository_file"},
                        })
        except Exception as e:
            logger.error(f"Error listing changed GitHub documents: {e}")
        return changed

    async def list_branches(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Return branches for a repository."""
        self._ensure_initialized()
        branches: List[Dict[str, Any]] = []
        page = 1
        try:
            while True:
                resp = await self.client.get(
                    f"{self.api_base}/repos/{owner}/{repo}/branches",
                    params={"per_page": 100, "page": page},
                )
                if resp.status_code != 200:
                    logger.warning(f"GitHub branch list failed for {owner}/{repo}: {resp.status_code} {resp.text}")
                    break
                data = resp.json()
                if not data:
                    break
                for item in data:
                    commit = item.get("commit") or {}
                    commit_info = commit.get("commit") or {}
                    branches.append(
                        {
                            "name": item.get("name"),
                            "commit_sha": commit.get("sha"),
                            "commit_message": commit_info.get("message"),
                            "commit_author": (commit_info.get("author") or {}).get("name"),
                            "commit_date": (commit_info.get("author") or {}).get("date"),
                            "protected": item.get("protected"),
                        }
                    )
                if len(data) < 100:
                    break
                page += 1
        except Exception as exc:
            logger.error(f"Error listing branches for {owner}/{repo}: {exc}")
        return branches

    async def compare_branches(self, owner: str, repo: str, base_branch: str, compare_branch: str) -> Dict[str, Any]:
        """Use GitHub compare API to diff two branches."""
        self._ensure_initialized()
        try:
            base = quote_plus(base_branch)
            head = quote_plus(compare_branch)
            resp = await self.client.get(f"{self.api_base}/repos/{owner}/{repo}/compare/{base}...{head}")
            if resp.status_code != 200:
                raise ValueError(f"GitHub compare API failed: {resp.status_code} {resp.text}")
            return resp.json()
        except Exception as exc:
            logger.error(f"Error comparing {owner}/{repo} {base_branch}..{compare_branch}: {exc}")
            raise

    # =========================================================================
    # Repository Metadata Methods (for report generation)
    # =========================================================================

    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository metadata including stars, forks, description, etc."""
        self._ensure_initialized()
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}")
            if resp.status_code != 200:
                logger.error(f"Failed to get repo info for {owner}/{repo}: {resp.status_code}")
                return {}
            data = resp.json()
            return {
                "name": data.get("name", ""),
                "full_name": data.get("full_name", f"{owner}/{repo}"),
                "description": data.get("description"),
                "url": data.get("html_url", f"https://github.com/{owner}/{repo}"),
                "default_branch": data.get("default_branch", "main"),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "watchers": data.get("watchers_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "license": (data.get("license") or {}).get("name"),
                "language": data.get("language"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "pushed_at": data.get("pushed_at"),
                "topics": data.get("topics", []),
                "visibility": data.get("visibility", "public"),
                "archived": data.get("archived", False),
                "is_fork": data.get("fork", False),
                "size": data.get("size", 0),  # Size in KB
            }
        except Exception as e:
            logger.error(f"Error getting repo info for {owner}/{repo}: {e}")
            return {}

    async def get_readme(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get README content and HTML."""
        self._ensure_initialized()
        try:
            # Get raw README content
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/readme")
            if resp.status_code != 200:
                logger.debug(f"No README found for {owner}/{repo}")
                return {"content": None, "html": None, "name": None}

            data = resp.json()
            content = None
            if data.get("encoding") == "base64" and data.get("content"):
                try:
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                except Exception:
                    pass

            # Get HTML-rendered README
            html_resp = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}/readme",
                headers={"Accept": "application/vnd.github.html+json"}
            )
            html_content = html_resp.text if html_resp.status_code == 200 else None

            return {
                "content": content,
                "html": html_content,
                "name": data.get("name", "README.md"),
                "path": data.get("path"),
                "url": data.get("html_url"),
            }
        except Exception as e:
            logger.error(f"Error getting README for {owner}/{repo}: {e}")
            return {"content": None, "html": None, "name": None}

    async def get_recent_commits(
        self,
        owner: str,
        repo: str,
        limit: int = 20,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent commits for a repository."""
        self._ensure_initialized()
        commits: List[Dict[str, Any]] = []
        try:
            params = {"per_page": min(limit, 100)}
            if branch:
                params["sha"] = branch

            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/commits", params=params)
            if resp.status_code != 200:
                logger.warning(f"Failed to get commits for {owner}/{repo}: {resp.status_code}")
                return commits

            for item in resp.json()[:limit]:
                commit_data = item.get("commit", {})
                author_info = commit_data.get("author", {})
                commits.append({
                    "sha": item.get("sha", "")[:8],  # Short SHA
                    "full_sha": item.get("sha", ""),
                    "message": commit_data.get("message", "").split("\n")[0],  # First line
                    "full_message": commit_data.get("message", ""),
                    "author": author_info.get("name", "Unknown"),
                    "author_email": author_info.get("email"),
                    "date": author_info.get("date"),
                    "url": item.get("html_url"),
                })
        except Exception as e:
            logger.error(f"Error getting commits for {owner}/{repo}: {e}")
        return commits

    async def get_contributors(self, owner: str, repo: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top contributors for a repository."""
        self._ensure_initialized()
        contributors: List[Dict[str, Any]] = []
        try:
            resp = await self._get(
                f"{self.api_base}/repos/{owner}/{repo}/contributors",
                params={"per_page": min(limit, 100)}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get contributors for {owner}/{repo}: {resp.status_code}")
                return contributors

            for item in resp.json()[:limit]:
                contributors.append({
                    "username": item.get("login", ""),
                    "contributions": item.get("contributions", 0),
                    "avatar_url": item.get("avatar_url"),
                    "profile_url": item.get("html_url"),
                    "type": item.get("type", "User"),
                })
        except Exception as e:
            logger.error(f"Error getting contributors for {owner}/{repo}: {e}")
        return contributors

    async def get_languages(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get language statistics for a repository."""
        self._ensure_initialized()
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/languages")
            if resp.status_code != 200:
                logger.warning(f"Failed to get languages for {owner}/{repo}: {resp.status_code}")
                return {"languages": {}, "total_bytes": 0, "percentages": {}}

            data = resp.json()
            total = sum(data.values()) if data else 0
            percentages = {}
            if total > 0:
                percentages = {lang: round((bytes_count / total) * 100, 1) for lang, bytes_count in data.items()}

            return {
                "languages": data,
                "total_bytes": total,
                "percentages": percentages,
            }
        except Exception as e:
            logger.error(f"Error getting languages for {owner}/{repo}: {e}")
            return {"languages": {}, "total_bytes": 0, "percentages": {}}

    async def get_file_tree(self, owner: str, repo: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get repository file tree structure."""
        self._ensure_initialized()
        try:
            resp = await self._get(
                f"{self.api_base}/repos/{owner}/{repo}/git/trees/HEAD",
                params={"recursive": 1}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get file tree for {owner}/{repo}: {resp.status_code}")
                return {"tree": None, "text": ""}

            tree_data = resp.json().get("tree", [])

            # Build tree structure
            root = {"name": repo, "type": "directory", "path": "", "children": []}
            nodes = {"/": root}

            for item in tree_data:
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
                    # Create missing parent directories
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
                    "size": item.get("size"),
                }
                if item.get("type") == "tree":
                    node["children"] = []

                node_path = "/" + path
                nodes[node_path] = node

                if parent_path in nodes:
                    nodes[parent_path]["children"].append(node)

            # Generate text representation
            text_lines = [f"{repo}/"]
            self._tree_to_text(root, "", text_lines)

            return {
                "tree": root,
                "text": "\n".join(text_lines),
                "total_files": len([t for t in tree_data if t.get("type") == "blob"]),
                "total_dirs": len([t for t in tree_data if t.get("type") == "tree"]),
            }
        except Exception as e:
            logger.error(f"Error getting file tree for {owner}/{repo}: {e}")
            return {"tree": None, "text": ""}

    def _tree_to_text(self, node: Dict, prefix: str, lines: List[str]) -> None:
        """Convert tree node to text lines recursively."""
        children = node.get("children", [])
        # Sort: directories first, then files, alphabetically
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

    async def get_open_issues(self, owner: str, repo: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get open issues for a repository."""
        self._ensure_initialized()
        issues: List[Dict[str, Any]] = []
        try:
            resp = await self._get(
                f"{self.api_base}/repos/{owner}/{repo}/issues",
                params={"state": "open", "per_page": min(limit, 100), "sort": "updated", "direction": "desc"}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get issues for {owner}/{repo}: {resp.status_code}")
                return issues

            for item in resp.json()[:limit]:
                # Skip pull requests
                if item.get("pull_request"):
                    continue
                issues.append({
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "state": item.get("state", "open"),
                    "author": (item.get("user") or {}).get("login", "Unknown"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": [l.get("name") for l in item.get("labels", [])],
                    "url": item.get("html_url"),
                    "comments": item.get("comments", 0),
                })
        except Exception as e:
            logger.error(f"Error getting issues for {owner}/{repo}: {e}")
        return issues

    async def get_open_pull_requests(self, owner: str, repo: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get open pull requests for a repository."""
        self._ensure_initialized()
        prs: List[Dict[str, Any]] = []
        try:
            resp = await self._get(
                f"{self.api_base}/repos/{owner}/{repo}/pulls",
                params={"state": "open", "per_page": min(limit, 100), "sort": "updated", "direction": "desc"}
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to get PRs for {owner}/{repo}: {resp.status_code}")
                return prs

            for item in resp.json()[:limit]:
                prs.append({
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "state": item.get("state", "open"),
                    "author": (item.get("user") or {}).get("login", "Unknown"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": [l.get("name") for l in item.get("labels", [])],
                    "source_branch": (item.get("head") or {}).get("ref", ""),
                    "target_branch": (item.get("base") or {}).get("ref", ""),
                    "url": item.get("html_url"),
                    "draft": item.get("draft", False),
                })
        except Exception as e:
            logger.error(f"Error getting PRs for {owner}/{repo}: {e}")
        return prs

    async def get_document_content(self, identifier: str) -> str:
        self._ensure_initialized()
        try:
            doc_type, full, tail = identifier.split(":", 2)
            owner, repo = full.split("/", 1)
            if doc_type == "file":
                return await self._get_file_content(owner, repo, tail)
            elif doc_type == "issue":
                number = int(tail)
                return await self._get_issue_content(owner, repo, number)
            elif doc_type == "pull_request":
                number = int(tail)
                return await self._get_pull_content(owner, repo, number)
            else:
                return ""
        except Exception as e:
            logger.error(f"Error getting GitHub document content {identifier}: {e}")
            return ""

    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        try:
            doc_type, full, tail = identifier.split(":", 2)
            owner, repo = full.split("/", 1)
            if doc_type == "file":
                return {"owner": owner, "repo": repo, "path": tail, "type": "repository_file"}
            elif doc_type == "issue":
                return {"owner": owner, "repo": repo, "issue_number": int(tail), "type": "issue"}
            else:
                return {}
        except Exception:
            return {}

    async def _list_repo_files(self, owner: str, repo: str) -> List[str]:
        # Use git trees API to list files
        files: List[str] = []
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/git/trees/HEAD", params={"recursive": 1})
            if resp.status_code != 200:
                return files
            data = resp.json()
            for entry in data.get("tree", []):
                if entry.get("type") == "blob" and entry.get("path"):
                    files.append(entry["path"])
        except Exception as e:
            logger.warning(f"Failed to list files for {owner}/{repo}: {e}")
        return files

    async def _get_file_content(self, owner: str, repo: str, path: str) -> str:
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/contents/{path}", params={"ref": "HEAD"})
            if resp.status_code != 200:
                return ""
            data = resp.json()
            content_b64 = data.get("content", "")
            if data.get("encoding") == "base64":
                try:
                    return base64.b64decode(content_b64).decode("utf-8", errors="ignore")
                except Exception:
                    return ""
            return content_b64
        except Exception as e:
            logger.warning(f"Failed to fetch file content {owner}/{repo}:{path}: {e}")
            return ""

    async def _list_repo_issues(self, owner: str, repo: str, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            page = 1
            while page <= self.max_pages:
                params = {"state": "all", "per_page": 100, "page": page}
                if since_iso:
                    params["since"] = since_iso
                resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/issues", params=params)
                if resp.status_code != 200:
                    break
                items = resp.json()
                if not items:
                    break
                for issue in items:
                    if issue.get("pull_request"):
                        continue
                    title = issue.get("title", "")
                    number = issue.get("number")
                    updated_at = issue.get("updated_at")
                    last_modified = None
                    if updated_at:
                        try:
                            last_modified = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        except Exception:
                            pass
                    docs.append({
                        "identifier": f"issue:{owner}/{repo}:{number}",
                        "title": f"Issue #{number}: {title}",
                        "url": issue.get("html_url"),
                        "author": (issue.get("user") or {}).get("login"),
                        "last_modified": last_modified,
                        "file_type": "issue",
                        "metadata": {
                            "owner": owner,
                            "repo": repo,
                            "issue_number": number,
                            "labels": [l.get("name") for l in issue.get("labels", [])],
                            "state": issue.get("state"),
                            "type": "issue",
                        },
                    })
                page += 1
        except Exception as e:
            logger.warning(f"Failed to list issues for {owner}/{repo}: {e}")
        return docs

    async def _list_repo_pulls(self, owner: str, repo: str, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            page = 1
            while page <= self.max_pages:
                params = {"state": "all", "per_page": 100, "page": page}
                # GitHub pulls API does not support 'since'; we sort by updated desc and rely on pagination limits
                params["sort"] = "updated"
                params["direction"] = "desc"
                resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/pulls", params=params)
                if resp.status_code != 200:
                    break
                items = resp.json()
                if not items:
                    break
                for pr in items:
                    updated_at = pr.get("updated_at")
                    if since_iso and updated_at and updated_at < since_iso:
                        # Early stop condition if sorted desc
                        page = self.max_pages + 1
                        break
                    last_modified = None
                    if updated_at:
                        try:
                            last_modified = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        except Exception:
                            pass
                    number = pr.get("number")
                    docs.append({
                        "identifier": f"pull_request:{owner}/{repo}:{number}",
                        "title": f"PR #{number}: {pr.get('title','')}",
                        "url": pr.get("html_url"),
                        "author": (pr.get("user") or {}).get("login"),
                        "last_modified": last_modified,
                        "file_type": "pull_request",
                        "metadata": {
                            "owner": owner,
                            "repo": repo,
                            "pull_number": number,
                            "state": pr.get("state"),
                            "labels": [l.get("name") for l in pr.get("labels", [])],
                            "type": "pull_request",
                        },
                    })
                page += 1
        except Exception as e:
            logger.warning(f"Failed to list PRs for {owner}/{repo}: {e}")
        return docs

    async def _get_pull_content(self, owner: str, repo: str, number: int) -> str:
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/pulls/{number}")
            if resp.status_code != 200:
                return ""
            pr = resp.json()
            parts = [
                f"Title: {pr.get('title','')}",
                f"State: {pr.get('state','')}",
                f"Author: {(pr.get('user') or {}).get('login','')}",
                f"Body: {pr.get('body','')}",
                f"Base: {(pr.get('base') or {}).get('ref','')} | Head: {(pr.get('head') or {}).get('ref','')}",
            ]
            return "\n\n".join([p for p in parts if p])
        except Exception as e:
            logger.warning(f"Failed to fetch PR content for {owner}/{repo}#{number}: {e}")
            return ""

    async def _list_repo_wiki(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            wiki_repo = f"{repo}.wiki"
            resp = await self._get(f"{self.api_base}/repos/{owner}/{wiki_repo}/git/trees/HEAD", params={"recursive": 1})
            if resp.status_code != 200:
                return docs
            data = resp.json()
            for entry in data.get("tree", []):
                if entry.get("type") == "blob" and entry.get("path"):
                    path = entry["path"]
                    ext = "." + path.split(".")[-1] if "." in path else ""
                    if self.file_extensions and ext not in self.file_extensions:
                        continue
                    if self._should_ignore(path):
                        continue
                    docs.append({
                        "identifier": f"file:{owner}/{wiki_repo}:{path}",
                        "title": path,
                        "url": f"https://github.com/{owner}/{repo}/wiki/{path}",
                        "file_type": ext.lstrip('.'),
                        "metadata": {"owner": owner, "repo": wiki_repo, "path": path, "type": "wiki_file"},
                    })
        except Exception as e:
            logger.debug(f"No wiki repo or failed to list wiki for {owner}/{repo}: {e}")
        return docs

    async def _get_issue_content(self, owner: str, repo: str, number: int) -> str:
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/issues/{number}")
            if resp.status_code != 200:
                return ""
            issue = resp.json()
            parts = [
                f"Title: {issue.get('title', '')}",
                f"State: {issue.get('state', '')}",
                f"Author: {(issue.get('user') or {}).get('login', '')}",
                f"Labels: {', '.join([l.get('name') for l in issue.get('labels', [])])}",
                f"Body: {issue.get('body', '')}",
            ]
            return "\n\n".join([p for p in parts if p])
        except Exception as e:
            logger.warning(f"Failed to fetch issue content for {owner}/{repo}#{number}: {e}")
            return ""

    async def cleanup(self):
        if self.client:
            await self.client.aclose()

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """HTTP GET with basic rate-limit handling."""
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code in (429, 403):
                # Check rate limit headers
                try:
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
                    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                except Exception:
                    remaining, reset = 0, 0
                if remaining == 0 and reset:
                    import time
                    now = int(time.time())
                    sleep_for = max(0, min(15, reset - now))  # cap to 15s to avoid long sleeps
                    if sleep_for > 0:
                        import asyncio
                        await asyncio.sleep(sleep_for)
                        return await self.client.get(url, params=params)
            return resp
        except Exception as e:
            logger.warning(f"GitHub GET failed: {e}")
            raise

    async def _list_changed_files(self, owner: str, repo: str, since_iso: str) -> List[str]:
        paths: List[str] = []
        seen = set()
        try:
            page = 1
            pages = min(self.max_pages, 5)
            while page <= pages:
                resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/commits", params={"since": since_iso, "per_page": 50, "page": page})
                if resp.status_code != 200:
                    break
                commits = resp.json()
                if not commits:
                    break
                for c in commits:
                    sha = c.get("sha")
                    if not sha:
                        continue
                    detail = await self._get(f"{self.api_base}/repos/{owner}/{repo}/commits/{sha}")
                    if detail.status_code != 200:
                        continue
                    for f in detail.json().get("files", []):
                        p = f.get("filename")
                        if p and p not in seen and not self._should_ignore(p):
                            seen.add(p)
                            paths.append(p)
                page += 1
        except Exception as e:
            logger.warning(f"Failed to list changed files for {owner}/{repo}: {e}")
        return paths

    def _should_ignore(self, path: str) -> bool:
        try:
            from fnmatch import fnmatch
            for pat in self.ignore_globs:
                if fnmatch(path, pat):
                    return True
        except Exception:
            return False
        return False

    async def _merge_gitignore(self, owner: str, repo: str) -> None:
        """Fetch root .gitignore and merge into ignore_globs."""
        try:
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/contents/.gitignore", params={"ref": "HEAD"})
            if resp.status_code != 200:
                return
            data = resp.json()
            content_b64 = data.get("content", "")
            if data.get("encoding") == "base64":
                import base64
                raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            else:
                raw = content_b64
            patterns: List[str] = []
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                # Convert some .gitignore styles to fnmatch-friendly globs
                if s.startswith('/'):
                    s = s[1:]
                if s.endswith('/'):
                    s = s + '**'
                if not s.startswith('**/') and not s.startswith('*') and '/' in s:
                    s = '**/' + s
                patterns.append(s)
            # Merge uniquely
            merged = set(self.ignore_globs or [])
            for p in patterns:
                merged.add(p)
            self.ignore_globs = list(merged)
        except Exception as e:
            logger.debug(f"Failed to merge .gitignore for {owner}/{repo}: {e}")

    async def _merge_nested_gitignores(self, owner: str, repo: str) -> None:
        """Fetch all nested .gitignore files and merge with directory-anchored patterns."""
        try:
            # Use tree to find .gitignore files
            resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/git/trees/HEAD", params={"recursive": 1})
            if resp.status_code != 200:
                return
            tree = resp.json().get("tree", [])
            gitignores = [e.get("path") for e in tree if e.get("type") == "blob" and e.get("path", "").endswith(".gitignore")]
            if not gitignores:
                return
            merged = set(self.ignore_globs or [])
            for path in gitignores:
                # Skip root .gitignore (handled already)
                if path == ".gitignore":
                    continue
                # Directory for this .gitignore
                import os
                dir_prefix = os.path.dirname(path)
                if not dir_prefix:
                    continue
                # Fetch content
                file_resp = await self._get(f"{self.api_base}/repos/{owner}/{repo}/contents/{path}", params={"ref": "HEAD"})
                if file_resp.status_code != 200:
                    continue
                data = file_resp.json()
                content_b64 = data.get("content", "")
                raw = ""
                if data.get("encoding") == "base64":
                    import base64
                    raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore") if content_b64 else ""
                else:
                    raw = content_b64
                for line in raw.splitlines():
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    # Anchor pattern to directory
                    if s.startswith('/'):
                        s = s[1:]
                    if s.endswith('/'):
                        s = s + '**'
                    # Prefix directory
                    anchored = f"{dir_prefix}/{s}"
                    # Normalize to glob covering subpaths
                    if not anchored.startswith('**/') and '/' in anchored and not anchored.startswith(dir_prefix + '/**'):
                        anchored = '**/' + anchored
                    merged.add(anchored)
            self.ignore_globs = list(merged)
        except Exception as e:
            logger.debug(f"Failed nested .gitignore merge for {owner}/{repo}: {e}")
