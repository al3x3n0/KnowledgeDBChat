"""
Repository Analysis Service.

Aggregates repository data from GitHub/GitLab connectors into structured analysis results.
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import DocumentSource
from app.schemas.repo_report import (
    RepoInfo,
    CommitInfo,
    IssueInfo,
    PullRequestInfo,
    ContributorInfo,
    LanguageStats,
    FileTreeNode,
    RepoInsights,
    RepoAnalysisResult,
)
from app.services.connectors.github_connector import GitHubConnector
from app.services.connectors.gitlab_connector import GitLabConnector
from app.services.llm_service import LLMService, UserLLMSettings


class RepoAnalysisService:
    """
    Service for analyzing GitHub/GitLab repositories.

    Supports two modes:
    1. From existing DocumentSource - uses saved configuration
    2. From ad-hoc URL - one-time analysis with provided credentials
    """

    def __init__(self):
        self.llm_service = LLMService()

    async def analyze_from_source(
        self,
        source_id: UUID,
        db: AsyncSession,
        sections: List[str],
        progress_callback: Optional[callable] = None,
        user_id: Optional[UUID] = None,
    ) -> RepoAnalysisResult:
        """
        Analyze a repository from an existing DocumentSource.

        Args:
            source_id: ID of the DocumentSource
            db: Database session
            sections: List of sections to include in analysis
            progress_callback: Optional callback for progress updates

        Returns:
            RepoAnalysisResult with all requested data
        """
        # Load source configuration
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise ValueError(f"DocumentSource {source_id} not found")

        source_type = source.source_type.lower()
        config = source.config or {}

        if source_type == "github":
            return await self._analyze_github(
                config=config,
                sections=sections,
                progress_callback=progress_callback,
                user_id=user_id,
                db=db,
            )
        elif source_type == "gitlab":
            return await self._analyze_gitlab(
                config=config,
                sections=sections,
                progress_callback=progress_callback,
                user_id=user_id,
                db=db,
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    async def analyze_from_url(
        self,
        repo_url: str,
        token: Optional[str],
        sections: List[str],
        progress_callback: Optional[callable] = None,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> RepoAnalysisResult:
        """
        Analyze a repository from an ad-hoc URL.

        Args:
            repo_url: Repository URL (GitHub or GitLab)
            token: Access token for private repos
            sections: List of sections to include
            progress_callback: Optional callback for progress updates

        Returns:
            RepoAnalysisResult with all requested data
        """
        repo_type, owner, repo = self._parse_repo_url(repo_url)

        if repo_type == "github":
            config = {
                "token": token,
                "repos": [f"{owner}/{repo}"]
            }
            return await self._analyze_github(
                config=config,
                sections=sections,
                progress_callback=progress_callback,
                owner_override=owner,
                repo_override=repo,
                user_id=user_id,
                db=db,
            )
        elif repo_type == "gitlab":
            # Extract GitLab URL base
            gitlab_base = self._extract_gitlab_base(repo_url)
            project_path = f"{owner}/{repo}"
            config = {
                "gitlab_url": gitlab_base,
                "token": token,
                "projects": [{"id": project_path, "name": repo}]
            }
            return await self._analyze_gitlab(
                config=config,
                sections=sections,
                progress_callback=progress_callback,
                project_override=project_path,
                user_id=user_id,
                db=db,
            )
        else:
            raise ValueError(f"Could not determine repository type from URL: {repo_url}")

    def _parse_repo_url(self, url: str) -> tuple[str, str, str]:
        """
        Parse a repository URL to extract type, owner, and repo name.

        Returns:
            Tuple of (repo_type, owner, repo_name)
        """
        # GitHub patterns
        github_patterns = [
            r"github\.com[:/]([^/]+)/([^/?#\s]+)",
            r"api\.github\.com/repos/([^/]+)/([^/?#\s]+)",
        ]
        for pattern in github_patterns:
            match = re.search(pattern, url)
            if match:
                return ("github", match.group(1), match.group(2).removesuffix(".git"))

        # GitLab patterns
        gitlab_patterns = [
            r"gitlab\.com[:/]([^/]+)/([^/?#\s]+)",
            r"gitlab\.[^/]+[:/]([^/]+)/([^/?#\s]+)",
        ]
        for pattern in gitlab_patterns:
            match = re.search(pattern, url)
            if match:
                return ("gitlab", match.group(1), match.group(2).removesuffix(".git"))

        raise ValueError(f"Could not parse repository URL: {url}")

    def _extract_gitlab_base(self, url: str) -> str:
        """Extract GitLab instance base URL."""
        match = re.match(r"(https?://[^/]+)", url)
        if match:
            return match.group(1)
        return "https://gitlab.com"

    async def _analyze_github(
        self,
        config: Dict[str, Any],
        sections: List[str],
        progress_callback: Optional[callable] = None,
        owner_override: Optional[str] = None,
        repo_override: Optional[str] = None,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> RepoAnalysisResult:
        """Analyze a GitHub repository."""
        connector = GitHubConnector()
        try:
            repos = config.get("repos", [])
            logger.info(f"Initializing GitHub connector for repos: {repos}")
            if not await connector.initialize(config):
                repo_str = repos[0] if repos else "unknown"
                has_token = bool(config.get("token"))
                raise ValueError(
                    f"Failed to initialize GitHub connector for {repo_str}. "
                    f"Token provided: {has_token}. "
                    "Check if the repository exists and is accessible."
                )

            # Determine owner and repo
            if owner_override and repo_override:
                owner, repo = owner_override, repo_override
            else:
                repos = config.get("repos", [])
                if not repos:
                    raise ValueError("No repository configured")
                if isinstance(repos[0], str) and "/" in repos[0]:
                    owner, repo = repos[0].split("/", 1)
                else:
                    owner = repos[0].get("owner", "")
                    repo = repos[0].get("repo", "")

            return await self._collect_github_data(connector, owner, repo, sections, progress_callback, user_id, db)
        finally:
            await connector.cleanup()

    async def _collect_github_data(
        self,
        connector: GitHubConnector,
        owner: str,
        repo: str,
        sections: List[str],
        progress_callback: Optional[callable] = None,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> RepoAnalysisResult:
        """Collect data from GitHub repository."""
        # Get basic repo info
        if progress_callback:
            await progress_callback(10, "Fetching repository info")

        repo_data = await connector.get_repository_info(owner, repo)
        if not repo_data:
            raise ValueError(f"Could not fetch repository info for {owner}/{repo}")

        repo_info = RepoInfo(
            name=repo_data.get("name", repo),
            full_name=repo_data.get("full_name", f"{owner}/{repo}"),
            description=repo_data.get("description"),
            url=repo_data.get("url", f"https://github.com/{owner}/{repo}"),
            default_branch=repo_data.get("default_branch", "main"),
            stars=repo_data.get("stars", 0),
            forks=repo_data.get("forks", 0),
            watchers=repo_data.get("watchers", 0),
            license=repo_data.get("license"),
            language=repo_data.get("language"),
            created_at=self._parse_datetime(repo_data.get("created_at")),
            updated_at=self._parse_datetime(repo_data.get("updated_at")),
        )

        result = RepoAnalysisResult(repo_info=repo_info)

        # README
        if "readme" in sections:
            if progress_callback:
                await progress_callback(15, "Fetching README")
            readme_data = await connector.get_readme(owner, repo)
            result.readme_content = readme_data.get("content")
            result.readme_html = readme_data.get("html")

        # File structure
        if "file_structure" in sections:
            if progress_callback:
                await progress_callback(25, "Analyzing file structure")
            tree_data = await connector.get_file_tree(owner, repo)
            result.file_tree_text = tree_data.get("text")
            if tree_data.get("tree"):
                result.file_tree = self._dict_to_file_tree(tree_data["tree"])

        # Commits
        if "commits" in sections:
            if progress_callback:
                await progress_callback(35, "Fetching recent commits")
            commits_data = await connector.get_recent_commits(owner, repo, limit=20)
            result.commits = [
                CommitInfo(
                    sha=c.get("sha", ""),
                    message=c.get("message", ""),
                    author=c.get("author", "Unknown"),
                    date=self._parse_datetime(c.get("date")) or datetime.utcnow(),
                    url=c.get("url"),
                )
                for c in commits_data
            ]

        # Issues
        if "issues" in sections:
            if progress_callback:
                await progress_callback(45, "Fetching open issues")
            issues_data = await connector.get_open_issues(owner, repo, limit=20)
            result.issues = [
                IssueInfo(
                    number=i.get("number", 0),
                    title=i.get("title", ""),
                    state=i.get("state", "open"),
                    author=i.get("author", "Unknown"),
                    created_at=self._parse_datetime(i.get("created_at")) or datetime.utcnow(),
                    labels=i.get("labels", []),
                    url=i.get("url"),
                )
                for i in issues_data
            ]

        # Pull requests
        if "pull_requests" in sections:
            if progress_callback:
                await progress_callback(55, "Fetching open pull requests")
            prs_data = await connector.get_open_pull_requests(owner, repo, limit=20)
            result.pull_requests = [
                PullRequestInfo(
                    number=pr.get("number", 0),
                    title=pr.get("title", ""),
                    state=pr.get("state", "open"),
                    author=pr.get("author", "Unknown"),
                    created_at=self._parse_datetime(pr.get("created_at")) or datetime.utcnow(),
                    labels=pr.get("labels", []),
                    source_branch=pr.get("source_branch", ""),
                    target_branch=pr.get("target_branch", ""),
                    url=pr.get("url"),
                )
                for pr in prs_data
            ]

        # Code statistics
        if "code_stats" in sections:
            if progress_callback:
                await progress_callback(60, "Analyzing code statistics")
            lang_data = await connector.get_languages(owner, repo)
            result.language_stats = LanguageStats(
                languages=lang_data.get("languages", {}),
                total_bytes=lang_data.get("total_bytes", 0),
                percentages=lang_data.get("percentages", {}),
            )

        # Contributors
        if "contributors" in sections:
            if progress_callback:
                await progress_callback(65, "Fetching contributors")
            contrib_data = await connector.get_contributors(owner, repo, limit=10)
            result.contributors = [
                ContributorInfo(
                    username=c.get("username", ""),
                    name=c.get("name"),
                    contributions=c.get("contributions", 0),
                    avatar_url=c.get("avatar_url"),
                )
                for c in contrib_data
            ]

        # LLM insights
        if "architecture" in sections or "technology_stack" in sections:
            if progress_callback:
                await progress_callback(70, "Generating insights")
            result.insights = await self._generate_insights(
                repo_info=result.repo_info,
                readme=result.readme_content,
                file_tree=result.file_tree_text,
                language_stats=result.language_stats,
                include_architecture="architecture" in sections,
                include_tech_stack="technology_stack" in sections,
                user_id=user_id,
                db=db,
            )

        if progress_callback:
            await progress_callback(80, "Analysis complete")

        return result

    async def _analyze_gitlab(
        self,
        config: Dict[str, Any],
        sections: List[str],
        progress_callback: Optional[callable] = None,
        project_override: Optional[str] = None,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> RepoAnalysisResult:
        """Analyze a GitLab repository."""
        connector = GitLabConnector()
        try:
            if not await connector.initialize(config):
                raise ValueError("Failed to initialize GitLab connector")

            # Determine project
            if project_override:
                project_id = project_override
            else:
                projects = config.get("projects", [])
                if not projects:
                    raise ValueError("No project configured")
                project_id = projects[0].get("id") or projects[0].get("name")

            return await self._collect_gitlab_data(connector, project_id, sections, progress_callback, user_id, db)
        finally:
            await connector.cleanup()

    async def _collect_gitlab_data(
        self,
        connector: GitLabConnector,
        project_id: str,
        sections: List[str],
        progress_callback: Optional[callable] = None,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None
    ) -> RepoAnalysisResult:
        """Collect data from GitLab repository."""
        # Get basic repo info
        if progress_callback:
            await progress_callback(10, "Fetching repository info")

        repo_data = await connector.get_repository_info(project_id)
        if not repo_data:
            raise ValueError(f"Could not fetch repository info for {project_id}")

        repo_info = RepoInfo(
            name=repo_data.get("name", str(project_id)),
            full_name=repo_data.get("full_name", str(project_id)),
            description=repo_data.get("description"),
            url=repo_data.get("url", ""),
            default_branch=repo_data.get("default_branch", "main"),
            stars=repo_data.get("stars", 0),
            forks=repo_data.get("forks", 0),
            watchers=0,
            license=None,
            language=None,
            created_at=self._parse_datetime(repo_data.get("created_at")),
            updated_at=self._parse_datetime(repo_data.get("updated_at")),
        )

        result = RepoAnalysisResult(repo_info=repo_info)

        # README
        if "readme" in sections:
            if progress_callback:
                await progress_callback(15, "Fetching README")
            readme_data = await connector.get_readme(project_id)
            result.readme_content = readme_data.get("content")

        # File structure
        if "file_structure" in sections:
            if progress_callback:
                await progress_callback(25, "Analyzing file structure")
            tree_data = await connector.get_file_tree(project_id)
            result.file_tree_text = tree_data.get("text")
            if tree_data.get("tree"):
                result.file_tree = self._dict_to_file_tree(tree_data["tree"])

        # Commits
        if "commits" in sections:
            if progress_callback:
                await progress_callback(35, "Fetching recent commits")
            commits_data = await connector.get_recent_commits(project_id, limit=20)
            result.commits = [
                CommitInfo(
                    sha=c.get("sha", ""),
                    message=c.get("message", ""),
                    author=c.get("author", "Unknown"),
                    date=self._parse_datetime(c.get("date")) or datetime.utcnow(),
                    url=c.get("url"),
                )
                for c in commits_data
            ]

        # Issues
        if "issues" in sections:
            if progress_callback:
                await progress_callback(45, "Fetching open issues")
            issues_data = await connector.get_open_issues(project_id, limit=20)
            result.issues = [
                IssueInfo(
                    number=i.get("number", 0),
                    title=i.get("title", ""),
                    state=i.get("state", "open"),
                    author=i.get("author", "Unknown"),
                    created_at=self._parse_datetime(i.get("created_at")) or datetime.utcnow(),
                    labels=i.get("labels", []),
                    url=i.get("url"),
                )
                for i in issues_data
            ]

        # Merge requests (GitLab equivalent of pull requests)
        if "pull_requests" in sections:
            if progress_callback:
                await progress_callback(55, "Fetching open merge requests")
            mrs_data = await connector.get_open_merge_requests(project_id, limit=20)
            result.pull_requests = [
                PullRequestInfo(
                    number=mr.get("number", 0),
                    title=mr.get("title", ""),
                    state=mr.get("state", "open"),
                    author=mr.get("author", "Unknown"),
                    created_at=self._parse_datetime(mr.get("created_at")) or datetime.utcnow(),
                    labels=mr.get("labels", []),
                    source_branch=mr.get("source_branch", ""),
                    target_branch=mr.get("target_branch", ""),
                    url=mr.get("url"),
                )
                for mr in mrs_data
            ]

        # Code statistics
        if "code_stats" in sections:
            if progress_callback:
                await progress_callback(60, "Analyzing code statistics")
            lang_data = await connector.get_languages(project_id)
            result.language_stats = LanguageStats(
                languages=lang_data.get("languages", {}),
                total_bytes=lang_data.get("total_bytes", 0),
                percentages=lang_data.get("percentages", {}),
            )

        # Contributors
        if "contributors" in sections:
            if progress_callback:
                await progress_callback(65, "Fetching contributors")
            contrib_data = await connector.get_contributors(project_id, limit=10)
            result.contributors = [
                ContributorInfo(
                    username=c.get("username", ""),
                    name=c.get("name"),
                    contributions=c.get("contributions", 0),
                    avatar_url=c.get("avatar_url"),
                )
                for c in contrib_data
            ]

        # LLM insights
        if "architecture" in sections or "technology_stack" in sections:
            if progress_callback:
                await progress_callback(70, "Generating insights")
            result.insights = await self._generate_insights(
                repo_info=result.repo_info,
                readme=result.readme_content,
                file_tree=result.file_tree_text,
                language_stats=result.language_stats,
                include_architecture="architecture" in sections,
                include_tech_stack="technology_stack" in sections,
                user_id=user_id,
                db=db,
            )

        if progress_callback:
            await progress_callback(80, "Analysis complete")

        return result

    async def _generate_insights(
        self,
        repo_info: RepoInfo,
        readme: Optional[str],
        file_tree: Optional[str],
        language_stats: Optional[LanguageStats],
        include_architecture: bool,
        include_tech_stack: bool,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> RepoInsights:
        """Generate LLM-powered insights about the repository."""
        insights = RepoInsights()

        # Load user LLM settings
        user_settings = None
        if user_id and db:
            try:
                from app.models.memory import UserPreferences
                result = await db.execute(
                    select(UserPreferences).where(UserPreferences.user_id == user_id)
                )
                user_prefs = result.scalar_one_or_none()
                if user_prefs:
                    user_settings = UserLLMSettings.from_preferences(user_prefs)
                    logger.info(f"Loaded user LLM settings for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to load user LLM settings: {e}")

        # Build context
        context_parts = [
            f"Repository: {repo_info.full_name}",
            f"Description: {repo_info.description or 'No description'}",
        ]

        if language_stats and language_stats.percentages:
            lang_str = ", ".join([f"{lang}: {pct}%" for lang, pct in list(language_stats.percentages.items())[:5]])
            context_parts.append(f"Languages: {lang_str}")

        if file_tree:
            # Truncate file tree if too long
            tree_truncated = file_tree[:3000] + "..." if len(file_tree) > 3000 else file_tree
            context_parts.append(f"File structure:\n{tree_truncated}")

        if readme:
            # Truncate README if too long
            readme_truncated = readme[:2000] + "..." if len(readme) > 2000 else readme
            context_parts.append(f"README excerpt:\n{readme_truncated}")

        context = "\n\n".join(context_parts)

        try:
            if include_architecture:
                arch_prompt = f"""Analyze this repository and provide a brief architecture summary (2-3 paragraphs).
Focus on:
- Overall project structure and organization
- Key architectural patterns used
- Main components and their relationships

Repository context:
{context}

Provide a concise architecture summary:"""

                arch_response = await self.llm_service.generate_response(
                    query=arch_prompt,
                    task_type="summarization",
                    user_id=user_id,
                    db=db,
                    user_settings=user_settings,
                )
                insights.architecture_summary = arch_response

                # Extract key features
                features_prompt = f"""Based on this repository, list 3-5 key features or capabilities.
Format as a simple list, one feature per line.

Repository: {repo_info.full_name}
Description: {repo_info.description or 'No description'}

Key features:"""

                features_response = await self.llm_service.generate_response(
                    query=features_prompt,
                    task_type="summarization",
                    user_id=user_id,
                    db=db,
                    user_settings=user_settings,
                )
                features_text = features_response
                insights.key_features = [
                    line.strip().lstrip("•-*").strip()
                    for line in features_text.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ][:5]

            if include_tech_stack:
                tech_prompt = f"""Identify the technology stack used in this repository.
List technologies, frameworks, and tools detected.
Format as a simple list, one technology per line.

Repository context:
{context}

Technology stack:"""

                tech_response = await self.llm_service.generate_response(
                    query=tech_prompt,
                    task_type="summarization",
                    user_id=user_id,
                    db=db,
                    user_settings=user_settings,
                )
                tech_text = tech_response
                insights.technology_stack = [
                    line.strip().lstrip("•-*").strip()
                    for line in tech_text.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ][:10]

        except Exception as e:
            logger.error(f"Error generating insights: {e}")

        return insights

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object."""
        if not dt_str:
            return None
        try:
            # Handle various formats
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except Exception:
            return None

    def _dict_to_file_tree(self, data: Dict) -> FileTreeNode:
        """Convert dictionary to FileTreeNode."""
        children = []
        for child in data.get("children", []):
            children.append(self._dict_to_file_tree(child))

        return FileTreeNode(
            name=data.get("name", ""),
            type=data.get("type", "file"),
            path=data.get("path", ""),
            children=children,
            size=data.get("size"),
            language=data.get("language"),
        )


# Singleton instance
repo_analysis_service = RepoAnalysisService()
