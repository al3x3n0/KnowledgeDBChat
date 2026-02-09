"""
GitLab Architecture Diagram Generator Service.

Analyzes GitLab repositories to generate architecture diagrams using LLM.
"""

import base64
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.mermaid_renderer import get_mermaid_renderer, MermaidRenderError


class GitLabArchitectureService:
    """
    Service for generating architecture diagrams from GitLab repositories.

    Analyzes repository structure, README, config files, and code to understand
    the architecture and generate Mermaid diagrams.
    """

    # File patterns for architecture analysis
    ARCHITECTURE_FILES = [
        "README.md", "README.rst", "README.txt", "README",
        "ARCHITECTURE.md", "DESIGN.md", "OVERVIEW.md",
        "docker-compose.yml", "docker-compose.yaml",
        "Dockerfile", "Makefile",
        "package.json", "requirements.txt", "go.mod", "Cargo.toml",
        "pyproject.toml", "setup.py", "pom.xml", "build.gradle",
        ".env.example", "env.example",
    ]

    # Directories that indicate architectural components
    COMPONENT_DIRS = [
        "src", "app", "lib", "pkg", "cmd", "internal", "api",
        "services", "controllers", "handlers", "models", "views",
        "components", "modules", "core", "common", "utils",
        "frontend", "backend", "web", "mobile", "cli",
        "infra", "infrastructure", "deploy", "k8s", "helm",
    ]

    # File extensions for code analysis
    CODE_EXTENSIONS = [
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rs",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ]

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self, gitlab_url: str, token: str) -> httpx.AsyncClient:
        """Get or create HTTP client for GitLab API."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def analyze_repository(
        self,
        gitlab_url: str,
        token: str,
        project_id: str,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a GitLab repository structure.

        Returns:
            Dict containing:
            - project_info: Basic project information
            - directory_structure: Top-level directory structure
            - architecture_files: Content of architecture-relevant files
            - components: Detected components/services
            - dependencies: Detected dependencies
        """
        client = await self._get_client(gitlab_url, token)
        base_url = gitlab_url.rstrip("/")

        # Get project info
        project_info = await self._get_project_info(client, base_url, project_id)
        if not project_info:
            raise ValueError(f"Could not access project: {project_id}")

        default_branch = branch or project_info.get("default_branch", "main")

        # Get repository tree
        tree = await self._get_repo_tree(client, base_url, project_id, default_branch)

        # Analyze directory structure
        directory_structure = self._analyze_directory_structure(tree)

        # Get architecture-relevant files
        architecture_files = await self._get_architecture_files(
            client, base_url, project_id, default_branch, tree
        )

        # Detect components from directory structure
        components = self._detect_components(tree)

        # Extract dependencies
        dependencies = self._extract_dependencies(architecture_files)

        return {
            "project_info": {
                "name": project_info.get("name"),
                "description": project_info.get("description"),
                "default_branch": default_branch,
                "web_url": project_info.get("web_url"),
            },
            "directory_structure": directory_structure,
            "architecture_files": architecture_files,
            "components": components,
            "dependencies": dependencies,
        }

    async def generate_architecture_diagram(
        self,
        gitlab_url: str,
        token: str,
        project_id: str,
        branch: Optional[str] = None,
        diagram_type: str = "auto",
        focus: Optional[str] = None,
        detail_level: str = "medium",
    ) -> Dict[str, Any]:
        """
        Generate an architecture diagram from a GitLab repository.

        Args:
            gitlab_url: GitLab instance URL
            token: GitLab access token
            project_id: Project ID or path
            branch: Branch to analyze (defaults to default branch)
            diagram_type: Type of diagram (flowchart, architecture, c4, etc.)
            focus: Specific aspect to focus on
            detail_level: Level of detail (high, medium, low)

        Returns:
            Dict with diagram code, rendered images, and analysis summary
        """
        # Analyze repository
        analysis = await self.analyze_repository(
            gitlab_url, token, project_id, branch
        )

        # Build prompt for LLM
        prompt = self._build_diagram_prompt(
            analysis, diagram_type, focus, detail_level
        )

        # Generate diagram with LLM
        mermaid_code = await self._generate_mermaid_with_llm(prompt)

        # Render diagram
        renderer = get_mermaid_renderer()
        png_bytes = None
        svg_content = None
        render_error = None

        try:
            png_bytes = await renderer.render_to_png(mermaid_code)
        except MermaidRenderError as e:
            render_error = str(e)
            logger.warning(f"Failed to render PNG: {e}")

        try:
            svg_content = await renderer.render_to_svg(mermaid_code)
        except MermaidRenderError as e:
            if not render_error:
                render_error = str(e)
            logger.warning(f"Failed to render SVG: {e}")

        return {
            "project": analysis["project_info"],
            "mermaid_code": mermaid_code,
            "png_base64": base64.b64encode(png_bytes).decode() if png_bytes else None,
            "svg": svg_content,
            "render_error": render_error,
            "analysis_summary": {
                "components_detected": len(analysis["components"]),
                "architecture_files_analyzed": len(analysis["architecture_files"]),
                "dependencies": analysis["dependencies"],
            },
            "diagram_type": diagram_type,
            "focus": focus,
        }

    async def _get_project_info(
        self, client: httpx.AsyncClient, base_url: str, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get GitLab project information."""
        try:
            import urllib.parse
            encoded_id = urllib.parse.quote_plus(str(project_id))
            response = await client.get(f"{base_url}/api/v4/projects/{encoded_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting project info: {e}")
            return None

    async def _get_repo_tree(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        project_id: str,
        branch: str,
        max_items: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get repository tree recursively."""
        import urllib.parse
        encoded_id = urllib.parse.quote_plus(str(project_id))

        all_items = []
        page = 1

        while len(all_items) < max_items:
            try:
                response = await client.get(
                    f"{base_url}/api/v4/projects/{encoded_id}/repository/tree",
                    params={
                        "recursive": True,
                        "per_page": 100,
                        "page": page,
                        "ref": branch,
                    }
                )

                if response.status_code != 200:
                    break

                items = response.json()
                if not items:
                    break

                all_items.extend(items)
                page += 1

            except Exception as e:
                logger.error(f"Error getting repo tree: {e}")
                break

        return all_items

    async def _get_architecture_files(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        project_id: str,
        branch: str,
        tree: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Get content of architecture-relevant files."""
        import urllib.parse
        encoded_id = urllib.parse.quote_plus(str(project_id))

        files_content = {}

        # Find architecture files in tree
        architecture_paths = []
        for item in tree:
            if item["type"] != "blob":
                continue
            path = item["path"]
            filename = path.split("/")[-1]

            # Check if it's a known architecture file
            if filename in self.ARCHITECTURE_FILES:
                architecture_paths.append(path)
            # Also include root config files
            elif "/" not in path and any(
                filename.endswith(ext) for ext in [".yml", ".yaml", ".json", ".toml"]
            ):
                architecture_paths.append(path)

        # Limit to prevent too many API calls
        for path in architecture_paths[:20]:
            try:
                encoded_path = urllib.parse.quote(path, safe='')
                response = await client.get(
                    f"{base_url}/api/v4/projects/{encoded_id}/repository/files/{encoded_path}",
                    params={"ref": branch}
                )

                if response.status_code == 200:
                    file_data = response.json()
                    content = file_data.get("content", "")

                    if file_data.get("encoding") == "base64":
                        content = base64.b64decode(content).decode("utf-8", errors="ignore")

                    # Limit content size
                    if len(content) < 50000:
                        files_content[path] = content

            except Exception as e:
                logger.warning(f"Error getting file {path}: {e}")

        return files_content

    def _analyze_directory_structure(
        self, tree: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the directory structure of the repository."""
        directories = set()
        files_by_dir = {}
        file_types = {}

        for item in tree:
            path = item["path"]

            if item["type"] == "tree":
                directories.add(path)
            else:
                # Extract directory
                parts = path.split("/")
                if len(parts) > 1:
                    dir_path = "/".join(parts[:-1])
                    if dir_path not in files_by_dir:
                        files_by_dir[dir_path] = []
                    files_by_dir[dir_path].append(parts[-1])

                # Count file types
                ext = "." + path.split(".")[-1] if "." in path else "no_ext"
                file_types[ext] = file_types.get(ext, 0) + 1

        # Get top-level directories
        top_level_dirs = [d for d in directories if "/" not in d]

        return {
            "top_level_dirs": sorted(top_level_dirs),
            "total_directories": len(directories),
            "total_files": sum(len(files) for files in files_by_dir.values()),
            "file_types": dict(sorted(file_types.items(), key=lambda x: -x[1])[:10]),
            "files_by_directory": {
                k: v for k, v in sorted(files_by_dir.items())[:30]
            },
        }

    def _detect_components(self, tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect application components from directory structure."""
        components = []

        # Check for known component directories
        for item in tree:
            if item["type"] != "tree":
                continue

            path = item["path"]
            name = path.split("/")[-1]

            # Top-level or second-level component directories
            depth = path.count("/")
            if depth > 1:
                continue

            if name.lower() in self.COMPONENT_DIRS:
                component_type = self._infer_component_type(name, path, tree)
                components.append({
                    "name": name,
                    "path": path,
                    "type": component_type,
                })

        # Check for microservices pattern (services/*)
        services_paths = [
            item["path"] for item in tree
            if item["type"] == "tree" and item["path"].startswith("services/")
            and item["path"].count("/") == 1
        ]
        for svc_path in services_paths:
            svc_name = svc_path.split("/")[-1]
            components.append({
                "name": svc_name,
                "path": svc_path,
                "type": "microservice",
            })

        return components

    def _infer_component_type(
        self, name: str, path: str, tree: List[Dict[str, Any]]
    ) -> str:
        """Infer the type of component based on name and contents."""
        name_lower = name.lower()

        type_mapping = {
            "frontend": "frontend",
            "backend": "backend",
            "api": "api",
            "web": "frontend",
            "mobile": "mobile",
            "cli": "cli",
            "cmd": "cli",
            "services": "services",
            "controllers": "controllers",
            "handlers": "handlers",
            "models": "data",
            "views": "presentation",
            "components": "ui_components",
            "infra": "infrastructure",
            "infrastructure": "infrastructure",
            "deploy": "deployment",
            "k8s": "kubernetes",
            "helm": "kubernetes",
            "core": "core",
            "lib": "library",
            "pkg": "package",
            "internal": "internal",
        }

        return type_mapping.get(name_lower, "module")

    def _extract_dependencies(
        self, architecture_files: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract dependencies from various config files."""
        dependencies = {
            "python": [],
            "node": [],
            "go": [],
            "docker_services": [],
        }

        for path, content in architecture_files.items():
            filename = path.split("/")[-1]

            # Python dependencies
            if filename == "requirements.txt":
                deps = [
                    line.split("==")[0].split(">=")[0].split("[")[0].strip()
                    for line in content.split("\n")
                    if line.strip() and not line.startswith("#") and not line.startswith("-")
                ]
                dependencies["python"].extend(deps[:30])

            # Node dependencies
            elif filename == "package.json":
                try:
                    import json
                    pkg = json.loads(content)
                    deps = list(pkg.get("dependencies", {}).keys())[:20]
                    dev_deps = list(pkg.get("devDependencies", {}).keys())[:10]
                    dependencies["node"].extend(deps + dev_deps)
                except Exception:
                    pass

            # Go dependencies
            elif filename == "go.mod":
                for line in content.split("\n"):
                    if line.strip().startswith("require") or "\t" in line:
                        parts = line.strip().split()
                        if parts and not parts[0] in ("require", "(", ")"):
                            dependencies["go"].append(parts[0])

            # Docker Compose services
            elif filename in ("docker-compose.yml", "docker-compose.yaml"):
                try:
                    import yaml
                    compose = yaml.safe_load(content)
                    services = compose.get("services", {})
                    dependencies["docker_services"] = list(services.keys())
                except Exception:
                    pass

        return {k: v for k, v in dependencies.items() if v}

    def _build_diagram_prompt(
        self,
        analysis: Dict[str, Any],
        diagram_type: str,
        focus: Optional[str],
        detail_level: str,
    ) -> str:
        """Build the prompt for LLM to generate Mermaid diagram."""

        project_info = analysis["project_info"]
        components = analysis["components"]
        dependencies = analysis["dependencies"]
        architecture_files = analysis["architecture_files"]
        dir_structure = analysis["directory_structure"]

        # Build context from analysis
        context_parts = []

        context_parts.append(f"Project: {project_info['name']}")
        if project_info.get("description"):
            context_parts.append(f"Description: {project_info['description']}")

        context_parts.append(f"\nTop-level directories: {', '.join(dir_structure['top_level_dirs'])}")
        context_parts.append(f"Total files: {dir_structure['total_files']}")
        context_parts.append(f"File types: {dir_structure['file_types']}")

        if components:
            context_parts.append(f"\nDetected components:")
            for comp in components:
                context_parts.append(f"  - {comp['name']} ({comp['type']}): {comp['path']}")

        if dependencies:
            context_parts.append(f"\nDependencies:")
            for dep_type, deps in dependencies.items():
                if deps:
                    context_parts.append(f"  {dep_type}: {', '.join(deps[:15])}")

        # Include relevant file contents
        for path, content in list(architecture_files.items())[:5]:
            if len(content) < 3000:
                context_parts.append(f"\n--- {path} ---")
                context_parts.append(content[:3000])

        context = "\n".join(context_parts)

        # Determine diagram type
        if diagram_type == "auto":
            if "docker_services" in dependencies and len(dependencies["docker_services"]) > 2:
                diagram_type = "flowchart"
            elif len(components) > 5:
                diagram_type = "flowchart"
            else:
                diagram_type = "flowchart"

        # Build detail instructions
        detail_instructions = {
            "high": "Include all major components, their relationships, data flows, external integrations, and internal modules. Show detailed connections.",
            "medium": "Show the main components and their primary relationships. Include key data flows and external dependencies.",
            "low": "Show only the high-level architecture with main components and critical connections.",
        }

        focus_instruction = ""
        if focus:
            focus_instruction = f"\nFocus specifically on: {focus}"

        prompt = f"""Analyze this repository structure and generate a Mermaid {diagram_type} diagram showing the architecture.

{context}

Requirements:
1. Generate valid Mermaid diagram code for a {diagram_type}
2. {detail_instructions.get(detail_level, detail_instructions['medium'])}
3. Use clear, descriptive labels for components
4. Show relationships and data flows between components
5. Group related components logically
6. Include external services if detected (databases, APIs, etc.)
{focus_instruction}

IMPORTANT:
- Output ONLY the Mermaid code, no explanations
- Start with the diagram type declaration (e.g., 'flowchart TB' or 'flowchart LR')
- Use proper Mermaid syntax
- Ensure all node IDs are valid (alphanumeric, underscores)

Generate the Mermaid diagram:"""

        return prompt

    async def _generate_mermaid_with_llm(self, prompt: str) -> str:
        """Generate Mermaid code using LLM."""
        try:
            response = await llm_service.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
            )

            mermaid_code = response.strip()

            # Clean up the response
            if "```mermaid" in mermaid_code:
                mermaid_code = mermaid_code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in mermaid_code:
                mermaid_code = mermaid_code.split("```")[1].split("```")[0].strip()

            return mermaid_code

        except Exception as e:
            logger.error(f"Error generating Mermaid with LLM: {e}")
            # Return a basic fallback diagram
            return """flowchart TB
    subgraph Repository
        A[Source Code] --> B[Build]
        B --> C[Application]
    end
    C --> D[Users]"""


# Singleton instance
_service: Optional[GitLabArchitectureService] = None


def get_gitlab_architecture_service() -> GitLabArchitectureService:
    """Get the singleton GitLabArchitectureService instance."""
    global _service
    if _service is None:
        _service = GitLabArchitectureService()
    return _service
