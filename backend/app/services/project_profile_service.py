"""Helpers for inferring and formatting repository project profiles."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob


def infer_project_profile_from_paths(paths: List[str]) -> Dict[str, Any]:
    """Infer a lightweight project profile from repository-like file paths."""
    cleaned: List[str] = []
    for raw in paths:
        path = str(raw or "").strip().replace("\\", "/")
        if not path:
            continue
        if "://" in path:
            continue
        cleaned.append(path.lstrip("./"))
    if not cleaned:
        return {}

    ext_counter: Counter[str] = Counter()
    top_level_counter: Counter[str] = Counter()
    marker_paths: set[str] = set()
    test_paths: List[str] = []
    marker_dir_map: Dict[str, set[str]] = {}

    marker_files = {
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
    }

    for path in cleaned:
        parts = [p for p in path.split("/") if p]
        if not parts:
            continue
        top_level_counter.update([parts[0].lower()])
        lower_path = path.lower()
        lower_name = parts[-1].lower()

        if (
            lower_name in marker_files
            or lower_name.endswith(".csproj")
            or lower_name.endswith(".sln")
        ):
            marker_paths.add(path)
            marker_dir = "/".join(parts[:-1]).strip("/") or "."
            dirs = marker_dir_map.get(lower_name)
            if dirs is None:
                dirs = set()
                marker_dir_map[lower_name] = dirs
            dirs.add(marker_dir)

        if (
            lower_path.startswith("tests/")
            or lower_path.startswith("test/")
            or "/tests/" in lower_path
            or "/test/" in lower_path
            or "/__tests__/" in lower_path
            or "/spec/" in lower_path
            or lower_name.endswith("_test.py")
            or lower_name.endswith(".test.ts")
            or lower_name.endswith(".test.tsx")
            or lower_name.endswith(".spec.ts")
            or lower_name.endswith(".spec.tsx")
        ):
            if len(test_paths) < 24:
                test_paths.append(path)

        if "." in lower_name:
            ext = lower_name.rsplit(".", 1)[-1]
            if ext:
                ext_counter.update([ext])

    ext_to_stack = {
        "py": "python",
        "ipynb": "python",
        "ts": "typescript",
        "tsx": "typescript",
        "js": "javascript",
        "jsx": "javascript",
        "go": "go",
        "rs": "rust",
        "java": "java",
        "kt": "kotlin",
        "cs": "dotnet",
        "csproj": "dotnet",
        "php": "php",
        "rb": "ruby",
        "swift": "swift",
        "scala": "scala",
        "cpp": "cpp",
        "c": "c",
    }

    detected_stack: List[str] = []
    seen_stack: set[str] = set()
    for ext, _count in ext_counter.most_common(16):
        stack = ext_to_stack.get(ext)
        if stack and stack not in seen_stack:
            seen_stack.add(stack)
            detected_stack.append(stack)

    marker_names = {p.split("/")[-1].lower() for p in marker_paths}
    if "package.json" in marker_names and "node" not in seen_stack:
        detected_stack.append("node")
        seen_stack.add("node")
    if "pyproject.toml" in marker_names or "requirements.txt" in marker_names:
        if "python" not in seen_stack:
            detected_stack.append("python")
            seen_stack.add("python")
    if "go.mod" in marker_names and "go" not in seen_stack:
        detected_stack.append("go")
        seen_stack.add("go")
    if "cargo.toml" in marker_names and "rust" not in seen_stack:
        detected_stack.append("rust")
        seen_stack.add("rust")
    if (
        any(name.endswith(".csproj") or name.endswith(".sln") for name in marker_names)
        and "dotnet" not in seen_stack
    ):
        detected_stack.append("dotnet")
        seen_stack.add("dotnet")

    def _iter_marker_dirs(*names: str) -> List[str]:
        dirs: set[str] = set()
        for name in names:
            dirs.update(marker_dir_map.get(name, set()))
        return sorted(dirs)

    def _find_test_target(base_dir: str) -> Optional[str]:
        if base_dir in {"", "."}:
            root_candidates = ["tests", "test"]
            for candidate in root_candidates:
                if any(str(path).startswith(f"{candidate}/") for path in test_paths):
                    return candidate
            return None

        prefixes = [f"{base_dir}/tests/", f"{base_dir}/test/"]
        for prefix in prefixes:
            if any(str(path).startswith(prefix) for path in test_paths):
                return prefix[:-1]
        return base_dir

    command_groups: Dict[str, List[str]] = {
        "install": [],
        "build": [],
        "test": [],
        "test_fallback": [],
    }

    def _append_group(group: str, commands: List[str]) -> None:
        rows = command_groups.get(group)
        if not isinstance(rows, list):
            rows = []
            command_groups[group] = rows
        for cmd in commands:
            text = str(cmd or "").strip()
            if text and text not in rows:
                rows.append(text)

    def _node_pkg_manager_for_dir(base_dir: str) -> str:
        if base_dir in marker_dir_map.get("pnpm-lock.yaml", set()):
            return "pnpm"
        if base_dir in marker_dir_map.get("yarn.lock", set()):
            return "yarn"
        return "npm"

    for base_dir in _iter_marker_dirs("package.json"):
        pkg_manager = _node_pkg_manager_for_dir(base_dir)
        if pkg_manager == "pnpm":
            install_cmd = (
                "pnpm install" if base_dir == "." else f"cd {base_dir} && pnpm install"
            )
            build_cmd = (
                "pnpm build" if base_dir == "." else f"cd {base_dir} && pnpm build"
            )
            test_cmd = (
                "CI=true pnpm test -- --watchAll=false"
                if base_dir == "."
                else f"cd {base_dir} && CI=true pnpm test -- --watchAll=false"
            )
            fallback_cmd = (
                "pnpm test" if base_dir == "." else f"cd {base_dir} && pnpm test"
            )
        elif pkg_manager == "yarn":
            install_cmd = (
                "yarn install" if base_dir == "." else f"cd {base_dir} && yarn install"
            )
            build_cmd = (
                "yarn build" if base_dir == "." else f"cd {base_dir} && yarn build"
            )
            test_cmd = (
                "CI=true yarn test --watchAll=false"
                if base_dir == "."
                else f"cd {base_dir} && CI=true yarn test --watchAll=false"
            )
            fallback_cmd = (
                "yarn test" if base_dir == "." else f"cd {base_dir} && yarn test"
            )
        else:
            install_cmd = (
                "npm install" if base_dir == "." else f"npm --prefix {base_dir} install"
            )
            build_cmd = (
                "npm run build"
                if base_dir == "."
                else f"npm --prefix {base_dir} run build"
            )
            test_cmd = (
                "CI=true npm test -- --watchAll=false"
                if base_dir == "."
                else f"CI=true npm --prefix {base_dir} test -- --watchAll=false"
            )
            fallback_cmd = (
                "npm test" if base_dir == "." else f"npm --prefix {base_dir} test"
            )

        _append_group("install", [install_cmd])
        _append_group("build", [build_cmd])
        _append_group("test", [test_cmd])
        _append_group("test_fallback", [fallback_cmd])

    for base_dir in _iter_marker_dirs("pyproject.toml", "requirements.txt"):
        test_target = _find_test_target(base_dir)
        poetry_managed = base_dir in marker_dir_map.get("poetry.lock", set())
        if poetry_managed:
            primary = (
                f"poetry run pytest -q {test_target}"
                if test_target
                else (
                    "poetry run pytest -q"
                    if base_dir == "."
                    else f"cd {base_dir} && poetry run pytest -q"
                )
            )
            fallback = (
                f"python -m pytest -q {test_target}"
                if test_target
                else (
                    "python -m pytest -q"
                    if base_dir == "."
                    else f"python -m pytest -q {base_dir}"
                )
            )
            install_cmd = (
                "poetry install"
                if base_dir == "."
                else f"cd {base_dir} && poetry install"
            )
            _append_group("install", [install_cmd])
            _append_group("test", [primary])
            _append_group(
                "test_fallback", [fallback, fallback.replace("python -m", "python3 -m")]
            )
        else:
            primary = (
                f"python -m pytest -q {test_target}"
                if test_target
                else (
                    "python -m pytest -q"
                    if base_dir == "."
                    else f"python -m pytest -q {base_dir}"
                )
            )
            _append_group("test", [primary])
            _append_group(
                "test_fallback",
                [
                    primary.replace("python -m", "python3 -m"),
                    primary.replace("python -m pytest", "pytest"),
                ],
            )

    for base_dir in _iter_marker_dirs("go.mod"):
        if base_dir == ".":
            _append_group("test", ["go test ./..."])
            _append_group("build", ["go build ./..."])
        else:
            _append_group("test", [f"cd {base_dir} && go test ./..."])
            _append_group("build", [f"cd {base_dir} && go build ./..."])

    dotnet_markers = [
        path
        for path in sorted(marker_paths)
        if path.lower().endswith(".csproj") or path.lower().endswith(".sln")
    ]
    if dotnet_markers:
        for marker in dotnet_markers:
            _append_group("build", [f'dotnet build "{marker}"'])
            _append_group("test", [f'dotnet test "{marker}"'])
            _append_group("test_fallback", ["dotnet test"])

    for base_dir in _iter_marker_dirs("makefile"):
        _append_group(
            "test", ["make test" if base_dir == "." else f"make -C {base_dir} test"]
        )

    suggested_commands: List[str] = []
    for group in ("install", "build", "test", "test_fallback"):
        suggested_commands.extend(command_groups.get(group) or [])
    deduped_commands: List[str] = []
    for cmd in suggested_commands:
        if cmd not in deduped_commands:
            deduped_commands.append(cmd)

    bootstrap_notes: List[str] = []
    if command_groups.get("install"):
        bootstrap_notes.append(
            "Install dependencies before running build/test commands in a fresh environment."
        )
    if command_groups.get("test_fallback"):
        bootstrap_notes.append(
            "If the primary test command fails due to missing toolchain binaries, try a fallback test command."
        )

    return {
        "sampled_files": len(cleaned),
        "detected_stack": detected_stack[:8],
        "top_extensions": [
            {"ext": ext, "count": count} for ext, count in ext_counter.most_common(12)
        ],
        "top_directories": [
            {"dir": name, "count": count}
            for name, count in top_level_counter.most_common(12)
        ],
        "marker_files": sorted(list(marker_paths))[:24],
        "test_paths": test_paths[:24],
        "command_groups": {k: v[:8] for k, v in command_groups.items() if v},
        "bootstrap_notes": bootstrap_notes[:6],
        "suggested_commands": deduped_commands[:10],
    }


async def build_project_profile(
    job: AgentJob,
    db: AsyncSession,
    *,
    source_id: Optional[str] = None,
    max_files: int = 400,
) -> Dict[str, Any]:
    """Build project profile from ingested source metadata and document paths."""
    from app.models.document import Document, DocumentSource

    max_files = max(50, min(int(max_files or 400), 2000))
    source_uuid: Optional[UUID] = None
    if source_id:
        try:
            source_uuid = UUID(str(source_id).strip())
        except Exception:
            source_uuid = None

    source_obj = None
    if source_uuid:
        source_obj = await db.get(DocumentSource, source_uuid)

    rows = []
    query = select(
        Document.source_identifier, Document.file_path, Document.title
    ).order_by(desc(Document.updated_at))
    if source_uuid:
        query = query.where(Document.source_id == source_uuid)
    query = query.limit(max_files)
    try:
        res = await db.execute(query)
        rows = res.all()
    except Exception:
        rows = []

    paths: List[str] = []
    for source_identifier, file_path, title in rows:
        candidate = str(file_path or "").strip() or str(source_identifier or "").strip()
        if candidate:
            paths.append(candidate)
        elif title:
            paths.append(str(title))

    inferred = infer_project_profile_from_paths(paths)
    profile = {
        "source_id": str(source_obj.id)
        if source_obj
        else (str(source_uuid) if source_uuid else None),
        "source_name": str(source_obj.name) if source_obj else None,
        "source_type": str(source_obj.source_type) if source_obj else None,
        "generated_at": datetime.utcnow().isoformat(),
        "goal_hint": str(job.goal or "")[:240],
        **(inferred if isinstance(inferred, dict) else {}),
    }
    if source_obj and isinstance(source_obj.config, dict):
        source_cfg = source_obj.config
        repo_url = str(
            source_cfg.get("repo_url") or source_cfg.get("url") or ""
        ).strip()
        default_branch = str(
            source_cfg.get("branch") or source_cfg.get("default_branch") or ""
        ).strip()
        if repo_url:
            profile["repository_url"] = repo_url
        if default_branch:
            profile["default_branch"] = default_branch
    return profile


def format_project_profile_for_prompt(state: Dict[str, Any]) -> str:
    """Render project profile context for planner prompt."""
    profile = (
        state.get("project_profile")
        if isinstance(state.get("project_profile"), dict)
        else {}
    )
    if not profile:
        return ""

    lines: List[str] = ["PROJECT PROFILE:"]
    source_name = str(profile.get("source_name") or "").strip()
    source_type = str(profile.get("source_type") or "").strip()
    if source_name or source_type:
        lines.append(
            f"- Source: {source_name or 'unknown'} ({source_type or 'unknown'})"
        )

    stack = profile.get("detected_stack")
    if isinstance(stack, list) and stack:
        lines.append(f"- Detected stack: {', '.join([str(x) for x in stack[:8]])}")

    commands = profile.get("suggested_commands")
    if isinstance(commands, list) and commands:
        lines.append(
            f"- Suggested commands: {', '.join([str(x) for x in commands[:6]])}"
        )

    command_groups = profile.get("command_groups")
    if isinstance(command_groups, dict):
        bootstrap_cmds = (
            command_groups.get("install")
            if isinstance(command_groups.get("install"), list)
            else []
        )
        test_cmds = (
            command_groups.get("test")
            if isinstance(command_groups.get("test"), list)
            else []
        )
        fallback_cmds = (
            command_groups.get("test_fallback")
            if isinstance(command_groups.get("test_fallback"), list)
            else []
        )
        if bootstrap_cmds:
            lines.append(
                f"- Bootstrap: {', '.join([str(x) for x in bootstrap_cmds[:4]])}"
            )
        if test_cmds:
            lines.append(
                f"- Preferred verification: {', '.join([str(x) for x in test_cmds[:4]])}"
            )
        if fallback_cmds:
            lines.append(
                f"- Verification fallback: {', '.join([str(x) for x in fallback_cmds[:4]])}"
            )

    markers = profile.get("marker_files")
    if isinstance(markers, list) and markers:
        lines.append("- Key files:")
        for item in markers[:8]:
            lines.append(f"  - {str(item)}")

    notes = profile.get("bootstrap_notes")
    if isinstance(notes, list) and notes:
        lines.append("- Bootstrap notes:")
        for item in notes[:4]:
            lines.append(f"  - {str(item)}")

    test_paths = profile.get("test_paths")
    if isinstance(test_paths, list) and test_paths:
        lines.append("- Test locations:")
        for item in test_paths[:6]:
            lines.append(f"  - {str(item)}")

    return "\n".join(lines)
