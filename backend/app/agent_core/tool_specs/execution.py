"""Code execution, sandboxes, custom tools and workspace snapshots.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="create_custom_tool",
        description="Create a reusable custom tool for later use by you, by workflows, "
            "or by future jobs. Use it when you find yourself repeating the "
            "same shaped work. The tool is owned by this user and persists "
            "after the job ends. Types: transform (Jinja2/JSONPath over "
            "inputs), llm_prompt (templated model call), webhook (HTTP call to "
            "an external API), python (sandboxed, no subprocess/filesystem/"
            "network).",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Tool name, unique for this user",
                },
                "description": {
                    "type": "string",
                    "description": "What the tool does and when to use it",
                },
                "tool_type": {
                    "type": "string",
                    "description": ("One of: transform, llm_prompt, webhook, python"),
                },
                "parameters_schema": {
                    "type": "object",
                    "description": "JSON Schema for the tool's inputs",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Type-specific configuration. llm_prompt: "
                        "{'user_prompt': '...'} and optional 'system_prompt'. "
                        "python: {'code': '...'}. webhook: {'url': ..., "
                        "'method': ...}. transform: {'expression': ...}. "
                        "Templates use Jinja2, so reference inputs as "
                        "{{ input_name }} with double braces; single braces are "
                        "left as literal text."
                    ),
                },
            },
            "required": ["name", "tool_type", "config"],
        },
        effects="write",
        pii_risk="medium",
    ),
    ToolSpec(
        name="run_custom_tool",
        description="Execute a user-defined custom tool by name. Custom tools include webhooks, data transformers, Python scripts, and LLM prompts.",
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the custom tool to execute",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters for the tool",
                },
            },
            "required": ["tool_name"],
        },
        effects="write",
        pii_risk="medium",
    ),
    ToolSpec(
        name="list_custom_tools",
        description="List available custom tools that can be executed.",
        parameters={
            "type": "object",
            "properties": {
                "tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "external_agent",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                        "workflow_runner",
                    ],
                    "description": "Filter by tool type (optional)",
                }
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="project_bootstrap",
        description="Build a lightweight project profile from ingested repository files (stack, key files, test paths, and suggested commands).",
        parameters={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Optional document source UUID to scope profiling",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum source files to sample (default: 400, max: 2000)",
                    "default": 400,
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="execute_python",
        description="Run Python code in a RestrictedPython sandbox. Output via the "
            "'result' variable. Only whitelisted pure-Python modules are "
            "importable: there is no subprocess, no filesystem and no network, "
            "so this CANNOT compile code or invoke a toolchain. Use "
            "compile_c_snippet or benchmark_c_snippet for compiler work.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (max 30, default 10)",
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="medium",
        pii_risk="medium",
    ),
    ToolSpec(
        name="execute_data_pipeline",
        description="Process data with pandas/numpy operations in a Docker sandbox. Pass data via 'input_data' variable, output via 'result' variable.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using pandas/numpy for data processing",
                },
                "input_data": {
                    "type": "object",
                    "description": "Data to process (passed as dict to the code)",
                },
                "input_document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs whose content should be loaded as input",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout (max 300, default 60)",
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        job_types=('data_analysis',),
    ),
    ToolSpec(
        name="write_and_run_script",
        description="Write a Python script, execute it in a Docker sandbox, and return the results. For multi-file or complex logic.",
        parameters={
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "Filename for the script (e.g., 'analysis.py')",
                },
                "script_content": {
                    "type": "string",
                    "description": "Full script content",
                },
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "pip packages to install from whitelist (pandas, numpy, scipy, etc.)",
                },
                "arguments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments to pass to the script",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout (max 300, default 120)",
                },
                "input_data": {
                    "type": "object",
                    "description": "Data passed as /workspace/input.json",
                },
            },
            "required": ["script_name", "script_content"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="high",
        job_types=('data_analysis',),
    ),
    ToolSpec(
        name="clone_and_index_repo",
        description="Clone a git repository into a temporary coding workspace and index its file tree. Returns a workspace_id for subsequent file operations.",
        parameters={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of a git DocumentSource in KB (preferred)",
                },
                "repo_url": {
                    "type": "string",
                    "description": "Git clone URL (alternative to source_id, requires code execution enabled)",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to check out (default: main/master)",
                },
            },
            "required": [],
        },
        network="egress",
        cost_tier="medium",
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="browse_repo_files",
        description="List files and directories in the coding workspace with optional glob filtering.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: root)",
                    "default": ".",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter (e.g., '**/*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return (default 200)",
                    "default": 200,
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="read_file",
        description="Read file contents from the coding workspace, optionally limited to a line range.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path in workspace",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-based, optional)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (1-based, optional)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max chars to return (default 20000)",
                    "default": 20000,
                },
            },
            "required": ["path"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="write_file",
        description="Write or overwrite a file in the coding workspace.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {"type": "string", "description": "Relative file path"},
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if missing (default true)",
                    "default": True,
                },
            },
            "required": ["path", "content"],
        },
        effects="write",
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="apply_patch",
        description="Apply a unified diff to files in the coding workspace. Supports fuzzy hunk matching.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "diff": {"type": "string", "description": "Unified diff text"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate patch without applying (default false)",
                    "default": False,
                },
            },
            "required": ["diff"],
        },
        effects="write",
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="run_command",
        description="Run a shell command in the coding workspace. Gated by unsafe_code_execution_enabled feature flag.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (max 120, default 30)",
                    "default": 30,
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables",
                },
            },
            "required": ["command"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="search_code",
        description="Search for text patterns in workspace files using regex (grep-like).",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search in (default: root)",
                    "default": ".",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob to limit files (e.g., '*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches to return (default 50)",
                    "default": 50,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default 2)",
                    "default": 2,
                },
            },
            "required": ["pattern"],
        },
        pii_risk="medium",
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="get_workspace_status",
        description="Show modified, added, and deleted files in the coding workspace compared to original state.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "show_diff_summary": {
                    "type": "boolean",
                    "description": "Include change statistics (default true)",
                    "default": True,
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="create_workspace_checkpoint",
        description="Create a bounded recovery checkpoint of the active coding workspace before risky edits.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "label": {
                    "type": "string",
                    "description": "Short reason for the checkpoint",
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="list_workspace_checkpoints",
        description="List available recovery checkpoints for the active coding workspace.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="restore_workspace_checkpoint",
        description="Restore a recovery checkpoint. By default, first preserves the current workspace as another checkpoint.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint ID returned by create/list checkpoint",
                },
                "preserve_current": {
                    "type": "boolean",
                    "description": "Checkpoint the current state before restore (default true)",
                    "default": True,
                },
            },
            "required": ["checkpoint_id"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="hydrate_candidate_snapshot",
        description="Load the system-provided immutable candidate snapshot into the active workspace after verifying its baseline and file hashes.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "snapshot_id": {
                    "type": "string",
                    "description": "Candidate snapshot ID when multiple system-provided candidates are available",
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="persist_durable_workspace_checkpoint",
        description="Persist the active mutation-owner workspace as an immutable restart-safe session checkpoint.",
        parameters={
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "label": {
                    "type": "string",
                    "description": "Short reason for preserving this state",
                },
            },
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="list_durable_workspace_checkpoints",
        description="List restart-safe checkpoints bound to the current coding session.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="restore_durable_workspace_checkpoint",
        description="Restore a restart-safe checkpoint belonging to the current coding session into a clean reconstructed workspace.",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint_id": {
                    "type": "string",
                    "description": "Durable checkpoint ID from the current job session",
                },
            },
            "required": ["checkpoint_id"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="retrieve_repo_symbols",
        description="Search for code symbols (functions, classes, methods) in the coding workspace. Returns ranked matches with file locations and line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (function names, class names, keywords)",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
                "language_filter": {
                    "type": "string",
                    "enum": ["python", "typescript", "javascript"],
                    "description": "Filter by programming language (optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max symbols to return (default 20)",
                },
            },
            "required": ["query"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="get_symbol_context",
        description="Get a symbol's full definition, surrounding code context, and related symbols in the same file.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol (function, class, method)",
                },
                "file_path": {
                    "type": "string",
                    "description": "File path containing the symbol",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
            },
            "required": ["symbol_name", "file_path"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="find_tests_for_symbol",
        description="Find test files and test functions that reference or cover a given code symbol.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol to find tests for",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
            },
            "required": ["symbol_name"],
        },
        job_types=('analysis', 'coding'),
    ),
    ToolSpec(
        name="capture_snapshot",
        description="Capture a named snapshot of current workspace state metrics (findings count, progress, tool stats, etc.) for later comparison or drift detection.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Snapshot label (e.g. 'after_search', 'before_synthesis')",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional state keys to capture beyond default metrics",
                },
            },
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="compare_snapshots",
        description="Compare two named snapshots and return a structured diff showing what changed between them (findings delta, progress change, new tools used, etc.).",
        parameters={
            "type": "object",
            "properties": {
                "snapshot_a": {
                    "type": "string",
                    "description": "Name of the earlier snapshot",
                },
                "snapshot_b": {
                    "type": "string",
                    "description": "Name of the later snapshot",
                },
            },
            "required": ["snapshot_a", "snapshot_b"],
        },
    ),
    ToolSpec(
        name="detect_drift",
        description="Compare current state against a named baseline snapshot and flag significant changes or problems (stalling, progress regression, high tool failure rates).",
        parameters={
            "type": "object",
            "properties": {
                "baseline": {
                    "type": "string",
                    "description": "Name of the baseline snapshot to compare against",
                },
                "thresholds": {
                    "type": "object",
                    "description": 'Custom thresholds for drift alerts (e.g. {"stalled_iterations": 3, "goal_progress_drop": 10})',
                },
            },
            "required": ["baseline"],
        },
    ),
)
