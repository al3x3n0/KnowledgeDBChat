"""
MCP tool for executing bounded commands inside Docker containers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.schemas.docker_tool import DockerToolConfig, DockerToolExecutionInput
from app.services.docker_tool_executor import docker_executor


class DockerExecuteTool:
    name = "docker_execute"
    description = "Run commands in Docker with limits and optional stdin"

    input_schema = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Docker image to execute, e.g. python:3.11-slim",
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command argv list to run inside container",
            },
            "stdin_data": {"type": "string", "description": "Optional stdin payload"},
            "input_content": {
                "type": "string",
                "description": "Optional input file content",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 120,
                "minimum": 1,
                "maximum": 3600,
            },
            "memory_limit": {"type": "string", "default": "512m"},
            "cpu_limit": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.1,
                "maximum": 8.0,
            },
            "network_enabled": {"type": "boolean", "default": False},
            "environment": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional environment variables",
            },
            "working_dir": {"type": "string", "default": "/workspace"},
        },
        "required": ["image", "command"],
    }

    async def execute(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,  # unused, kept for MCP signature consistency
        image: str,
        command: List[str],
        stdin_data: Optional[str] = None,
        input_content: Optional[str] = None,
        timeout_seconds: int = 120,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        network_enabled: bool = False,
        environment: Optional[Dict[str, str]] = None,
        working_dir: str = "/workspace",
    ) -> Dict[str, Any]:
        auth.require_scope("write")
        if network_enabled:
            auth.require_scope("admin")

        if not docker_executor.is_docker_available():
            return {"error": "Docker is not available on server"}

        safe_image = str(image or "").strip()
        if not safe_image:
            return {"error": "image is required"}
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) and x.strip() for x in command)
        ):
            return {"error": "command must be a non-empty string array"}

        logger.info(
            f"MCP docker_execute: image={safe_image}, user={auth.user.username}"
        )

        # Attempt to ensure image exists; if pull fails, execution may still fail with a detailed error.
        await docker_executor.pull_image(safe_image)

        config = DockerToolConfig(
            image=safe_image,
            command=[c.strip() for c in command],
            input_mode="both" if input_content is not None else "stdin",
            output_mode="stdout",
            timeout_seconds=max(1, min(int(timeout_seconds), 3600)),
            memory_limit=str(memory_limit or "512m"),
            cpu_limit=max(0.1, min(float(cpu_limit), 8.0)),
            environment=environment or {},
            working_dir=str(working_dir or "/workspace"),
            network_enabled=bool(network_enabled),
        )
        execution_input = DockerToolExecutionInput(
            stdin_data=stdin_data,
            input_content=input_content,
            document_ids=None,
            environment_overrides=None,
        )

        result = await docker_executor.execute(
            config=config,
            execution_input=execution_input,
            user_id=auth.user_id,
        )

        return {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }
