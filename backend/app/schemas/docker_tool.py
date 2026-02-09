"""
Pydantic schemas for Docker container tool configuration.
"""

from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field


class DockerToolConfig(BaseModel):
    """
    Configuration for a Docker container tool.

    This defines how a Docker container should be executed as part of a workflow.
    """
    # Container image
    image: str = Field(..., description="Docker image name (e.g., 'python:3.11-slim', 'node:18')")

    # Command and entrypoint
    command: Optional[List[str]] = Field(None, description="Command to run in the container")
    entrypoint: Optional[List[str]] = Field(None, description="Override the container entrypoint")

    # Input/Output modes
    input_mode: Literal["stdin", "file", "both"] = Field(
        "stdin",
        description="How input is provided: stdin, file, or both"
    )
    output_mode: Literal["stdout", "file", "both"] = Field(
        "stdout",
        description="How output is captured: stdout, file, or both"
    )

    # Input/Output file paths (inside container)
    input_file_path: Optional[str] = Field(
        "/workspace/input.txt",
        description="Path inside container for input file (when input_mode includes 'file')"
    )
    output_file_path: Optional[str] = Field(
        "/workspace/output.txt",
        description="Path inside container for output file (when output_mode includes 'file')"
    )

    # Resource limits
    timeout_seconds: int = Field(
        300,
        ge=1,
        le=3600,
        description="Maximum execution time in seconds (1s - 1 hour)"
    )
    memory_limit: str = Field(
        "512m",
        description="Memory limit (e.g., '256m', '1g', '2g')"
    )
    cpu_limit: float = Field(
        1.0,
        ge=0.1,
        le=8.0,
        description="CPU limit (e.g., 0.5, 1.0, 2.0)"
    )

    # Environment variables
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set in the container"
    )

    # Working directory
    working_dir: str = Field(
        "/workspace",
        description="Working directory inside the container"
    )

    # Network access
    network_enabled: bool = Field(
        False,
        description="Whether to enable network access (security consideration)"
    )

    # User to run as
    user: Optional[str] = Field(
        None,
        description="User to run the container as (e.g., 'nobody', '1000:1000')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": "python:3.11-slim",
                "command": ["python", "/workspace/script.py"],
                "input_mode": "both",
                "output_mode": "both",
                "input_file_path": "/workspace/input.json",
                "output_file_path": "/workspace/output.json",
                "timeout_seconds": 300,
                "memory_limit": "1g",
                "cpu_limit": 1.0,
                "environment": {"PYTHONUNBUFFERED": "1"},
                "working_dir": "/workspace",
                "network_enabled": False
            }
        }


class DockerToolExecutionInput(BaseModel):
    """
    Input for executing a Docker container tool.
    """
    stdin_data: Optional[str] = Field(
        None,
        description="Data to send to container via stdin"
    )
    input_content: Optional[str] = Field(
        None,
        description="Content to write to input file"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Document IDs to download and mount in the container"
    )
    environment_overrides: Optional[Dict[str, str]] = Field(
        None,
        description="Additional environment variables to set"
    )


class DockerToolExecutionResult(BaseModel):
    """
    Result from executing a Docker container tool.
    """
    success: bool = Field(..., description="Whether execution succeeded")
    exit_code: int = Field(..., description="Container exit code")
    stdout: str = Field("", description="Standard output from container")
    stderr: str = Field("", description="Standard error from container")
    output_content: Optional[str] = Field(
        None,
        description="Content read from output file (if output_mode includes 'file')"
    )
    duration_seconds: float = Field(..., description="Execution duration")
    error: Optional[str] = Field(None, description="Error message if execution failed")
