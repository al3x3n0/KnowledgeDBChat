"""
Docker container tool executor service.

Executes Docker containers as workflow tools with stdin/stdout and volume mount support.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Any
from uuid import UUID
from pathlib import Path
from loguru import logger

from app.schemas.docker_tool import DockerToolConfig, DockerToolExecutionInput, DockerToolExecutionResult
from app.services.storage_service import StorageService


class DockerToolExecutor:
    """
    Executes Docker containers as workflow tools.

    Features:
    - Stdin/stdout communication
    - Input/output file support
    - Document volume mounting
    - Resource limits (memory, CPU, timeout)
    - Network isolation
    """

    def __init__(self):
        self.storage = StorageService()

    async def execute(
        self,
        config: DockerToolConfig,
        execution_input: DockerToolExecutionInput,
        user_id: Optional[UUID] = None
    ) -> DockerToolExecutionResult:
        """
        Execute a Docker container with the given configuration.

        Args:
            config: Docker tool configuration
            execution_input: Input data for the container
            user_id: User ID for document access

        Returns:
            DockerToolExecutionResult with output and status
        """
        start_time = time.time()
        workspace_dir = None

        try:
            # Create temporary workspace directory
            workspace_dir = tempfile.mkdtemp(prefix="docker_tool_")
            logger.info(f"Created workspace directory: {workspace_dir}")

            # Download documents if provided
            if execution_input.document_ids:
                await self._download_documents(
                    execution_input.document_ids,
                    workspace_dir,
                    user_id
                )

            # Write input file if needed
            if config.input_mode in ("file", "both") and execution_input.input_content:
                input_path = os.path.join(workspace_dir, "input.txt")
                with open(input_path, "w") as f:
                    f.write(execution_input.input_content)
                logger.debug(f"Wrote input file: {input_path}")

            # Build docker run command
            cmd = self._build_docker_command(
                config=config,
                workspace_dir=workspace_dir,
                environment_overrides=execution_input.environment_overrides
            )

            logger.info(f"Executing Docker command: {' '.join(cmd)}")

            # Prepare stdin data
            stdin_data = None
            if config.input_mode in ("stdin", "both") and execution_input.stdin_data:
                stdin_data = execution_input.stdin_data.encode()

            # Execute the container
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=stdin_data),
                        timeout=config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    duration = time.time() - start_time
                    return DockerToolExecutionResult(
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr="",
                        duration_seconds=duration,
                        error=f"Container timed out after {config.timeout_seconds} seconds"
                    )

                exit_code = process.returncode
                stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
                stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            except FileNotFoundError:
                duration = time.time() - start_time
                return DockerToolExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_seconds=duration,
                    error="Docker command not found. Is Docker installed?"
                )
            except Exception as e:
                duration = time.time() - start_time
                return DockerToolExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_seconds=duration,
                    error=f"Failed to execute Docker command: {str(e)}"
                )

            # Read output file if needed
            output_content = None
            if config.output_mode in ("file", "both"):
                output_filename = os.path.basename(config.output_file_path or "output.txt")
                output_path = os.path.join(workspace_dir, output_filename)
                if os.path.exists(output_path):
                    with open(output_path, "r") as f:
                        output_content = f.read()
                    logger.debug(f"Read output file: {output_path}")

            duration = time.time() - start_time

            return DockerToolExecutionResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                output_content=output_content,
                duration_seconds=duration,
                error=None if exit_code == 0 else f"Container exited with code {exit_code}"
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"Docker tool execution failed: {e}")
            return DockerToolExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                duration_seconds=duration,
                error=str(e)
            )

        finally:
            # Cleanup workspace directory
            if workspace_dir and os.path.exists(workspace_dir):
                try:
                    shutil.rmtree(workspace_dir)
                    logger.debug(f"Cleaned up workspace: {workspace_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup workspace {workspace_dir}: {e}")

    def _build_docker_command(
        self,
        config: DockerToolConfig,
        workspace_dir: str,
        environment_overrides: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Build the docker run command with all options.

        Args:
            config: Docker tool configuration
            workspace_dir: Path to workspace directory
            environment_overrides: Additional environment variables

        Returns:
            List of command arguments
        """
        cmd = ["docker", "run", "--rm"]

        # Resource limits
        cmd.extend(["--memory", config.memory_limit])
        cmd.extend(["--cpus", str(config.cpu_limit)])

        # Network mode
        if not config.network_enabled:
            cmd.extend(["--network", "none"])

        # User
        if config.user:
            cmd.extend(["--user", config.user])

        # Working directory
        cmd.extend(["--workdir", config.working_dir])

        # Mount workspace directory
        # Map the host workspace to the container working directory
        cmd.extend(["-v", f"{workspace_dir}:{config.working_dir}"])

        # Environment variables from config
        for key, value in config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Environment variable overrides
        if environment_overrides:
            for key, value in environment_overrides.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Override entrypoint if specified
        if config.entrypoint:
            cmd.extend(["--entrypoint", config.entrypoint[0]])
            if len(config.entrypoint) > 1:
                # Extra entrypoint args will be added after image
                pass

        # Add the image
        cmd.append(config.image)

        # Add entrypoint args (if any beyond the first)
        if config.entrypoint and len(config.entrypoint) > 1:
            cmd.extend(config.entrypoint[1:])

        # Add command
        if config.command:
            cmd.extend(config.command)

        return cmd

    async def _download_documents(
        self,
        document_ids: List[str],
        workspace_dir: str,
        user_id: Optional[UUID] = None
    ) -> None:
        """
        Download documents from MinIO to the workspace directory.

        Args:
            document_ids: List of document IDs to download
            workspace_dir: Directory to download documents to
            user_id: User ID for access control
        """
        from app.core.database import AsyncSessionLocal
        from app.models.document import Document
        from sqlalchemy import select

        await self.storage.initialize()

        async with AsyncSessionLocal() as db:
            for doc_id in document_ids:
                try:
                    # Get document from database
                    result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = result.scalar_one_or_none()

                    if not doc:
                        logger.warning(f"Document {doc_id} not found, skipping")
                        continue

                    # Get the file path from document
                    if not doc.file_path:
                        logger.warning(f"Document {doc_id} has no file path, skipping")
                        continue

                    # Download file from MinIO
                    try:
                        content = await self.storage.get_file_content(doc.file_path)

                        # Determine filename
                        filename = doc.source_identifier or f"{doc_id}"
                        if not Path(filename).suffix and doc.file_type:
                            # Add extension based on file type
                            ext_map = {
                                "text/plain": ".txt",
                                "text/markdown": ".md",
                                "application/json": ".json",
                                "application/pdf": ".pdf",
                            }
                            ext = ext_map.get(doc.file_type, ".txt")
                            filename += ext

                        # Write to workspace
                        filepath = os.path.join(workspace_dir, filename)
                        with open(filepath, "wb") as f:
                            f.write(content)

                        logger.info(f"Downloaded document {doc_id} to {filepath}")

                    except FileNotFoundError:
                        logger.warning(f"File not found in storage for document {doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to download document {doc_id}: {e}")

                except Exception as e:
                    logger.warning(f"Error processing document {doc_id}: {e}")

    async def pull_image(self, image: str) -> bool:
        """
        Pull a Docker image if not already present.

        Args:
            image: Docker image name

        Returns:
            True if image is available, False otherwise
        """
        try:
            # Check if image exists locally
            check_cmd = ["docker", "image", "inspect", image]
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True
            )

            if check_result.returncode == 0:
                logger.debug(f"Image {image} already present locally")
                return True

            # Pull the image
            logger.info(f"Pulling Docker image: {image}")
            pull_cmd = ["docker", "pull", image]
            pull_result = subprocess.run(
                pull_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for pull
            )

            if pull_result.returncode == 0:
                logger.info(f"Successfully pulled image: {image}")
                return True
            else:
                logger.error(f"Failed to pull image {image}: {pull_result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pulling image: {image}")
            return False
        except Exception as e:
            logger.error(f"Error pulling image {image}: {e}")
            return False

    def is_docker_available(self) -> bool:
        """
        Check if Docker is available and running.

        Returns:
            True if Docker is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception:
            return False


# Singleton instance
docker_executor = DockerToolExecutor()
