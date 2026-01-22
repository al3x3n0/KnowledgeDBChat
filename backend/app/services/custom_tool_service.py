"""
Custom Tool Service for executing user-defined tools.

Supports:
- Webhook: HTTP requests to external APIs
- Transform: Data transformation using Jinja2/JSONPath
- Python: Sandboxed Python code execution
- LLM Prompt: LLM calls with templated prompts
"""

import json
import time
import asyncio
import httpx
from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from jinja2 import Environment, BaseLoader, sandbox, TemplateSyntaxError
from jsonpath_ng import parse as jsonpath_parse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.workflow import UserTool
from app.models.user import User
from app.services.llm_service import LLMService, UserLLMSettings
from app.models.memory import UserPreferences


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class CustomToolService:
    """Service for executing user-defined custom tools."""

    def __init__(self):
        self.jinja_env = sandbox.SandboxedEnvironment(loader=BaseLoader())
        # Add safe filters
        self.jinja_env.filters['tojson'] = json.dumps

    async def execute_tool(
        self,
        tool: UserTool,
        inputs: Dict[str, Any],
        user: User,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Execute a user-defined tool.

        Args:
            tool: The UserTool to execute
            inputs: Input parameters for the tool
            user: The user executing the tool
            db: Database session

        Returns:
            Dict containing the tool output

        Raises:
            ToolExecutionError: If execution fails
        """
        start_time = time.time()

        try:
            # Validate inputs against schema if provided
            if tool.parameters_schema:
                self._validate_inputs(inputs, tool.parameters_schema)

            # Route to appropriate handler
            if tool.tool_type == "webhook":
                result = await self._execute_webhook(tool.config, inputs)
            elif tool.tool_type == "transform":
                result = await self._execute_transform(tool.config, inputs)
            elif tool.tool_type == "python":
                result = await self._execute_python(tool.config, inputs, user)
            elif tool.tool_type == "llm_prompt":
                result = await self._execute_llm_prompt(tool.config, inputs, user, db)
            else:
                raise ToolExecutionError(f"Unknown tool type: {tool.tool_type}")

            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Tool '{tool.name}' executed successfully in {execution_time_ms}ms")

            return {
                "success": True,
                "output": result,
                "execution_time_ms": execution_time_ms
            }

        except ToolExecutionError:
            raise
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Tool '{tool.name}' execution failed: {e}")
            raise ToolExecutionError(f"Tool execution failed: {str(e)}")

    def _validate_inputs(self, inputs: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate inputs against JSON Schema (basic validation)."""
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in inputs:
                raise ToolExecutionError(f"Missing required input: {field}")

        for field, value in inputs.items():
            if field in properties:
                prop_type = properties[field].get("type")
                if prop_type == "string" and not isinstance(value, str):
                    raise ToolExecutionError(f"Input '{field}' must be a string")
                elif prop_type == "integer" and not isinstance(value, int):
                    raise ToolExecutionError(f"Input '{field}' must be an integer")
                elif prop_type == "number" and not isinstance(value, (int, float)):
                    raise ToolExecutionError(f"Input '{field}' must be a number")
                elif prop_type == "boolean" and not isinstance(value, bool):
                    raise ToolExecutionError(f"Input '{field}' must be a boolean")
                elif prop_type == "array" and not isinstance(value, list):
                    raise ToolExecutionError(f"Input '{field}' must be an array")
                elif prop_type == "object" and not isinstance(value, dict):
                    raise ToolExecutionError(f"Input '{field}' must be an object")

    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the given context."""
        try:
            tpl = self.jinja_env.from_string(template)
            return tpl.render(input=context, **context)
        except TemplateSyntaxError as e:
            raise ToolExecutionError(f"Template syntax error: {e}")
        except Exception as e:
            raise ToolExecutionError(f"Template rendering failed: {e}")

    async def _execute_webhook(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a webhook tool - make HTTP request to external API.

        Config:
            method: HTTP method (GET, POST, etc.)
            url: URL template
            headers: Headers dict
            body_template: Request body template (Jinja2)
            response_path: JSONPath to extract from response
            timeout_seconds: Request timeout
        """
        method = config.get("method", "POST")
        url_template = config.get("url", "")
        headers = config.get("headers", {})
        body_template = config.get("body_template")
        response_path = config.get("response_path")
        timeout = config.get("timeout_seconds", 30)

        # Render URL
        url = self._render_template(url_template, inputs)

        # Render headers
        rendered_headers = {}
        for key, value in headers.items():
            rendered_headers[key] = self._render_template(value, inputs)

        # Render body
        body = None
        if body_template:
            body = self._render_template(body_template, inputs)

        # Make request
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if method in ["GET", "DELETE"]:
                    response = await client.request(method, url, headers=rendered_headers)
                else:
                    # Try to parse body as JSON, fall back to raw string
                    try:
                        json_body = json.loads(body) if body else None
                        response = await client.request(
                            method, url, headers=rendered_headers, json=json_body
                        )
                    except json.JSONDecodeError:
                        response = await client.request(
                            method, url, headers=rendered_headers, content=body
                        )

                response.raise_for_status()

            except httpx.TimeoutException:
                raise ToolExecutionError(f"Request timed out after {timeout}s")
            except httpx.HTTPStatusError as e:
                raise ToolExecutionError(f"HTTP {e.response.status_code}: {e.response.text[:500]}")
            except Exception as e:
                raise ToolExecutionError(f"Request failed: {str(e)}")

        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"text": response.text}

        # Extract using JSONPath if specified
        if response_path and isinstance(result, dict):
            try:
                jsonpath_expr = jsonpath_parse(response_path)
                matches = [match.value for match in jsonpath_expr.find(result)]
                if len(matches) == 1:
                    result = matches[0]
                elif matches:
                    result = matches
            except Exception as e:
                logger.warning(f"JSONPath extraction failed: {e}")

        return result

    async def _execute_transform(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a transform tool - transform data using templates.

        Config:
            transform_type: "jinja2", "jsonpath", or "javascript"
            template: The transformation template/expression
        """
        transform_type = config.get("transform_type", "jinja2")
        template = config.get("template", "")

        if transform_type == "jinja2":
            rendered = self._render_template(template, inputs)
            # Try to parse as JSON, fall back to string
            try:
                return json.loads(rendered)
            except json.JSONDecodeError:
                return {"result": rendered}

        elif transform_type == "jsonpath":
            try:
                jsonpath_expr = jsonpath_parse(template)
                matches = [match.value for match in jsonpath_expr.find(inputs)]
                if len(matches) == 1:
                    return {"result": matches[0]}
                return {"result": matches}
            except Exception as e:
                raise ToolExecutionError(f"JSONPath error: {e}")

        elif transform_type == "javascript":
            # JavaScript execution would require a separate runtime
            # For now, we'll just raise an error
            raise ToolExecutionError("JavaScript transforms are not yet supported")

        else:
            raise ToolExecutionError(f"Unknown transform type: {transform_type}")

    async def _execute_python(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """
        Execute a Python tool - run sandboxed Python code.

        Config:
            code: Python code to execute
            timeout_seconds: Execution timeout
            allowed_imports: List of allowed module names

        Security:
            Uses RestrictedPython for sandboxed execution.
            Limited to basic operations and whitelisted imports.
        """
        code = config.get("code", "")
        timeout = config.get("timeout_seconds", 10)
        allowed_imports = set(config.get("allowed_imports", []))

        # Default safe imports
        safe_imports = {"json", "re", "datetime", "math", "collections", "itertools"}
        allowed = safe_imports.union(allowed_imports)

        try:
            # Import RestrictedPython for sandboxed execution
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Eval import default_guarded_getiter
            from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

            # Compile the code in restricted mode
            byte_code = compile_restricted(code, '<user_tool>', 'exec')

            if byte_code.errors:
                raise ToolExecutionError(f"Code compilation errors: {byte_code.errors}")

            # Prepare safe globals
            _globals = safe_globals.copy()
            _globals['_getiter_'] = default_guarded_getiter
            _globals['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
            _globals['_getattr_'] = safer_getattr

            # Add allowed imports
            for module_name in allowed:
                try:
                    _globals[module_name] = __import__(module_name)
                except ImportError:
                    pass

            # Prepare locals
            _locals = {
                'input': inputs,
                'output': {},
                'result': None
            }

            # Execute with timeout
            def run_code():
                exec(byte_code, _globals, _locals)
                return _locals.get('output') or _locals.get('result')

            # Run in thread with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=timeout
            )

            return result if isinstance(result, dict) else {"result": result}

        except asyncio.TimeoutError:
            raise ToolExecutionError(f"Execution timed out after {timeout}s")
        except ImportError:
            # RestrictedPython not installed - fall back to error
            raise ToolExecutionError(
                "Python execution is not available. Install RestrictedPython package."
            )
        except Exception as e:
            raise ToolExecutionError(f"Python execution failed: {str(e)}")

    async def _execute_llm_prompt(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        user: User,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Execute an LLM prompt tool - call LLM with templated prompt.

        Config:
            system_prompt: Optional system prompt
            user_prompt: User prompt template
            output_format: "text" or "json"
            model_override: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        """
        system_prompt = config.get("system_prompt")
        user_prompt = config.get("user_prompt", "")
        output_format = config.get("output_format", "text")
        model_override = config.get("model_override")
        temperature = config.get("temperature")
        max_tokens = config.get("max_tokens")

        # Render the user prompt
        rendered_prompt = self._render_template(user_prompt, inputs)

        # Load user's LLM settings
        user_settings = None
        try:
            prefs_result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user.id)
            )
            user_prefs = prefs_result.scalar_one_or_none()
            if user_prefs:
                user_settings = UserLLMSettings.from_preferences(user_prefs)
        except Exception as e:
            logger.warning(f"Could not load user LLM preferences: {e}")

        # Apply overrides
        if user_settings and model_override:
            user_settings.model = model_override

        # Build messages
        messages = []
        if system_prompt:
            rendered_system = self._render_template(system_prompt, inputs)
            messages.append({"role": "system", "content": rendered_system})
        messages.append({"role": "user", "content": rendered_prompt})

        # Call LLM
        llm_service = LLMService()
        try:
            response = await llm_service.generate_response(
                messages=messages,
                user_settings=user_settings,
                task_type="chat"  # Use chat model by default
            )

            content = response.get("content", "")

            # Parse JSON if requested
            if output_format == "json":
                try:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        content = json_match.group(1)
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"text": content, "parse_error": "Could not parse as JSON"}

            return {"text": content}

        except Exception as e:
            raise ToolExecutionError(f"LLM call failed: {str(e)}")


# Create singleton instance
custom_tool_service = CustomToolService()
