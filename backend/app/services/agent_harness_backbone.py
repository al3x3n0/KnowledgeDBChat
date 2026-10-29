"""The DeepSeek Harness as an optional backbone for an autonomous job.

`config.backbone = "deepseek_harness"` hands the conversation with the model
to DeepSeek's own agent runtime (github.com/deepseek-ai/deepseek-harness)
instead of this platform's observe -> think -> act -> evaluate loop. The
point is a comparison: the same goals, tools and contracts, driven by their
loop and by ours, scored by the eval scorecard.

What changes hands and what does not:

* **The harness decides** which tool to call next, with what arguments, and
  when to stop talking. It keeps its reasoning across tool calls, compacts
  its own context and may call tools in parallel.
* **This platform still executes and judges.** Every tool call the harness
  makes arrives over MCP and runs as one iteration of the job's own runtime
  adapter -- act, evaluate, iteration complete -- so the approval gates,
  scopes, audit, findings, goal contract, checkpoints and budgets are the
  ones every job has. What is skipped is what is *our loop's* judgement: the
  planner, the critic, plan-step enforcement and the automatic verification,
  summarise and recovery calls (`_AutonomousRuntimeAdapter.externally_driven`).
* **The contract still ends the run.** The harness has no step budget of its
  own. When evaluation says stop, or a budget is spent, every further tool
  call is answered "the run is over" and the harness is told to write its
  summary; one that keeps calling is terminated.

Three isolation choices, each forced by something found while building it:

* The SDK needs pydantic>=2.12 and the backend pins an older one, so the
  harness runs under its own interpreter (`AGENT_DEEPSEEK_HARNESS_PYTHON`)
  through `deepseek_harness_driver.py`, never imported here.
* The SDK copies its parent's whole environment into the runtime, and a
  worker's environment holds the database URL and the signing key. The
  driver is started with an allowlist.
* The harness uploads its session log to DeepSeek and sends telemetry by
  default. Both are off, always (`session-log-deepseek` disabled,
  `DSH_TELEMETRY_MODE=DISABLED`), and its shell, file and MCP-resource tools
  are removed, so the only tools it has are the ones this job may call.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from app.core.config import settings
from app.utils.datetimes import as_aware_utc, utc_now

BACKBONE_KEY = "backbone"
DEEPSEEK_HARNESS = "deepseek_harness"
SERVER_NAME = "kdbc"

#: Rows of the harness's `sdk-minimal` profile that give it tools or send data
#: anywhere. Removing them leaves a model loop whose only tools are ours.
DISABLED_ROWS = (
    "persistent-bash",
    "persistent-pwsh",
    "terminal-bash",
    "terminal-pwsh",
    "tool-fs",
    "tool-working-directory",
    "mcp-resources",
    "session-log-deepseek",
)

#: What the driver's environment may hold. Everything else in the worker's
#: environment -- database URL, signing key, other providers' keys -- stays out.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")

MAX_RESULT_CHARS = 12_000
#: Tool calls answered "the run is over" before the harness is terminated.
CALLS_AFTER_STOP = 3
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

STOP_TEXT = (
    "The run is over ({reason}). Do not call any more tools. Reply now with a "
    "short final summary of what was established and what was not."
)

SYSTEM_PROMPT = (
    "You are an autonomous research agent working inside a research platform. "
    "You act only through the tools you are given; each call is executed and "
    "recorded by the platform, which also decides when the work is complete. "
    "Results that establish something must be recorded through the tools "
    "(findings), not only stated in your reply."
)


class HarnessUnavailable(RuntimeError):
    """The backbone was asked for and cannot run here."""


def backbone_of(job: Any) -> str:
    config = getattr(job, "config", None)
    value = config.get(BACKBONE_KEY) if isinstance(config, dict) else None
    return str(value or "").strip().lower()


def unavailable_reason() -> Optional[str]:
    """Why the harness backbone cannot run in this process, or None."""
    if not settings.AGENT_DEEPSEEK_HARNESS_ENABLED:
        return "AGENT_DEEPSEEK_HARNESS_ENABLED is off"
    python = settings.AGENT_DEEPSEEK_HARNESS_PYTHON
    if not python or not Path(python).exists():
        return (
            f"no harness interpreter at {python!r} (build the image with "
            "WITH_DEEPSEEK_HARNESS=true, or set AGENT_DEEPSEEK_HARNESS_PYTHON)"
        )
    if not settings.DEEPSEEK_API_KEY:
        return "DEEPSEEK_API_KEY is not set"
    return None


def build_patch(url: str, token: str) -> str:
    """The profile patch: our MCP server in, every other tool and upload out."""
    rows = [f"- id: {row}\n  disabled: true" for row in DISABLED_ROWS]
    rows.append(
        "- insert:\n"
        "    - id: mcp-kdbc\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        f"        serverName: {SERVER_NAME}\n"
        "        transport: streamable-http\n"
        f"        url: {json.dumps(url)}\n"
        "        headers:\n"
        f"          Authorization: {json.dumps('Bearer ' + token)}\n"
        "        failOnStartupError: true\n"
        "        toolCallTimeoutMs: 1800000\n"
        "        reconnect:\n"
        "          enabled: false"
    )
    return "\n".join(rows) + "\n"


def driver_environment(api_key: str) -> Dict[str, str]:
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env["DEEPSEEK_API_KEY"] = api_key
    return env


# --------------------------------------------------------------------------
# MCP, the part of it the harness uses: JSON-RPC over HTTP POST, stateless,
# answered with JSON (Streamable HTTP allows a JSON body in place of a stream).
# --------------------------------------------------------------------------

ToolCall = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def mcp_tool(definition: Dict[str, Any]) -> Dict[str, Any]:
    """One of our tool definitions in MCP's shape."""
    parameters = definition.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        parameters = {"type": "object", "properties": {}}
    return {
        "name": definition["name"],
        "description": str(definition.get("description") or "")[:4000],
        "inputSchema": parameters,
    }


def _rpc_result(msg_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


async def handle_rpc(
    message: Any, *, tools: List[Dict[str, Any]], call: ToolCall
) -> Optional[Dict[str, Any]]:
    """Answer one JSON-RPC message, or None for a notification."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _rpc_error(None, -32600, "invalid request")
    method = str(message.get("method") or "")
    msg_id = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if "id" not in message:
        return None  # notifications/initialized and the like
    if method == "initialize":
        asked = str(params.get("protocolVersion") or "")
        return _rpc_result(
            msg_id,
            {
                "protocolVersion": asked
                if asked in PROTOCOL_VERSIONS
                else PROTOCOL_VERSIONS[1],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "1"},
            },
        )
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, {"tools": tools})
    if method == "tools/call":
        name = str(params.get("name") or "")
        if name not in {tool["name"] for tool in tools}:
            return _rpc_error(msg_id, -32602, f"unknown tool: {name}")
        arguments = params.get("arguments")
        answer = await call(name, arguments if isinstance(arguments, dict) else {})
        return _rpc_result(
            msg_id,
            {
                "content": [{"type": "text", "text": answer["text"]}],
                "isError": bool(answer.get("is_error")),
            },
        )
    return _rpc_error(msg_id, -32601, f"method not found: {method}")


def mcp_app(*, token: str, tools: List[Dict[str, Any]], call: ToolCall):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def endpoint(request: Request) -> Response:
        if request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.method != "POST":
            # No server-initiated stream: allowed by the transport.
            return Response(status_code=405)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(
                _rpc_error(None, -32700, "parse error"), status_code=400
            )
        if isinstance(body, list):
            answers = [await handle_rpc(m, tools=tools, call=call) for m in body]
            answers = [a for a in answers if a is not None]
            return JSONResponse(answers) if answers else Response(status_code=202)
        answer = await handle_rpc(body, tools=tools, call=call)
        return JSONResponse(answer) if answer is not None else Response(status_code=202)

    return Starlette(
        routes=[Route("/mcp", endpoint, methods=["GET", "POST", "DELETE"])]
    )


class _QuietServer:
    """uvicorn without its signal handlers: in a prefork Celery child the task
    runs on the main thread, where uvicorn would take over SIGTERM/SIGINT."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        class _Server(uvicorn.Server):
            def install_signal_handlers(self) -> None:  # noqa: D401
                return None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.server = _Server(
            uvicorn.Config(app, log_level="warning", lifespan="off", access_log=False)
        )
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self.server.serve(sockets=[self.sock]))
        for _ in range(200):
            if self.server.started:
                return
            if self.task.done():
                self.task.result()
            await asyncio.sleep(0.025)
        raise HarnessUnavailable("the MCP server for the harness did not start")

    async def stop(self) -> None:
        self.server.should_exit = True
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.task.cancel()
        self.sock.close()


# --------------------------------------------------------------------------
# The runner: the same `run(adapter)` contract as AgentRuntimeRunner.
# --------------------------------------------------------------------------


def _result_text(bundle: Dict[str, Any]) -> str:
    result = bundle.get("action_result")
    text = json.dumps(result, default=str) if not isinstance(result, str) else result
    if len(text) > MAX_RESULT_CHARS:
        text = (
            text[:MAX_RESULT_CHARS]
            + f"... [{len(text) - MAX_RESULT_CHARS} more characters]"
        )
    return text


def _contract_line(state: Dict[str, Any]) -> str:
    contract = state.get("goal_contract_last")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        return ""
    if contract.get("satisfied"):
        return "\n[platform] The goal contract is satisfied."
    missing = [str(m)[:80] for m in (contract.get("missing") or [])][:6]
    return "\n[platform] Goal contract still requires: " + "; ".join(missing)


@dataclass
class _Run:
    stopped: Optional[str] = None
    calls_after_stop: int = 0
    terminal: Any = None
    llm_turns: int = 0
    usage: Dict[str, float] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)
    stderr_tail: List[str] = field(default_factory=list)


class HarnessBackboneRunner:
    """Run a job with the DeepSeek Harness choosing its tool calls."""

    def __init__(
        self,
        *,
        driver_python: Optional[str] = None,
        driver_script: Optional[str] = None,
    ) -> None:
        # Both are for tests: a stand-in driver speaks the same JSON lines.
        self.driver_python = driver_python
        self.driver_script = driver_script

    async def run(self, adapter: Any) -> Any:
        reason = unavailable_reason() if self.driver_python is None else None
        if reason:
            raise HarnessUnavailable(f"DeepSeek harness backbone: {reason}")
        job, state = adapter.job, adapter.state
        executor = adapter.executor
        adapter.externally_driven = True

        tools = self._tool_menu(executor, job, state)
        if not tools:
            raise HarnessUnavailable(
                "the job may call no tools the harness can be given"
            )
        run = _Run()
        lock = asyncio.Lock()
        llm_base = int(job.llm_calls_used or 0)

        async def call(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
            async with lock:  # one AsyncSession, so one call at a time
                job.llm_calls_used = llm_base + run.llm_turns
                if run.stopped:
                    run.calls_after_stop += 1
                    return {
                        "text": STOP_TEXT.format(reason=run.stopped),
                        "is_error": True,
                    }
                if not await adapter.can_continue():
                    run.stopped = "a budget or the time limit was reached"
                    return {
                        "text": STOP_TEXT.format(reason=run.stopped),
                        "is_error": True,
                    }
                return await self._iteration(adapter, run, name, params)

        token = secrets.token_urlsafe(32)
        server = _QuietServer(mcp_app(token=token, tools=tools, call=call))
        await server.start()
        home = Path(tempfile.mkdtemp(prefix="dsh-"))
        try:
            await self._drive(adapter, run, server.port, token, home)
        finally:
            await server.stop()
            shutil.rmtree(home, ignore_errors=True)

        job.llm_calls_used = llm_base + run.llm_turns
        job.add_log_entry(
            {
                "phase": "harness_session",
                "backbone": DEEPSEEK_HARNESS,
                "model": self._model(),
                "llm_turns": run.llm_turns,
                "usage": run.usage,
                "stopped_by_platform": run.stopped,
                "calls_after_stop": run.calls_after_stop,
                "finish_reason": run.final.get("finish_reason"),
                "error": run.final.get("error"),
                "final_response": str(run.final.get("final_response") or "")[:1500],
            }
        )
        state["harness_final_response"] = str(run.final.get("final_response") or "")[
            :4000
        ]
        if run.terminal is not None:
            return run.terminal
        return await adapter.build_run_result()

    async def _iteration(
        self, adapter: Any, run: _Run, name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        decision = {
            "goal_achieved": False,
            "should_stop": False,
            "reasoning": "chosen by the DeepSeek harness",
            "action": {"tool": name, "params": params, "purpose": "deepseek_harness"},
        }
        try:
            await adapter.on_iteration_start()
            started = time.monotonic()
            bundle = await adapter.act_phase(decision)
            act_ms = int((time.monotonic() - started) * 1000)
            if isinstance(bundle, dict) and bundle.get("terminal_result") is not None:
                run.terminal = bundle["terminal_result"]
                run.stopped = "the run is waiting for a person's approval"
                return {"text": STOP_TEXT.format(reason=run.stopped), "is_error": True}
            started = time.monotonic()
            evaluation = await adapter.evaluate_phase(decision, bundle)
            bundle["phase_ms"] = {
                "act": act_ms,
                "evaluate": int((time.monotonic() - started) * 1000),
            }
            await adapter.on_iteration_complete({}, decision, bundle, evaluation)
        except Exception as exc:  # noqa: BLE001 - the adapter decides
            if bool(getattr(exc, "fatal", False)):
                raise
            if not await adapter.on_iteration_error(exc):
                run.stopped = "the run failed"
            return {
                "text": f"The platform could not run {name}: {exc}"[:2000],
                "is_error": True,
            }

        text = _result_text(bundle) + _contract_line(adapter.state)
        if bool((evaluation or {}).get("should_stop")):
            run.stopped = str(
                (evaluation or {}).get("stop_reason") or "the platform stopped it"
            )
            text += "\n" + STOP_TEXT.format(reason=run.stopped)
        result = bundle.get("action_result")
        failed = isinstance(result, dict) and result.get("success") is False
        return {"text": text, "is_error": failed}

    def _tool_menu(
        self, executor: Any, job: Any, state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from app.services.agent_tools import get_tool_by_name

        profile = state.get("skill_profile")
        if not isinstance(profile, dict):
            profile = executor._resolve_agent_skill_profile(job, state=state)
        names = executor._get_tools_for_job_type(
            job.job_type, job.config, profile=profile
        )
        return [mcp_tool(d) for d in (get_tool_by_name(n) for n in names) if d]

    def _model(self) -> str:
        return settings.AGENT_DEEPSEEK_HARNESS_MODEL or settings.DEEPSEEK_MODEL

    def _prompt(self, adapter: Any) -> str:
        executor, job, state = adapter.executor, adapter.job, adapter.state
        build = getattr(executor, "_build_thinking_prompt_stable", None)
        stable = ""
        if callable(build):
            profile = (
                state.get("skill_profile")
                if isinstance(state.get("skill_profile"), dict)
                else None
            )
            try:
                stable = build(job, adapter.agent_def, state, profile=profile)
            except Exception as exc:  # noqa: BLE001 - the goal alone still runs
                logger.warning(
                    f"Harness prompt for job {job.id} without the stable prompt: {exc}"
                )
        return (
            f"{stable}\n\n"
            "You are driving this job through the platform's tools. Call them "
            "directly; ignore any instruction above to reply with a JSON "
            "decision. After each call the platform says what the goal "
            "contract still requires. When it says the run is over, or the "
            "goal is reached or cannot be, stop calling tools and reply with a "
            f"short final summary.\n\nGoal: {job.goal}"
        )

    async def _drive(
        self, adapter: Any, run: _Run, port: int, token: str, home: Path
    ) -> None:
        job = adapter.job
        python = self.driver_python or settings.AGENT_DEEPSEEK_HARNESS_PYTHON
        driver = Path(
            self.driver_script or Path(__file__).with_name("deepseek_harness_driver.py")
        )
        patch = home / "kdbc.patch.yml"
        patch.write_text(
            build_patch(f"http://127.0.0.1:{port}/mcp", token), encoding="utf-8"
        )
        work = home / "work"
        work.mkdir()
        elapsed = (utc_now() - as_aware_utc(adapter.start_time)).total_seconds()
        remaining = max(60, int(adapter.max_runtime.total_seconds() - elapsed))
        spec = {
            "model": self._model(),
            "reasoning_effort": settings.AGENT_DEEPSEEK_HARNESS_REASONING_EFFORT
            or None,
            "base_url": settings.AGENT_DEEPSEEK_HARNESS_BASE_URL or None,
            "patch": str(patch),
            "dsh_home": str(home / "home"),
            "cwd": str(work),
            "prompt": self._prompt(adapter),
            "system_prompt": SYSTEM_PROMPT,
            "timeout_seconds": remaining,
        }
        proc = await asyncio.create_subprocess_exec(
            python,
            "-I",
            str(driver),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=driver_environment(settings.DEEPSEEK_API_KEY),
        )
        assert proc.stdin and proc.stdout and proc.stderr
        proc.stdin.write(json.dumps(spec).encode())
        await proc.stdin.drain()
        proc.stdin.close()

        async def read_stdout() -> None:
            async for raw in proc.stdout:
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                if line.get("event") == "assistant":
                    run.llm_turns += 1
                elif line.get("event") in ("result", "error"):
                    run.final = line
                    for key, value in (line.get("usage") or {}).items():
                        run.usage[key] = value

        async def read_stderr() -> None:
            async for raw in proc.stderr:
                run.stderr_tail = (
                    run.stderr_tail + [raw.decode(errors="replace")[:400]]
                )[-20:]

        async def watchdog() -> None:
            # A harness that ignores "the run is over" is terminated.
            while proc.returncode is None:
                if run.calls_after_stop >= CALLS_AFTER_STOP:
                    proc.terminate()
                    return
                await asyncio.sleep(1)

        readers = asyncio.gather(read_stdout(), read_stderr())
        guard = asyncio.create_task(watchdog())
        try:
            await asyncio.wait_for(readers, timeout=remaining + 60)
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            run.stopped = run.stopped or "the time limit was reached"
            proc.kill()
            await proc.wait()
        finally:
            guard.cancel()
        if proc.returncode not in (0, None) and not run.final.get("error"):
            run.final.setdefault(
                "error", "".join(run.stderr_tail)[-1500:] or f"exit {proc.returncode}"
            )
        logger.info(
            f"Harness session for job {job.id}: {run.llm_turns} model turns, "
            f"exit {proc.returncode}, stopped={run.stopped!r}"
        )


__all__ = [
    "BACKBONE_KEY",
    "DEEPSEEK_HARNESS",
    "HarnessBackboneRunner",
    "HarnessUnavailable",
    "backbone_of",
    "build_patch",
    "handle_rpc",
    "unavailable_reason",
]
