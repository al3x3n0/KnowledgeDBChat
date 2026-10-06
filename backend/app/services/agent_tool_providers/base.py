"""The provider protocol, the function-table provider and the registry every
tool surface dispatches through.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Protocol


def _unimplemented_tool(tool_name: str) -> Dict[str, Any]:
    """Report a capability that does not exist, instead of faking success.

    These handlers used to return {"success": True, ...} with invented fields —
    "relationship_created": True from a function that created nothing,
    "Comparison would be generated here" as an actual result. The agent believed
    the work happened and recorded it as evidence, which corrupts every
    downstream claim built on it.

    Returning an error marks the call failed (success is derived as `not
    error`), so the loop's existing tool-failure handling takes over and the
    agent can pick a different route rather than proceeding on a fiction.
    """
    return {
        "error": (
            f"Tool '{tool_name}' is not implemented. It is advertised but has no "
            "behaviour behind it; do not retry, choose a different approach."
        ),
        "unimplemented": True,
    }


@dataclass(slots=True)
class AgentToolExecutionContext:
    """Execution context for app-side tool providers."""

    mode: str
    db: Any
    service: Any
    user_id: Any = None
    job: Any = None
    state: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AgentToolProvider(Protocol):
    @property
    def supported_tools(self) -> set[str]:
        ...

    def can_handle(self, tool_name: str, context: AgentToolExecutionContext) -> bool:
        ...

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        ...


class FunctionToolProvider:
    """Simple provider backed by async callables."""

    def __init__(
        self,
        *,
        name: str,
        handlers: Dict[
            str, Callable[[Dict[str, Any], AgentToolExecutionContext], Awaitable[Any]]
        ],
        modes: Optional[Iterable[str]] = None,
    ) -> None:
        self.name = name
        self._handlers = dict(handlers)
        self._modes = set(modes or [])

    @property
    def supported_tools(self) -> set[str]:
        return set(self._handlers.keys())

    def can_handle(self, tool_name: str, context: AgentToolExecutionContext) -> bool:
        if self._modes and context.mode not in self._modes:
            return False
        return tool_name in self._handlers

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        job_config = (
            context.job.config
            if isinstance(getattr(context.job, "config", None), dict)
            else {}
        )

        def _tool_set(value: Any) -> set[str]:
            if isinstance(value, list):
                return {str(item).strip() for item in value if str(item).strip()}
            if isinstance(value, str):
                return {item.strip() for item in value.split(",") if item.strip()}
            return set()

        allowed_tools = _tool_set(
            job_config.get("allowed_tools") or job_config.get("tool_allowlist")
        )
        blocked_tools = _tool_set(
            job_config.get("blocked_tools") or job_config.get("tool_denylist")
        )
        if tool_name in blocked_tools or (
            allowed_tools and tool_name not in allowed_tools
        ):
            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' is not permitted by this agent's "
                    "enforced tool policy"
                ),
            }
        return await self._handlers[tool_name](params, context)


class AgentToolRegistry:
    """Resolves and executes tool providers."""

    def __init__(self, providers: Optional[Iterable[AgentToolProvider]] = None) -> None:
        self._providers = list(providers or [])

    def register(self, provider: AgentToolProvider) -> None:
        self._providers.append(provider)

    def resolve(
        self, tool_name: str, context: AgentToolExecutionContext
    ) -> Optional[AgentToolProvider]:
        for provider in self._providers:
            if provider.can_handle(tool_name, context):
                return provider
        return None

    async def try_execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> tuple[bool, Any]:
        provider = self.resolve(tool_name, context)
        if provider is None:
            return False, None
        await self._verify_instrument(provider, tool_name, context)
        result = await self._execute_maybe_replicated(
            provider, tool_name, params, context
        )
        result = self._check_measures_what_it_names(tool_name, params, result)
        await self._record_evidence(tool_name, params, result, context)
        return True, result

    @staticmethod
    async def _execute_maybe_replicated(
        provider: AgentToolProvider,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        """Take a nondeterministic measurement several times, once.

        Only tools whose answers actually move are replicated -- callgrind
        counts, llvm-mca and gem5 are deterministic, and calling them three
        times buys the same number at three times the cost.

        A control call is never itself replicated: the controls already run a
        median over 31 rounds internally, and replicating them would multiply
        the cost of verifying the instrument by the cost of using it.
        """
        from loguru import logger

        from app.services import agent_measurement_replication as replication
        from app.services import agent_tool_controls as controls

        if not replication.is_replicated(tool_name):
            return await provider.execute(tool_name, params, context)
        if controls.is_control_call(params):
            return await provider.execute(tool_name, params, context)

        async def _once() -> Any:
            return await provider.execute(tool_name, dict(params), context)

        try:
            return await replication.run_replicated(_once, tool_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not replicate {tool_name}: {exc}")
            return await provider.execute(tool_name, params, context)

    @staticmethod
    async def _verify_instrument(
        provider: AgentToolProvider,
        tool_name: str,
        context: AgentToolExecutionContext,
    ) -> None:
        """Run this tool's controls before its first use in the run.

        Here for the same reason evidence capture is here: every call passes
        through this point, so the run cannot use a measurement tool without
        first establishing that the tool works. A control the agent has to
        remember is a control that is missing from whichever run mattered.

        Only the *opening* half of the bracket can be automated -- nothing at
        call time knows which measurement is the last one. The closing half is
        the evaluate phase's job, and `validity.instruments_verified` refuses
        the run until it has happened.

        Never allowed to fail the call it precedes. A failing control does not
        stop the tool; it records that nothing the tool produces in this window
        is evidence, which the contract then acts on.
        """
        from loguru import logger

        from app.services import agent_tool_controls as controls

        if not controls.is_controlled(tool_name):
            return
        state = getattr(context, "state", None)
        if not isinstance(state, dict):
            return
        if not controls.needs_pre_control(state, tool_name):
            return

        async def _call(name: str, params: Dict[str, Any]) -> Any:
            return await provider.execute(name, params, context)

        try:
            verdicts = await controls.run_controls(
                _call, tool_name, state, when="before"
            )
            failed = [v for v in verdicts if not v.get("passed")]
            if failed:
                logger.warning(
                    f"Instrument control failed for {tool_name}: "
                    f"{failed[0].get('reason')}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not run controls for {tool_name}: {exc}")

    @staticmethod
    def _check_measures_what_it_names(
        tool_name: str, params: Dict[str, Any], result: Any
    ) -> Any:
        """Ask whether the call measured the thing it named.

        The one failure controls and replication both miss, because it is
        neither broken nor noisy: a chain that reaches infinity is precise,
        stable, reproducible and about something else.

        Attached to the result and to its findings rather than raised. A tool
        that refused would strand a run mid-measurement over an analysis this
        module can only sometimes perform; the contract is where the judgement
        belongs.
        """
        from loguru import logger

        from app.services import agent_measurement_replication as replication
        from app.services import agent_measurement_sanity as sanity

        if not replication.is_replicated(tool_name):
            return result
        if not isinstance(result, dict):
            return result

        try:
            verdict = sanity.check(params, result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not sanity-check {tool_name}: {exc}")
            return result

        if not verdict.get("checked"):
            return result
        if not verdict.get("sound"):
            logger.warning(
                f"{tool_name} may not have measured what it named: "
                f"{'; '.join(verdict.get('problems') or [])[:300]}"
            )

        enriched = dict(result)
        data = (
            dict(enriched.get("data") or {})
            if isinstance(enriched.get("data"), dict)
            else {}
        )
        data["measurement_sanity"] = verdict
        enriched["data"] = data
        findings = enriched.get("findings")
        if isinstance(findings, list):
            enriched["findings"] = [
                {**f, "measurement_sanity": verdict} if isinstance(f, dict) else f
                for f in findings
            ]
        return enriched

    @staticmethod
    async def _record_evidence(
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        context: AgentToolExecutionContext,
    ) -> None:
        """Add this call to the run's bundle, as it happens.

        Here rather than in each tool because every call passes through this
        point: a bundle that depends on tools opting in is a bundle missing
        whichever tool was added last. Failures are recorded too -- a run that
        cited a measurement from a call that had failed is only visible if the
        failure is in the record.

        Never allowed to affect the call itself. A bundle is a description of
        the run, not a participant in it.
        """
        from loguru import logger

        from app.services import agent_evidence_bundle as bundle

        if tool_name not in bundle.EVIDENCE_TOOLS:
            return
        # A replay re-runs calls already in the bundle; recording them again
        # appended every one to the bundle being verified.
        if (getattr(context, "extra", None) or {}).get("replaying_bundle"):
            return
        job_id = getattr(getattr(context, "job", None), "id", None)
        if not job_id:
            return
        try:
            image = ""
            if isinstance(result, dict):
                image = str((result.get("data") or {}).get("image") or "")
            if not image:
                image = str(params.get("image") or "")
            image_id = await bundle.resolve_image_id(image) if image else ""
            bundle.record_entry(
                job_id=str(job_id),
                tool=tool_name,
                params=params if isinstance(params, dict) else {"params": params},
                result=result,
                image_id=image_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"evidence bundle: skipped {tool_name}: {exc}")
