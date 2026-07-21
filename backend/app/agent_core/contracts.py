"""Protocol contracts for the extracted agent core."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def generate_response(
        self,
        *,
        system_prompt: str,
        user_message: str,
        user_settings: Any = None,
        routing: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    async def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 200,
    ) -> str: ...


@runtime_checkable
class ToolCatalog(Protocol):
    def iter_tools(self) -> Iterable[Any]: ...

    def get_tool(self, tool_name: str) -> Optional[Any]: ...


@runtime_checkable
class ToolExecutor(Protocol):
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class PolicyEvaluator(Protocol):
    async def evaluate(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Any: ...


@runtime_checkable
class MemoryProvider(Protocol):
    async def load_context(self, subject_id: str) -> Dict[str, Any]: ...


@runtime_checkable
class StateStore(Protocol):
    async def load(self, run_id: str) -> Dict[str, Any]: ...

    async def save(self, run_id: str, state: Dict[str, Any]) -> None: ...


@runtime_checkable
class EventPublisher(Protocol):
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None: ...


@runtime_checkable
class AgentLoader(Protocol):
    async def load_agents(self) -> Dict[str, Any]: ...


@runtime_checkable
class RuntimeObserver(Protocol):
    async def observe_phase(self) -> Dict[str, Any]: ...


@runtime_checkable
class RuntimeThinker(Protocol):
    async def think_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class RuntimeActor(Protocol):
    async def act_phase(self, decision: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class RuntimeEvaluator(Protocol):
    async def evaluate_phase(
        self,
        decision: Dict[str, Any],
        action_bundle: Dict[str, Any],
    ) -> Dict[str, Any]: ...


@runtime_checkable
class RuntimeLoopAdapter(
    RuntimeObserver,
    RuntimeThinker,
    RuntimeActor,
    RuntimeEvaluator,
    Protocol,
):
    async def can_continue(self) -> bool: ...

    async def on_iteration_start(self) -> None: ...

    async def on_iteration_complete(
        self,
        observation: Dict[str, Any],
        decision: Dict[str, Any],
        action_bundle: Dict[str, Any],
        evaluation: Dict[str, Any],
    ) -> None: ...

    async def on_iteration_error(self, exc: Exception) -> bool: ...

    async def build_run_result(self) -> Any: ...
