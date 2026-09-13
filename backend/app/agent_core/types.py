"""Runtime DTOs for the extracted agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class AgentSpec:
    name: str
    display_name: str
    system_prompt: str
    capabilities: List[str] = field(default_factory=list)
    tool_whitelist: Optional[List[str]] = None
    priority: int = 50
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobSpec:
    goal: str
    job_type: str = "custom"
    config: Dict[str, Any] = field(default_factory=dict)
    goal_criteria: Optional[Dict[str, Any]] = None
    max_iterations: int = 100
    iteration: int = 0


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    purpose: str = ""


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(slots=True)
class RunState:
    run_id: str
    status: str = "pending"
    progress: int = 0
    iteration: int = 0
    current_phase: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation:
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionSpec:
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    purpose: str = ""


@dataclass(slots=True)
class Decision:
    goal_achieved: bool = False
    should_stop: bool = False
    stop_reason: str = ""
    reasoning: str = ""
    assessment: Any = None
    action: Optional[ActionSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionResult:
    success: bool = False
    output: Dict[str, Any] = field(default_factory=dict)
    terminal: bool = False
    terminal_result: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class EvaluationResult:
    progress: int = 0
    should_stop: bool = False
    stop_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeIterationResult:
    iteration: int
    observation: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeRunResult:
    status: str
    iterations_completed: int = 0
    stop_reason: str = ""
    result: Dict[str, Any] = field(default_factory=dict)


def agent_spec_from_model(agent: Any) -> AgentSpec:
    """Best-effort mapper from ORM-like agent model to a core DTO."""
    return AgentSpec(
        name=str(getattr(agent, "name", "") or ""),
        display_name=str(
            getattr(agent, "display_name", "") or getattr(agent, "name", "") or ""
        ),
        system_prompt=str(getattr(agent, "system_prompt", "") or ""),
        capabilities=list(getattr(agent, "capabilities", None) or []),
        tool_whitelist=(
            list(getattr(agent, "tool_whitelist", None))
            if getattr(agent, "tool_whitelist", None) is not None
            else None
        ),
        priority=int(getattr(agent, "priority", 50) or 50),
        is_active=bool(getattr(agent, "is_active", True)),
        metadata={
            "id": str(getattr(agent, "id", "") or ""),
            "routing_defaults": getattr(agent, "routing_defaults", None),
        },
    )


def job_spec_from_model(job: Any) -> JobSpec:
    """Best-effort mapper from ORM-like job model to a core DTO."""
    return JobSpec(
        goal=str(getattr(job, "goal", "") or ""),
        job_type=str(getattr(job, "job_type", "custom") or "custom"),
        config=dict(getattr(job, "config", None) or {}),
        goal_criteria=getattr(job, "goal_criteria", None),
        max_iterations=int(getattr(job, "max_iterations", 100) or 100),
        iteration=int(getattr(job, "iteration", 0) or 0),
    )
