"""Shared contracts for autonomous-job operator action handlers."""

from dataclasses import dataclass
from typing import Any, Callable


class JobActionError(Exception):
    """Application error translated to an HTTP response by the API boundary."""

    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class JobActionDependencies:
    is_job_visible: Callable[..., bool]
    approval_payload_from_results: Callable[..., Any]
    load_latest_checkpoint: Callable[..., Any]
    append_operator_intervention: Callable[..., Any]
    append_step_event: Callable[..., Any]
    normalize_checkpoint_action_patch: Callable[..., Any]
    apply_checkpoint_action_patch: Callable[..., Any]
    set_current_plan_step_status: Callable[..., Any]
    append_approval_event: Callable[..., Any]
    sync_execution_strategy_state: Callable[..., Any]
    quick_start_relaunch_dispatcher: Any
    infer_coding_swarm_preset_key: Callable[..., Any]
    extract_swarm_collaboration: Callable[..., Any]
    build_swarm_collaboration_payload: Callable[..., Any]
    store_swarm_collaboration: Callable[..., Any]
    execute_agent_job_task: Any
    generate_job_summary: Any
