"""
LangGraph-backed issue -> PR draft orchestration service.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypedDict, TypeVar
from uuid import UUID

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.feature_flags import get_flag as get_feature_flag
from app.models.memory import UserPreferences
from app.schemas.langgraph_issue_pr import (
    ChecklistItem,
    EventLogItem,
    ExecutorOutput,
    LangGraphIssuePrRequest,
    LangGraphIssuePrResponse,
    PlannerOutput,
    PlanStep,
    PolicyCheck,
    PrDraftPackage,
    RequiredFix,
    ResearcherOutput,
    ReviewerOutput,
    ReviewFailure,
)
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.repo_symbol_index_service import RepoSymbolIndexService

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__end__"
    StateGraph = Any


TModel = TypeVar("TModel", bound=BaseModel)


class IssuePrState(TypedDict, total=False):
    request: LangGraphIssuePrRequest
    planner: PlannerOutput
    researcher: ResearcherOutput
    executor: ExecutorOutput
    reviewer: ReviewerOutput
    status: str
    reason: str
    pr_draft: Optional[PrDraftPackage]
    revision_count: int
    max_revision_loops: int
    reviewer_min_confidence: float
    needs_human: bool
    event_log: List[EventLogItem]
    user_id: Optional[UUID]
    db: Optional[AsyncSession]
    user_settings: Optional[UserLLMSettings]
    required_fixes: List[RequiredFix]
    repo_context_snapshot: Dict[str, Any]
    repo_context_meta: Dict[str, Any]
    use_symbol_retrieval: bool


class LangGraphIssuePrService:
    _status_reason_codes: Set[str] = {
        "passed",
        "policy_escalation",
        "manual_escalation",
        "human_review_required",
        "revision_exhausted",
        "needs_revision",
        "escalated_blocked",
        "blocked_unknown",
        "unknown",
    }
    _repo_snapshot_cache: Dict[str, Dict[str, Any]] = {}
    _repo_snapshot_cache_ttl_seconds: int = 300
    _repo_snapshot_cache_max_entries: int = 64

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()
        self.repo_symbol_index_service = RepoSymbolIndexService()
        self._compiled_graph: Any = None

    async def run(
        self,
        request: LangGraphIssuePrRequest,
        *,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> LangGraphIssuePrResponse:
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "LangGraph is not installed. Install `langgraph` in backend dependencies."
            )
        graph = self._get_graph()
        user_settings = await self._load_user_settings(db=db, user_id=user_id)
        use_symbol_retrieval = await self._resolve_use_symbol_retrieval(request)
        repo_snapshot, repo_meta = self._get_or_collect_repo_context_with_meta(
            request, use_symbol_retrieval=use_symbol_retrieval
        )
        pre_events: List[EventLogItem] = [
            EventLogItem(
                ts=datetime.now(timezone.utc),
                agent="orchestrator",
                action="repo_context",
                result=(
                    f"{str(repo_meta.get('cache_status', 'unknown'))}:"
                    f"{int(repo_meta.get('scan_ms', 0) or 0)}ms"
                ),
                ref=(
                    f"scanned:{int((repo_snapshot or {}).get('scanned_files', 0) or 0)},"
                    f"matches:{len((repo_snapshot or {}).get('matched_files', []) or [])}"
                ),
            )
        ]
        state: IssuePrState = {
            "request": request,
            "revision_count": 0,
            "max_revision_loops": int(request.max_revision_loops),
            "reviewer_min_confidence": float(request.reviewer_min_confidence),
            "event_log": pre_events,
            "needs_human": False,
            "user_id": user_id,
            "db": db,
            "user_settings": user_settings,
            "required_fixes": [],
            "repo_context_snapshot": repo_snapshot,
            "repo_context_meta": repo_meta,
            "use_symbol_retrieval": use_symbol_retrieval,
        }
        result = await graph.ainvoke(state)
        planner = result.get("planner") or self._fallback_planner(request)
        researcher = result.get("researcher") or self._fallback_researcher(request)
        executor = result.get("executor") or self._fallback_executor(state)
        reviewer = result.get("reviewer") or ReviewerOutput(decision="escalate")
        return LangGraphIssuePrResponse(
            status=str(result.get("status") or "blocked"),
            reason=str(result.get("reason") or ""),
            status_reason_code=self._coerce_status_reason_code(
                str(result.get("status_reason_code") or "").strip()
                or self._derive_status_reason_code(
                    status=str(result.get("status") or "blocked"),
                    reviewer=reviewer,
                    reason=str(result.get("reason") or ""),
                    needs_human=bool(result.get("status") == "needs_human_review"),
                )
            ),
            planner=planner,
            researcher=researcher,
            executor=executor,
            reviewer=reviewer,
            pr_draft=result.get("pr_draft"),
            repo_context_meta=(
                result.get("repo_context_meta")
                if isinstance(result.get("repo_context_meta"), dict)
                else (state.get("repo_context_meta") or {})
            ),
            repo_context_summary=(
                result.get("repo_context_summary")
                if isinstance(result.get("repo_context_summary"), dict)
                else self._build_repo_context_summary(
                    state.get("repo_context_snapshot") or {}
                )
            ),
            confidence_breakdown=(
                result.get("confidence_breakdown")
                if isinstance(result.get("confidence_breakdown"), dict)
                else self._build_confidence_breakdown(
                    planner, researcher, executor, reviewer
                )
            ),
            decision_trace=(
                result.get("decision_trace")
                if isinstance(result.get("decision_trace"), list)
                else self._build_decision_trace(
                    reviewer=reviewer,
                    reason=str(result.get("reason") or ""),
                    needs_human=bool(result.get("status") == "needs_human_review"),
                )
            ),
            event_log=result.get("event_log") or [],
        )

    async def _load_user_settings(
        self, *, db: Optional[AsyncSession], user_id: Optional[UUID]
    ) -> Optional[UserLLMSettings]:
        if db is None or user_id is None:
            return None
        try:
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            prefs = result.scalar_one_or_none()
            if prefs:
                return UserLLMSettings.from_preferences(prefs)
        except Exception as exc:
            logger.warning(
                f"Failed to load user LLM settings for issue-pr orchestration: {exc}"
            )
        return None

    async def _resolve_use_symbol_retrieval(
        self, request: LangGraphIssuePrRequest
    ) -> bool:
        repo_ctx = (
            request.repo_context if isinstance(request.repo_context, dict) else {}
        )
        explicit = repo_ctx.get("use_symbol_retrieval")
        if isinstance(explicit, bool):
            return explicit
        try:
            feature = await get_feature_flag("repo_symbol_retrieval_enabled")
            if feature is not None:
                return bool(feature)
        except Exception:
            pass
        return bool(getattr(settings, "REPO_SYMBOL_RETRIEVAL_ENABLED", False))

    def _get_graph(self) -> Any:
        if self._compiled_graph is not None:
            return self._compiled_graph
        graph = StateGraph(IssuePrState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("finalize", self._finalize_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "executor")
        graph.add_edge("executor", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {"executor": "executor", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        self._compiled_graph = graph.compile()
        return self._compiled_graph

    async def _invoke_structured_agent(
        self,
        *,
        state: IssuePrState,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[TModel],
        max_attempts: int = 3,
    ) -> TModel:
        last_error: Optional[str] = None
        previous_output = ""
        for attempt in range(1, max_attempts + 1):
            repair_hint = ""
            if attempt > 1:
                repair_hint = (
                    f"\n\nPrevious output was invalid. Error: {last_error or 'unknown'}\n"
                    f"Previous output:\n{previous_output[:2000]}\n"
                    f"Return ONLY valid JSON matching the required schema."
                )
            raw = await self.llm_service.generate_response(
                query=f"{user_prompt}{repair_hint}",
                system_prompt=system_prompt,
                user_settings=state.get("user_settings"),
                task_type="code_agent",
                user_id=state.get("user_id"),
                db=state.get("db"),
                temperature=0.1,
                max_tokens=1800,
            )
            previous_output = raw or ""
            try:
                payload = self._extract_json(raw or "")
                return output_model.model_validate(payload)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"LangGraph {agent_name} parse failure attempt {attempt}/{max_attempts}: {exc}"
                )
        raise ValueError(
            f"{agent_name} did not return valid structured output: {last_error or 'unknown'}"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(cleaned[start : end + 1])

    def _append_event(
        self,
        state: IssuePrState,
        *,
        agent: str,
        action: str,
        result: str,
        ref: str = "",
    ) -> List[EventLogItem]:
        history = list(state.get("event_log") or [])
        history.append(
            EventLogItem(
                ts=datetime.now(timezone.utc),
                agent=agent,
                action=action,
                result=result,
                ref=ref,
            )
        )
        return history

    def _fallback_planner(self, request: LangGraphIssuePrRequest) -> PlannerOutput:
        issue = request.issue
        return PlannerOutput(
            plan_steps=[
                PlanStep(
                    id="S1",
                    action=f"Triage issue {issue.id} and confirm expected behavior",
                    rationale="Align implementation with issue intent before code changes.",
                ),
                PlanStep(
                    id="S2",
                    action="Implement minimal fix in impacted modules",
                    rationale="Keep patch scope tight to reduce regression risk.",
                ),
                PlanStep(
                    id="S3",
                    action="Add or update regression tests for the failing path",
                    rationale="Prevent recurrence and validate behavior.",
                ),
            ],
            acceptance_criteria=(
                [f"Issue '{issue.title}' is resolved."] + request.constraints
            )[:12],
            test_plan=["Run focused unit tests for touched modules."],
            risks=[
                {
                    "risk": "Scope drift during implementation",
                    "severity": "medium",
                    "mitigation": "Keep to minimal diff and acceptance criteria.",
                }
            ],
            out_of_scope=["Broad refactors unrelated to issue acceptance criteria."],
        )

    def _fallback_researcher(
        self, request: LangGraphIssuePrRequest
    ) -> ResearcherOutput:
        repo_snapshot = self._get_or_collect_repo_context(
            request, use_symbol_retrieval=False
        )
        top_app_file = ""
        top_test_file = ""
        matched_files = repo_snapshot.get("matched_files")
        matched_tests = repo_snapshot.get("matched_tests")
        if isinstance(matched_files, list) and matched_files:
            top_app_file = str((matched_files[0] or {}).get("path") or "")
        if isinstance(matched_tests, list) and matched_tests:
            top_test_file = str((matched_tests[0] or {}).get("path") or "")
        issue = request.issue
        return ResearcherOutput(
            findings=[
                {
                    "claim": "Issue likely maps to backend service and endpoint contract behavior.",
                    "evidence": issue.body[:900] or issue.title,
                    "file_path": top_app_file or "backend/app/services",
                }
            ],
            related_tests=[
                {
                    "file_path": top_test_file or "backend/tests",
                    "why": "Likely location for regression coverage based on issue keywords.",
                }
            ],
            unknowns=[
                "Exact failing file requires stack trace or reproduction command."
            ],
            risk_flags=[],
        )

    def _fallback_executor(self, state: IssuePrState) -> ExecutorOutput:
        attempt = int(state.get("revision_count", 0)) + 1
        repo_snapshot = state.get("repo_context_snapshot") or {}
        suggested_commands = (
            repo_snapshot.get("suggested_test_commands")
            if isinstance(repo_snapshot.get("suggested_test_commands"), list)
            else []
        )
        test_cmd = (
            str(suggested_commands[0]).strip()
            if suggested_commands
            else "pytest -q backend/tests"
        )
        return ExecutorOutput(
            changes=[
                {
                    "file": "backend/app/services/<target_service>.py",
                    "summary": "Apply minimal fix for issue behavior under target condition.",
                }
            ],
            tests_added=[
                {
                    "file": "backend/tests/test_<target_area>.py",
                    "summary": "Add regression test for issue reproduction path.",
                }
            ],
            commands_run=[
                {
                    "cmd": test_cmd,
                    "result": "pass" if attempt > 1 else "fail",
                    "output_ref": f"attempt_{attempt}_test_log",
                }
            ],
            assumptions=["Issue can be resolved without schema/database migration."],
            confidence=0.86 if attempt > 1 else 0.72,
        )

    async def _planner_node(self, state: IssuePrState) -> Dict[str, Any]:
        request = state["request"]
        issue = request.issue
        prompt = (
            "Task: produce planning output for issue->PR workflow.\n"
            "Return JSON with keys: plan_steps[{id,action,rationale}], acceptance_criteria[], "
            "test_plan[], risks[{risk,severity,mitigation}], out_of_scope[].\n"
            f"Issue ID: {issue.id}\n"
            f"Issue Title: {issue.title}\n"
            f"Issue Body:\n{issue.body[:6000]}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            "Focus on minimal implementation and regression test coverage."
        )
        try:
            planner = await self._invoke_structured_agent(
                state=state,
                agent_name="planner",
                system_prompt="You are a planning agent. Return only strict JSON.",
                user_prompt=prompt,
                output_model=PlannerOutput,
            )
            result_tag = "done"
        except Exception as exc:
            planner = self._fallback_planner(request)
            result_tag = f"fallback:{str(exc)[:80]}"
        return {
            "planner": planner,
            "event_log": self._append_event(
                state,
                agent="planner",
                action="plan",
                result=result_tag,
                ref=f"issue:{issue.id}",
            ),
        }

    async def _researcher_node(self, state: IssuePrState) -> Dict[str, Any]:
        request = state["request"]
        planner = state.get("planner") or self._fallback_planner(request)
        repo_snapshot = state.get(
            "repo_context_snapshot"
        ) or self._get_or_collect_repo_context(
            request, use_symbol_retrieval=bool(state.get("use_symbol_retrieval", False))
        )
        prompt = (
            "Task: produce evidence-backed repository research output.\n"
            "Return JSON with keys: findings[{claim,evidence,file_path}], related_tests[{file_path,why}], "
            "unknowns[], risk_flags[{flag,severity,evidence}].\n"
            f"Issue Title: {request.issue.title}\n"
            f"Issue Body:\n{request.issue.body[:6000]}\n"
            f"Plan steps: {planner.model_dump_json()}\n"
            f"Repository context snapshot: {json.dumps(repo_snapshot, ensure_ascii=False)}\n"
            "Include concrete file paths when possible."
        )
        try:
            researcher = await self._invoke_structured_agent(
                state=state,
                agent_name="researcher",
                system_prompt="You are a repository research agent. Return only strict JSON.",
                user_prompt=prompt,
                output_model=ResearcherOutput,
            )
            result_tag = "done"
        except Exception as exc:
            researcher = self._fallback_researcher(request)
            result_tag = f"fallback:{str(exc)[:80]}"
        return {
            "researcher": researcher,
            "event_log": self._append_event(
                state,
                agent="researcher",
                action="collect_evidence",
                result=result_tag,
                ref="repo_scan",
            ),
        }

    async def _executor_node(self, state: IssuePrState) -> Dict[str, Any]:
        request = state["request"]
        planner = state.get("planner") or self._fallback_planner(request)
        researcher = state.get("researcher") or self._fallback_researcher(request)
        required_fixes = state.get("required_fixes") or []
        repo_snapshot = state.get(
            "repo_context_snapshot"
        ) or self._get_or_collect_repo_context(
            request, use_symbol_retrieval=bool(state.get("use_symbol_retrieval", False))
        )
        prompt = (
            "Task: produce execution output for issue->PR workflow.\n"
            "Return JSON with keys: changes[{file,summary}], tests_added[{file,summary}], "
            "commands_run[{cmd,result,output_ref}], assumptions[], confidence.\n"
            f"Issue Title: {request.issue.title}\n"
            f"Planner output: {planner.model_dump_json()}\n"
            f"Research output: {researcher.model_dump_json()}\n"
            f"Repository context snapshot: {json.dumps(repo_snapshot, ensure_ascii=False)}\n"
            f"Required fixes from reviewer: {json.dumps([fix.model_dump() for fix in required_fixes])}\n"
            "Use realistic command entries and confidence in [0,1]."
        )
        try:
            executor = await self._invoke_structured_agent(
                state=state,
                agent_name="executor",
                system_prompt="You are an implementation agent. Return only strict JSON.",
                user_prompt=prompt,
                output_model=ExecutorOutput,
            )
            result_tag = "done"
        except Exception as exc:
            executor = self._fallback_executor(state)
            result_tag = f"fallback:{str(exc)[:80]}"
        attempt = int(state.get("revision_count", 0)) + 1
        return {
            "executor": executor,
            "event_log": self._append_event(
                state,
                agent="executor",
                action="implement",
                result=result_tag,
                ref=f"attempt:{attempt}",
            ),
        }

    def _is_high_impact(self, state: IssuePrState) -> bool:
        policy_profile = state["request"].policy_profile or {}
        if bool(policy_profile.get("high_impact_action", False)):
            return True
        risky_keywords = {
            "deploy",
            "production",
            "secret",
            "auth",
            "credential",
            "payment",
            "delete",
            "drop table",
        }
        executor = state.get("executor") or ExecutorOutput()
        text = " ".join(
            [item.summary for item in executor.changes]
            + [item.file for item in executor.changes]
            + [run.cmd for run in executor.commands_run]
        ).lower()
        return any(keyword in text for keyword in risky_keywords)

    async def _reviewer_node(self, state: IssuePrState) -> Dict[str, Any]:
        request = state["request"]
        planner = state.get("planner") or self._fallback_planner(request)
        researcher = state.get("researcher") or self._fallback_researcher(request)
        executor = state.get("executor") or self._fallback_executor(state)
        repo_snapshot = state.get(
            "repo_context_snapshot"
        ) or self._get_or_collect_repo_context(
            request, use_symbol_retrieval=bool(state.get("use_symbol_retrieval", False))
        )
        min_conf = float(state.get("reviewer_min_confidence", 0.75))
        prompt = (
            "Task: review execution output against acceptance criteria and policy.\n"
            "Return JSON with keys: decision(pass|revise|escalate), failures[{criterion,reason,evidence}], "
            "required_fixes[{file,change_request}], policy_checks[{check,status,evidence}], confidence.\n"
            f"Acceptance criteria: {json.dumps(planner.acceptance_criteria, ensure_ascii=False)}\n"
            f"Planner output: {planner.model_dump_json()}\n"
            f"Research output: {researcher.model_dump_json()}\n"
            f"Executor output: {executor.model_dump_json()}\n"
            f"Repository context snapshot: {json.dumps(repo_snapshot, ensure_ascii=False)}\n"
            f"Minimum confidence threshold: {min_conf}\n"
            "Prefer revise for fixable issues and escalate for policy/high-impact risks."
        )
        try:
            reviewer = await self._invoke_structured_agent(
                state=state,
                agent_name="reviewer",
                system_prompt="You are a QA and policy reviewer. Return only strict JSON.",
                user_prompt=prompt,
                output_model=ReviewerOutput,
            )
            result_tag = "done"
        except Exception as exc:
            reviewer = ReviewerOutput(
                decision="escalate",
                failures=[
                    ReviewFailure(
                        criterion="Structured review output",
                        reason="Reviewer output parsing failed.",
                        evidence=str(exc)[:500],
                    )
                ],
                required_fixes=[],
                policy_checks=[],
                confidence=0.0,
            )
            result_tag = f"fallback:{str(exc)[:80]}"

        revision_count = int(state.get("revision_count", 0))
        max_loops = int(state.get("max_revision_loops", 2))
        needs_human = False

        if self._is_high_impact(state):
            needs_human = True
            reviewer = reviewer.model_copy(
                update={
                    "decision": "escalate",
                    "policy_checks": [
                        *reviewer.policy_checks,
                        PolicyCheck(
                            check="High impact action gate",
                            status="fail",
                            evidence="Potential high-impact change detected; human approval required.",
                        ),
                    ],
                }
            )

        if reviewer.decision == "pass" and float(reviewer.confidence) < min_conf:
            if revision_count < max_loops:
                reviewer = reviewer.model_copy(
                    update={
                        "decision": "revise",
                        "required_fixes": [
                            *reviewer.required_fixes,
                            RequiredFix(
                                file="backend/tests",
                                change_request=(
                                    f"Increase evidence quality and confidence to at least {min_conf:.2f}."
                                ),
                            ),
                        ],
                    }
                )
            else:
                reviewer = reviewer.model_copy(update={"decision": "escalate"})
                needs_human = True

        if reviewer.decision == "revise":
            revision_count += 1

        return {
            "reviewer": reviewer,
            "revision_count": revision_count,
            "needs_human": needs_human or bool(reviewer.decision == "escalate"),
            "required_fixes": reviewer.required_fixes,
            "event_log": self._append_event(
                state,
                agent="reviewer",
                action="review",
                result=f"{reviewer.decision}:{result_tag}",
                ref=f"revision:{revision_count}",
            ),
        }

    def _route_after_review(self, state: IssuePrState) -> str:
        reviewer = state.get("reviewer")
        if not reviewer:
            return "finalize"
        if reviewer.decision == "revise":
            revision_count = int(state.get("revision_count", 0))
            max_loops = int(state.get("max_revision_loops", 2))
            if revision_count <= max_loops:
                return "executor"
        return "finalize"

    async def _finalize_node(self, state: IssuePrState) -> Dict[str, Any]:
        reviewer = state.get("reviewer") or ReviewerOutput(decision="escalate")
        request = state["request"]
        planner = state.get("planner") or self._fallback_planner(request)
        executor = state.get("executor") or self._fallback_executor(state)
        status = "blocked"
        reason = "Workflow ended without a passing review."
        pr_draft: Optional[PrDraftPackage] = None

        if reviewer.decision == "pass":
            if bool(state.get("needs_human", False)):
                status = "needs_human_review"
                reason = "Reviewer passed but policy requires human approval."
            else:
                status = "pr_ready"
                reason = "PR draft package is ready."
            scope = (request.issue.labels[0] if request.issue.labels else "agent")[:40]
            pr_draft = PrDraftPackage(
                title=f"fix({scope}): {request.issue.title[:220]}",
                body_sections={
                    "Summary": f"Fixes issue {request.issue.id}: {request.issue.title}",
                    "Root Cause": "Issue-specific mismatch identified in planning/research phases.",
                    "Changes": "; ".join([item.summary for item in executor.changes])
                    or "Code changes applied.",
                    "Test Plan": "; ".join(planner.test_plan) or "Run targeted tests.",
                    "Risks": "; ".join(
                        [str(r.get("risk") or "") for r in planner.risks]
                    )
                    or "No major risks.",
                    "Rollback": "Revert the patch commit if regressions appear.",
                },
                checklist={
                    "acceptance_criteria": [
                        ChecklistItem(item=criterion, status="pass")
                        for criterion in planner.acceptance_criteria
                    ],
                    "policy": [
                        ChecklistItem(item=item.check, status=item.status)
                        for item in reviewer.policy_checks
                    ],
                },
                artifacts=[
                    {"type": "diff", "ref": "generated_diff_summary"},
                    {"type": "test_report", "ref": "pytest_targeted_report"},
                ],
            )
        elif reviewer.decision == "escalate":
            status = "needs_human_review"
            reason = "Reviewer escalated for human review."
        else:
            status = "blocked"
            reason = "Revision loop exhausted before passing review."

        logger.info(f"LangGraph issue-pr workflow finished with status={status}")
        return {
            "status": status,
            "reason": reason,
            "status_reason_code": self._derive_status_reason_code(
                status=status,
                reviewer=reviewer,
                reason=reason,
                needs_human=bool(status == "needs_human_review"),
            ),
            "pr_draft": pr_draft,
            "repo_context_meta": state.get("repo_context_meta") or {},
            "repo_context_summary": self._build_repo_context_summary(
                state.get("repo_context_snapshot") or {}
            ),
            "confidence_breakdown": self._build_confidence_breakdown(
                planner,
                state.get("researcher") or self._fallback_researcher(request),
                executor,
                reviewer,
            ),
            "decision_trace": self._build_decision_trace(
                reviewer=reviewer,
                reason=reason,
                needs_human=bool(status == "needs_human_review"),
            ),
            "event_log": self._append_event(
                state,
                agent="orchestrator",
                action="finalize",
                result=status,
                ref=request.issue.id,
            ),
        }

    def _build_repo_context_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(snapshot, dict):
            return {}
        matched_files = (
            snapshot.get("matched_files")
            if isinstance(snapshot.get("matched_files"), list)
            else []
        )
        matched_tests = (
            snapshot.get("matched_tests")
            if isinstance(snapshot.get("matched_tests"), list)
            else []
        )
        symbol_matches = (
            snapshot.get("symbol_matches")
            if isinstance(snapshot.get("symbol_matches"), list)
            else []
        )
        snippet_matches = (
            snapshot.get("snippet_matches")
            if isinstance(snapshot.get("snippet_matches"), list)
            else []
        )
        suggested = (
            snapshot.get("suggested_test_commands")
            if isinstance(snapshot.get("suggested_test_commands"), list)
            else []
        )

        def _top_paths(items: List[Dict[str, Any]], limit: int = 5) -> List[str]:
            out: List[str] = []
            for item in items[:limit]:
                path = str((item or {}).get("path") or "").strip()
                if path:
                    out.append(path)
            return out

        return {
            "repo_root": str(snapshot.get("repo_root") or ""),
            "scanned_files": int(snapshot.get("scanned_files", 0) or 0),
            "symbol_scan_files": int(snapshot.get("symbol_scan_files", 0) or 0),
            "symbol_retrieval_enabled": bool(
                snapshot.get("symbol_retrieval_enabled", False)
            ),
            "keywords": [str(token) for token in (snapshot.get("keywords") or [])[:10]],
            "top_files": _top_paths(matched_files, 5),
            "top_tests": _top_paths(matched_tests, 5),
            "suggested_test_commands": [str(cmd) for cmd in suggested[:5]],
            "top_symbols": [
                {
                    "path": str((row or {}).get("path") or ""),
                    "symbol": str((row or {}).get("symbol") or ""),
                    "kind": str((row or {}).get("kind") or ""),
                    "score": int((row or {}).get("score", 0) or 0),
                }
                for row in symbol_matches[:5]
            ],
            "top_snippets": [
                {
                    "path": str((row or {}).get("path") or ""),
                    "symbol": str((row or {}).get("symbol") or ""),
                    "kind": str((row or {}).get("kind") or ""),
                    "why_relevant": str((row or {}).get("why_relevant") or ""),
                }
                for row in snippet_matches[:3]
            ],
        }

    def _build_confidence_breakdown(
        self,
        planner: PlannerOutput,
        researcher: ResearcherOutput,
        executor: ExecutorOutput,
        reviewer: ReviewerOutput,
    ) -> Dict[str, float]:
        planner_conf = min(
            1.0,
            0.25
            + min(len(planner.plan_steps), 4) * 0.15
            + min(len(planner.acceptance_criteria), 4) * 0.04,
        )
        researcher_conf = min(
            1.0,
            0.2
            + min(len(researcher.findings), 4) * 0.16
            + min(len(researcher.related_tests), 3) * 0.08,
        )
        executor_conf = max(0.0, min(1.0, float(executor.confidence)))
        reviewer_conf = max(0.0, min(1.0, float(reviewer.confidence)))
        overall = max(
            0.0,
            min(
                1.0,
                (planner_conf * 0.20)
                + (researcher_conf * 0.20)
                + (executor_conf * 0.30)
                + (reviewer_conf * 0.30),
            ),
        )
        return {
            "planner": round(planner_conf, 3),
            "researcher": round(researcher_conf, 3),
            "executor": round(executor_conf, 3),
            "reviewer": round(reviewer_conf, 3),
            "overall": round(overall, 3),
        }

    def _build_decision_trace(
        self,
        *,
        reviewer: ReviewerOutput,
        reason: str,
        needs_human: bool,
    ) -> List[str]:
        trace: List[str] = []
        trace.append(f"reviewer_decision:{reviewer.decision}")
        trace.append(f"reviewer_confidence:{round(float(reviewer.confidence), 3)}")
        if reason:
            trace.append(f"reason:{reason[:180]}")
        if reviewer.failures:
            first_failure = reviewer.failures[0]
            trace.append(
                f"top_failure:{first_failure.criterion[:80]}:{first_failure.reason[:120]}"
            )
        if reviewer.required_fixes:
            first_fix = reviewer.required_fixes[0]
            trace.append(
                f"required_fix:{first_fix.file[:80]}:{first_fix.change_request[:120]}"
            )
        if reviewer.policy_checks:
            failed_policies = [
                item.check
                for item in reviewer.policy_checks
                if str(item.status).lower() == "fail"
            ]
            if failed_policies:
                trace.append(f"policy_failed:{';'.join(failed_policies[:3])[:180]}")
        if needs_human:
            trace.append("escalation:human_review_required")
        return trace

    def _derive_status_reason_code(
        self,
        *,
        status: str,
        reviewer: ReviewerOutput,
        reason: str,
        needs_human: bool,
    ) -> str:
        reason_l = (reason or "").lower()
        if status == "pr_ready":
            return "passed"
        if status == "needs_human_review":
            if "high-impact" in reason_l or "high impact" in reason_l:
                return "policy_escalation"
            if reviewer.decision == "escalate" or needs_human:
                return "manual_escalation"
            return "human_review_required"
        if status == "blocked":
            if "revision loop exhausted" in reason_l:
                return "revision_exhausted"
            if reviewer.decision == "revise":
                return "needs_revision"
            if reviewer.decision == "escalate":
                return "escalated_blocked"
            return "blocked_unknown"
        return "unknown"

    def _coerce_status_reason_code(self, value: str) -> str:
        code = str(value or "").strip()
        if code in self._status_reason_codes:
            return code
        return "unknown"

    def _build_repo_cache_key(self, request: LangGraphIssuePrRequest) -> str:
        return self._build_repo_cache_key_with_mode(request, use_symbol_retrieval=False)

    def _build_repo_cache_key_with_mode(
        self, request: LangGraphIssuePrRequest, *, use_symbol_retrieval: bool
    ) -> str:
        issue = request.issue
        payload = {
            "issue_id": issue.id,
            "issue_title": issue.title,
            "issue_body": issue.body[:4000],
            "labels": issue.labels,
            "constraints": request.constraints,
            "repo_context": request.repo_context,
            "use_symbol_retrieval": bool(use_symbol_retrieval),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return digest

    def _prune_repo_context_cache(self) -> None:
        now = time.time()
        expired_keys = []
        for key, entry in self._repo_snapshot_cache.items():
            created_at = float(entry.get("created_at", 0))
            if (now - created_at) > float(self._repo_snapshot_cache_ttl_seconds):
                expired_keys.append(key)
        for key in expired_keys:
            self._repo_snapshot_cache.pop(key, None)
        if len(self._repo_snapshot_cache) <= self._repo_snapshot_cache_max_entries:
            return
        ordered = sorted(
            self._repo_snapshot_cache.items(),
            key=lambda item: float(item[1].get("created_at", 0)),
        )
        overflow = (
            len(self._repo_snapshot_cache) - self._repo_snapshot_cache_max_entries
        )
        for idx in range(max(0, overflow)):
            self._repo_snapshot_cache.pop(str(ordered[idx][0]), None)

    def _get_or_collect_repo_context(
        self, request: LangGraphIssuePrRequest, *, use_symbol_retrieval: bool
    ) -> Dict[str, Any]:
        snapshot, _meta = self._get_or_collect_repo_context_with_meta(
            request, use_symbol_retrieval=use_symbol_retrieval
        )
        return snapshot

    def _get_or_collect_repo_context_with_meta(
        self,
        request: LangGraphIssuePrRequest,
        *,
        use_symbol_retrieval: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        self._prune_repo_context_cache()
        key = self._build_repo_cache_key_with_mode(
            request, use_symbol_retrieval=use_symbol_retrieval
        )
        now = time.time()
        cached = self._repo_snapshot_cache.get(key)
        if cached:
            created_at = float(cached.get("created_at", 0))
            if (now - created_at) <= float(self._repo_snapshot_cache_ttl_seconds):
                snapshot = cached.get("snapshot")
                if isinstance(snapshot, dict):
                    return (
                        snapshot,
                        {
                            "cache_status": "hit",
                            "scan_ms": 0,
                            "cached_age_ms": int((now - created_at) * 1000),
                        },
                    )
        started = time.perf_counter()
        snapshot = self._collect_repo_context(
            request, use_symbol_retrieval=use_symbol_retrieval
        )
        scan_ms = int((time.perf_counter() - started) * 1000)
        self._repo_snapshot_cache[key] = {
            "created_at": now,
            "snapshot": snapshot,
        }
        return (
            snapshot,
            {
                "cache_status": "miss",
                "scan_ms": scan_ms,
                "cached_age_ms": 0,
            },
        )

    def _collect_repo_context(
        self, request: LangGraphIssuePrRequest, *, use_symbol_retrieval: bool
    ) -> Dict[str, Any]:
        repo_ctx = (
            request.repo_context if isinstance(request.repo_context, dict) else {}
        )
        repo_root = str(repo_ctx.get("repo_root") or ".").strip() or "."
        include_paths = repo_ctx.get("include_paths")
        include = (
            [str(item).strip() for item in include_paths]
            if isinstance(include_paths, list)
            else []
        )
        max_scan_files = int(repo_ctx.get("max_scan_files", 2000) or 2000)
        max_candidates = int(repo_ctx.get("max_candidates", 12) or 12)
        max_content_scan_files = int(repo_ctx.get("max_content_scan_files", 300) or 300)
        content_scan_bytes = int(repo_ctx.get("content_scan_bytes", 8192) or 8192)
        max_file_size_bytes = int(
            repo_ctx.get("max_file_size_bytes", 512 * 1024) or (512 * 1024)
        )

        keywords = self._extract_issue_keywords(request)
        root_path = Path(repo_root).resolve()
        if not root_path.exists():
            return {
                "repo_root": repo_root,
                "error": "repo_root_not_found",
                "keywords": keywords,
                "matched_files": [],
                "matched_tests": [],
            }

        app_candidates: List[Dict[str, Any]] = []
        test_candidates: List[Dict[str, Any]] = []
        scanned = 0
        scanned_content = 0
        include_prefixes = [token for token in include if token]
        for file_path in root_path.rglob("*"):
            if scanned >= max_scan_files:
                break
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(root_path).as_posix()
            if include_prefixes and not any(
                rel.startswith(prefix) for prefix in include_prefixes
            ):
                continue
            ext = file_path.suffix.lower()
            if ext not in {".py", ".ts", ".tsx", ".js", ".go", ".md", ".yaml", ".yml"}:
                continue
            scanned += 1
            score = self._score_path(rel, keywords)
            content_score = 0
            if scanned_content < max_content_scan_files:
                content_score = self._score_file_content(
                    file_path=file_path,
                    keywords=keywords,
                    max_bytes=content_scan_bytes,
                    max_file_size_bytes=max_file_size_bytes,
                )
                scanned_content += 1
            score += content_score
            if score <= 0:
                continue
            item = {
                "path": rel,
                "score": score,
                "path_score": max(score - content_score, 0),
                "content_score": content_score,
            }
            if self._looks_like_test_path(rel):
                test_candidates.append(item)
            else:
                app_candidates.append(item)

        app_candidates.sort(
            key=lambda value: (-int(value.get("score", 0)), value.get("path", ""))
        )
        test_candidates.sort(
            key=lambda value: (-int(value.get("score", 0)), value.get("path", ""))
        )
        top_tests = test_candidates[:max_candidates]
        suggested_test_commands = self._build_test_commands(top_tests)
        result: Dict[str, Any] = {
            "repo_root": str(root_path),
            "scanned_files": scanned,
            "scanned_file_content": scanned_content,
            "keywords": keywords,
            "matched_files": app_candidates[:max_candidates],
            "matched_tests": top_tests,
            "suggested_test_commands": suggested_test_commands,
            "symbol_retrieval_enabled": bool(use_symbol_retrieval),
        }
        if use_symbol_retrieval:
            symbol_pack = self.repo_symbol_index_service.retrieve(
                repo_root=root_path,
                query_keywords=keywords,
                include_paths=include_prefixes,
                max_scan_files=max_scan_files,
                max_symbols=max_candidates * 2,
                max_snippets=max_candidates,
            )
            result["symbol_matches"] = symbol_pack.get("symbol_matches", [])
            result["snippet_matches"] = symbol_pack.get("snippet_matches", [])
            result["related_symbol_tests"] = symbol_pack.get("related_tests", [])
            result["symbol_scan_files"] = int(
                symbol_pack.get("symbol_scan_files", 0) or 0
            )
        return result

    def _extract_issue_keywords(self, request: LangGraphIssuePrRequest) -> List[str]:
        issue = request.issue
        text = f"{issue.title} {issue.body} {' '.join(issue.labels)} {' '.join(request.constraints)}".lower()
        tokens = re.findall(r"[a-z0-9_./-]{3,}", text)
        stop_words = {
            "the",
            "and",
            "with",
            "for",
            "from",
            "that",
            "this",
            "when",
            "must",
            "should",
            "issue",
            "error",
            "fails",
            "fail",
            "test",
            "tests",
        }
        out: List[str] = []
        seen: Set[str] = set()
        for token in tokens:
            if token in stop_words:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token[:60])
            if len(out) >= 20:
                break
        return out

    def _score_path(self, rel_path: str, keywords: List[str]) -> int:
        path_l = rel_path.lower()
        score = 0
        for token in keywords:
            if token in path_l:
                score += 3 if len(token) > 4 else 2
            for part in token.replace("-", "_").split("_"):
                piece = part.strip()
                if piece and len(piece) > 2 and piece in path_l:
                    score += 1
        if self._looks_like_test_path(rel_path):
            score += 1
        if "/app/" in path_l or path_l.startswith("backend/app/"):
            score += 1
        if "/tests/" in path_l or path_l.startswith("backend/tests/"):
            score += 1
        return score

    def _looks_like_test_path(self, rel_path: str) -> bool:
        text = rel_path.lower()
        return (
            "/tests/" in text
            or "__tests__" in text
            or text.endswith("_test.py")
            or text.endswith(".test.ts")
            or text.endswith(".test.tsx")
        )

    def _score_file_content(
        self,
        *,
        file_path: Path,
        keywords: List[str],
        max_bytes: int,
        max_file_size_bytes: int,
    ) -> int:
        try:
            size = file_path.stat().st_size
            if size <= 0 or size > max_file_size_bytes:
                return 0
            raw = file_path.read_bytes()[:max_bytes]
            text = raw.decode("utf-8", errors="ignore").lower()
        except Exception:
            return 0

        score = 0
        for token in keywords:
            if not token:
                continue
            if token in text:
                score += 2 if len(token) > 4 else 1
            for part in token.replace("-", "_").split("_"):
                piece = part.strip()
                if piece and len(piece) > 2 and piece in text:
                    score += 1

        if "def test_" in text or "pytest" in text or "test(" in text:
            score += 1
        if "todo" in text or "fixme" in text:
            score += 1
        return score

    def _build_test_commands(self, matched_tests: List[Dict[str, Any]]) -> List[str]:
        commands: List[str] = []
        for item in matched_tests:
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            path_lower = path.lower()
            cmd = ""
            if path_lower.endswith(".py"):
                cmd = f"pytest -q {path}"
            elif path_lower.endswith(
                (".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx")
            ):
                cmd = f"cd frontend && npm test -- {path}"
            elif path_lower.endswith(".go"):
                cmd = "go test ./..."
            if cmd and cmd not in commands:
                commands.append(cmd)
            if len(commands) >= 5:
                break
        return commands
