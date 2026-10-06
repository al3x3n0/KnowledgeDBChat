"""Autonomous-job tools: the ``reasoning`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_reasoning_provider(executor: Any) -> FunctionToolProvider:
    """Structured reasoning tools for AutonomousAgentExecutor."""

    async def _reflect(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        reflections = state.get("reflections")
        if not isinstance(reflections, list):
            reflections = []
        topic = str(params.get("topic") or "").strip()
        assessment = str(params.get("assessment") or "").strip()
        if not topic or not assessment:
            return {"error": "topic and assessment are required"}
        entry = {
            "iteration": int(job.iteration or 0),
            "topic": topic[:300],
            "assessment": assessment[:500],
            "blind_spots": [
                str(b)[:200]
                for b in (params.get("blind_spots") or [])
                if isinstance(b, str)
            ][:10],
            "suggested_corrections": [
                str(c)[:200]
                for c in (params.get("suggested_corrections") or [])
                if isinstance(c, str)
            ][:10],
            "timestamp": datetime.utcnow().isoformat(),
        }
        reflections.append(entry)
        state["reflections"] = reflections[-50:]
        return {
            "success": True,
            "data": {"reflection_count": len(state["reflections"]), "recorded": entry},
        }

    async def _hypothesize(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        hypotheses = state.get("hypotheses")
        if not isinstance(hypotheses, list):
            hypotheses = []
        hyp_id = str(params.get("hypothesis_id") or "").strip()
        valid_statuses = {"proposed", "testing", "supported", "refuted", "inconclusive"}
        given_status = str(params.get("status") or "").strip()
        if given_status and given_status not in valid_statuses:
            # Refused, not coerced: an unknown status used to become
            # "proposed" and overwrite a hypothesis already settled.
            return {
                "error": f"status must be one of {sorted(valid_statuses)}, "
                f"not {given_status!r}"
            }
        status = given_status or "proposed"
        if not hyp_id and not str(params.get("hypothesis") or "").strip():
            return {"error": "hypothesis is required"}
        result: Dict[str, Any] = {}
        if hyp_id:
            updated = False
            for hypothesis in hypotheses:
                if isinstance(hypothesis, dict) and hypothesis.get("id") == hyp_id:
                    # Only when given: adding a rationale must not demote a
                    # supported hypothesis back to "proposed".
                    if given_status:
                        hypothesis["status"] = status
                    if params.get("rationale"):
                        hypothesis["rationale"] = str(params["rationale"])[:400]
                    if params.get("testable_predictions"):
                        hypothesis["testable_predictions"] = [
                            str(p)[:200] for p in params["testable_predictions"]
                        ][:10]
                    hypothesis["updated_at"] = datetime.utcnow().isoformat()
                    updated = True
                    result["data"] = {"hypothesis": hypothesis, "action": "updated"}
                    break
            if not updated:
                result["error"] = f"Hypothesis {hyp_id} not found"
                result["data"] = {
                    "available_ids": [
                        h.get("id") for h in hypotheses if isinstance(h, dict)
                    ]
                }
        else:
            # A counter, not the list length: the list is capped at 30, so
            # every hypothesis after the thirtieth was "h-31".
            counter = int(state.get("hypothesis_counter") or len(hypotheses)) + 1
            state["hypothesis_counter"] = counter
            hyp_id = f"h-{counter}"
            entry = {
                "id": hyp_id,
                "hypothesis": str(params.get("hypothesis", ""))[:500],
                "rationale": str(params.get("rationale") or "")[:400],
                "testable_predictions": [
                    str(p)[:200] for p in (params.get("testable_predictions") or [])
                ][:10],
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
            }
            hypotheses.append(entry)
            result["data"] = {"hypothesis": entry, "action": "created"}
        state["hypotheses"] = hypotheses[-30:]
        result["success"] = not result.get("error")
        return result

    async def _weigh_evidence(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ledger = state.get("evidence_ledger")
        if not isinstance(ledger, list):
            ledger = []
        claim = str(params.get("claim") or "").strip()
        verdict = str(params.get("verdict") or "").strip()
        valid_verdicts = {
            "strongly_supported",
            "weakly_supported",
            "neutral",
            "weakly_refuted",
            "strongly_refuted",
        }
        if not claim or not verdict:
            return {"error": "claim and verdict are required"}
        if verdict not in valid_verdicts:
            return {
                "error": f"verdict must be one of {sorted(valid_verdicts)}, "
                f"not {verdict!r}"
            }
        linked_id = str(params.get("hypothesis_id") or "").strip()
        known_ids = {
            h.get("id") for h in state.get("hypotheses") or [] if isinstance(h, dict)
        }
        if linked_id and linked_id not in known_ids:
            return {
                "error": f"Hypothesis {linked_id} not found",
                "data": {"available_ids": sorted(str(i) for i in known_ids)},
            }

        def _strength(item: Dict[str, Any]) -> float:
            try:
                value = float(item.get("strength", 0.5))
            except (TypeError, ValueError):
                return 0.5
            if value != value:  # NaN compares unequal to itself
                return 0.5
            return max(0.0, min(1.0, value))

        ev_for = params.get("evidence_for") or []
        ev_against = params.get("evidence_against") or []
        entry = {
            "claim": claim[:500],
            "hypothesis_id": str(params.get("hypothesis_id") or "").strip() or None,
            "evidence_for": [
                {
                    "statement": str(e.get("statement", ""))[:300],
                    "source_document_id": str(e.get("source_document_id") or ""),
                    "strength": _strength(e),
                }
                for e in ev_for
                if isinstance(e, dict)
            ][:10],
            "evidence_against": [
                {
                    "statement": str(e.get("statement", ""))[:300],
                    "source_document_id": str(e.get("source_document_id") or ""),
                    "strength": _strength(e),
                }
                for e in ev_against
                if isinstance(e, dict)
            ][:10],
            "verdict": verdict,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for_score = (
            sum(e["strength"] for e in entry["evidence_for"])
            if entry["evidence_for"]
            else 0
        )
        against_score = (
            sum(e["strength"] for e in entry["evidence_against"])
            if entry["evidence_against"]
            else 0
        )
        entry["aggregate_score"] = round(for_score - against_score, 3)
        ledger.append(entry)
        state["evidence_ledger"] = ledger[-100:]
        hyp_id = entry.get("hypothesis_id")
        if hyp_id:
            for hypothesis in state.get("hypotheses") or []:
                if isinstance(hypothesis, dict) and hypothesis.get("id") == hyp_id:
                    if verdict in {"strongly_supported", "weakly_supported"}:
                        hypothesis["status"] = "supported"
                    elif verdict in {"strongly_refuted", "weakly_refuted"}:
                        hypothesis["status"] = "refuted"
                    break
        return {
            "success": True,
            "data": {"entry": entry, "ledger_size": len(state["evidence_ledger"])},
        }

    async def _critique_plan(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        critiques = state.get("plan_critiques")
        if not isinstance(critiques, list):
            critiques = []
        severity = str(params.get("severity") or "moderate").strip()
        if severity not in {"minor", "moderate", "major"}:
            return {
                "error": f"severity must be minor, moderate or major, not {severity!r}"
            }
        plan_summary = str(params.get("plan_summary") or "").strip()
        if not plan_summary or not params.get("weaknesses"):
            return {"error": "plan_summary and weaknesses are required"}
        entry = {
            "iteration": int(job.iteration or 0),
            "plan_summary": plan_summary[:500],
            "weaknesses": [
                str(w)[:200]
                for w in (params.get("weaknesses") or [])
                if isinstance(w, str)
            ][:10],
            "missing_steps": [
                str(s)[:200]
                for s in (params.get("missing_steps") or [])
                if isinstance(s, str)
            ][:10],
            "assumptions_challenged": [
                str(a)[:200]
                for a in (params.get("assumptions_challenged") or [])
                if isinstance(a, str)
            ][:10],
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }
        critiques.append(entry)
        state["plan_critiques"] = critiques[-20:]
        if severity == "major":
            notes = state.get("critic_notes")
            if not isinstance(notes, list):
                notes = []
            notes.append(
                {
                    "trajectory_assessment": f"Plan critique (major): {entry['plan_summary'][:200]}",
                    "pivot": "; ".join(entry["weaknesses"][:3]),
                    "recommended_tools": [],
                    "source": "critique_plan_tool",
                }
            )
            state["critic_notes"] = notes[-6:]
        return {
            "success": True,
            "data": {
                "critique": entry,
                "critiques_count": len(state["plan_critiques"]),
            },
        }

    return FunctionToolProvider(
        name="autonomous_reasoning_tools",
        modes={"autonomous"},
        handlers={
            "reflect": _reflect,
            "hypothesize": _hypothesize,
            "weigh_evidence": _weigh_evidence,
            "critique_plan": _critique_plan,
        },
    )
