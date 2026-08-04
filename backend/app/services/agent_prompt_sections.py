"""Pure renderers for the volatile sections of the agent thinking prompt.

Each function turns a slice of runtime state into a compact text block. They are
deliberately free of services and database access: prompt text is contract
surface, and it should be checkable without booting the loop.

These sections belong to the *volatile* half of the thinking prompt. The stable
half is byte-stable per job because it keys the provider prompt cache, so
nothing here may be moved into it.
"""

from __future__ import annotations

from typing import Any

from app.services import agent_tool_scoring


def format_execution_plan(state: dict[str, Any]) -> str:
    """Render current plan context for decision prompts."""
    plan = state.get("execution_plan")
    if not isinstance(plan, list) or not plan:
        return ""

    idx = int(state.get("plan_step_index", 0) or 0)
    idx = max(0, min(idx, len(plan) - 1))
    mode = str(state.get("execution_mode") or "adaptive").strip().lower()

    lines = ["EXECUTION PLAN (Plan-Then-Act):"]
    lines.append(f"- Execution mode: {mode}")
    current = plan[idx] if isinstance(plan[idx], dict) else {}
    lines.append(
        f"- Current step {idx + 1}/{len(plan)}: "
        f"{str(current.get('title') or 'Untitled')[:220]}"
    )
    objective = str(current.get("objective") or "").strip()
    if objective:
        lines.append(f"- Current objective: {objective[:400]}")
    exit_criteria = str(current.get("exit_criteria") or "").strip()
    if exit_criteria:
        lines.append(f"- Exit criteria: {exit_criteria[:300]}")
    tools = current.get("suggested_tools") if isinstance(current, dict) else []
    if isinstance(tools, list) and tools:
        lines.append(f"- Suggested tools: {', '.join([str(t) for t in tools[:8]])}")

    completed_titles: list[str] = []
    for step in plan:
        if not isinstance(step, dict):
            continue
        if str(step.get("status") or "").lower() == "done":
            title = str(step.get("title") or "").strip()
            if title:
                completed_titles.append(title[:120])
    if completed_titles:
        lines.append(f"- Completed steps: {len(completed_titles)}")
    return "\n".join(lines)


def format_causal_experiment_plan(state: dict[str, Any]) -> str:
    """Render causal experiment context for research decisions."""
    plan = state.get("causal_experiment_plan")
    if not isinstance(plan, dict):
        return ""
    hypotheses = (
        plan.get("hypotheses") if isinstance(plan.get("hypotheses"), list) else []
    )
    experiments = (
        plan.get("experiments") if isinstance(plan.get("experiments"), list) else []
    )
    if not hypotheses or not experiments:
        return ""

    lines = ["CAUSAL EXPERIMENT PLAN:"]
    lines.append(f"- Hypotheses: {len(hypotheses)}")
    for hyp in hypotheses[:3]:
        if not isinstance(hyp, dict):
            continue
        hid = str(hyp.get("id") or "").strip()
        statement = str(hyp.get("statement") or "").strip()
        if statement:
            lines.append(f"  - {hid or 'H?'}: {statement[:220]}")

    priority = (
        plan.get("priority_order")
        if isinstance(plan.get("priority_order"), list)
        else []
    )
    exp_map = {
        str(e.get("id") or "").strip(): e
        for e in experiments
        if isinstance(e, dict) and str(e.get("id") or "").strip()
    }
    ordered = [
        eid
        for eid in [str(x).strip() for x in priority if str(x).strip()]
        if eid in exp_map
    ]
    if not ordered:
        ordered = list(exp_map.keys())
    next_ids = ordered[:2]
    if next_ids:
        lines.append(f"- Next experiment IDs: {', '.join(next_ids)}")
    for eid in next_ids:
        exp = exp_map.get(eid) if isinstance(exp_map.get(eid), dict) else {}
        if not exp:
            continue
        name = str(exp.get("name") or "").strip()
        hid = str(exp.get("hypothesis_id") or "").strip()
        lines.append(f"  - {eid} ({hid}): {name[:180]}")
        expected = (
            exp.get("expected_evidence")
            if isinstance(exp.get("expected_evidence"), dict)
            else {}
        )
        supports = (
            expected.get("supports")
            if isinstance(expected.get("supports"), list)
            else []
        )
        falsifies = (
            expected.get("falsifies")
            if isinstance(expected.get("falsifies"), list)
            else []
        )
        if supports:
            lines.append(f"    support signal: {str(supports[0])[:180]}")
        if falsifies:
            lines.append(f"    falsify signal: {str(falsifies[0])[:180]}")
    return "\n".join(lines)


def format_subgoals(state: dict[str, Any]) -> str:
    """Render subgoal context for prompts."""
    subgoals = state.get("subgoals")
    if not isinstance(subgoals, list) or not subgoals:
        return ""

    idx = int(state.get("subgoal_index", 0) or 0)
    idx = max(0, min(idx, len(subgoals) - 1))
    current = subgoals[idx] if isinstance(subgoals[idx], dict) else {}

    lines = ["SUBGOALS:"]
    lines.append(
        f"- Current subgoal {idx + 1}/{len(subgoals)}: "
        f"{str(current.get('title') or '').strip()[:220]}"
    )
    done = 0
    for subgoal in subgoals:
        if (
            isinstance(subgoal, dict)
            and str(subgoal.get("status") or "").lower() == "done"
        ):
            done += 1
    lines.append(f"- Subgoals completed: {done}")
    return "\n".join(lines)


def format_critic(state: dict[str, Any]) -> str:
    """Render the latest critic guidance for prompts."""
    notes = state.get("critic_notes")
    if not isinstance(notes, list) or not notes:
        return ""

    latest = notes[-1] if isinstance(notes[-1], dict) else {}
    if not isinstance(latest, dict):
        return ""

    lines = ["CRITIC FEEDBACK:"]
    assessment = str(latest.get("trajectory_assessment") or "").strip()
    pivot = str(latest.get("pivot") or "").strip()
    if assessment:
        lines.append(f"- Assessment: {assessment[:320]}")
    severity = str(latest.get("severity") or "").strip()
    if severity:
        lines.append(f"- Severity: {severity[:40]}")
    try:
        confidence = float(latest.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    lines.append(f"- Confidence: {max(0.0, min(1.0, confidence)):.2f}")
    if pivot:
        lines.append(f"- Pivot: {pivot[:280]}")
    tools = latest.get("recommended_tools")
    if isinstance(tools, list) and tools:
        lines.append(f"- Recommended tools: {', '.join([str(t) for t in tools[:6]])}")
    risks = latest.get("risks")
    if isinstance(risks, list) and risks:
        lines.append(f"- Top risk: {str(risks[0])[:220]}")
    return "\n".join(lines)


def format_tool_stats(state: dict[str, Any]) -> str:
    """Render per-tool outcomes as prompt hints."""
    current_stats = (
        state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {}
    )
    prior_stats = (
        state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {}
    )
    merged_stats = agent_tool_scoring.merge_tool_stats(prior_stats, current_stats)
    if not merged_stats:
        return ""

    scored: list[tuple[str, int, int, float]] = []
    for tool, raw in merged_stats.items():
        if not isinstance(raw, dict):
            continue
        successes = int(raw.get("success", 0) or 0)
        failures = int(raw.get("failure", 0) or 0)
        if successes + failures <= 0:
            continue
        scored.append(
            (str(tool), successes, failures, agent_tool_scoring.tool_success_ratio(raw))
        )

    if not scored:
        return ""

    scored.sort(key=lambda row: (row[3], row[1], -row[2]), reverse=True)
    best = scored[:3]
    worst = sorted(scored, key=lambda row: (row[3], -row[2], row[1]))[:3]

    lines = ["ADAPTIVE TOOL HINTS:"]
    if prior_stats:
        lines.append(f"- Historical priors loaded for {len(prior_stats)} tools.")
    if best:
        lines.append("- Strong tools:")
        for tool, successes, failures, _ in best:
            lines.append(f"  - {tool}: success={successes}, failure={failures}")
    if worst:
        lines.append("- Weak tools (avoid repeats unless needed):")
        for tool, successes, failures, _ in worst:
            lines.append(f"  - {tool}: success={successes}, failure={failures}")
    return "\n".join(lines)


def format_skill_profile(state: dict[str, Any]) -> str:
    """Render active role profile for the planner prompt."""
    profile = (
        state.get("skill_profile")
        if isinstance(state.get("skill_profile"), dict)
        else {}
    )
    if not profile:
        return ""
    lines = [
        "ROLE PROFILE: "
        f"{str(profile.get('display_name') or profile.get('role') or '').strip()}",
    ]
    directives = profile.get("prompt_directives")
    if isinstance(directives, list):
        for directive in directives[:4]:
            text = str(directive or "").strip()
            if text:
                lines.append(f"- {text}")
    preferred = profile.get("preferred_tools")
    if isinstance(preferred, list) and preferred:
        lines.append(f"- Preferred tools: {', '.join([str(t) for t in preferred[:8]])}")
    discouraged = profile.get("discouraged_tools")
    if isinstance(discouraged, list) and discouraged:
        lines.append(
            f"- Discouraged tools: {', '.join([str(t) for t in discouraged[:6]])}"
        )
    return "\n".join(lines)


def format_feedback_learning(state: dict[str, Any]) -> str:
    """Render compact human-feedback guidance for prompt conditioning."""
    feedback = (
        state.get("feedback_learning")
        if isinstance(state.get("feedback_learning"), dict)
        else {}
    )
    if not feedback:
        return ""
    if int(feedback.get("feedback_count", 0) or 0) <= 0:
        return ""
    lines = ["HUMAN FEEDBACK LEARNING:"]
    average = feedback.get("avg_rating")
    if average is not None:
        try:
            lines.append(f"- Average rating context: {float(average):.2f}/5")
        except Exception:
            pass
    preferred = feedback.get("preferred_tools")
    if isinstance(preferred, list) and preferred:
        lines.append(f"- Prefer tools: {', '.join([str(t) for t in preferred[:6]])}")
    avoid = feedback.get("discouraged_tools")
    if isinstance(avoid, list) and avoid:
        lines.append(f"- Avoid tools: {', '.join([str(t) for t in avoid[:6]])}")
    highlights = feedback.get("highlights")
    if isinstance(highlights, list) and highlights:
        lines.append(f"- Recent feedback note: {str(highlights[0])[:260]}")
    return "\n".join(lines)


def format_execution_graph(runtime: Any) -> str:
    """Render compact live execution-graph diagnostics for the planner prompt."""
    if not isinstance(runtime, dict):
        return ""

    dag_stats = (
        runtime.get("dag_stats") if isinstance(runtime.get("dag_stats"), dict) else {}
    )
    health = (
        runtime.get("graph_health")
        if isinstance(runtime.get("graph_health"), dict)
        else {}
    )
    total_nodes = int(dag_stats.get("total_nodes", 0) or 0)
    total_edges = int(dag_stats.get("total_edges", 0) or 0)
    if total_nodes <= 0 and total_edges <= 0:
        return ""

    lines: list[str] = ["EXECUTION GRAPH:"]
    lines.append(
        f"- Health: {str(health.get('status') or 'unknown')} "
        f"(severity={int(health.get('severity_score', 0) or 0)})"
    )
    reasons = health.get("reasons") if isinstance(health.get("reasons"), list) else []
    if reasons:
        lines.append(f"- Health reasons: {', '.join([str(x) for x in reasons[:6]])}")
    lines.append(
        f"- Nodes={total_nodes}, edges={total_edges}, "
        f"critical_path={int(dag_stats.get('critical_path_length', 0) or 0)}"
    )
    lines.append(
        "- Verify/summarize: "
        f"{int(runtime.get('verification_successes', 0) or 0)}/"
        f"{int(runtime.get('verification_attempts', 0) or 0)} "
        "verifications succeeded; "
        f"{int(runtime.get('summarization_successes', 0) or 0)}/"
        f"{int(runtime.get('summarization_attempts', 0) or 0)} "
        "summaries succeeded"
    )
    recommendations = (
        runtime.get("recommended_actions")
        if isinstance(runtime.get("recommended_actions"), list)
        else []
    )
    if recommendations:
        lines.append("- Recommended actions:")
        for item in recommendations[:4]:
            lines.append(f"  - {str(item)[:220]}")
    return "\n".join(lines)
