"""Pure normalization and deterministic fallbacks for agent plans.

Planner output arrives from an LLM, so it is untrusted shape: missing keys,
alternate key names, wrong types, unbounded lists. These functions clamp it into
the stable schema the runtime stores in job state, and supply deterministic
plans when planning is unavailable — an agent with no plan must still have a
sane sequence to follow rather than improvising from nothing.

Extracted from ``autonomous_agent_executor``; behaviour is byte-identical to the
inline versions it replaced. Nothing here touches a service, a database, or an
``AgentJob``: the fallbacks take the job's goal or type as plain values.
"""

from __future__ import annotations

from typing import Any, Dict, List


def normalize_execution_plan(
    payload: Dict[str, Any],
    max_steps: int = 6,
) -> List[Dict[str, Any]]:
    """Normalize planner output into stable step objects."""
    if not isinstance(payload, dict):
        return []
    raw_steps = payload.get("plan_steps")
    if not isinstance(raw_steps, list):
        raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_steps:
        if isinstance(item, str):
            title = item.strip()
            if not title:
                continue
            step = {
                "title": title[:220],
                "objective": title[:350],
                "exit_criteria": "",
                "suggested_tools": [],
                "status": "pending",
            }
            normalized.append(step)
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            objective = str(item.get("objective") or item.get("purpose") or "").strip()
            exit_criteria = str(
                item.get("exit_criteria") or item.get("done_when") or ""
            ).strip()
            suggested_tools = item.get("suggested_tools")
            if not isinstance(suggested_tools, list):
                suggested_tools = item.get("tools")
            if not isinstance(suggested_tools, list):
                suggested_tools = []
            suggested_tools = [
                str(x).strip() for x in suggested_tools if str(x).strip()
            ]
            if not title and objective:
                title = objective[:180]
            if not title:
                continue
            normalized.append(
                {
                    "title": title[:220],
                    "objective": objective[:500],
                    "exit_criteria": exit_criteria[:300],
                    "suggested_tools": suggested_tools[:6],
                    "status": "pending",
                }
            )
        if len(normalized) >= max_steps:
            break

    return normalized


def fallback_execution_plan(job_type: str, max_steps: int = 6) -> List[Dict[str, Any]]:
    """Create a deterministic fallback plan when LLM planning is unavailable."""
    steps: List[Dict[str, Any]] = [
        {
            "title": "Scope the goal and constraints",
            "objective": "Clarify objective, success criteria, and important constraints.",
            "exit_criteria": "Clear objective statement and constraints captured.",
            "suggested_tools": ["write_progress_report"],
            "status": "pending",
        },
        {
            "title": "Collect high-signal internal evidence",
            "objective": "Find relevant documents and supporting context in the knowledge base.",
            "exit_criteria": "At least one relevant document identified and inspected.",
            "suggested_tools": ["search_documents", "read_document_content"],
            "status": "pending",
        },
    ]

    if str(job_type or "") in {"research", "monitor", "knowledge_expansion"}:
        steps.append(
            {
                "title": "Expand with external research",
                "objective": "Complement internal evidence with current papers when appropriate.",
                "exit_criteria": "Relevant external papers gathered or explicitly deemed unnecessary.",
                "suggested_tools": ["search_arxiv", "find_related_papers"],
                "status": "pending",
            }
        )

    steps.extend(
        [
            {
                "title": "Synthesize findings",
                "objective": "Convert evidence into conclusions, gaps, and next actions.",
                "exit_criteria": "Findings are organized and attributable to sources.",
                "suggested_tools": [
                    "save_research_finding",
                    "create_synthesis_document",
                ],
                "status": "pending",
            },
            {
                "title": "Publish results",
                "objective": "Produce a final output artifact and concise status summary.",
                "exit_criteria": "Final artifact/report produced and progress reported.",
                "suggested_tools": [
                    "create_document_from_text",
                    "write_progress_report",
                ],
                "status": "pending",
            },
        ]
    )
    return steps[:max_steps]


def normalize_causal_experiment_plan(
    payload: Dict[str, Any],
    *,
    max_hypotheses: int = 4,
    max_experiments: int = 6,
) -> Dict[str, Any]:
    """Normalize causal experiment planner output into stable schema."""
    if not isinstance(payload, dict):
        return {}

    hypotheses_raw = payload.get("hypotheses")
    if not isinstance(hypotheses_raw, list):
        hypotheses_raw = []
    hypotheses: List[Dict[str, Any]] = []
    for i, item in enumerate(hypotheses_raw, start=1):
        if isinstance(item, str):
            statement = item.strip()
            if not statement:
                continue
            hypotheses.append(
                {
                    "id": f"H{i}",
                    "statement": statement[:320],
                    "rationale": "",
                    "confidence": 0.5,
                }
            )
        elif isinstance(item, dict):
            statement = str(
                item.get("statement") or item.get("hypothesis") or ""
            ).strip()
            if not statement:
                continue
            hid = str(item.get("id") or f"H{i}").strip()[:24] or f"H{i}"
            rationale = str(item.get("rationale") or item.get("because") or "").strip()[
                :320
            ]
            try:
                conf = float(item.get("confidence", 0.5) or 0.5)
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            hypotheses.append(
                {
                    "id": hid,
                    "statement": statement[:320],
                    "rationale": rationale,
                    "confidence": conf,
                }
            )
        if len(hypotheses) >= max(1, min(max_hypotheses, 12)):
            break

    if not hypotheses:
        return {}
    hyp_ids = [
        str(h.get("id") or "") for h in hypotheses if str(h.get("id") or "").strip()
    ]

    experiments_raw = payload.get("experiments")
    if not isinstance(experiments_raw, list):
        experiments_raw = []
    experiments: List[Dict[str, Any]] = []
    for i, item in enumerate(experiments_raw, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("title") or f"Experiment {i}").strip()
        eid = str(item.get("id") or f"E{i}").strip()[:24] or f"E{i}"
        hypothesis_id = str(
            item.get("hypothesis_id") or item.get("hypothesis") or ""
        ).strip()
        if hypothesis_id not in hyp_ids:
            hypothesis_id = hyp_ids[min(i - 1, len(hyp_ids) - 1)]
        minimal_design = str(
            item.get("minimal_design")
            or item.get("design")
            or item.get("purpose")
            or ""
        ).strip()

        required_data = item.get("required_data")
        if not isinstance(required_data, list):
            required_data = item.get("data")
        if not isinstance(required_data, list):
            required_data = []
        required_data = [str(x).strip()[:140] for x in required_data if str(x).strip()][
            :8
        ]

        steps = item.get("steps")
        if not isinstance(steps, list):
            steps = []
        steps = [str(x).strip()[:180] for x in steps if str(x).strip()][:8]

        success_criteria = item.get("success_criteria")
        if not isinstance(success_criteria, list):
            success_criteria = item.get("metrics")
        if not isinstance(success_criteria, list):
            success_criteria = []
        success_criteria = [
            str(x).strip()[:180] for x in success_criteria if str(x).strip()
        ][:8]

        expected = item.get("expected_evidence")
        if not isinstance(expected, dict):
            expected = {}
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
        ambiguous = (
            expected.get("ambiguous")
            if isinstance(expected.get("ambiguous"), list)
            else []
        )
        expected_norm = {
            "supports": [str(x).strip()[:180] for x in supports if str(x).strip()][:6],
            "falsifies": [str(x).strip()[:180] for x in falsifies if str(x).strip()][
                :6
            ],
            "ambiguous": [str(x).strip()[:180] for x in ambiguous if str(x).strip()][
                :6
            ],
        }

        effort = (
            str(item.get("estimated_effort") or item.get("effort") or "medium")
            .strip()
            .lower()
        )
        if effort not in {"low", "medium", "high"}:
            effort = "medium"

        experiments.append(
            {
                "id": eid,
                "hypothesis_id": hypothesis_id,
                "name": name[:220],
                "minimal_design": minimal_design[:360],
                "required_data": required_data,
                "steps": steps,
                "success_criteria": success_criteria,
                "expected_evidence": expected_norm,
                "estimated_effort": effort,
            }
        )
        if len(experiments) >= max(1, min(max_experiments, 20)):
            break

    if not experiments:
        return {}

    priority_raw = payload.get("priority_order")
    if not isinstance(priority_raw, list):
        priority_raw = []
    exp_ids = [str(e.get("id") or "") for e in experiments]
    priority = [str(x).strip() for x in priority_raw if str(x).strip() in set(exp_ids)]
    if not priority:
        priority = exp_ids[:]

    decision_rules = payload.get("decision_rules")
    if not isinstance(decision_rules, list):
        decision_rules = []
    decision_rules = [str(x).strip()[:220] for x in decision_rules if str(x).strip()][
        :8
    ]
    if not decision_rules:
        decision_rules = [
            "If >=70% of support criteria are met, treat hypothesis as provisionally supported.",
            "If any falsification criterion is strongly observed, deprioritize that hypothesis.",
        ]

    assumptions = payload.get("assumptions")
    if not isinstance(assumptions, list):
        assumptions = []
    assumptions = [str(x).strip()[:180] for x in assumptions if str(x).strip()][:8]

    return {
        "hypotheses": hypotheses,
        "experiments": experiments,
        "priority_order": priority,
        "decision_rules": decision_rules,
        "assumptions": assumptions,
    }


def fallback_causal_experiment_plan(
    goal: str,
    *,
    max_hypotheses: int = 3,
    max_experiments: int = 4,
) -> Dict[str, Any]:
    """Deterministic fallback when LLM causal planning is unavailable."""
    goal = str(goal or "").strip()[:220]
    hypotheses = [
        {
            "id": "H1",
            "statement": f"A focused approach derived from '{goal}' improves the target outcome versus baseline.",
            "rationale": "Primary causal claim from the stated goal.",
            "confidence": 0.55,
        },
        {
            "id": "H2",
            "statement": "Removing the key proposed factor will reduce outcome quality.",
            "rationale": "Ablation-style falsifiability check for causal contribution.",
            "confidence": 0.45,
        },
    ][: max(1, min(max_hypotheses, 8))]

    experiments = [
        {
            "id": "E1",
            "hypothesis_id": "H1",
            "name": "Minimal baseline comparison",
            "minimal_design": "Compare baseline process against the proposed intervention on a small representative sample.",
            "required_data": [
                "Representative sample",
                "Baseline output",
                "Intervention output",
            ],
            "steps": [
                "Define baseline and intervention",
                "Run both on same sample",
                "Measure delta on core metric",
            ],
            "success_criteria": ["Intervention outperforms baseline on primary metric"],
            "expected_evidence": {
                "supports": ["Consistent metric lift over baseline"],
                "falsifies": ["No lift or negative lift vs baseline"],
                "ambiguous": ["Mixed outcomes across segments"],
            },
            "estimated_effort": "low",
        },
        {
            "id": "E2",
            "hypothesis_id": "H2",
            "name": "Ablation stress test",
            "minimal_design": "Remove or weaken the suspected causal factor and re-evaluate outcome quality.",
            "required_data": [
                "Intervention variant without factor",
                "Evaluation rubric",
            ],
            "steps": [
                "Define ablated variant",
                "Run same evaluation",
                "Compare to full intervention",
            ],
            "success_criteria": ["Ablated variant underperforms full intervention"],
            "expected_evidence": {
                "supports": ["Meaningful drop after removing factor"],
                "falsifies": ["No meaningful drop after ablation"],
                "ambiguous": ["Drop only on subset of conditions"],
            },
            "estimated_effort": "medium",
        },
    ][: max(1, min(max_experiments, 12))]

    return {
        "hypotheses": hypotheses,
        "experiments": experiments,
        "priority_order": [str(e.get("id") or "") for e in experiments],
        "decision_rules": [
            "Prioritize the experiment with the highest falsifiability and lowest effort first.",
            "Advance only hypotheses with supporting evidence and no strong falsification signal.",
        ],
        "assumptions": ["Primary metric is stable and measurable on available data."],
        "source": "fallback",
    }
