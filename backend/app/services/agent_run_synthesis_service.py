"""Turn a run's findings into an answer to the goal it was given.

A run recorded nine codegen measurements and ended with a list of them. The
measurements were correct and nobody had said what they meant, so the job's
own report of itself was a table with no conclusion.

The conclusion is written once at finalization, from the findings already
recorded. It is deliberately not a place to gather more evidence: if what the
run collected does not answer the goal, saying so is the useful output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

MAX_FINDINGS_IN_PROMPT = 40
MAX_FINDING_CHARS = 300

CONCLUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Direct answer to the goal, or a plain statement "
            "that the evidence does not answer it.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Findings that support the answer, quoted from the list.",
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the run did not establish.",
        },
    },
    "required": ["answer", "confidence"],
}

SYSTEM_PROMPT = (
    "You are summarizing a completed autonomous run for an engineer who will "
    "act on it.\n"
    "Answer the goal using only the findings given. Cite the findings you "
    "relied on.\n"
    "If the findings do not answer the goal, say so plainly and list what is "
    "missing: a wrong confident answer is worse than an admitted gap.\n"
    "Do not invent measurements, numbers or sources.\n\n"
    # The schema-constrained path is not always available: DeepSeek answered a
    # schema request with HTTP 400, the caller fell back to plain prompting,
    # and the model returned a correct answer in markdown that JSON extraction
    # then discarded. The shape has to be stated in the prompt as well.
    "Respond with a single JSON object and nothing else. No markdown, no "
    "code fences, no commentary.\n"
    "Shape:\n"
    '{"answer": "<direct answer to the goal>", '
    '"confidence": "high|medium|low", '
    '"evidence": ["<finding you relied on>"], '
    '"gaps": ["<what the run did not establish>"]}'
)


MAX_METRICS_PER_FINDING = 6
# Containers a tool uses for the numbers it measured, alongside the title.
METRIC_CONTAINERS = ("reported_metrics", "codegen", "metrics")


def _format_number(value: float) -> str:
    text = f"{value:.6g}"
    return text


def summarize_finding_metrics(finding: Dict[str, Any]) -> str:
    """Render the numbers a finding carries, so the answer can cite them.

    Only a finding's title used to reach this prompt, so every measurement a
    tool recorded in a field rather than in its title was invisible here: runs
    concluded "GFLOP/s values were not recorded" about findings that carried
    five of them. Ranges rather than a single figure, because which end of a
    series is the good one depends on the metric, and picking for the model
    would be this module inventing an interpretation.
    """
    pairs: List[str] = []

    def _add(name: str, value: Any) -> None:
        if len(pairs) >= MAX_METRICS_PER_FINDING:
            return
        if isinstance(value, bool) or name.startswith("_"):
            return
        if isinstance(value, (int, float)):
            pairs.append(f"{name}={_format_number(float(value))}")
            return
        if isinstance(value, list):
            numbers = [v for v in value if isinstance(v, (int, float))]
            numbers = [float(v) for v in numbers if not isinstance(v, bool)]
            if not numbers:
                return
            low, high = min(numbers), max(numbers)
            span = (
                _format_number(low)
                if low == high
                else f"{_format_number(low)}..{_format_number(high)}"
            )
            pairs.append(f"{name}={span} (n={len(numbers)})")

    for container in METRIC_CONTAINERS:
        values = finding.get(container)
        if isinstance(values, dict):
            for name, value in values.items():
                _add(str(name), value)
    for name, value in finding.items():
        if name in METRIC_CONTAINERS:
            continue
        # Lists count here too, not only inside a metric container. A
        # benchmark_measurement records `all_ms: [54, 18, 16, 15, 30]` at the
        # top level, and reading only the scalars beside it dropped the series
        # — the same way this function's own reason for existing was findings
        # whose numbers lived in a field rather than in the title. `_add`
        # ignores a list with no numbers in it, so a list of strings still
        # contributes nothing.
        if isinstance(value, list):
            _add(str(name), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            _add(str(name), value)

    return ", ".join(pairs)


def summarize_findings_for_prompt(findings: Any) -> List[str]:
    """Render findings as short lines, keeping whatever identifies them."""
    rows: List[str] = []
    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, dict):
            continue
        title = str(
            finding.get("title")
            or finding.get("summary")
            or finding.get("content")
            or ""
        ).strip()
        if not title:
            continue
        kind = str(finding.get("type") or "finding").strip()
        row = f"[{kind}] {title[:MAX_FINDING_CHARS]}"
        metrics = summarize_finding_metrics(finding)
        if metrics:
            row += f" | {metrics}"
        rows.append(row)
        if len(rows) >= MAX_FINDINGS_IN_PROMPT:
            break
    return rows


def build_conclusion_prompt(goal: str, findings: Any) -> str:
    lines = summarize_findings_for_prompt(findings)
    return (
        f"GOAL:\n{str(goal or '').strip()[:2000]}\n\n"
        f"FINDINGS RECORDED ({len(lines)}):\n" + "\n".join(f"- {row}" for row in lines)
    )


def conclusion_without_evidence(reason: str) -> Dict[str, Any]:
    """The honest result when there is nothing to conclude from."""
    return {
        "answer": None,
        "confidence": "low",
        "evidence": [],
        "gaps": [reason],
        "generated_by": "no_evidence",
    }


async def synthesize_conclusion(
    executor: Any,
    job: Any,
    state: Dict[str, Any],
    db: Any = None,
) -> Optional[Dict[str, Any]]:
    """Answer the job's goal from its findings. Never raises."""
    findings = state.get("findings") if isinstance(state.get("findings"), list) else []
    rows = summarize_findings_for_prompt(findings)
    if not rows:
        return conclusion_without_evidence("The run recorded no findings.")

    try:
        from app.services import llm_structured

        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=CONCLUSION_SCHEMA,
            system_prompt=SYSTEM_PROMPT,
            user_message=build_conclusion_prompt(getattr(job, "goal", ""), findings),
            task_type="summarization",
            temperature=0.1,
            max_tokens=900,
            db=db,
            snapshot_context={
                "job_id": str(getattr(job, "id", "") or "") or None,
                "iteration": int(getattr(job, "iteration", 0) or 0),
                "phase": "conclusion",
            },
        )
    except Exception as exc:
        # Finalization must complete regardless: an optional summary that
        # aborts the run would also skip the chain trigger after it.
        logger.warning(
            f"Conclusion synthesis failed for job {getattr(job, 'id', '')}: {exc}"
        )
        return {
            "answer": None,
            "confidence": "low",
            "evidence": [],
            "gaps": [f"Conclusion could not be generated: {str(exc)[:200]}"],
            "generated_by": "error",
        }

    if not isinstance(payload, dict) or not str(payload.get("answer") or "").strip():
        return conclusion_without_evidence(
            "The model did not return a usable conclusion."
        )

    return {
        "answer": str(payload.get("answer"))[:4000],
        "confidence": str(payload.get("confidence") or "low"),
        "evidence": [str(x)[:300] for x in (payload.get("evidence") or [])][:10],
        "gaps": [str(x)[:300] for x in (payload.get("gaps") or [])][:10],
        "generated_by": "llm",
        "findings_considered": len(rows),
        "findings_total": len([f for f in findings if isinstance(f, dict)]),
    }


def conclusion_line(conclusion: Optional[Dict[str, Any]]) -> str:
    """One-line rendering for logs and compact views."""
    if not isinstance(conclusion, dict):
        return ""
    answer = str(conclusion.get("answer") or "").strip()
    if not answer:
        gaps = conclusion.get("gaps") or []
        return f"No conclusion: {gaps[0]}" if gaps else "No conclusion."
    return f"{answer[:300]} (confidence: {conclusion.get('confidence')})"


__all__ = [
    "CONCLUSION_SCHEMA",
    "build_conclusion_prompt",
    "conclusion_line",
    "conclusion_without_evidence",
    "summarize_findings_for_prompt",
    "synthesize_conclusion",
]
