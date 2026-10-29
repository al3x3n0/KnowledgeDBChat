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


#: Fields that carry a finding's verdict. Booleans are left out of the
#: metrics line, and `check_implementation` records no title: its findings
#: were dropped whole, so a run whose contract was met by six verified
#: implementations was concluded "The run recorded no findings" (first eval
#: launch, 2026-10-29), and another as having "no reference implementation".
VERDICT_FIELDS = ("verified", "verdict", "intact", "reproduced")


def _verdict_words(finding: Dict[str, Any]) -> str:
    words = []
    for name in VERDICT_FIELDS:
        value = finding.get(name)
        if isinstance(value, bool):
            words.append(name if value else f"not {name}")
        elif isinstance(value, str) and value.strip():
            words.append(f"{name}: {value.strip()[:40]}")
    return ", ".join(words)


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
            or finding.get("subject")
            or ""
        ).strip()
        verdicts = _verdict_words(finding)
        if not title and not verdicts:
            continue
        kind = str(finding.get("type") or "finding").strip()
        row = f"[{kind}] {title[:MAX_FINDING_CHARS]}".rstrip()
        if verdicts:
            row += f" ({verdicts})"
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


#: A second request after an empty reply. The first is sized for the answer;
#: reasoning models charge their thinking to the same budget, and a run's
#: conclusion came back empty having spent all of it reasoning.
RETRY_MAX_TOKENS = 4000
RETRY_NOTE = (
    "\n\nYour previous reply contained no usable answer. Reply now with the "
    "JSON object only, the answer in at most 120 words."
)
MAX_FINDINGS_AS_EVIDENCE = 10


def findings_only_conclusion(
    rows: List[str], reason: str, generated_by: str = "findings_only"
) -> Dict[str, Any]:
    """What a person reads when no model answer could be written: no answer
    (this module does not invent one), and the findings the run did record,
    so the result is the evidence rather than a bare apology."""
    return {
        "answer": None,
        "confidence": "low",
        "evidence": list(rows[:MAX_FINDINGS_AS_EVIDENCE]),
        "gaps": [reason],
        "generated_by": generated_by,
        "findings_considered": len(rows),
    }


def _usable(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(str(payload.get("answer") or "").strip())


def conclusion_without_evidence(reason: str) -> Dict[str, Any]:
    """The honest result when there is nothing to conclude from."""
    return {
        "answer": None,
        "confidence": "low",
        "evidence": [],
        "gaps": [reason],
        "generated_by": "no_evidence",
    }


async def conclude(
    llm_service: Any,
    *,
    goal: str,
    findings: Any,
    db: Any = None,
    snapshot_context: Optional[Dict[str, Any]] = None,
    extra_gaps: Optional[List[str]] = None,
    user_settings: Any = None,
    user_id: Any = None,
) -> Dict[str, Any]:
    """Answer a goal from a set of findings. Never raises.

    The core of `synthesize_conclusion`, taken apart from the job that used to
    be the only caller. A campaign concludes over the findings of *many* jobs
    and has no executor, no iteration and no single job id -- and the rule it
    needs is the same one: if what was collected does not answer the goal, say
    so rather than manufacture something that sounds like an answer.

    `extra_gaps` is for what the caller knows and the findings cannot say --
    that a campaign ran out of budget, say, which makes any answer partial for
    a reason no individual finding records.

    `user_settings` and `user_id` are the owner's: the conclusion is a model
    call made for them, like every other one in the run. Without them it went
    to the deployment's default provider whatever provider the owner had
    chosen, and its usage was attributed to nobody.
    """
    rows = summarize_findings_for_prompt(findings)
    if not rows:
        result = conclusion_without_evidence("No findings were recorded.")
        result["gaps"] = list(result["gaps"]) + list(extra_gaps or [])
        return result

    from app.services import llm_structured

    async def _ask(max_tokens: int, note: str = "", phase_suffix: str = "") -> Any:
        context = dict(snapshot_context or {})
        if phase_suffix and context.get("phase"):
            context["phase"] = f"{context['phase']}{phase_suffix}"
        return await llm_structured.ask_for_json(
            llm_service,
            schema=CONCLUSION_SCHEMA,
            system_prompt=SYSTEM_PROMPT,
            user_message=build_conclusion_prompt(goal, findings) + note,
            task_type="summarization",
            temperature=0.1,
            max_tokens=max_tokens,
            user_settings=user_settings,
            user_id=user_id,
            db=db,
            snapshot_context=context or None,
        )

    try:
        payload = await _ask(900)
        if not _usable(payload):
            payload = await _ask(RETRY_MAX_TOKENS, RETRY_NOTE, "_retry")
    except Exception as exc:
        # The caller is finalizing something. An optional summary that aborts
        # would also skip whatever runs after it.
        logger.warning(f"Conclusion synthesis failed: {exc}")
        result = findings_only_conclusion(
            rows,
            f"Conclusion could not be generated: {str(exc)[:200]}",
            generated_by="error",
        )
        result["gaps"] += list(extra_gaps or [])
        return result

    if not _usable(payload):
        result = findings_only_conclusion(
            rows,
            "The model returned no usable conclusion, twice; the findings "
            "below are what the run recorded, without an interpretation.",
        )
        result["gaps"] += list(extra_gaps or [])
        return result

    return {
        "answer": str(payload.get("answer"))[:4000],
        "confidence": str(payload.get("confidence") or "low"),
        "evidence": [str(x)[:300] for x in (payload.get("evidence") or [])][:10],
        "gaps": [str(x)[:300] for x in (payload.get("gaps") or [])][:10]
        + list(extra_gaps or []),
        "generated_by": "llm",
        "findings_considered": len(rows),
        "findings_total": len([f for f in findings if isinstance(f, dict)]),
    }


def _budget_exhausted_conclusion(rows: Optional[List[str]] = None) -> Dict[str, Any]:
    return findings_only_conclusion(
        list(rows or []),
        "Model-call budget exhausted before conclusion generation could finish.",
        generated_by="budget_exhausted",
    )


class _BudgetedConclusionLLM:
    """Share the job allowance across structured and prompted JSON requests."""

    def __init__(self, service: Any, job: Any) -> None:
        self.service = service
        self.job = job
        self.exhausted = False

    async def _call(self, name: str, **kwargs: Any) -> Any:
        method = getattr(self.service, name, None)
        if not callable(method):
            # Missing schema support is not an attempted provider request.
            raise AttributeError(f"Model service has no {name}")
        used = int(getattr(self.job, "llm_calls_used", 0) or 0)
        limit = int(getattr(self.job, "max_llm_calls", 0) or 0)
        if used >= limit:
            self.exhausted = True
            raise RuntimeError("Conclusion model-call budget exhausted")
        self.job.llm_calls_used = used + 1
        return await method(**kwargs)

    async def generate_structured(self, **kwargs: Any) -> Any:
        return await self._call("generate_structured", **kwargs)

    async def generate_response(self, **kwargs: Any) -> Any:
        return await self._call("generate_response", **kwargs)


async def synthesize_conclusion(
    executor: Any,
    job: Any,
    state: Dict[str, Any],
    db: Any = None,
) -> Optional[Dict[str, Any]]:
    """Answer the job's goal from its findings. Never raises."""
    findings = state.get("findings") if isinstance(state.get("findings"), list) else []
    if not summarize_findings_for_prompt(findings):
        # Kept as its own message: "the run recorded no findings" is about a
        # job, and a campaign saying that would be wrong -- its jobs may each
        # have recorded plenty.
        return conclusion_without_evidence("The run recorded no findings.")

    rows = summarize_findings_for_prompt(findings)
    if int(getattr(job, "llm_calls_used", 0) or 0) >= int(
        getattr(job, "max_llm_calls", 0) or 0
    ):
        return _budget_exhausted_conclusion(rows)

    from app.services.llm_service import load_user_llm_settings

    owner = getattr(job, "user_id", None)
    model = _BudgetedConclusionLLM(executor.llm_service, job)
    result = await conclude(
        model,
        goal=getattr(job, "goal", ""),
        findings=findings,
        db=db,
        user_settings=await load_user_llm_settings(db, owner),
        user_id=owner,
        snapshot_context={
            "job_id": str(getattr(job, "id", "") or "") or None,
            "iteration": int(getattr(job, "iteration", 0) or 0),
            "phase": "conclusion",
        },
    )
    return _budget_exhausted_conclusion(rows) if model.exhausted else result


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
    "conclude",
    "build_conclusion_prompt",
    "conclusion_line",
    "conclusion_without_evidence",
    "findings_only_conclusion",
    "summarize_findings_for_prompt",
    "synthesize_conclusion",
]
