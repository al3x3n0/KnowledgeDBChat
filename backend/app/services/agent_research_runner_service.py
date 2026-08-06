"""Deterministic runner services extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.services import llm_json
from app.services.ai_hub_dataset_preset_service import ai_hub_dataset_preset_service
from app.services.ai_hub_eval_service import ai_hub_eval_service
from app.services.autonomy_service import (
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.research_opportunity_service import (
    collect_research_opportunity_linked_ids,
    compute_research_opportunity_evidence_revision,
    compute_research_portfolio_config_revision,
    list_normalized_research_opportunities,
    merge_operator_fields,
    normalize_research_opportunity,
    summarize_portfolio_operator_reviews,
    summarize_research_opportunity_autonomy_states,
    summarize_research_opportunity_stages,
)


def _extract_json(text: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM response.

    Delegates to ``llm_json`` so this runner tolerates exactly what the rest of
    the agent stack tolerates. The shared implementation scans balanced brace
    spans, so replies that put a second object after the first — which this
    runner used to drop — now parse.
    """
    return llm_json.extract_json_object(text)


class AgentResearchRunnerService:
    async def run_ai_hub_scientist(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: propose an AI Hub plugin bundle for the current deployment.

        Produces `job.results.ai_hub_bundle` with enabled preset IDs + eval template IDs and a demo plan.
        """
        from app.core.feature_flags import get_str as get_feature_str
        from app.core.feature_flags import set_str as set_feature_str
        from app.schemas.customer_profile import CustomerProfile

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "ai_hub_scientist", "result": details}
            )

        # Customer profile (deployment-level) optionally overrides defaults.
        customer_profile_raw = await get_feature_str("ai_hub_customer_profile")
        customer_profile: CustomerProfile | None = None
        if customer_profile_raw:
            try:
                customer_profile = CustomerProfile.model_validate(
                    json.loads(customer_profile_raw)
                )
            except Exception:
                customer_profile = None

        workflows = (job.config or {}).get("workflows")
        if not workflows and customer_profile and customer_profile.preferred_workflows:
            workflows = customer_profile.preferred_workflows
        workflows = workflows or ["triage", "extraction", "literature"]
        workflows = [str(x).strip().lower() for x in workflows if str(x).strip()]

        apply_now = bool((job.config or {}).get("apply", False))
        customer_context = str((job.config or {}).get("customer_context") or "").strip()
        if not customer_context and customer_profile and customer_profile.notes:
            customer_context = str(customer_profile.notes).strip()

        _emit(
            10,
            "planning",
            f"Building AI Hub bundle for workflows: {', '.join(workflows) or 'all'}",
        )
        await db.commit()

        # --- Build a lightweight customer signal (no LLM) ---
        STOPWORDS = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "over",
            "under",
            "when",
            "where",
            "what",
            "which",
            "while",
            "your",
            "you",
            "are",
            "our",
            "their",
            "they",
            "them",
            "then",
            "than",
            "also",
            "only",
            "just",
            "more",
            "most",
            "less",
            "very",
            "use",
            "using",
            "used",
            "make",
            "made",
            "help",
            "helps",
            "via",
            "can",
            "could",
            "should",
            "would",
            "may",
            "might",
            "will",
            "data",
            "dataset",
            "datasets",
            "model",
            "models",
            "train",
            "training",
            "eval",
            "evaluate",
            "evaluation",
            "assistant",
            "job",
            "jobs",
            "v1",
            "v2",
            "version",
            "note",
            "notes",
            "paper",
            "papers",
            "doc",
            "docs",
            "document",
            "documents",
        }

        def _tokens(text: str) -> list[str]:
            raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
            out: list[str] = []
            for w in raw:
                w = w.strip("_-")
                if len(w) < 3:
                    continue
                if w in STOPWORDS:
                    continue
                out.append(w)
            return out

        async def _collect_customer_corpus() -> tuple[str, dict]:
            """
            Collect lightweight signals from this deployment + user.
            Assumes this deployment typically maps to one customer/workspace.
            """
            evidence: dict[str, Any] = {"signals": []}
            parts: list[str] = []

            if customer_profile and customer_profile.name:
                parts.append(str(customer_profile.name))
                evidence["signals"].append(
                    {
                        "source": "customer_profile.name",
                        "chars": len(customer_profile.name),
                    }
                )
            if customer_profile and customer_profile.keywords:
                # Keywords are strong signal, but still lightweight and user-controlled.
                parts.extend([" ".join(customer_profile.keywords)] * 4)
                evidence["signals"].append(
                    {
                        "source": "customer_profile.keywords",
                        "count": len(customer_profile.keywords),
                    }
                )

            # User-provided context gets extra weight.
            if customer_context:
                parts.extend([customer_context] * 3)
                evidence["signals"].append(
                    {
                        "source": "job.config.customer_context",
                        "chars": len(customer_context),
                    }
                )

            # Recent docs (titles + tags).
            try:
                from app.models.document import Document, DocumentSource

                docs_result = await db.execute(
                    select(Document.title, Document.tags, Document.source_id)
                    .order_by(Document.created_at.desc())
                    .limit(200)
                )
                rows = docs_result.all()
                parts.extend([r[0] for r in rows if r and r[0]])
                tags: list[str] = []
                src_ids = []
                for _, t, sid in rows:
                    if isinstance(t, list):
                        tags.extend([str(x) for x in t if x])
                    if sid:
                        src_ids.append(sid)
                if tags:
                    evidence["top_tags"] = [
                        k for k, _ in Counter([x.lower() for x in tags]).most_common(20)
                    ]
                if src_ids:
                    src_result = await db.execute(
                        select(DocumentSource.source_type).where(
                            DocumentSource.id.in_(src_ids)
                        )
                    )
                    types = [r[0] for r in src_result.all() if r and r[0]]
                    if types:
                        evidence["source_types"] = [
                            k for k, _ in Counter(types).most_common(10)
                        ]
                evidence["signals"].append(
                    {"source": "recent_documents", "count": len(rows)}
                )
            except Exception as e:
                evidence["signals"].append(
                    {"source": "recent_documents", "error": str(e)}
                )

            # Reading lists (names + descriptions).
            try:
                from app.models.reading_list import ReadingList

                rl_result = await db.execute(
                    select(ReadingList.name, ReadingList.description)
                    .where(ReadingList.user_id == job.user_id)
                    .order_by(ReadingList.updated_at.desc())
                    .limit(30)
                )
                rls = rl_result.all()
                for name, description in rls:
                    if name:
                        parts.append(str(name))
                    if description:
                        parts.append(str(description))
                evidence["signals"].append(
                    {"source": "reading_lists", "count": len(rls)}
                )
            except Exception as e:
                evidence["signals"].append({"source": "reading_lists", "error": str(e)})

            # Research notes (titles + tags).
            try:
                from app.models.research_note import ResearchNote

                rn_result = await db.execute(
                    select(ResearchNote.title, ResearchNote.tags)
                    .where(ResearchNote.user_id == job.user_id)
                    .order_by(ResearchNote.updated_at.desc())
                    .limit(20)
                )
                notes = rn_result.all()
                note_tags: list[str] = []
                for title, tags in notes:
                    if title:
                        parts.append(str(title))
                    if isinstance(tags, list):
                        note_tags.extend([str(x) for x in tags if x])
                if note_tags:
                    evidence["note_tags"] = [
                        k
                        for k, _ in Counter([x.lower() for x in note_tags]).most_common(
                            20
                        )
                    ]
                evidence["signals"].append(
                    {"source": "research_notes", "count": len(notes)}
                )
            except Exception as e:
                evidence["signals"].append(
                    {"source": "research_notes", "error": str(e)}
                )

            return "\n".join(parts), evidence

        corpus, evidence = await _collect_customer_corpus()
        customer_freq = Counter(_tokens(corpus))
        top_keywords = [k for k, _ in customer_freq.most_common(30)]

        # Infer a coarse domain label (used for naming + suggested new plugins)
        kw = set(top_keywords[:50])
        if {"security", "vulnerability", "cve", "malware", "threat"} & kw:
            domain = "Security"
        elif {"robot", "robotics", "slam", "control", "motion"} & kw:
            domain = "Robotics"
        elif {"genome", "protein", "bio", "rna", "sequencing"} & kw:
            domain = "Bio"
        elif {
            "compiler",
            "llvm",
            "clang",
            "microarchitecture",
            "perf",
            "benchmark",
        } & kw:
            domain = "Compiler/Performance"
        elif {"hardware", "rtl", "verilog", "silicon", "chip"} & kw:
            domain = "Hardware"
        else:
            domain = "Research"

        # --- Score plugins against customer keywords ---
        presets = ai_hub_dataset_preset_service.list_presets()
        evals = ai_hub_eval_service.list_templates()

        # Load feedback aggregates to bias scoring (learning loop).
        # Bias is scoped by customer profile name when available.
        feedback_bias: dict[tuple[str, str, str], dict[str, int]] = {}
        try:
            from app.models.ai_hub_recommendation_feedback import (
                AIHubRecommendationFeedback,
            )

            profile_id = customer_profile.id if customer_profile else None
            q = (
                select(
                    AIHubRecommendationFeedback.workflow,
                    AIHubRecommendationFeedback.item_type,
                    AIHubRecommendationFeedback.item_id,
                    AIHubRecommendationFeedback.decision,
                    func.count().label("cnt"),
                )
                .where(
                    AIHubRecommendationFeedback.customer_profile_id == profile_id
                    if profile_id is not None
                    else AIHubRecommendationFeedback.customer_profile_id.is_(None)
                )
                .group_by(
                    AIHubRecommendationFeedback.workflow,
                    AIHubRecommendationFeedback.item_type,
                    AIHubRecommendationFeedback.item_id,
                    AIHubRecommendationFeedback.decision,
                )
            )
            res = await db.execute(q)
            for wf, itype, iid, decision, cnt in res.all():
                key = (str(wf), str(itype), str(iid))
                bucket = feedback_bias.get(key) or {"accept": 0, "reject": 0}
                if str(decision) == "accept":
                    bucket["accept"] += int(cnt or 0)
                elif str(decision) == "reject":
                    bucket["reject"] += int(cnt or 0)
                feedback_bias[key] = bucket
        except Exception:
            feedback_bias = {}

        def _plugin_tokens_preset(p: Any) -> set[str]:
            text = f"{p.id}\n{p.name}\n{p.description}\n{getattr(p, 'dataset_type', '')}\n{getattr(p, 'generation_prompt', '')}"
            return set(_tokens(text))

        def _plugin_tokens_eval(t: Any) -> set[str]:
            cases_text = "\n".join(
                [
                    str(c.get("prompt") or "")
                    for c in (t.cases or [])
                    if isinstance(c, dict)
                ]
            )
            rubric_text = json.dumps(t.rubric or {}, ensure_ascii=False)
            text = f"{t.id}\n{t.name}\n{t.description}\n{t.judge_preamble}\n{rubric_text}\n{cases_text}"
            return set(_tokens(text))

        def _feedback_weight(
            workflow_name: str, item_type: str, item_id: str
        ) -> dict[str, int]:
            bucket = feedback_bias.get((workflow_name, item_type, item_id)) or {
                "accept": 0,
                "reject": 0,
            }
            accepts = int(bucket.get("accept", 0))
            rejects = int(bucket.get("reject", 0))
            # Keep weights moderate; keyword overlap remains primary signal.
            bias = accepts * 20 - rejects * 30
            return {"accepts": accepts, "rejects": rejects, "bias": bias}

        def _score(
            plugin_tokens: set[str], *, workflow_name: str, item_type: str, item_id: str
        ) -> dict[str, Any]:
            overlap = [w for w in plugin_tokens if w in customer_freq]
            overlap_sorted = sorted(
                overlap, key=lambda w: customer_freq.get(w, 0), reverse=True
            )
            base = sum(int(customer_freq.get(w, 0)) for w in overlap_sorted)
            fb = _feedback_weight(workflow_name, item_type, item_id)
            return {
                "score": base + fb["bias"],
                "base_score": base,
                "feedback_bias": fb["bias"],
                "feedback_accepts": fb["accepts"],
                "feedback_rejects": fb["rejects"],
                "overlap": overlap_sorted[:10],
                "overlap_count": len(overlap),
            }

        # Categorize existing plugins into workflows
        def _workflow_for_preset(preset_id: str) -> str:
            pid = (preset_id or "").lower()
            if "triage" in pid or "regression" in pid:
                return "triage"
            if "repro" in pid or "checklist" in pid:
                return "extraction"
            if "gap" in pid or "hypoth" in pid:
                return "literature"
            return "other"

        def _workflow_for_eval(eval_id: str) -> str:
            eid = (eval_id or "").lower()
            if "triage" in eid or "regression" in eid:
                return "triage"
            if "extraction" in eid:
                return "extraction"
            if "literature" in eid:
                return "literature"
            return "other"

        scored_presets: list[dict[str, Any]] = []
        for p in presets:
            wf = _workflow_for_preset(p.id)
            scored_presets.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "workflow": wf,
                    **_score(
                        _plugin_tokens_preset(p),
                        workflow_name=wf,
                        item_type="dataset_preset",
                        item_id=p.id,
                    ),
                }
            )
        scored_evals: list[dict[str, Any]] = []
        for t in evals:
            wf = _workflow_for_eval(t.id)
            scored_evals.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "workflow": wf,
                    **_score(
                        _plugin_tokens_eval(t),
                        workflow_name=wf,
                        item_type="eval_template",
                        item_id=t.id,
                    ),
                }
            )

        def _pick_best(
            scored: list[dict[str, Any]], workflow_name: str
        ) -> Optional[dict[str, Any]]:
            candidates = [x for x in scored if x.get("workflow") == workflow_name]
            candidates.sort(
                key=lambda x: (x.get("score", 0), x.get("overlap_count", 0)),
                reverse=True,
            )
            if not candidates:
                return None

            # Guardrail: avoid items the customer has repeatedly rejected, unless nothing else fits.
            def is_blocked(c: dict[str, Any]) -> bool:
                try:
                    rejects = int(c.get("feedback_rejects") or 0)
                    accepts = int(c.get("feedback_accepts") or 0)
                except Exception:
                    rejects = 0
                    accepts = 0
                # Conservative: block only when there is strong negative signal and no positive signal.
                return rejects >= 3 and accepts == 0

            unblocked = [c for c in candidates if not is_blocked(c)]
            if unblocked:
                candidates = unblocked
            best = candidates[0]
            # Require at least a weak match to claim it's customer-specific
            if (
                best.get("overlap_count", 0) < 3
                and best.get("score", 0) < 5
                and customer_context
            ):
                return None
            return best

        dataset_preset_ids: list[str] = []
        eval_template_ids: list[str] = []
        rationale: list[dict[str, Any]] = []
        recommended_new: list[dict[str, Any]] = []
        selected_by_workflow: dict[str, dict[str, Optional[str]]] = {}

        def _dedupe_preserve_order(items: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for x in items:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        def _best_scored_id_for_workflow(
            scored: list[dict[str, Any]], workflow_name: str
        ) -> Optional[str]:
            candidates = [x for x in scored if x.get("workflow") == workflow_name]
            candidates.sort(
                key=lambda x: (x.get("score", 0), x.get("overlap_count", 0)),
                reverse=True,
            )
            return candidates[0]["id"] if candidates else None

        def _representative_id_for_workflow(
            *,
            workflow_name: str,
            item_key: str,
            selected: dict[str, dict[str, Optional[str]]],
            allowlist_ids: list[str],
            classifier: Callable[[str], str],
            scored: list[dict[str, Any]],
        ) -> Optional[str]:
            # 1) Per-workflow explicit selection (best match)
            explicit = (selected.get(workflow_name) or {}).get(item_key)
            if explicit:
                return explicit

            # 2) If allowlist contains workflow-scoped items, pick first (stable)
            for iid in allowlist_ids:
                try:
                    if classifier(iid) == workflow_name:
                        return iid
                except Exception:
                    continue

            # 3) Fall back to "best available" for that workflow, even if low-signal
            return _best_scored_id_for_workflow(scored, workflow_name)

        for wf in workflows:
            if wf not in {"triage", "extraction", "literature"}:
                continue
            best_preset = _pick_best(scored_presets, wf)
            best_eval = _pick_best(scored_evals, wf)
            selected_by_workflow.setdefault(
                wf, {"dataset_preset_id": None, "eval_template_id": None}
            )

            if best_preset:
                dataset_preset_ids.append(best_preset["id"])
                selected_by_workflow[wf]["dataset_preset_id"] = best_preset["id"]
                rationale.append(
                    {
                        "type": "dataset_preset",
                        "workflow": wf,
                        "id": best_preset["id"],
                        "score": best_preset["score"],
                        "base_score": best_preset.get(
                            "base_score", best_preset["score"]
                        ),
                        "feedback_bias": best_preset.get("feedback_bias", 0),
                        "feedback_accepts": best_preset.get("feedback_accepts", 0),
                        "feedback_rejects": best_preset.get("feedback_rejects", 0),
                        "matched_terms": best_preset["overlap"],
                    }
                )
            else:
                recommended_new.append(
                    {
                        "type": "dataset_preset",
                        "workflow": wf,
                        "id_suggestion": f"{wf}_{domain.lower().replace('/', '_').replace(' ', '_')}_v1".lower(),
                        "name_suggestion": f"{wf.title()} ({domain}) (v1)",
                        "why": "No existing preset matched customer keywords strongly enough.",
                        "skeleton": {
                            "id": "<replace>",
                            "name": "<replace>",
                            "description": f"Customer-specific {wf} dataset generation preset for {domain}.",
                            "dataset_type": "instruction",
                            "generation_prompt": "You are generating training data for a domain-specific research assistant. Generate {num} instruction/answer pairs from the document. Output JSON array with 'instruction' and 'output' only.",
                        },
                    }
                )

            if best_eval:
                eval_template_ids.append(best_eval["id"])
                selected_by_workflow[wf]["eval_template_id"] = best_eval["id"]
                rationale.append(
                    {
                        "type": "eval_template",
                        "workflow": wf,
                        "id": best_eval["id"],
                        "score": best_eval["score"],
                        "base_score": best_eval.get("base_score", best_eval["score"]),
                        "feedback_bias": best_eval.get("feedback_bias", 0),
                        "feedback_accepts": best_eval.get("feedback_accepts", 0),
                        "feedback_rejects": best_eval.get("feedback_rejects", 0),
                        "matched_terms": best_eval["overlap"],
                    }
                )
            else:
                recommended_new.append(
                    {
                        "type": "eval_template",
                        "workflow": wf,
                        "id_suggestion": f"{wf}_{domain.lower().replace('/', '_').replace(' ', '_')}_v1".lower(),
                        "name_suggestion": f"{wf.title()} Eval ({domain}) (v1)",
                        "why": "No existing eval template matched customer keywords strongly enough.",
                        "skeleton": {
                            "id": "<replace>",
                            "name": "<replace>",
                            "description": f"Customer-specific {wf} eval for {domain}.",
                            "version": 1,
                            "judge_preamble": "You are an evaluator for a domain-specific research assistant. Penalize hallucinations; prefer actionable next steps.",
                            "rubric": {
                                "scale": "1-5",
                                "criteria": [
                                    "Actionability",
                                    "Fidelity",
                                    "Clarity",
                                    "Rigor",
                                ],
                            },
                            "cases": [
                                {
                                    "id": f"{wf}_001",
                                    "prompt": "Write a realistic test prompt for this customer/workflow.",
                                }
                            ],
                        },
                    }
                )

        # If nothing selected, prefer presets/evals aligned to requested workflows.
        # Avoid proposing an empty allowlist (empty allowlist == "all enabled" in this product).
        has_customer_signal = bool(customer_context) or bool(top_keywords)
        if not dataset_preset_ids and presets:
            if not has_customer_signal:
                dataset_preset_ids = [p.id for p in presets]
            else:
                requested = set(
                    [
                        w
                        for w in workflows
                        if w in {"triage", "extraction", "literature"}
                    ]
                )
                dataset_preset_ids = [
                    p.id for p in presets if _workflow_for_preset(p.id) in requested
                ]
                if not dataset_preset_ids:
                    dataset_preset_ids = [p.id for p in presets]
        if not eval_template_ids and evals:
            if not has_customer_signal:
                eval_template_ids = [t.id for t in evals]
            else:
                requested = set(
                    [
                        w
                        for w in workflows
                        if w in {"triage", "extraction", "literature"}
                    ]
                )
                eval_template_ids = [
                    t.id for t in evals if _workflow_for_eval(t.id) in requested
                ]
                if not eval_template_ids:
                    eval_template_ids = [t.id for t in evals]

        dataset_preset_ids = _dedupe_preserve_order(dataset_preset_ids)
        eval_template_ids = _dedupe_preserve_order(eval_template_ids)

        _emit(60, "composing", "Prepared bundle configuration and demo plan")
        await db.commit()

        workflow_specs = {
            "triage": {
                "title": "Triage",
                "happy_path": [
                    "Generate dataset",
                    "Train",
                    "Deploy adapter",
                    "Run eval",
                    "Use in Chat",
                ],
            },
            "extraction": {
                "title": "Extraction",
                "happy_path": [
                    "Generate dataset",
                    "Train",
                    "Deploy adapter",
                    "Run eval",
                    "Use in Chat",
                ],
            },
            "literature": {
                "title": "Literature",
                "happy_path": [
                    "Synthesize",
                    "Save note",
                    "Generate dataset",
                    "Train",
                    "Deploy adapter",
                    "Run eval",
                    "Use in Chat",
                ],
            },
        }

        demo_plan: list[dict[str, Any]] = []
        demo_workflows = [w for w in workflows if w in workflow_specs]
        for idx, wf in enumerate(demo_workflows):
            preset_id = _representative_id_for_workflow(
                workflow_name=wf,
                item_key="dataset_preset_id",
                selected=selected_by_workflow,
                allowlist_ids=dataset_preset_ids,
                classifier=_workflow_for_preset,
                scored=scored_presets,
            )
            eval_id = _representative_id_for_workflow(
                workflow_name=wf,
                item_key="eval_template_id",
                selected=selected_by_workflow,
                allowlist_ids=eval_template_ids,
                classifier=_workflow_for_eval,
                scored=scored_evals,
            )
            spec = workflow_specs[wf]
            demo_plan.append(
                {
                    "name": f"Workflow {chr(65 + idx)} — {spec['title']}",
                    "workflow": wf,
                    "preset_id": preset_id,
                    "eval_template_id": eval_id,
                    "happy_path": spec["happy_path"],
                }
            )

        profile_name = (
            (customer_profile.name if customer_profile else "")
            if customer_profile
            else ""
        )
        ai_hub_bundle = {
            "bundle_name": f"{profile_name} Bundle"
            if profile_name
            else (f"{domain} Bundle" if domain != "Research" else "Research Bundle"),
            "inferred_domain": domain,
            "customer_profile": customer_profile.model_dump()
            if customer_profile
            else None,
            "customer_keywords": top_keywords[:20],
            "customer_evidence": evidence,
            "enabled_dataset_presets": dataset_preset_ids,
            "enabled_eval_templates": eval_template_ids,
            "workflows": workflows,
            "env": {
                "AI_HUB_DATASET_ENABLED_PRESET_IDS": ",".join(dataset_preset_ids),
                "AI_HUB_EVAL_ENABLED_TEMPLATE_IDS": ",".join(eval_template_ids),
            },
            "selection_rationale": rationale,
            "recommended_new_plugins": recommended_new,
            "success_metrics": {
                "triage": "median time-to-triage; severe-regression misses",
                "extraction": "field-level correctness; unknowns explicitly marked",
                "literature": "precision@k on read/skim/skip against historical decisions",
            },
            "demo_plan": demo_plan,
        }

        job.results = {
            "summary": f"Proposed an AI Hub bundle (presets + evals) and a {len(demo_plan)}-workflow happy-path demo plan.",
            "actions_count": len(demo_plan),
            "findings_count": 0,
            "ai_hub_bundle": ai_hub_bundle,
        }

        if apply_now:
            user_result = await db.execute(select(User).where(User.id == job.user_id))
            user = user_result.scalar_one_or_none()
            if not user or not user.is_admin():
                job.add_log_entry(
                    {"phase": "apply", "error": "Apply requested but user is not admin"}
                )
            else:
                # Store as CSV in feature flags (empty string means "all enabled" semantics)
                await set_feature_str(
                    "ai_hub_enabled_dataset_presets", ",".join(dataset_preset_ids)
                )
                await set_feature_str(
                    "ai_hub_enabled_eval_templates", ",".join(eval_template_ids)
                )
                job.add_log_entry(
                    {
                        "phase": "apply",
                        "result": "Applied bundle to feature-flag allowlists",
                    }
                )

        _emit(100, "completed", "Bundle proposal ready")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def run_research_inbox_monitor(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: monitor internal KB + arXiv, write new items into `research_inbox_items`.

        Intended for scheduled runs. Dedupes per-user on (item_type, item_key).
        """
        import re

        from app.core.feature_flags import get_str as get_feature_str
        from app.models.research_inbox import ResearchInboxItem
        from app.schemas.customer_profile import CustomerProfile

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "research_inbox_monitor", "result": details}
            )

        def _safe_text(x: Any) -> str:
            try:
                return str(x or "")
            except Exception:
                return ""

        def _parse_iso_dt(s: str) -> Optional[datetime]:
            ss = (s or "").strip()
            if not ss:
                return None
            if ss.endswith("Z"):
                ss = ss[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(ss)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        def _tokens(text: str) -> list[str]:
            raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
            out: list[str] = []
            stop = {
                "the",
                "and",
                "for",
                "with",
                "from",
                "that",
                "this",
                "into",
                "over",
                "under",
                "when",
                "where",
                "what",
                "which",
                "while",
                "your",
                "you",
                "are",
                "our",
                "their",
                "they",
                "them",
                "then",
                "than",
                "also",
                "only",
                "just",
                "more",
                "most",
                "less",
                "use",
                "using",
                "used",
                "make",
                "made",
                "help",
                "helps",
                "via",
                "can",
                "could",
                "should",
                "would",
                "may",
                "might",
                "will",
                "data",
                "dataset",
                "datasets",
                "model",
                "models",
                "train",
                "training",
                "eval",
                "evaluate",
                "evaluation",
                "assistant",
                "job",
                "jobs",
                "paper",
                "papers",
                "doc",
                "docs",
                "document",
                "documents",
                "research",
                "monitor",
            }
            for w in raw:
                w = w.strip("_-")
                if len(w) < 3:
                    continue
                if w in stop:
                    continue
                out.append(w)
            return out

        async def _load_feedback_bias_tokens(
            *, customer: Optional[str]
        ) -> tuple[list[str], set[str], list[str], dict]:
            """
            Load positive/negative token sets from the persisted monitor profile if present,
            otherwise derive from inbox items.

            Returns: (positive_tokens, negative_tokens_set, debug_info)
            """
            debug: dict[str, Any] = {
                "source": None,
                "positive_tokens": [],
                "negative_tokens": [],
                "token_scores": {},
                "muted_patterns": [],
                "phrase_scores": {},
                "source_type_scores": {},
                "recommendation_scores": {},
                "outcome_counters": {},
            }
            muted_patterns: list[str] = []

            # Prefer persisted profile.
            try:
                from app.services.research_monitor_profile_service import (
                    research_monitor_profile_service,
                )

                prof = await research_monitor_profile_service.get_profile(
                    db=db, user_id=job.user_id, customer=customer
                )
                if prof:
                    raw_scores = (
                        prof.token_scores
                        if isinstance(getattr(prof, "token_scores", None), dict)
                        else {}
                    )
                    scores = {
                        str(k): int(v)
                        for k, v in (raw_scores or {}).items()
                        if isinstance(v, (int, float))
                    }
                    positive = [
                        t
                        for t, s in sorted(
                            scores.items(), key=lambda kv: kv[1], reverse=True
                        )
                        if s >= 3
                    ][:20]
                    negative = {
                        t
                        for t, s in sorted(scores.items(), key=lambda kv: kv[1])
                        if s <= -3
                    }
                    negative -= set(positive)
                    try:
                        muted = getattr(prof, "muted_tokens", None)
                        if isinstance(muted, list):
                            negative |= {
                                str(x).strip().lower() for x in muted if str(x).strip()
                            }
                    except Exception:
                        pass
                    try:
                        mp = getattr(prof, "muted_patterns", None)
                        if isinstance(mp, list):
                            muted_patterns = [
                                str(x).strip() for x in mp if str(x).strip()
                            ]
                    except Exception:
                        muted_patterns = []
                    debug["phrase_scores"] = (
                        dict(getattr(prof, "phrase_scores", {}))
                        if isinstance(getattr(prof, "phrase_scores", None), dict)
                        else {}
                    )
                    debug["source_type_scores"] = (
                        dict(getattr(prof, "source_type_scores", {}))
                        if isinstance(getattr(prof, "source_type_scores", None), dict)
                        else {}
                    )
                    debug["recommendation_scores"] = (
                        dict(getattr(prof, "recommendation_scores", {}))
                        if isinstance(
                            getattr(prof, "recommendation_scores", None), dict
                        )
                        else {}
                    )
                    debug["outcome_counters"] = (
                        dict(getattr(prof, "outcome_counters", {}))
                        if isinstance(getattr(prof, "outcome_counters", None), dict)
                        else {}
                    )
                    debug["source"] = "profile"
                    debug["token_scores"] = scores
                    debug["positive_tokens"] = positive[:10]
                    debug["negative_tokens"] = list(sorted(list(negative)))[:10]
                    debug["muted_patterns"] = muted_patterns[:10]
                    return (positive, negative, muted_patterns, debug)
            except Exception:
                pass

            # Fallback: derive from inbox history.
            try:
                stmt = (
                    select(
                        ResearchInboxItem.status,
                        ResearchInboxItem.title,
                        ResearchInboxItem.summary,
                    )
                    .where(
                        ResearchInboxItem.user_id == job.user_id,
                        ResearchInboxItem.status.in_(["accepted", "rejected"]),
                    )
                    .order_by(ResearchInboxItem.updated_at.desc())
                    .limit(250)
                )
                if customer:
                    stmt = stmt.where(ResearchInboxItem.customer == customer)

                res = await db.execute(stmt)
                rows = res.all()
            except Exception:
                return ([], set(), muted_patterns, debug)

            pos = Counter()
            neg = Counter()
            for status, title, summary in rows:
                text = f"{_safe_text(title)} {_safe_text(summary)}"
                toks = _tokens(text)
                if not toks:
                    continue
                if str(status) == "accepted":
                    pos.update(toks)
                elif str(status) == "rejected":
                    neg.update(toks)

            scores2: dict[str, int] = {}
            for t, c in pos.items():
                scores2[t] = scores2.get(t, 0) + int(c)
            for t, c in neg.items():
                scores2[t] = scores2.get(t, 0) - int(c)

            positive2 = [
                t
                for t, s in sorted(scores2.items(), key=lambda kv: kv[1], reverse=True)
                if s >= 3
            ][:20]
            negative2 = {
                t for t, s in sorted(scores2.items(), key=lambda kv: kv[1]) if s <= -3
            }
            negative2 -= set(positive2)
            debug["source"] = "inbox_derived"
            debug["token_scores"] = scores2
            debug["positive_tokens"] = positive2[:10]
            debug["negative_tokens"] = list(sorted(list(negative2)))[:10]
            return (positive2, negative2, muted_patterns, debug)

        def _score_discovery_candidate(
            *, item_type: str, title: str, summary: str, bias: dict | None
        ) -> tuple[int, list[str]]:
            if not isinstance(bias, dict):
                return 0, []
            token_scores = (
                bias.get("token_scores")
                if isinstance(bias.get("token_scores"), dict)
                else {}
            )
            phrase_scores = (
                bias.get("phrase_scores")
                if isinstance(bias.get("phrase_scores"), dict)
                else {}
            )
            source_type_scores = (
                bias.get("source_type_scores")
                if isinstance(bias.get("source_type_scores"), dict)
                else {}
            )
            text = f"{title or ''} {summary or ''}".strip()
            tokens = _tokens(text)
            phrases = [
                f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)
            ]
            score = 0
            reasons: list[str] = []
            if item_type and item_type in source_type_scores:
                delta = int(source_type_scores.get(item_type) or 0)
                score += delta * 6
                reasons.append(f"source_type:{item_type}:{delta}")
            token_delta = (
                sum(int(token_scores.get(token) or 0) for token in tokens[:10])
                if isinstance(token_scores, dict)
                else 0
            )
            if token_delta:
                score += token_delta
                reasons.append("token_bias")
            phrase_delta = (
                sum(int(phrase_scores.get(phrase) or 0) for phrase in phrases[:6])
                if isinstance(phrase_scores, dict)
                else 0
            )
            if phrase_delta:
                score += phrase_delta * 2
                reasons.append("phrase_bias")
            return int(score), reasons[:4]

        async def _create_inbox_item(
            *,
            item_type: str,
            item_key: str,
            title: str,
            summary: Optional[str],
            url: Optional[str],
            published_at: Optional[datetime],
            customer: Optional[str],
            metadata: dict,
        ) -> bool:
            it = ResearchInboxItem(
                user_id=job.user_id,
                job_id=job.id,
                customer=customer,
                item_type=item_type,
                item_key=item_key,
                title=title or item_key,
                summary=summary,
                url=url,
                published_at=published_at,
                discovered_at=datetime.utcnow(),
                status="new",
                feedback=None,
                item_metadata=metadata,
            )
            db.add(it)
            try:
                await db.flush()
                return True
            except IntegrityError:
                await db.rollback()
                return False

        # Customer profile (deployment-level) + optional per-job context.
        customer_profile_raw = await get_feature_str("ai_hub_customer_profile")
        customer_profile: CustomerProfile | None = None
        if customer_profile_raw:
            try:
                customer_profile = CustomerProfile.model_validate(
                    json.loads(customer_profile_raw)
                )
            except Exception:
                customer_profile = None

        customer_context = _safe_text(
            (job.config or {}).get("customer_context")
        ).strip()
        if not customer_context and customer_profile and customer_profile.notes:
            customer_context = _safe_text(customer_profile.notes).strip()

        customer_name = (
            _safe_text(
                getattr(customer_profile, "name", "") if customer_profile else ""
            ).strip()
            or None
        )
        customer_tag = (
            _safe_text((job.config or {}).get("customer") or customer_name).strip()
            or None
        )

        prefer_sources = (job.config or {}).get("prefer_sources")
        if not isinstance(prefer_sources, list):
            prefer_sources = ["documents", "arxiv"]
        prefer_sources = [
            str(s).strip().lower() for s in prefer_sources if str(s).strip()
        ]

        max_documents = int((job.config or {}).get("max_documents") or 8)
        max_papers = int((job.config or {}).get("max_papers") or 8)
        max_documents = max(0, min(max_documents, 50))
        max_papers = max(0, min(max_papers, 50))

        monitor_queries = (job.config or {}).get("monitor_queries")
        if not isinstance(monitor_queries, list):
            monitor_queries = []
        monitor_queries = [
            str(q).strip()
            for q in monitor_queries
            if isinstance(q, (str, int, float)) and str(q).strip()
        ]

        use_feedback_bias = bool((job.config or {}).get("use_feedback_bias", True))
        positive_tokens: list[str] = []
        negative_tokens: set[str] = set()
        muted_patterns: list[str] = []
        bias_debug: dict = {}
        if use_feedback_bias:
            (
                positive_tokens,
                negative_tokens,
                muted_patterns,
                bias_debug,
            ) = await _load_feedback_bias_tokens(customer=customer_tag)

        def _is_muted(text: str) -> bool:
            if not muted_patterns:
                return False
            t = (text or "").lower()
            for p in muted_patterns:
                pp = (p or "").strip().lower()
                if not pp:
                    continue
                if pp in t:
                    return True
            return False

        # Filter manual queries too
        if monitor_queries and muted_patterns:
            monitor_queries = [q for q in monitor_queries if not _is_muted(q)]

        if not monitor_queries:
            goal = _safe_text(job.goal).strip()
            kws: list[str] = []
            if customer_profile and isinstance(customer_profile.keywords, list):
                kws = [
                    str(x).strip() for x in customer_profile.keywords if str(x).strip()
                ]

            seed = " ".join([goal, customer_context, " ".join(kws[:12])]).strip()
            if positive_tokens:
                seed = (seed + " " + " ".join(positive_tokens)).strip()
            toks = [t for t in _tokens(seed) if t not in negative_tokens]

            derived: list[str] = []
            if goal:
                derived.append(goal[:200])
            if customer_tag:
                derived.append(
                    f"{customer_tag} {goal[:160]}".strip()[:200]
                    if goal
                    else customer_tag[:200]
                )
            if toks:
                derived.append(" ".join(toks[:10])[:200])

            seen: set[str] = set()
            deduped: list[str] = []
            for q in derived:
                q = (q or "").strip()
                if not q or q in seen:
                    continue
                if _is_muted(q):
                    continue
                seen.add(q)
                deduped.append(q)
            monitor_queries = deduped[:5]

        job.iteration = int(job.iteration or 0) + 1
        _emit(
            5,
            "planning",
            f"Monitoring {len(monitor_queries)} queries (sources: {', '.join(prefer_sources) or 'none'})",
        )
        await db.commit()

        created = 0
        skipped = 0
        created_doc_ids: list[str] = []

        if "documents" in prefer_sources and max_documents > 0:
            for idx, q in enumerate(monitor_queries):
                _emit(10 + idx * 10, "searching_documents", f"Searching KB: {q[:120]}")
                await db.commit()
                try:
                    docs, _total, _took = await executor.search_service.search(
                        query=q,
                        mode="smart",
                        page=1,
                        page_size=max_documents,
                        db=db,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Research inbox KB search failed for job {job.id}: {exc}"
                    )
                    continue

                for d in docs or []:
                    if not isinstance(d, dict):
                        continue
                    doc_id = _safe_text(d.get("id")).strip()
                    if not doc_id:
                        continue
                    title_text = _safe_text(d.get("title")).strip()
                    summary_text = _safe_text(d.get("snippet")).strip()
                    discovery_score, discovery_reasons = _score_discovery_candidate(
                        item_type="document",
                        title=title_text,
                        summary=summary_text,
                        bias=bias_debug,
                    )
                    if (
                        _is_muted(f"{title_text} {summary_text}")
                        or discovery_score <= -6
                    ):
                        skipped += 1
                        continue
                    ok = await _create_inbox_item(
                        item_type="document",
                        item_key=doc_id,
                        title=title_text or doc_id,
                        summary=summary_text or None,
                        url=_safe_text(d.get("url")).strip() or None,
                        published_at=None,
                        customer=customer_tag,
                        metadata={
                            "query": q,
                            "document_id": doc_id,
                            "source": d.get("source"),
                            "source_type": d.get("source_type"),
                            "relevance_score": d.get("relevance_score"),
                            "discovery_score": discovery_score,
                            "discovery_reasons": discovery_reasons,
                            "score_explained": bool(discovery_reasons),
                            "bias": bias_debug or None,
                        },
                    )
                    if ok:
                        created += 1
                        created_doc_ids.append(doc_id)
                    else:
                        skipped += 1

        if "arxiv" in prefer_sources and max_papers > 0:
            for idx, q in enumerate(monitor_queries):
                _emit(60 + idx * 8, "searching_arxiv", f"Searching arXiv: {q[:120]}")
                await db.commit()

                toks = [t for t in _tokens(q) if t not in negative_tokens]
                if positive_tokens:
                    # Add a couple of learned positives into the arXiv query deterministically.
                    for t in positive_tokens[:4]:
                        if t not in toks:
                            toks.append(t)
                if toks:
                    arxiv_q = " AND ".join([f"all:{t}" for t in toks[:6]])
                else:
                    phrase = " ".join(re.findall(r"[a-zA-Z0-9_\\-]+", q))[:120].strip()
                    if not phrase:
                        continue
                    arxiv_q = f'all:"{phrase}"'

                try:
                    res = await executor.arxiv_service.search(
                        query=arxiv_q,
                        start=0,
                        max_results=max_papers,
                        sort_by="submittedDate",
                        sort_order="descending",
                    )
                except Exception as exc:
                    logger.warning(
                        f"Research inbox arXiv search failed for job {job.id}: {exc}"
                    )
                    continue

                for it in res.items or []:
                    if not isinstance(it, dict):
                        continue
                    arxiv_id = _safe_text(it.get("id")).strip()
                    if not arxiv_id:
                        continue
                    title_text = _safe_text(it.get("title")).strip()
                    summary_text = _safe_text(it.get("summary")).strip()
                    discovery_score, discovery_reasons = _score_discovery_candidate(
                        item_type="arxiv",
                        title=title_text,
                        summary=summary_text,
                        bias=bias_debug,
                    )
                    if (
                        _is_muted(f"{title_text} {summary_text}")
                        or discovery_score <= -6
                    ):
                        skipped += 1
                        continue
                    ok = await _create_inbox_item(
                        item_type="arxiv",
                        item_key=arxiv_id,
                        title=title_text or arxiv_id,
                        summary=summary_text or None,
                        url=_safe_text(it.get("pdf_url") or it.get("entry_url")).strip()
                        or None,
                        published_at=_parse_iso_dt(_safe_text(it.get("published")))
                        or None,
                        customer=customer_tag,
                        metadata={
                            "query": q,
                            "arxiv_query": arxiv_q,
                            "arxiv_id": arxiv_id,
                            "entry_url": it.get("entry_url"),
                            "pdf_url": it.get("pdf_url"),
                            "authors": it.get("authors"),
                            "categories": it.get("categories"),
                            "primary_category": it.get("primary_category"),
                            "updated": it.get("updated"),
                            "doi": it.get("doi"),
                            "comments": it.get("comments"),
                            "discovery_score": discovery_score,
                            "discovery_reasons": discovery_reasons,
                            "score_explained": bool(discovery_reasons),
                            "bias": bias_debug or None,
                        },
                    )
                    if ok:
                        created += 1
                    else:
                        skipped += 1

        await db.commit()

        auto_add = bool((job.config or {}).get("auto_add_to_reading_list", False))
        reading_list_name = _safe_text(
            (job.config or {}).get("reading_list_name")
        ).strip()
        if auto_add and reading_list_name and created_doc_ids:
            try:
                from app.models.document import Document
                from app.models.reading_list import ReadingList, ReadingListItem

                rl_result = await db.execute(
                    select(ReadingList).where(
                        ReadingList.user_id == job.user_id,
                        ReadingList.name == reading_list_name,
                    )
                )
                rl = rl_result.scalar_one_or_none()
                if not rl:
                    rl = ReadingList(
                        user_id=job.user_id,
                        name=reading_list_name,
                        description="Auto-populated by Research Inbox monitor",
                    )
                    db.add(rl)
                    await db.commit()
                    await db.refresh(rl)

                max_pos_res = await db.execute(
                    select(func.max(ReadingListItem.position)).where(
                        ReadingListItem.reading_list_id == rl.id
                    )
                )
                max_pos = int(max_pos_res.scalar() or 0)
                added = 0

                for doc_id in created_doc_ids[:200]:
                    try:
                        doc_uuid = UUID(str(doc_id))
                    except Exception:
                        continue
                    doc = await db.get(Document, doc_uuid)
                    if not doc:
                        continue

                    exists = await db.execute(
                        select(func.count())
                        .select_from(ReadingListItem)
                        .where(
                            ReadingListItem.reading_list_id == rl.id,
                            ReadingListItem.document_id == doc.id,
                        )
                    )
                    if int(exists.scalar() or 0) > 0:
                        continue

                    item = ReadingListItem(
                        reading_list_id=rl.id,
                        document_id=doc.id,
                        status="to-read",
                        priority=0,
                        position=max_pos + 1,
                        notes="Added automatically by Research Inbox monitor",
                    )
                    db.add(item)
                    try:
                        await db.flush()
                    except IntegrityError:
                        await db.rollback()
                        continue

                    max_pos += 1
                    added += 1

                await db.commit()
                if job.results is None:
                    job.results = {}
                job.results["reading_list"] = {
                    "name": reading_list_name,
                    "items_added": added,
                }
            except Exception as exc:
                logger.warning(
                    f"Failed to auto-populate reading list for inbox monitor: {exc}"
                )

        persist = bool((job.config or {}).get("persist_artifacts", False))
        if persist and created > 0:
            try:
                from app.models.document import Document

                notes_source = (
                    await executor.document_service._get_or_create_agent_notes_source(
                        db
                    )
                )
                now = datetime.utcnow()
                iso_year, iso_week, _ = now.isocalendar()
                customer_slug = (customer_tag or "default").lower()
                customer_slug = (
                    re.sub(r"[^a-z0-9_\\-]+", "-", customer_slug).strip("-")[:64]
                    or "default"
                )
                source_identifier = (
                    f"research_inbox_weekly:{customer_slug}:{iso_year}-W{iso_week:02d}"
                )

                existing = await db.execute(
                    select(Document)
                    .where(
                        Document.source_id == notes_source.id,
                        Document.source_identifier == source_identifier,
                    )
                    .limit(1)
                )
                doc = existing.scalar_one_or_none()

                header = (
                    f"# Research Inbox Weekly Brief — {customer_tag}"
                    if customer_tag
                    else "# Research Inbox Weekly Brief"
                )
                section_lines: list[str] = [
                    header,
                    "",
                    "## Run",
                    f"- Timestamp: {now.isoformat()}Z",
                    f"- Queries: {', '.join(monitor_queries)[:800]}",
                    f"- New items created: {created}",
                    f"- Duplicates skipped: {skipped}",
                ]
                if customer_context:
                    section_lines.append(
                        f"- Customer context: {customer_context[:500]}"
                    )
                section_lines.append("")

                content = "\n".join(section_lines).strip() + "\n"

                if doc:
                    doc.content = (doc.content or "").rstrip() + "\n\n" + content
                    doc.updated_at = datetime.utcnow()
                else:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    doc = Document(
                        title=header.lstrip("# ").strip()[:240]
                        or "Research Inbox Weekly Brief",
                        content=content,
                        content_hash=content_hash,
                        url=None,
                        file_path=None,
                        file_type="text/markdown",
                        file_size=len(content.encode("utf-8")),
                        source_id=notes_source.id,
                        source_identifier=source_identifier,
                        author=None,
                        tags=["autonomous_job", "research_inbox", "monitor"],
                        extra_metadata={
                            "origin": "autonomous_job",
                            "job_id": str(job.id),
                            "job_type": job.job_type,
                            "customer": customer_tag,
                        },
                        is_processed=False,
                    )
                    db.add(doc)

                await db.commit()
                await db.refresh(doc)

                try:
                    await executor.document_service.reprocess_document(
                        doc.id, db, user_id=job.user_id
                    )
                except Exception:
                    pass

                if job.results is None:
                    job.results = {}
                job.results["weekly_brief_document"] = {
                    "id": str(doc.id),
                    "title": doc.title,
                    "source_identifier": source_identifier,
                }
            except Exception as exc:
                logger.warning(
                    f"Failed to persist weekly brief for inbox monitor: {exc}"
                )

        job.results = job.results or {}
        job.results.update(
            {
                "summary": f"Research inbox monitor: {created} new items ({skipped} duplicates) across {len(monitor_queries)} queries.",
                "monitor": {
                    "queries": monitor_queries,
                    "prefer_sources": prefer_sources,
                    "items_created": created,
                    "items_skipped": skipped,
                },
                "customer_profile": customer_profile.model_dump()
                if customer_profile
                else None,
                "customer_context": customer_context,
            }
        )

        _emit(100, "completed", job.results.get("summary", "Completed"))
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def run_research_engineer_scientist(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: "AI Scientist" planning step for ResearchEngineer chains.

        Produces:
          - job.results.research_engineer_plan (hypothesis + experiments + metrics + risks)
          - optional: appends a LaTeX section into a LatexProject if job.config.latex_project_id is set

        Expects (optional):
          - job.config.search_query (string): KB query to ground on (defaults to job.goal)
          - job.config.max_documents (int): number of KB docs to include (default 8)
          - job.config.latex_project_id (UUID): LaTeX Studio project to update
        """
        import json
        from uuid import UUID as _UUID

        from app.models.document import Document
        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "research_engineer_scientist",
                    "result": details,
                }
            )

        def _bib_key_from_uuid(doc_id: _UUID) -> str:
            return f"KDB:{str(doc_id)}"

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = source or ""
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        config = job.config if isinstance(job.config, dict) else {}
        search_query = (
            str((config or {}).get("search_query") or "").strip()
            or str(job.goal or "").strip()
        )
        max_docs = int((config or {}).get("max_documents") or 8)
        max_docs = max(1, min(max_docs, 20))
        latex_project_id = (config or {}).get("latex_project_id")

        _emit(10, "planning", "Selecting Knowledge DB sources")
        await db.commit()

        docs: list[Document] = []
        try:
            results, _total, _took = await executor.search_service.search(
                query=search_query,
                mode="smart",
                page=1,
                page_size=max_docs,
                db=db,
            )
            ids = [
                r.get("id")
                for r in (results or [])
                if isinstance(r, dict) and r.get("id")
            ]
            for doc_id in ids[:max_docs]:
                try:
                    d = await db.get(Document, _UUID(str(doc_id)))
                except Exception:
                    d = None
                if d:
                    docs.append(d)
        except Exception:
            docs = []

        cite_map = []
        for d in docs:
            try:
                cite_map.append(
                    {
                        "doc_id": str(d.id),
                        "cite_key": _bib_key_from_uuid(d.id),
                        "title": (d.title or "").strip()[:200],
                        "url": (d.url or "").strip(),
                        "snippet": (
                            (d.summary or d.content or "")[:600]
                            if (d.summary or d.content)
                            else ""
                        ),
                    }
                )
            except Exception:
                continue

        _emit(
            35, "drafting", f"Drafting a research plan from {len(cite_map)} KB sources"
        )
        await db.commit()

        user_settings = await executor._load_user_settings(job.user_id, db)
        prompt = (
            "You are an AI Scientist working with an engineering teammate.\n"
            "Goal: produce a minimal, testable experiment plan and a short LaTeX section to insert into a paper.\n\n"
            "Output MUST be valid JSON only.\n"
            "JSON keys:\n"
            "- hypothesis (string)\n"
            "- plan (string)\n"
            "- experiments (array of {name, procedure, expected_outcome})\n"
            "- metrics (array of strings)\n"
            "- risks (array of strings)\n"
            "- code_change_goal (string)\n"
            "- code_search_query (string)\n"
            "- latex_section_tex (string)  # LaTeX snippet without preamble; may include \\cite{...}\n"
            "- cited_document_ids (array of doc_id strings)\n\n"
            f"USER GOAL:\n{(job.goal or '').strip()}\n\n"
            f"KB CONTEXT (use cite_key for citations):\n{json.dumps(cite_map, ensure_ascii=False)}\n"
        )
        response = await executor.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=1600,
            user_settings=user_settings,
            task_type="research_engineer_scientist",
            user_id=job.user_id,
            db=db,
            routing=executor._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Scientist step did not return valid JSON"
            await db.commit()
            return {"status": "failed", "error": job.error}

        cited_ids = (
            payload.get("cited_document_ids")
            if isinstance(payload.get("cited_document_ids"), list)
            else []
        )
        cited_ids = [str(x) for x in cited_ids if str(x).strip()]

        latex_section = str(payload.get("latex_section_tex") or "").strip()
        if not latex_section:
            latex_section = (
                "\\section{Hypothesis and Experiment Plan}\n"
                + (str(payload.get("hypothesis") or "").strip() + "\n\n")
                + (str(payload.get("plan") or "").strip() + "\n")
            ).strip()

        latex_updated = False
        latex_project_uuid = None
        if latex_project_id:
            try:
                latex_project_uuid = _UUID(str(latex_project_id))
            except Exception:
                latex_project_uuid = None
        if latex_project_uuid:
            project = await db.get(LatexProject, latex_project_uuid)
            if project and project.user_id == job.user_id:
                project.tex_source = _insert_before_end_document(
                    project.tex_source or "", latex_section
                )
                await db.commit()
                latex_updated = True

        job.results = job.results or {}
        job.results["research_engineer_plan"] = {
            "search_query": search_query,
            "hypothesis": str(payload.get("hypothesis") or "").strip(),
            "plan": str(payload.get("plan") or "").strip(),
            "experiments": payload.get("experiments")
            if isinstance(payload.get("experiments"), list)
            else [],
            "metrics": payload.get("metrics")
            if isinstance(payload.get("metrics"), list)
            else [],
            "risks": payload.get("risks")
            if isinstance(payload.get("risks"), list)
            else [],
            "code_change_goal": str(payload.get("code_change_goal") or "").strip(),
            "code_search_query": str(payload.get("code_search_query") or "").strip(),
            "cited_document_ids": cited_ids,
            "latex_project_id": str(latex_project_uuid) if latex_project_uuid else None,
            "latex_updated": latex_updated,
        }

        _emit(100, "completed", "Scientist plan ready")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        return {"status": "completed", "results": job.results}

    async def run_domain_research_orchestrator(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Deterministic runner for KB-first domain research with note persistence."""
        from app.models.document import Document
        from app.models.domain_research_profile import DomainResearchProfile
        from app.models.experiment import ExperimentPlan
        from app.models.research_inbox import ResearchInboxItem

        def _emit(progress: int, phase: str, details: str) -> None:
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "domain_research_orchestrator",
                    "result": details,
                }
            )

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _normalize_key(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
                "_"
            )

        def _signal_clusters_from_ideas(
            signals: list[str], ranked_ideas: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            buckets: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw in list(signals or []) + [
                str(idea.get("title") or "") for idea in ranked_ideas[:6]
            ]:
                text = str(raw or "").strip()
                if not text:
                    continue
                key = _normalize_key(text)[:64] or f"cluster_{len(buckets) + 1}"
                if key in seen:
                    continue
                seen.add(key)
                buckets.append(
                    {
                        "id": key,
                        "label": text[:180],
                        "source_count": 1,
                    }
                )
                if len(buckets) >= 8:
                    break
            return buckets

        def _track_keyword_sets(track: str) -> tuple[set[str], str]:
            if track == "compiler":
                return (
                    {
                        "llvm",
                        "mlir",
                        "pass",
                        "passes",
                        "ir",
                        "vectorization",
                        "vectorizer",
                        "codegen",
                        "scheduling",
                        "fusion",
                        "tiling",
                        "register",
                        "allocation",
                        "pipeline",
                        "kernel",
                    },
                    "Prioritize IR, passes, vectorization, codegen, kernels, compiler regressions, and optimization pipelines.",
                )
            if track == "microarchitecture":
                return (
                    {
                        "cache",
                        "ipc",
                        "branch",
                        "predictor",
                        "latency",
                        "bandwidth",
                        "simd",
                        "avx",
                        "sve",
                        "stall",
                        "pipeline",
                        "frontend",
                        "backend",
                        "throughput",
                        "memory",
                    },
                    "Prioritize cache behavior, branch behavior, SIMD/ISA usage, stalls, bandwidth, and pipeline efficiency.",
                )
            return (
                {
                    "benchmark",
                    "performance",
                    "compiler",
                    "kernel",
                    "cache",
                    "vectorization",
                    "latency",
                    "throughput",
                },
                "Optimize for novel, evidence-backed, testable ideas across the available technical evidence.",
            )

        def _track_fit_score(track: str, fields: list[str]) -> float:
            keywords, _track_prompt = _track_keyword_sets(track)
            text = " ".join(
                str(value or "").strip().lower()
                for value in fields
                if str(value or "").strip()
            )
            if not text:
                return 0.5 if track == "generic" else 0.35
            hits = sum(1 for keyword in keywords if keyword in text)
            base = 0.45 if track == "generic" else 0.35
            return round(min(1.0, base + (0.08 * hits)), 4)

        config = job.config if isinstance(job.config, dict) else {}
        profile_id_raw = str(config.get("profile_id") or "").strip()
        profile: Optional[DomainResearchProfile] = None
        previous_summary: dict[str, Any] = {}
        if profile_id_raw:
            try:
                profile = await db.get(DomainResearchProfile, UUID(profile_id_raw))
            except Exception:
                profile = None
            if profile is not None and profile.user_id != job.user_id:
                profile = None
            previous_summary = (
                profile.latest_summary
                if profile and isinstance(profile.latest_summary, dict)
                else {}
            )

        domain = str(config.get("domain") or "").strip() or "domain"
        objective = (
            str(config.get("objective") or "").strip() or str(job.goal or "").strip()
        )
        customer_context = str(config.get("customer_context") or "").strip()
        source_scope = (
            str(config.get("source_scope") or "kb_plus_arxiv").strip().lower()
        )
        track_type = (
            str(config.get("track_type") or "generic").strip().lower() or "generic"
        )
        research_mode = (
            str(config.get("research_mode") or "literature_to_hypothesis")
            .strip()
            .lower()
            or "literature_to_hypothesis"
        )
        max_docs = max(1, min(int(config.get("max_documents") or 10), 25))
        max_papers = max(0, min(int(config.get("max_papers") or 8), 25))
        report_format = (
            str(config.get("report_format") or "brief_and_report").strip().lower()
        )
        persist_artifacts = bool(config.get("persist_artifacts", True))
        (
            automation_profile,
            effective_policy,
        ) = resolve_domain_profile_automation_contract(
            automation_profile=config.get("automation_profile")
            or (profile.automation_profile if profile else None),
            automation_policy=(
                config.get("automation_policy")
                if isinstance(config.get("automation_policy"), dict)
                else {}
            ),
            current_snapshot=current_domain_profile_policy_snapshot(profile)
            if profile is not None
            else {
                "auto_launch_follow_up": True,
                "auto_create_experiment_plans": True,
                "confidence_threshold": 0.7,
            },
            explicit_updates=config,
        )
        auto_launch_follow_up = bool(
            effective_policy.get("auto_launch_follow_up", True)
        )
        auto_create_experiment_plans = bool(
            effective_policy.get("auto_create_experiment_plans", True)
        )
        auto_execute_validation_runs = bool(
            effective_policy.get("auto_execute_validation_runs", False)
        )
        confidence_threshold = max(
            0.0,
            min(_safe_float(effective_policy.get("confidence_threshold"), 0.7), 1.0),
        )
        experiment_readiness_threshold = max(
            0.0,
            min(
                _safe_float(
                    effective_policy.get("experiment_readiness_threshold"), 0.8
                ),
                1.0,
            ),
        )
        max_auto_follow_up_launches = max(
            0, min(int(effective_policy.get("max_auto_follow_up_launches") or 2), 10)
        )
        follow_up_review_mode = (
            str(effective_policy.get("follow_up_review_mode") or "auto_launch_safe")
            .strip()
            .lower()
        )
        if follow_up_review_mode not in {
            "auto_launch_safe",
            "queue_for_approval",
            "manual_only",
        }:
            follow_up_review_mode = "auto_launch_safe"
        sandbox_profile_id = (
            str(
                config.get("sandbox_profile_id")
                or (profile.sandbox_profile_id if profile else "")
                or ""
            ).strip()
            or None
        )
        raw_scoring_policy = (
            config.get("scoring_policy")
            if isinstance(config.get("scoring_policy"), dict)
            else {}
        )
        raw_selection_policy = (
            config.get("selection_policy")
            if isinstance(config.get("selection_policy"), dict)
            else {}
        )
        scoring_policy = {
            "weights": {
                "novelty": max(
                    0.0,
                    _safe_float(
                        (
                            (raw_scoring_policy.get("weights") or {})
                            if isinstance(raw_scoring_policy.get("weights"), dict)
                            else {}
                        ).get("novelty"),
                        0.4,
                    ),
                ),
                "evidence": max(
                    0.0,
                    _safe_float(
                        (
                            (raw_scoring_policy.get("weights") or {})
                            if isinstance(raw_scoring_policy.get("weights"), dict)
                            else {}
                        ).get("evidence"),
                        0.35,
                    ),
                ),
                "testability": max(
                    0.0,
                    _safe_float(
                        (
                            (raw_scoring_policy.get("weights") or {})
                            if isinstance(raw_scoring_policy.get("weights"), dict)
                            else {}
                        ).get("testability"),
                        0.25,
                    ),
                ),
            },
            "minimum_subscore": max(
                0.0,
                min(_safe_float(raw_scoring_policy.get("minimum_subscore"), 0.6), 1.0),
            ),
            "minimum_supporting_sources": max(
                1,
                min(int(raw_scoring_policy.get("minimum_supporting_sources") or 2), 6),
            ),
        }
        weight_total = sum(scoring_policy["weights"].values()) or 1.0
        scoring_policy["weights"] = {
            key: round(value / weight_total, 4)
            for key, value in scoring_policy["weights"].items()
        }
        selection_policy = {
            "max_candidates": max(
                1, min(int(raw_selection_policy.get("max_candidates") or 10), 20)
            ),
            "max_hypotheses": max(
                1, min(int(raw_selection_policy.get("max_hypotheses") or 3), 10)
            ),
        }
        monitor_queries = [
            str(q).strip()
            for q in (
                config.get("monitor_queries")
                if isinstance(config.get("monitor_queries"), list)
                else []
            )
            if str(q).strip()
        ][:12]
        repo_source_ids = [
            str(source_id).strip()
            for source_id in (
                config.get("repo_source_ids")
                if isinstance(config.get("repo_source_ids"), list)
                else []
            )
            if str(source_id).strip()
        ][:24]
        benchmark_queries = [
            str(query).strip()
            for query in (
                config.get("benchmark_queries")
                if isinstance(config.get("benchmark_queries"), list)
                else []
            )
            if str(query).strip()
        ][:16]
        if profile is None:
            try:
                profile_stmt = select(DomainResearchProfile).where(
                    DomainResearchProfile.user_id == job.user_id
                )
                if domain:
                    profile_stmt = profile_stmt.where(
                        DomainResearchProfile.domain == domain
                    )
                if objective:
                    profile_stmt = profile_stmt.where(
                        DomainResearchProfile.objective == objective
                    )
                if track_type:
                    profile_stmt = profile_stmt.where(
                        DomainResearchProfile.track_type == track_type
                    )
                if source_scope:
                    profile_stmt = profile_stmt.where(
                        DomainResearchProfile.source_scope == source_scope
                    )
                profile_stmt = profile_stmt.order_by(
                    desc(DomainResearchProfile.updated_at),
                    desc(DomainResearchProfile.created_at),
                ).limit(1)
                profile_result = await db.execute(profile_stmt)
                profile = profile_result.scalar_one_or_none()
            except Exception:
                profile = None
            previous_summary = (
                profile.latest_summary
                if profile and isinstance(profile.latest_summary, dict)
                else {}
            )
        if not sandbox_profile_id and profile is not None:
            sandbox_profile_id = str(profile.sandbox_profile_id or "").strip() or None
        search_query = (
            str(config.get("search_query") or "").strip()
            or f"{domain} {objective}".strip()
        )
        _track_keywords, track_prompt = _track_keyword_sets(track_type)

        _emit(10, "gathering", "Collecting Knowledge DB evidence")
        await db.commit()

        docs: list[dict[str, Any]] = []
        if source_scope in {"kb_only", "kb_plus_arxiv", "kb_plus_arxiv_plus_repo"}:
            try:
                search_results, _total, _took = await executor.search_service.search(
                    query=search_query,
                    mode="smart",
                    page=1,
                    page_size=max_docs,
                    db=db,
                )
                for row in (search_results or [])[:max_docs]:
                    if not isinstance(row, dict) or not row.get("id"):
                        continue
                    doc = None
                    try:
                        doc = await db.get(Document, UUID(str(row.get("id"))))
                    except Exception:
                        doc = None
                    title = str(
                        (row.get("title") or (doc.title if doc else "") or "")
                    ).strip()
                    snippet = str(
                        row.get("summary")
                        or row.get("snippet")
                        or (
                            doc.summary if doc and getattr(doc, "summary", None) else ""
                        )
                        or (
                            doc.content[:800]
                            if doc and getattr(doc, "content", None)
                            else ""
                        )
                        or ""
                    ).strip()
                    if not title and not snippet:
                        continue
                    docs.append(
                        {
                            "id": str(row.get("id")),
                            "title": title[:240] or "Knowledge DB document",
                            "summary": snippet[:1000],
                            "url": str(
                                (row.get("url") or (doc.url if doc else "") or "")
                            ).strip()
                            or None,
                            "source_type": "document",
                        }
                    )
            except Exception as exc:
                logger.warning(
                    f"Domain research KB search failed for job {job.id}: {exc}"
                )

        repo_documents: list[dict[str, Any]] = []
        if source_scope == "kb_plus_arxiv_plus_repo" and repo_source_ids:
            _emit(
                24,
                "gathering",
                f"Collecting repository evidence from {len(repo_source_ids)} repo sources",
            )
            await db.commit()
            per_query_limit = max(
                2, min(6, math.ceil(max_docs / max(1, len(repo_source_ids))))
            )
            repo_queries = (
                benchmark_queries[:8] or monitor_queries[:4] or [search_query[:240]]
            )
            seen_repo_doc_ids: set[str] = set()
            for source_id in repo_source_ids:
                for repo_query in repo_queries:
                    try:
                        (
                            search_results,
                            _total,
                            _took,
                        ) = await executor.search_service.search(
                            query=repo_query,
                            mode="exact",
                            page=1,
                            page_size=per_query_limit,
                            source_id=source_id,
                            db=db,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Domain research repo search failed for job {job.id} source {source_id}: {exc}"
                        )
                        continue
                    for row in search_results or []:
                        doc_id = str(row.get("id") or "").strip()
                        if not doc_id or doc_id in seen_repo_doc_ids:
                            continue
                        doc = None
                        try:
                            doc = await db.get(Document, UUID(doc_id))
                        except Exception:
                            doc = None
                        title = str(
                            (row.get("title") or (doc.title if doc else "") or "")
                        ).strip()
                        snippet = str(
                            row.get("snippet")
                            or row.get("summary")
                            or (
                                doc.summary
                                if doc and getattr(doc, "summary", None)
                                else ""
                            )
                            or (
                                doc.content[:1000]
                                if doc and getattr(doc, "content", None)
                                else ""
                            )
                            or ""
                        ).strip()
                        if not title and not snippet:
                            continue
                        seen_repo_doc_ids.add(doc_id)
                        repo_documents.append(
                            {
                                "id": doc_id,
                                "title": title[:240] or "Repository evidence",
                                "summary": snippet[:1000],
                                "url": str(
                                    (row.get("url") or (doc.url if doc else "") or "")
                                ).strip()
                                or None,
                                "source_type": "repo_document",
                                "source_id": source_id,
                                "source_name": str(row.get("source") or "").strip()
                                or None,
                                "file_path": str(
                                    (doc.file_path if doc else "") or ""
                                ).strip()
                                or None,
                                "benchmark_query": repo_query[:240],
                            }
                        )
                        if len(repo_documents) >= max_docs:
                            break
                    if len(repo_documents) >= max_docs:
                        break
                if len(repo_documents) >= max_docs:
                    break

        _emit(35, "gathering", "Collecting arXiv evidence")
        await db.commit()

        papers: list[dict[str, Any]] = []
        if (
            source_scope in {"arxiv_only", "kb_plus_arxiv", "kb_plus_arxiv_plus_repo"}
            and max_papers > 0
        ):
            try:
                arxiv_result = await executor.arxiv_search_service.search(
                    query=search_query,
                    max_results=max_papers,
                    sort_by="relevance",
                    sort_order="descending",
                )
                for item in (arxiv_result.items or [])[:max_papers]:
                    if not isinstance(item, dict):
                        continue
                    papers.append(
                        {
                            "arxiv_id": str(item.get("id") or "").strip(),
                            "title": str(item.get("title") or "").strip()[:240],
                            "summary": str(item.get("summary") or "").strip()[:1000],
                            "published": item.get("published"),
                            "categories": item.get("categories")
                            if isinstance(item.get("categories"), list)
                            else [],
                        }
                    )
            except Exception as exc:
                logger.warning(
                    f"Domain research arXiv search failed for job {job.id}: {exc}"
                )

        _emit(
            55,
            "synthesizing",
            f"Ranking ideas from {len(docs)} KB docs and {len(papers)} papers",
        )
        await db.commit()

        user_settings = await executor._load_user_settings(job.user_id, db)
        kb_context = [
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "summary": d.get("summary"),
            }
            for d in docs[:10]
        ]
        paper_context = [
            {
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "summary": p.get("summary"),
                "published": p.get("published"),
            }
            for p in papers[:10]
        ]
        repo_context = [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "file_path": item.get("file_path"),
                "benchmark_query": item.get("benchmark_query"),
            }
            for item in repo_documents[:10]
        ]
        prompt = (
            "You are a scientific research strategist for advanced systems teams.\n"
            "Given a domain and evidence, produce testable hypotheses backed by exact source references.\n"
            "Prefer non-obvious but practical ideas. Avoid repeating stale ideas unless new evidence materially changes the case.\n"
            "Return valid JSON only with keys:\n"
            "- domain_summary (string)\n"
            "- discovered_signals (array of strings)\n"
            "- proposed_ideas (array of {title, hypothesis, opportunity, supporting_evidence, confidence, next_steps, counterarguments})\n"
            "- ranked_opportunities (array of strings)\n"
            "- open_questions (array of strings)\n"
            "- brief_markdown (string)\n"
            "- report_markdown (string)\n\n"
            f"DOMAIN: {domain}\n"
            f"OBJECTIVE: {objective}\n"
            f"TRACK_TYPE: {track_type}\n"
            f"TRACK_GUIDANCE: {track_prompt}\n"
            f"RESEARCH_MODE: {research_mode}\n"
            f"CUSTOMER_CONTEXT: {customer_context}\n"
            f"MONITOR_QUERIES: {json.dumps(monitor_queries, ensure_ascii=False)}\n"
            f"BENCHMARK_QUERIES: {json.dumps(benchmark_queries, ensure_ascii=False)}\n"
            f"KB_EVIDENCE: {json.dumps(kb_context, ensure_ascii=False)}\n"
            f"REPO_EVIDENCE: {json.dumps(repo_context, ensure_ascii=False)}\n"
            f"ARXIV_EVIDENCE: {json.dumps(paper_context, ensure_ascii=False)}\n"
        )
        response = await executor.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=2200,
            user_settings=user_settings,
            task_type="domain_research_orchestrator",
            user_id=job.user_id,
            db=db,
            routing=executor._llm_routing_from_job_config(job.config),
        )
        payload = _extract_json(response) or {}
        if not isinstance(payload, dict):
            payload = {}

        previous_idea_titles = {
            _normalize_key(raw)
            for raw in (
                previous_summary.get("ranked_opportunities")
                if isinstance(previous_summary.get("ranked_opportunities"), list)
                else []
            )
            if str(raw or "").strip()
        }
        if isinstance(previous_summary.get("hypotheses"), list):
            for row in previous_summary.get("hypotheses") or []:
                if not isinstance(row, dict):
                    continue
                key = _normalize_key(row.get("title"))
                if key:
                    previous_idea_titles.add(key)

        source_rows: list[dict[str, Any]] = []
        for doc in docs:
            source_rows.append(
                {
                    "source_type": "document",
                    "id": str(doc.get("id") or "").strip(),
                    "title": str(doc.get("title") or "").strip(),
                    "summary": str(doc.get("summary") or "").strip(),
                    "url": str(doc.get("url") or "").strip() or None,
                }
            )
        for paper in papers:
            source_rows.append(
                {
                    "source_type": "paper",
                    "id": str(paper.get("arxiv_id") or "").strip(),
                    "title": str(paper.get("title") or "").strip(),
                    "summary": str(paper.get("summary") or "").strip(),
                    "published": paper.get("published"),
                }
            )
        for repo_doc in repo_documents:
            source_rows.append(
                {
                    "source_type": "repo_document",
                    "id": str(repo_doc.get("id") or "").strip(),
                    "title": str(repo_doc.get("title") or "").strip(),
                    "summary": str(repo_doc.get("summary") or "").strip(),
                    "url": repo_doc.get("url"),
                    "file_path": repo_doc.get("file_path"),
                    "source_id": repo_doc.get("source_id"),
                    "source_name": repo_doc.get("source_name"),
                }
            )

        def _match_evidence_sources(
            evidence_list: list[str], title: str, hypothesis: str
        ) -> list[dict[str, Any]]:
            haystacks = [str(title or "").strip(), str(hypothesis or "").strip()]
            haystacks.extend(
                [
                    str(item or "").strip()
                    for item in evidence_list
                    if str(item or "").strip()
                ]
            )
            refs: list[dict[str, Any]] = []
            seen_refs: set[str] = set()
            for source in source_rows:
                source_title = str(source.get("title") or "").strip()
                source_path = str(source.get("file_path") or "").strip()
                source_key = _normalize_key(source_title)
                if not source_key:
                    source_key = _normalize_key(source_path)
                if not source_key:
                    continue
                matched = False
                for text in haystacks:
                    lowered = str(text or "").strip().lower()
                    if not lowered:
                        continue
                    if source_title and (
                        source_title.lower() in lowered
                        or lowered in source_title.lower()
                    ):
                        matched = True
                        break
                    if source_path and (
                        source_path.lower() in lowered or lowered in source_path.lower()
                    ):
                        matched = True
                        break
                    overlap = set(source_key.split("_")) & set(
                        _normalize_key(lowered).split("_")
                    )
                    if len([token for token in overlap if token]) >= 3:
                        matched = True
                        break
                if not matched:
                    continue
                ref_key = (
                    f"{source.get('source_type')}:{source.get('id') or source_title}"
                )
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                refs.append(
                    {
                        "source_type": source.get("source_type"),
                        "id": source.get("id"),
                        "title": source_title,
                        "url": source.get("url"),
                        "published": source.get("published"),
                        "file_path": source.get("file_path"),
                        "source_name": source.get("source_name"),
                    }
                )
                if len(refs) >= 6:
                    break
            if len(refs) < scoring_policy["minimum_supporting_sources"]:
                for source in source_rows:
                    ref_key = f"{source.get('source_type')}:{source.get('id') or source.get('title')}"
                    if ref_key in seen_refs:
                        continue
                    refs.append(
                        {
                            "source_type": source.get("source_type"),
                            "id": source.get("id"),
                            "title": source.get("title"),
                            "url": source.get("url"),
                            "published": source.get("published"),
                        }
                    )
                    seen_refs.add(ref_key)
                    if len(refs) >= scoring_policy["minimum_supporting_sources"]:
                        break
            return refs[:6]

        def _build_candidate(
            item: dict[str, Any], idx: int
        ) -> Optional[dict[str, Any]]:
            if not isinstance(item, dict):
                return None
            title = str(item.get("title") or "").strip()
            hypothesis = str(item.get("hypothesis") or "").strip()
            opportunity = str(item.get("opportunity") or "").strip()
            if not title and not hypothesis and not opportunity:
                return None
            evidence = item.get("supporting_evidence")
            if isinstance(evidence, list):
                evidence_list = [str(x).strip() for x in evidence if str(x).strip()][:6]
            else:
                evidence_list = (
                    [str(evidence).strip()] if str(evidence or "").strip() else []
                )
            next_steps = [
                str(x).strip()
                for x in (
                    item.get("next_steps")
                    if isinstance(item.get("next_steps"), list)
                    else []
                )
                if str(x).strip()
            ][:5]
            counterarguments = [
                str(x).strip()
                for x in (
                    item.get("counterarguments")
                    if isinstance(item.get("counterarguments"), list)
                    else []
                )
                if str(x).strip()
            ][:4]
            normalized_title = (
                title or hypothesis[:180] or f"{domain} hypothesis {idx + 1}"
            )
            matched_sources = _match_evidence_sources(
                evidence_list, normalized_title, hypothesis
            )
            evidence_count = len(matched_sources)
            is_new = _normalize_key(normalized_title) not in previous_idea_titles
            novelty_score = 0.9 if is_new else 0.35
            evidence_score = min(1.0, 0.35 + 0.2 * min(evidence_count, 3))
            testability_score = 0.45
            if next_steps:
                testability_score += 0.1 * min(len(next_steps), 3)
            if hypothesis:
                testability_score += 0.1
            testability_score = min(1.0, testability_score)
            llm_confidence = max(
                0.0, min(_safe_float(item.get("confidence"), 0.55), 1.0)
            )
            track_fit_score = _track_fit_score(
                track_type,
                [
                    normalized_title,
                    hypothesis,
                    opportunity,
                    *evidence_list,
                    *[str(source.get("title") or "") for source in matched_sources],
                    *[str(source.get("file_path") or "") for source in matched_sources],
                ],
            )
            weighted = (
                novelty_score * scoring_policy["weights"]["novelty"]
                + evidence_score * scoring_policy["weights"]["evidence"]
                + testability_score * scoring_policy["weights"]["testability"]
            )
            overall_score = round(
                min(
                    1.0,
                    (weighted * 0.75)
                    + (llm_confidence * 0.15)
                    + (track_fit_score * 0.10),
                ),
                4,
            )
            return {
                "id": f"idea_{idx + 1}",
                "title": normalized_title,
                "hypothesis": hypothesis or opportunity,
                "opportunity": opportunity,
                "supporting_evidence": evidence_list,
                "supporting_sources": matched_sources,
                "counterarguments": counterarguments,
                "confidence": llm_confidence,
                "novelty_score": round(novelty_score, 4),
                "evidence_score": round(evidence_score, 4),
                "testability_score": round(testability_score, 4),
                "track_fit_score": round(track_fit_score, 4),
                "overall_score": overall_score,
                "passes_threshold": (
                    overall_score >= confidence_threshold
                    and novelty_score >= scoring_policy["minimum_subscore"]
                    and evidence_score >= scoring_policy["minimum_subscore"]
                    and testability_score >= scoring_policy["minimum_subscore"]
                    and evidence_count >= scoring_policy["minimum_supporting_sources"]
                ),
                "is_new": is_new,
                "next_steps": next_steps or ["Validate on a bounded benchmark slice"],
            }

        raw_ideas = (
            payload.get("proposed_ideas")
            if isinstance(payload.get("proposed_ideas"), list)
            else []
        )
        ideas: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_ideas[: selection_policy["max_candidates"]]):
            candidate = _build_candidate(item, idx)
            if candidate is not None:
                ideas.append(candidate)

        if not ideas:
            fallback_titles = [str(d.get("title") or "").strip() for d in docs[:2]] + [
                str(p.get("title") or "").strip() for p in papers[:2]
            ]
            fallback_titles = [title for title in fallback_titles if title]
            for idx, title in enumerate(fallback_titles[:3]):
                candidate = _build_candidate(
                    {
                        "title": title,
                        "hypothesis": f"Investigate whether '{title}' creates a practical opportunity in {domain}.",
                        "opportunity": f"Use {title} as a lead for {objective[:180]}",
                        "supporting_evidence": [title],
                        "confidence": 0.5,
                        "next_steps": [
                            "Validate relevance with a deeper literature review"
                        ],
                    },
                    idx,
                )
                if candidate is not None:
                    ideas.append(candidate)

        ideas = sorted(
            ideas,
            key=lambda idea: (
                _safe_float(idea.get("overall_score"), 0.0),
                _safe_float(idea.get("evidence_score"), 0.0),
                _safe_float(idea.get("novelty_score"), 0.0),
            ),
            reverse=True,
        )
        selected_hypotheses = [
            idea for idea in ideas if bool(idea.get("passes_threshold"))
        ][: selection_policy["max_hypotheses"]]
        if not selected_hypotheses:
            selected_hypotheses = ideas[
                : min(len(ideas), selection_policy["max_hypotheses"])
            ]

        discovered_signals = (
            payload.get("discovered_signals")
            if isinstance(payload.get("discovered_signals"), list)
            else []
        )
        discovered_signals = [
            str(x).strip() for x in discovered_signals if str(x).strip()
        ][:12]
        if not discovered_signals:
            discovered_signals = [idea["title"] for idea in selected_hypotheses[:5]]

        ranked_opportunities = (
            payload.get("ranked_opportunities")
            if isinstance(payload.get("ranked_opportunities"), list)
            else []
        )
        ranked_opportunities = [
            str(x).strip() for x in ranked_opportunities if str(x).strip()
        ][:8]
        if not ranked_opportunities:
            ranked_opportunities = [idea["title"] for idea in selected_hypotheses[:5]]

        open_questions = (
            payload.get("open_questions")
            if isinstance(payload.get("open_questions"), list)
            else []
        )
        open_questions = [str(x).strip() for x in open_questions if str(x).strip()][:8]
        current_idea_titles = {
            _normalize_key(idea.get("title"))
            for idea in ideas
            if str(idea.get("title") or "").strip()
        }
        new_idea_titles = sorted(
            str(idea.get("title") or "").strip()
            for idea in selected_hypotheses
            if _normalize_key(idea.get("title"))
            and _normalize_key(idea.get("title")) not in previous_idea_titles
        )[:8]
        prior_doc_ids = {
            str(raw).strip()
            for raw in (
                previous_summary.get("document_ids")
                if isinstance(previous_summary.get("document_ids"), list)
                else []
            )
            if str(raw).strip()
        }
        prior_paper_ids = {
            str(raw).strip()
            for raw in (
                previous_summary.get("paper_ids")
                if isinstance(previous_summary.get("paper_ids"), list)
                else []
            )
            if str(raw).strip()
        }
        prior_repo_document_ids = {
            str(raw).strip()
            for raw in (
                previous_summary.get("repo_document_ids")
                if isinstance(previous_summary.get("repo_document_ids"), list)
                else []
            )
            if str(raw).strip()
        }
        new_document_ids = [
            str(doc.get("id"))
            for doc in docs
            if str(doc.get("id") or "").strip()
            and str(doc.get("id")) not in prior_doc_ids
        ][:12]
        new_paper_ids = [
            str(paper.get("arxiv_id"))
            for paper in papers
            if str(paper.get("arxiv_id") or "").strip()
            and str(paper.get("arxiv_id")) not in prior_paper_ids
        ][:12]
        new_repo_document_ids = [
            str(doc.get("id"))
            for doc in repo_documents
            if str(doc.get("id") or "").strip()
            and str(doc.get("id")) not in prior_repo_document_ids
        ][:12]
        delta_since_last_run = {
            "had_previous_run": bool(previous_summary),
            "new_idea_titles": new_idea_titles,
            "new_document_ids": new_document_ids,
            "new_paper_ids": new_paper_ids,
            "new_repo_document_ids": new_repo_document_ids,
            "new_signal_count": len(new_idea_titles)
            + len(new_document_ids)
            + len(new_paper_ids)
            + len(new_repo_document_ids),
        }
        novelty_summary = {
            "new_idea_count": len(new_idea_titles),
            "repeated_idea_count": max(
                0, len(current_idea_titles) - len(new_idea_titles)
            ),
            "new_evidence_count": len(new_document_ids)
            + len(new_paper_ids)
            + len(new_repo_document_ids),
        }
        signal_clusters = _signal_clusters_from_ideas(
            discovered_signals, selected_hypotheses
        )
        idea_candidates = [
            {
                "idea_id": str(idea.get("id") or ""),
                "title": str(idea.get("title") or ""),
                "hypothesis": str(
                    idea.get("hypothesis")
                    or idea.get("opportunity")
                    or idea.get("title")
                    or ""
                ),
                "confidence": _safe_float(idea.get("confidence"), 0.0),
                "novelty": _safe_float(idea.get("novelty_score"), 0.0),
                "evidence_score": _safe_float(idea.get("evidence_score"), 0.0),
                "testability_score": _safe_float(idea.get("testability_score"), 0.0),
                "track_fit_score": _safe_float(idea.get("track_fit_score"), 0.0),
                "overall_score": _safe_float(idea.get("overall_score"), 0.0),
                "is_new": bool(idea.get("is_new")),
                "passes_threshold": bool(idea.get("passes_threshold")),
                "next_steps": idea.get("next_steps")
                if isinstance(idea.get("next_steps"), list)
                else [],
                "supporting_evidence": idea.get("supporting_evidence")
                if isinstance(idea.get("supporting_evidence"), list)
                else [],
                "supporting_sources": idea.get("supporting_sources")
                if isinstance(idea.get("supporting_sources"), list)
                else [],
            }
            for idea in ideas[:12]
        ]
        prior_profile_opportunities = {
            str(item.get("canonical_key") or ""): item
            for item in list_normalized_research_opportunities(
                previous_summary.get("opportunities")
            )
            if str(item.get("canonical_key") or "").strip()
        }
        opportunities = []
        for idx, item in enumerate(idea_candidates[:12]):
            normalized = normalize_research_opportunity(
                {
                    "opportunity_id": f"opp_{str(item.get('idea_id') or idx + 1)}",
                    "canonical_key": _normalize_key(item.get("title")),
                    "title": item.get("title"),
                    "hypothesis": item.get("hypothesis"),
                    "confidence": item.get("confidence"),
                    "novelty": item.get("novelty"),
                    "readiness": item.get("overall_score"),
                    "supporting_evidence": item.get("supporting_evidence"),
                    "supporting_sources": item.get("supporting_sources"),
                    "next_steps": item.get("next_steps"),
                    "source_job_ids": [str(job.id)],
                    "source_repo_ids": repo_source_ids[:8],
                    "track_type": track_type,
                    "decision_state": "pending_review",
                    "decision_source": "system",
                }
            )
            opportunities.append(
                merge_operator_fields(
                    normalized,
                    prior_profile_opportunities.get(
                        str(normalized.get("canonical_key") or "")
                    ),
                )
            )

        domain_summary = str(payload.get("domain_summary") or "").strip()
        if not domain_summary:
            domain_summary = (
                f"Surveyed {len(docs)} Knowledge DB documents and {len(papers)} arXiv papers for {domain}. "
                f"Surfaced {len(selected_hypotheses)} ranked hypotheses meeting the novelty/evidence/testability bar."
            )

        hypothesis_markdown_lines = []
        for index, idea in enumerate(selected_hypotheses, start=1):
            source_titles = [
                str(source.get("title") or "").strip()
                for source in (
                    idea.get("supporting_sources")
                    if isinstance(idea.get("supporting_sources"), list)
                    else []
                )
                if str(source.get("title") or "").strip()
            ][:3]
            hypothesis_markdown_lines.extend(
                [
                    f"### {index}. {str(idea.get('title') or '').strip()}",
                    f"- Claim: {str(idea.get('hypothesis') or idea.get('opportunity') or '').strip() or 'No claim provided.'}",
                    f"- Scores: overall {float(idea.get('overall_score') or 0.0):.2f} · novelty {float(idea.get('novelty_score') or 0.0):.2f} · evidence {float(idea.get('evidence_score') or 0.0):.2f} · testability {float(idea.get('testability_score') or 0.0):.2f}",
                    f"- Evidence: {', '.join(source_titles) if source_titles else 'Source evidence captured in payload'}",
                    f"- Next step: {str(((idea.get('next_steps') if isinstance(idea.get('next_steps'), list) else []) or ['Validate on a bounded benchmark slice'])[0])}",
                ]
            )
            counterarguments = (
                idea.get("counterarguments")
                if isinstance(idea.get("counterarguments"), list)
                else []
            )
            if counterarguments:
                hypothesis_markdown_lines.append(
                    f"- Counterarguments: {'; '.join([str(x) for x in counterarguments[:2]])}"
                )
            hypothesis_markdown_lines.append("")

        brief_markdown = str(payload.get("brief_markdown") or "").strip()
        if not brief_markdown:
            brief_markdown = (
                f"# Domain Research Brief — {domain}\n\n"
                f"## Objective\n{objective}\n\n"
                f"## Summary\n{domain_summary}\n\n"
                "## Ranked hypotheses\n" + "\n".join(hypothesis_markdown_lines[:20])
            ).strip()

        report_markdown = str(payload.get("report_markdown") or "").strip()
        if not report_markdown:
            report_lines = [
                f"# Domain Research Report — {domain}",
                "",
                "## Objective",
                objective,
                "",
                "## Domain summary",
                domain_summary,
                "",
                "## Discovered signals",
            ]
            report_lines.extend([f"- {signal}" for signal in discovered_signals[:8]])
            report_lines.extend(["", "## Ranked opportunities"])
            report_lines.extend([f"- {item}" for item in ranked_opportunities[:8]])
            report_lines.extend(["", "## Ranked hypotheses"])
            report_lines.extend(
                hypothesis_markdown_lines or ["- No hypotheses surfaced"]
            )
            report_lines.extend(["", "## Open questions"])
            report_lines.extend(
                [f"- {item}" for item in open_questions[:8]] or ["- None captured"]
            )
            report_markdown = "\n".join(report_lines).strip()

        memo_payload = {
            "research_mode": research_mode,
            "summary": domain_summary,
            "hypotheses": [
                {
                    "id": str(idea.get("id") or ""),
                    "rank": idx + 1,
                    "title": str(idea.get("title") or ""),
                    "claim": str(
                        idea.get("hypothesis") or idea.get("opportunity") or ""
                    ),
                    "rationale": str(
                        idea.get("opportunity") or idea.get("hypothesis") or ""
                    ),
                    "supporting_evidence": idea.get("supporting_evidence")
                    if isinstance(idea.get("supporting_evidence"), list)
                    else [],
                    "supporting_sources": idea.get("supporting_sources")
                    if isinstance(idea.get("supporting_sources"), list)
                    else [],
                    "counterarguments": idea.get("counterarguments")
                    if isinstance(idea.get("counterarguments"), list)
                    else [],
                    "novelty_score": _safe_float(idea.get("novelty_score"), 0.0),
                    "evidence_score": _safe_float(idea.get("evidence_score"), 0.0),
                    "testability_score": _safe_float(
                        idea.get("testability_score"), 0.0
                    ),
                    "track_fit_score": _safe_float(idea.get("track_fit_score"), 0.0),
                    "overall_score": _safe_float(idea.get("overall_score"), 0.0),
                    "recommended_next_step": str(
                        (
                            (
                                idea.get("next_steps")
                                if isinstance(idea.get("next_steps"), list)
                                else []
                            )
                            or ["Validate on a bounded benchmark slice"]
                        )[0]
                    ),
                }
                for idx, idea in enumerate(
                    selected_hypotheses[: selection_policy["max_hypotheses"]]
                )
            ],
            "scoring_policy": scoring_policy,
            "selection_policy": selection_policy,
            "validation_policy": {
                "confidence_threshold": confidence_threshold,
                "experiment_readiness_threshold": experiment_readiness_threshold,
                "max_auto_follow_up_launches": max_auto_follow_up_launches,
                "auto_create_experiment_plans": auto_create_experiment_plans,
                "auto_launch_follow_up": auto_launch_follow_up,
            },
            "track_type": track_type,
            "open_questions": open_questions[:8],
            "ranked_opportunities": ranked_opportunities[:8],
            "evidence_snapshot": {
                "documents": docs[:10],
                "repo_documents": repo_documents[:10],
                "papers": papers[:10],
            },
        }

        findings: list[dict[str, Any]] = []
        for doc in docs:
            findings.append(
                {
                    "type": "document",
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "summary": doc.get("summary"),
                }
            )
        for paper in papers:
            findings.append(
                {
                    "type": "paper",
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "published": paper.get("published"),
                    "summary": paper.get("summary"),
                }
            )
        for repo_doc in repo_documents:
            findings.append(
                {
                    "type": "repo_document",
                    "id": repo_doc.get("id"),
                    "title": repo_doc.get("title"),
                    "summary": repo_doc.get("summary"),
                    "file_path": repo_doc.get("file_path"),
                    "benchmark_query": repo_doc.get("benchmark_query"),
                }
            )
        for idea in ideas:
            findings.append(
                {
                    "type": "insight",
                    "title": idea.get("title"),
                    "category": "key_insight",
                    "confidence": idea.get("confidence"),
                    "overall_score": idea.get("overall_score"),
                }
            )

        artifacts: list[dict[str, Any]] = []
        created_note_ids: list[str] = []
        created_notes: list[Any] = []
        anchor_note: Optional[Any] = None
        note_source_document_ids = [
            item.get("id") for item in [*docs, *repo_documents] if item.get("id")
        ]
        if persist_artifacts:
            if report_format in {"brief_only", "brief_and_report"}:
                note = await executor._create_domain_research_note(
                    db=db,
                    job=job,
                    title=f"Domain Research Brief — {domain}",
                    content_markdown=brief_markdown,
                    tags=[
                        "autonomous_job",
                        "research",
                        "domain_research",
                        "research_memo",
                        "brief",
                        domain.lower().replace(" ", "_")[:60],
                    ],
                    source_document_ids=note_source_document_ids,
                    attribution={
                        "origin": "agent_job",
                        "job_id": str(job.id),
                        "launch_mode": "quick_start_domain_research",
                        "artifact_type": "brief",
                        "domain": domain,
                        "objective": objective,
                        "track_type": track_type,
                    },
                    structured_payload={**memo_payload, "artifact_type": "brief"},
                )
                if note is not None:
                    created_notes.append(note)
                    created_note_ids.append(str(note.id))
                    artifacts.append(
                        {
                            "type": "research_note",
                            "id": str(note.id),
                            "title": note.title,
                        }
                    )
                    if anchor_note is None:
                        anchor_note = note
            if report_format in {"report_only", "brief_and_report"}:
                note = await executor._create_domain_research_note(
                    db=db,
                    job=job,
                    title=f"Domain Research Report — {domain}",
                    content_markdown=report_markdown,
                    tags=[
                        "autonomous_job",
                        "research",
                        "domain_research",
                        "research_memo",
                        "report",
                        domain.lower().replace(" ", "_")[:60],
                    ],
                    source_document_ids=note_source_document_ids,
                    attribution={
                        "origin": "agent_job",
                        "job_id": str(job.id),
                        "launch_mode": "quick_start_domain_research",
                        "artifact_type": "report",
                        "domain": domain,
                        "objective": objective,
                        "track_type": track_type,
                    },
                    structured_payload={**memo_payload, "artifact_type": "report"},
                )
                if note is not None:
                    created_notes.append(note)
                    created_note_ids.append(str(note.id))
                    artifacts.append(
                        {
                            "type": "research_note",
                            "id": str(note.id),
                            "title": note.title,
                        }
                    )
                    anchor_note = note

        review_item_id: Optional[str] = None
        if anchor_note is not None:
            top_titles = [
                str(item.get("title") or "").strip()
                for item in selected_hypotheses[:3]
                if str(item.get("title") or "").strip()
            ]
            review_item = ResearchInboxItem(
                user_id=job.user_id,
                job_id=job.id,
                customer=(profile.title if profile else None) or domain[:255],
                item_type="hypothesis_memo",
                item_key=f"note:{anchor_note.id}",
                title=f"Review research memo: {anchor_note.title[:920]}",
                summary=(
                    f"{domain_summary} Top hypotheses: {', '.join(top_titles[:3])}."
                    if top_titles
                    else domain_summary
                )[:4000],
                url=f"/research-notes?note={anchor_note.id}",
                item_metadata={
                    "note_id": str(anchor_note.id),
                    "profile_id": str(profile.id) if profile else None,
                    "research_mode": research_mode,
                    "track_type": track_type,
                    "hypotheses": memo_payload.get("hypotheses"),
                },
            )
            db.add(review_item)
            await db.flush()
            review_item_id = str(review_item.id)
            artifacts.append(
                {
                    "type": "research_inbox_item",
                    "id": review_item_id,
                    "title": review_item.title,
                }
            )

        created_experiment_plan_ids: list[str] = []
        created_experiment_plans_by_key: dict[str, Any] = {}
        duplicate_idea_titles: list[str] = []
        if auto_create_experiment_plans and anchor_note is not None and idea_candidates:
            existing_stmt = (
                select(ExperimentPlan)
                .where(
                    ExperimentPlan.user_id == job.user_id,
                    ExperimentPlan.generator == "domain_research_orchestrator",
                )
                .order_by(ExperimentPlan.created_at.desc())
                .limit(60)
            )
            existing_plans = list((await db.execute(existing_stmt)).scalars().all())
            existing_keys = {
                _normalize_key(
                    (plan.generator_details or {}).get("idea_title") or plan.title
                )
                for plan in existing_plans
                if _normalize_key(
                    (plan.generator_details or {}).get("idea_title") or plan.title
                )
            }
            for candidate in selected_hypotheses[: selection_policy["max_hypotheses"]]:
                candidate_key = _normalize_key(candidate.get("title"))
                if not candidate_key:
                    continue
                if candidate_key in existing_keys:
                    duplicate_idea_titles.append(str(candidate.get("title") or ""))
                    continue
                plan = ExperimentPlan(
                    user_id=job.user_id,
                    research_note_id=anchor_note.id,
                    title=f"Experiment Plan: {str(candidate.get('title') or domain)[:460]}",
                    hypothesis_text=str(
                        candidate.get("hypothesis") or candidate.get("title") or ""
                    ).strip(),
                    plan={
                        "objective": objective,
                        "domain": domain,
                        "idea_title": str(candidate.get("title") or ""),
                        "supporting_evidence": candidate.get("supporting_evidence")
                        if isinstance(candidate.get("supporting_evidence"), list)
                        else [],
                        "supporting_sources": candidate.get("supporting_sources")
                        if isinstance(candidate.get("supporting_sources"), list)
                        else [],
                        "scores": {
                            "novelty": _safe_float(candidate.get("novelty_score"), 0.0),
                            "evidence": _safe_float(
                                candidate.get("evidence_score"), 0.0
                            ),
                            "testability": _safe_float(
                                candidate.get("testability_score"), 0.0
                            ),
                            "track_fit": _safe_float(
                                candidate.get("track_fit_score"), 0.0
                            ),
                            "overall": _safe_float(candidate.get("overall_score"), 0.0),
                        },
                        "next_steps": candidate.get("next_steps")
                        if isinstance(candidate.get("next_steps"), list)
                        else [],
                        "recommended_experiments": [
                            f"Validate {str(candidate.get('title') or 'the idea')} against current compiler or microarch baselines",
                            "Define measurable success criteria, benchmark scope, and instrumentation",
                            "Run a bounded feasibility experiment and record failure modes plus counterexamples",
                        ],
                    },
                    generator="domain_research_orchestrator",
                    generator_details={
                        "origin": "domain_research",
                        "job_id": str(job.id),
                        "profile_id": str(profile.id) if profile else None,
                        "source_research_note_id": str(anchor_note.id),
                        "source_hypothesis_id": str(candidate.get("id") or ""),
                        "generation_reason": "autonomous_research_memo",
                        "idea_id": str(candidate.get("id") or ""),
                        "idea_title": str(candidate.get("title") or ""),
                        "confidence": _safe_float(candidate.get("overall_score"), 0.0),
                        "domain": domain,
                        "track_type": track_type,
                        "research_note_id": str(anchor_note.id),
                        "source_document_ids": [
                            str(doc.get("id"))
                            for doc in [*docs, *repo_documents]
                            if str(doc.get("id") or "").strip()
                        ][:20],
                        "source_arxiv_ids": [
                            str(paper.get("arxiv_id"))
                            for paper in papers
                            if str(paper.get("arxiv_id") or "").strip()
                        ][:12],
                        "source_repo_ids": repo_source_ids[:24],
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
                db.add(plan)
                await db.flush()
                existing_keys.add(candidate_key)
                created_experiment_plan_ids.append(str(plan.id))
                created_experiment_plans_by_key[candidate_key] = plan
                artifacts.append(
                    {"type": "experiment_plan", "id": str(plan.id), "title": plan.title}
                )

        _emit(85, "persisting", "Persisting results and evaluating follow-up")
        await db.commit()

        follow_up_launches: list[dict[str, Any]] = []
        validation_launches: list[dict[str, Any]] = []
        created_validation_run_ids: list[str] = []
        launched_validation_job_ids: list[str] = []
        if auto_execute_validation_runs and anchor_note is not None:
            for candidate in selected_hypotheses[: selection_policy["max_hypotheses"]]:
                candidate_key = _normalize_key(candidate.get("title"))
                if not candidate_key:
                    continue
                plan = created_experiment_plans_by_key.get(candidate_key)
                if plan is None:
                    continue
                confidence = _safe_float(candidate.get("overall_score"), 0.0)
                if confidence < confidence_threshold:
                    continue
                readiness = min(
                    1.0,
                    confidence * 0.7
                    + _safe_float(candidate.get("testability_score"), 0.0) * 0.3,
                )
                if readiness < experiment_readiness_threshold:
                    continue
                decision = await executor._create_scientific_validation_run(
                    db=db,
                    parent_job=job,
                    experiment_plan=plan,
                    track_type=track_type,
                    objective=objective,
                    hypothesis_title=str(candidate.get("title") or ""),
                    hypothesis_text=str(
                        candidate.get("hypothesis") or candidate.get("title") or ""
                    ),
                    validation_policy=effective_policy,
                    sandbox_profile_id=sandbox_profile_id,
                    repo_source_ids=repo_source_ids,
                    benchmark_queries=benchmark_queries,
                    supporting_evidence=(
                        candidate.get("supporting_evidence")
                        if isinstance(candidate.get("supporting_evidence"), list)
                        else []
                    ),
                    supporting_sources=(
                        candidate.get("supporting_sources")
                        if isinstance(candidate.get("supporting_sources"), list)
                        else []
                    ),
                    profile_id=str(profile.id) if profile else None,
                    hypothesis_id=str(candidate.get("id") or "").strip() or None,
                    originating_job_id=str(job.id),
                )
                if decision.get("run_id"):
                    created_validation_run_ids.append(str(decision["run_id"]))
                    artifacts.append(
                        {
                            "type": "experiment_run",
                            "id": str(decision["run_id"]),
                            "title": f"Validation Run: {str(candidate.get('title') or domain)[:80]}",
                        }
                    )
                if decision.get("job_id"):
                    launched_validation_job_ids.append(str(decision["job_id"]))
                    artifacts.append(
                        {
                            "type": "agent_job",
                            "id": str(decision["job_id"]),
                            "title": f"Scientific Validation - {str(candidate.get('title') or domain)[:80]}",
                        }
                    )
                validation_launches.append(
                    {
                        "type": "scientific_validation",
                        "hypothesis_id": str(candidate.get("id") or "").strip() or None,
                        **decision,
                    }
                )

        follow_up_job_id: Optional[str] = None
        top_idea = (
            selected_hypotheses[0]
            if selected_hypotheses
            else (ideas[0] if ideas else {})
        )
        top_idea_key = (
            _normalize_key(top_idea.get("title")) if isinstance(top_idea, dict) else ""
        )
        top_idea_opportunity = next(
            (
                row
                for row in opportunities
                if isinstance(row, dict)
                and _normalize_key(row.get("title")) == top_idea_key
            ),
            None,
        )
        top_confidence = _safe_float(top_idea.get("overall_score"), 0.0)
        top_evidence_revision = (
            compute_research_opportunity_evidence_revision(
                normalize_research_opportunity(top_idea_opportunity)
            )
            if isinstance(top_idea_opportunity, dict)
            else ""
        )
        prior_top_idea = (
            prior_profile_opportunities.get(top_idea_key) if top_idea_key else None
        )
        prior_top_revision = str(
            (prior_top_idea or {}).get("evidence_revision") or ""
        ).strip()
        prior_top_review_status = (
            str((prior_top_idea or {}).get("follow_up_review_status") or "")
            .strip()
            .lower()
        )
        if isinstance(top_idea_opportunity, dict):
            top_idea_opportunity["follow_up_review_evidence_revision"] = (
                top_evidence_revision or None
            )
        should_launch_follow_up = (
            auto_launch_follow_up
            and selected_hypotheses
            and top_confidence >= confidence_threshold
            and max_auto_follow_up_launches > 0
        )
        if (
            isinstance(top_idea_opportunity, dict)
            and top_evidence_revision
            and prior_top_revision == top_evidence_revision
        ):
            if prior_top_review_status == "pending_approval":
                top_idea_opportunity["follow_up_review_status"] = "pending_approval"
                should_launch_follow_up = False
            elif prior_top_review_status == "rejected":
                top_idea_opportunity["follow_up_review_status"] = "rejected"
                should_launch_follow_up = False
        if (
            isinstance(top_idea_opportunity, dict)
            and should_launch_follow_up
            and follow_up_review_mode == "queue_for_approval"
        ):
            top_idea_opportunity["follow_up_review_status"] = "pending_approval"
            should_launch_follow_up = False
        elif (
            isinstance(top_idea_opportunity, dict)
            and should_launch_follow_up
            and follow_up_review_mode == "manual_only"
        ):
            top_idea_opportunity["follow_up_review_status"] = "manual_recommendation"
            should_launch_follow_up = False
        if should_launch_follow_up:
            child_job = await executor._create_domain_research_follow_up_job(
                db=db,
                job=job,
                domain=domain,
                objective=objective,
                customer_context=customer_context,
                track_type=track_type,
                source_scope=source_scope,
                top_idea=top_idea,
                docs=docs,
                repo_documents=repo_documents,
                papers=papers,
                repo_source_ids=repo_source_ids,
                benchmark_queries=benchmark_queries,
                automation_profile=automation_profile,
                automation_policy=effective_policy,
                sandbox_profile_id=sandbox_profile_id,
                profile_id=str(profile.id) if profile is not None else None,
            )
            if child_job is not None:
                follow_up_job_id = str(child_job.id)
                follow_up_launches.append(
                    {
                        "type": "deep_dive_chain",
                        "job_id": str(child_job.id),
                        "name": child_job.name,
                        "status": child_job.status,
                    }
                )
                artifacts.append(
                    {
                        "type": "agent_job",
                        "id": str(child_job.id),
                        "title": child_job.name,
                    }
                )
                if isinstance(top_idea_opportunity, dict):
                    top_idea_opportunity["follow_up_review_status"] = "approved_launch"
                    top_idea_opportunity["follow_up_review_evidence_revision"] = (
                        top_evidence_revision or None
                    )

        plan_ids_by_key = {
            str(key): [str(plan.id)]
            for key, plan in created_experiment_plans_by_key.items()
            if str(key).strip() and plan is not None
        }
        run_ids_by_idea_id: dict[str, list[str]] = {}
        for row in validation_launches:
            if not isinstance(row, dict):
                continue
            hypothesis_id = str(row.get("hypothesis_id") or "").strip()
            run_id = str(row.get("run_id") or "").strip()
            if not hypothesis_id or not run_id:
                continue
            run_ids_by_idea_id[hypothesis_id] = list(
                dict.fromkeys([*(run_ids_by_idea_id.get(hypothesis_id) or []), run_id])
            )[:8]

        opportunity_rows = []
        profile_config_revision = compute_research_portfolio_config_revision(
            automation_profile,
            effective_policy,
            sandbox_profile_id,
        )
        for row in opportunities:
            if not isinstance(row, dict):
                continue
            idea_key = _normalize_key(row.get("title"))
            idea_id = str(row.get("opportunity_id") or "").replace("opp_", "idea_")
            merged_row = {
                **row,
                "source_note_ids": list(
                    dict.fromkeys(
                        [
                            *(
                                [
                                    str(v)
                                    for v in (row.get("source_note_ids") or [])
                                    if str(v).strip()
                                ]
                            ),
                            *created_note_ids,
                        ]
                    )
                )[:8],
                "linked_experiment_plan_ids": list(
                    dict.fromkeys(
                        [
                            *(
                                [
                                    str(v)
                                    for v in (
                                        row.get("linked_experiment_plan_ids") or []
                                    )
                                    if str(v).strip()
                                ]
                            ),
                            *(plan_ids_by_key.get(idea_key) or []),
                        ]
                    )
                )[:8],
                "linked_validation_run_ids": list(
                    dict.fromkeys(
                        [
                            *(
                                [
                                    str(v)
                                    for v in (
                                        row.get("linked_validation_run_ids") or []
                                    )
                                    if str(v).strip()
                                ]
                            ),
                            *(run_ids_by_idea_id.get(idea_id) or []),
                        ]
                    )
                )[:8],
            }
            if follow_up_job_id and top_idea_key and idea_key == top_idea_key:
                merged_row["child_job_ids"] = list(
                    dict.fromkeys(
                        [
                            *(
                                [
                                    str(v)
                                    for v in (row.get("child_job_ids") or [])
                                    if str(v).strip()
                                ]
                            ),
                            follow_up_job_id,
                        ]
                    )
                )[:8]
            normalized_row = normalize_research_opportunity(merged_row)
            normalized_row[
                "evidence_revision"
            ] = compute_research_opportunity_evidence_revision(normalized_row)
            normalized_row["portfolio_config_revision"] = profile_config_revision
            normalized_row["last_evaluated_at"] = datetime.utcnow().isoformat()
            if not str(normalized_row.get("last_material_change_at") or "").strip():
                normalized_row["last_material_change_at"] = normalized_row[
                    "last_evaluated_at"
                ]
            opportunity_rows.append(normalize_research_opportunity(normalized_row))
        opportunities = opportunity_rows
        now_iso = datetime.utcnow().isoformat()
        pending_follow_up_approvals = [
            row
            for row in opportunities
            if str(row.get("follow_up_review_status") or "").strip()
            == "pending_approval"
        ][:12]
        manual_follow_up_recommendations = [
            row
            for row in opportunities
            if str(row.get("follow_up_review_status") or "").strip()
            == "manual_recommendation"
        ][:12]
        suppressed_relaunches = [
            {
                "opportunity_id": row.get("opportunity_id"),
                "canonical_key": row.get("canonical_key"),
                "title": row.get("title"),
                "reason_code": (
                    "follow_up_pending_approval"
                    if str(row.get("follow_up_review_status") or "").strip()
                    == "pending_approval"
                    else "manual_follow_up_recommendation"
                    if str(row.get("follow_up_review_status") or "").strip()
                    == "manual_recommendation"
                    else "operator_rejected_follow_up"
                ),
                "category": "follow_up_review",
            }
            for row in opportunities
            if str(row.get("follow_up_review_status") or "").strip()
            in {"pending_approval", "manual_recommendation", "rejected"}
            and not (row.get("child_job_ids") or [])
        ][:20]
        for row in opportunities:
            state = str(row.get("autonomy_state") or "").strip()
            if state:
                continue
            if row.get("child_job_ids") or row.get("linked_validation_run_ids"):
                row["autonomy_state"] = "active"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "active_hold"
                )
            elif str(row.get("follow_up_review_status") or "").strip() == "rejected":
                row["autonomy_state"] = "eligible"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "follow_up_rejected_hold"
                )
                row["last_decision_reason_code"] = (
                    row.get("last_decision_reason_code")
                    or "operator_rejected_follow_up"
                )
            elif (
                str(row.get("follow_up_review_status") or "").strip()
                == "pending_approval"
            ):
                row["autonomy_state"] = "eligible"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "follow_up_pending_approval"
                )
                row["last_decision_reason_code"] = (
                    row.get("last_decision_reason_code") or "follow_up_pending_approval"
                )
            elif (
                str(row.get("follow_up_review_status") or "").strip()
                == "manual_recommendation"
            ):
                row["autonomy_state"] = "eligible"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "follow_up_manual_recommendation"
                )
                row["last_decision_reason_code"] = (
                    row.get("last_decision_reason_code")
                    or "manual_follow_up_recommendation"
                )
            elif str(row.get("stage") or "").strip() == "completed":
                row["autonomy_state"] = "completed_waiting_change"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "completed_hold"
                )
                row["last_decision_reason_code"] = (
                    row.get("last_decision_reason_code") or "completed_current_evidence"
                )
            elif str(row.get("stage") or "").strip() == "blocked":
                row["autonomy_state"] = "blocked_structural"
                row["last_decision_type"] = (
                    row.get("last_decision_type") or "validation_blocked"
                )
                row["last_decision_reason_code"] = row.get(
                    "last_decision_reason_code"
                ) or row.get("last_blocked_reason_code")
            else:
                row["autonomy_state"] = "eligible"
            row["portfolio_config_revision"] = profile_config_revision
            row["last_evaluated_at"] = row.get("last_evaluated_at") or now_iso
            row["last_material_change_at"] = (
                row.get("last_material_change_at") or now_iso
            )
        autonomy_state_counts = summarize_research_opportunity_autonomy_states(
            opportunities
        )
        eligible_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "eligible"
        ][:12]
        cooldown_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "cooldown"
        ][:12]
        completed_waiting_change_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip()
            == "completed_waiting_change"
        ][:12]
        structural_blocked_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "blocked_structural"
        ][:12]
        follow_up_review_counts = {
            "pending_approval": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "pending_approval"
            ),
            "manual_recommendation": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "manual_recommendation"
            ),
            "rejected": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip() == "rejected"
            ),
            "approved_launch": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "approved_launch"
            ),
        }
        scheduler_state = (
            ((job.results or {}).get("execution_strategy") or {}).get("scheduler_state")
            if isinstance((job.results or {}).get("execution_strategy"), dict)
            else {}
        )
        scheduler_state = scheduler_state if isinstance(scheduler_state, dict) else {}
        scheduler_summary = {
            "schedule_type": str(job.schedule_type or "").strip() or None,
            "next_run_at": job.next_run_at.isoformat()
            if isinstance(job.next_run_at, datetime)
            else None,
            "last_evaluated_at": now_iso,
            "last_scheduled_at": str(
                scheduler_state.get("last_scheduled_at") or ""
            ).strip()
            or None,
            "last_dispatched_at": str(
                scheduler_state.get("last_dispatched_at") or ""
            ).strip()
            or None,
            "last_run_status": str(scheduler_state.get("last_run_status") or "").strip()
            or None,
            "pending_follow_up_approvals_count": len(pending_follow_up_approvals),
            "manual_follow_up_recommendations_count": len(
                manual_follow_up_recommendations
            ),
            "suppressed_relaunches_count": len(suppressed_relaunches),
            "launched_follow_up_job_count": len(follow_up_launches),
        }

        summary = (
            f"Domain research completed for {domain}: {len(docs)} KB docs, {len(repo_documents)} repo docs, "
            f"{len(papers)} papers, {len(selected_hypotheses)} promoted hypotheses, "
            f"{len(created_experiment_plan_ids)} experiment drafts."
        )
        domain_research_result = {
            "profile_id": str(profile.id) if profile else None,
            "domain": domain,
            "objective": objective,
            "customer_context": customer_context or None,
            "source_scope": source_scope,
            "track_type": track_type,
            "repo_source_ids": repo_source_ids[:24],
            "benchmark_queries": benchmark_queries[:16],
            "research_mode": research_mode,
            "domain_summary": domain_summary,
            "discovered_signals": discovered_signals,
            "signal_clusters": signal_clusters,
            "proposed_ideas": ideas,
            "hypotheses": memo_payload.get("hypotheses"),
            "idea_candidates": idea_candidates,
            "opportunities": opportunities,
            "ranked_opportunities": ranked_opportunities,
            "scoring_policy": scoring_policy,
            "selection_policy": selection_policy,
            "validation_policy": memo_payload.get("validation_policy"),
            "automation_profile": automation_profile,
            "automation_policy": effective_policy,
            "effective_policy": effective_policy,
            "autonomy_mode": automation_profile,
            "profile_config_revision": profile_config_revision,
            "sandbox_profile_id": sandbox_profile_id,
            "novelty_summary": novelty_summary,
            "delta_since_last_run": delta_since_last_run,
            "evidence_mix": {
                "documents": len(docs),
                "repo_documents": len(repo_documents),
                "papers": len(papers),
            },
            "open_questions": open_questions,
            "brief_markdown": brief_markdown,
            "report_markdown": report_markdown,
            "report_format": report_format,
            "review_item_id": review_item_id,
            "research_note_ids": created_note_ids[:12],
            "experiment_plan_ids": created_experiment_plan_ids[:12],
            "validation_run_ids": created_validation_run_ids[:12],
            "duplicate_suppressed_idea_titles": duplicate_idea_titles[:12],
            "validation_launches": validation_launches[:12],
            "follow_up_launches": follow_up_launches[:12],
            "stage_counts": summarize_research_opportunity_stages(opportunities),
            "autonomy_state_counts": autonomy_state_counts,
            "eligible_opportunities": eligible_opportunities,
            "cooldown_opportunities": cooldown_opportunities,
            "completed_waiting_change_opportunities": completed_waiting_change_opportunities,
            "structural_blocked_opportunities": structural_blocked_opportunities,
            "scheduler_summary": scheduler_summary,
            "pending_follow_up_approvals": pending_follow_up_approvals,
            "manual_follow_up_recommendations": manual_follow_up_recommendations,
            "suppressed_relaunches": suppressed_relaunches,
            "follow_up_review_counts": follow_up_review_counts,
        }
        domain_research_result.update(
            summarize_portfolio_operator_reviews(
                opportunities, effective_policy=effective_policy
            )
        )
        job.results = {
            "research": {
                "documents_found": len(docs),
                "repo_documents_found": len(repo_documents),
                "papers_found": len(papers),
                "insights_saved": len(selected_hypotheses),
                "top_documents": [
                    str(doc.get("title") or "")
                    for doc in docs[:6]
                    if str(doc.get("title") or "").strip()
                ],
                "top_repo_documents": [
                    str(doc.get("title") or doc.get("file_path") or "")
                    for doc in repo_documents[:6]
                    if str(doc.get("title") or doc.get("file_path") or "").strip()
                ],
                "top_papers": [
                    str(paper.get("title") or "")
                    for paper in papers[:6]
                    if str(paper.get("title") or "").strip()
                ],
                "top_insights": [
                    str(idea.get("title") or "")
                    for idea in selected_hypotheses[:8]
                    if str(idea.get("title") or "").strip()
                ],
                "research_note_ids": created_note_ids[:12],
                "experiment_plan_ids": created_experiment_plan_ids[:12],
                "validation_run_ids": created_validation_run_ids[:12],
            },
            "research_bundle": {
                "goal": str(job.goal or "").strip(),
                "top_documents": docs[:12],
                "top_repo_documents": repo_documents[:12],
                "top_papers": papers[:12],
                "key_insights": [
                    {
                        "id": idea.get("id"),
                        "title": idea.get("title"),
                        "category": "opportunity",
                        "confidence": idea.get("overall_score"),
                    }
                    for idea in selected_hypotheses[:12]
                ],
                "suggested_queries": monitor_queries[:12],
                "next_steps": [
                    step
                    for step in (
                        top_idea.get("next_steps")
                        if isinstance(top_idea.get("next_steps"), list)
                        else []
                    )[:5]
                ],
                "artifacts": artifacts[:50],
            },
            "domain_research": domain_research_result,
            "summary": summary,
            "findings_count": len(findings),
            "goal_progress": 100,
        }
        job.output_artifacts = artifacts
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        job.progress = 100
        job.current_phase = "completed"
        job.phase_details = summary
        if profile is not None:
            profile.latest_summary = {
                **domain_research_result,
                "document_ids": [
                    str(doc.get("id"))
                    for doc in docs
                    if str(doc.get("id") or "").strip()
                ][:20],
                "repo_document_ids": [
                    str(doc.get("id"))
                    for doc in repo_documents
                    if str(doc.get("id") or "").strip()
                ][:20],
                "paper_ids": [
                    str(paper.get("arxiv_id"))
                    for paper in papers
                    if str(paper.get("arxiv_id") or "").strip()
                ][:20],
                "summary": summary,
                "last_run_at": datetime.utcnow().isoformat(),
            }
            profile.automation_profile = automation_profile
            profile.automation_policy = effective_policy
            profile.validation_policy = effective_policy
            profile.auto_launch_follow_up = bool(
                effective_policy.get(
                    "auto_launch_follow_up", profile.auto_launch_follow_up
                )
            )
            profile.auto_create_experiment_plans = bool(
                effective_policy.get(
                    "auto_create_experiment_plans", profile.auto_create_experiment_plans
                )
            )
            profile.confidence_threshold = float(
                effective_policy.get(
                    "confidence_threshold", profile.confidence_threshold or 0.7
                )
            )
            profile.latest_note_ids = created_note_ids[:20]
            profile.latest_experiment_plan_ids = created_experiment_plan_ids[:20]
            profile.latest_validation_run_ids = created_validation_run_ids[:20]
            profile.latest_run_job_id = job.id
            profile.last_run_at = datetime.utcnow()
            if str(profile.status or "").strip().lower() != "paused":
                profile.status = (
                    "running"
                    if str(job.schedule_type or "").strip().lower() == "continuous"
                    else "completed"
                )
        await db.commit()
        for validation_job_id in launched_validation_job_ids:
            try:
                from app.tasks.agent_job_tasks import execute_agent_job_task

                execute_agent_job_task.delay(validation_job_id, str(job.user_id))
            except Exception:
                continue
        if follow_up_job_id:
            from app.tasks.agent_job_tasks import execute_agent_job_task

            execute_agent_job_task.delay(follow_up_job_id, str(job.user_id))

        return {"status": "completed", "results": job.results}

    async def run_research_fleet_orchestrator(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        from app.models.domain_research_profile import DomainResearchProfile
        from app.models.experiment import ExperimentPlan
        from app.models.research_note import ResearchNote
        from app.models.research_portfolio import ResearchPortfolio

        def _emit(progress: int, phase: str, details: str) -> None:
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "research_fleet_orchestrator",
                    "result": details,
                }
            )

        def _normalize_key(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
                "_"
            )

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _policy(raw: Any) -> dict[str, Any]:
            from app.services.scientific_validation_service import (
                resolve_portfolio_automation_policy,
            )

            return resolve_portfolio_automation_policy(portfolio_profile, raw)

        def _profile(raw: Any) -> str:
            from app.services.scientific_validation_service import (
                normalize_portfolio_automation_profile,
            )

            return normalize_portfolio_automation_profile(raw, default="balanced")

        cfg = job.config if isinstance(job.config, dict) else {}
        portfolio_id_raw = str(cfg.get("research_portfolio_id") or "").strip()
        if not portfolio_id_raw:
            job.error = "Missing config.research_portfolio_id"
            job.status = AgentJobStatus.FAILED.value
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            portfolio_id = UUID(portfolio_id_raw)
        except Exception:
            job.error = "Invalid research_portfolio_id"
            job.status = AgentJobStatus.FAILED.value
            await db.commit()
            return {"status": "failed", "error": job.error}

        portfolio = await db.get(ResearchPortfolio, portfolio_id)
        if portfolio is None or portfolio.user_id != job.user_id:
            job.error = "Research portfolio not found"
            job.status = AgentJobStatus.FAILED.value
            await db.commit()
            return {"status": "failed", "error": job.error}

        portfolio_profile = _profile(
            cfg.get("automation_profile")
            or getattr(portfolio, "automation_profile", None)
        )
        portfolio_policy = _policy(
            cfg.get("automation_policy")
            if isinstance(cfg.get("automation_policy"), dict)
            else portfolio.automation_policy
        )
        sandbox_profile_id = (
            str(
                cfg.get("sandbox_profile_id") or portfolio.sandbox_profile_id or ""
            ).strip()
            or None
        )
        linked_ids = [
            str(v).strip()
            for v in (
                cfg.get("linked_profile_ids")
                if isinstance(cfg.get("linked_profile_ids"), list)
                else portfolio.linked_profile_ids or []
            )
            if str(v).strip()
        ]

        _emit(10, "loading", f"Loading {len(linked_ids)} linked domain profiles")
        await db.commit()

        profiles: list[DomainResearchProfile] = []
        for raw in linked_ids:
            try:
                loaded = await db.get(DomainResearchProfile, UUID(raw))
            except Exception:
                loaded = None
            if loaded is not None and loaded.user_id == job.user_id:
                profiles.append(loaded)

        _emit(35, "ranking", f"Normalizing opportunities from {len(profiles)} profiles")
        await db.commit()

        recent_plans_stmt = (
            select(ExperimentPlan)
            .where(ExperimentPlan.user_id == job.user_id)
            .order_by(ExperimentPlan.created_at.desc())
            .limit(int(portfolio_policy["duplicate_window_items"]))
        )
        recent_plans = list((await db.execute(recent_plans_stmt)).scalars().all())
        recent_plan_keys = {
            _normalize_key(
                (plan.generator_details or {}).get("idea_title") or plan.title
            )
            for plan in recent_plans
            if _normalize_key(
                (plan.generator_details or {}).get("idea_title") or plan.title
            )
        }

        prior_opportunities = (
            portfolio.opportunities if isinstance(portfolio.opportunities, list) else []
        )
        prior_keys = {
            _normalize_key(
                (row or {}).get("title")
                or (row or {}).get("idea_title")
                or (row or {}).get("canonical_key")
            )
            for row in prior_opportunities
            if isinstance(row, dict)
        }
        prior_by_key = {
            str(item.get("canonical_key") or ""): item
            for item in list_normalized_research_opportunities(prior_opportunities)
            if str(item.get("canonical_key") or "").strip()
        }
        grouped: dict[str, dict[str, Any]] = {}
        suppressed_duplicates: list[dict[str, Any]] = []
        linked_note_ids: list[str] = []
        linked_plan_ids: list[str] = []

        for profile in profiles:
            summary = (
                profile.latest_summary
                if isinstance(profile.latest_summary, dict)
                else {}
            )
            idea_candidates = (
                summary.get("opportunities")
                if isinstance(summary.get("opportunities"), list)
                else summary.get("idea_candidates")
                if isinstance(summary.get("idea_candidates"), list)
                else []
            )
            note_ids = [
                str(v) for v in (profile.latest_note_ids or []) if str(v).strip()
            ]
            plan_ids = [
                str(v)
                for v in (profile.latest_experiment_plan_ids or [])
                if str(v).strip()
            ]
            validation_run_ids = [
                str(v)
                for v in (profile.latest_validation_run_ids or [])
                if str(v).strip()
            ]
            profile_repo_ids = [
                str(v) for v in (profile.repo_source_ids or []) if str(v).strip()
            ]
            for v in note_ids:
                if v not in linked_note_ids:
                    linked_note_ids.append(v)
            for v in plan_ids:
                if v not in linked_plan_ids:
                    linked_plan_ids.append(v)
            for idx, candidate in enumerate(idea_candidates[:12]):
                if not isinstance(candidate, dict):
                    continue
                title = str(candidate.get("title") or "").strip()
                if not title:
                    continue
                key = _normalize_key(title)
                if not key:
                    continue
                confidence = max(
                    0.0, min(_safe_float(candidate.get("confidence"), 0.0), 1.0)
                )
                novelty = max(0.0, min(_safe_float(candidate.get("novelty"), 0.5), 1.0))
                readiness = round(min(1.0, confidence * 0.65 + novelty * 0.35), 3)
                evidence = (
                    candidate.get("supporting_evidence")
                    if isinstance(candidate.get("supporting_evidence"), list)
                    else []
                )
                next_steps = (
                    candidate.get("next_steps")
                    if isinstance(candidate.get("next_steps"), list)
                    else []
                )
                current = grouped.get(key)
                item = {
                    "opportunity_id": f"opp_{key[:48]}",
                    "canonical_key": key,
                    "title": title,
                    "hypothesis": str(candidate.get("hypothesis") or title).strip(),
                    "confidence": confidence,
                    "novelty": novelty,
                    "readiness": readiness,
                    "stage": "discovered",
                    "supporting_evidence": [
                        str(v).strip() for v in evidence if str(v).strip()
                    ][:8],
                    "next_steps": [
                        str(v).strip() for v in next_steps if str(v).strip()
                    ][:6],
                    "source_profile_ids": [str(profile.id)],
                    "source_job_ids": [str(profile.latest_run_job_id)]
                    if profile.latest_run_job_id
                    else [],
                    "source_note_ids": note_ids[:8],
                    "linked_experiment_plan_ids": plan_ids[:8],
                    "linked_validation_run_ids": validation_run_ids[:8],
                    "source_repo_ids": profile_repo_ids[:8],
                    "track_type": str(profile.track_type or "generic"),
                    "decision_state": "pending_review",
                    "decision_source": "system",
                }
                item = merge_operator_fields(
                    normalize_research_opportunity(item), prior_by_key.get(key)
                )
                if current is None:
                    grouped[key] = item
                    continue
                current["source_profile_ids"] = sorted(
                    {*current.get("source_profile_ids", []), str(profile.id)}
                )
                current["source_job_ids"] = sorted(
                    {
                        *current.get("source_job_ids", []),
                        *(item.get("source_job_ids") or []),
                    }
                )
                current["source_note_ids"] = sorted(
                    {*current.get("source_note_ids", []), *note_ids}
                )[:8]
                current["linked_experiment_plan_ids"] = sorted(
                    {*current.get("linked_experiment_plan_ids", []), *plan_ids}
                )[:8]
                current["linked_validation_run_ids"] = sorted(
                    {*current.get("linked_validation_run_ids", []), *validation_run_ids}
                )[:8]
                current["source_repo_ids"] = sorted(
                    {*current.get("source_repo_ids", []), *profile_repo_ids}
                )[:8]
                current["supporting_evidence"] = list(
                    dict.fromkeys(
                        [
                            *current.get("supporting_evidence", []),
                            *(item.get("supporting_evidence") or []),
                        ]
                    )
                )[:8]
                current["next_steps"] = list(
                    dict.fromkeys(
                        [
                            *current.get("next_steps", []),
                            *(item.get("next_steps") or []),
                        ]
                    )
                )[:6]
                if readiness > _safe_float(current.get("readiness"), 0.0):
                    item["source_profile_ids"] = sorted(
                        {
                            *current.get("source_profile_ids", []),
                            *(item.get("source_profile_ids") or []),
                        }
                    )[:8]
                    item["source_job_ids"] = sorted(
                        {
                            *current.get("source_job_ids", []),
                            *(item.get("source_job_ids") or []),
                        }
                    )[:8]
                    item["source_note_ids"] = sorted(
                        {
                            *current.get("source_note_ids", []),
                            *(item.get("source_note_ids") or []),
                        }
                    )[:8]
                    item["linked_experiment_plan_ids"] = sorted(
                        {
                            *current.get("linked_experiment_plan_ids", []),
                            *(item.get("linked_experiment_plan_ids") or []),
                        }
                    )[:8]
                    item["linked_validation_run_ids"] = sorted(
                        {
                            *current.get("linked_validation_run_ids", []),
                            *(item.get("linked_validation_run_ids") or []),
                        }
                    )[:8]
                    item["source_repo_ids"] = sorted(
                        {
                            *current.get("source_repo_ids", []),
                            *(item.get("source_repo_ids") or []),
                        }
                    )[:8]
                    item["supporting_evidence"] = list(
                        dict.fromkeys(
                            [
                                *current.get("supporting_evidence", []),
                                *(item.get("supporting_evidence") or []),
                            ]
                        )
                    )[:8]
                    item["next_steps"] = list(
                        dict.fromkeys(
                            [
                                *current.get("next_steps", []),
                                *(item.get("next_steps") or []),
                            ]
                        )
                    )[:6]
                    suppressed_duplicates.append(
                        {
                            "canonical_key": key,
                            "suppressed_title": str(current.get("title") or ""),
                            "kept_title": title,
                            "reason": "higher_readiness_variant",
                            "profile_id": str(profile.id),
                        }
                    )
                    grouped[key] = item
                else:
                    suppressed_duplicates.append(
                        {
                            "canonical_key": key,
                            "suppressed_title": title,
                            "kept_title": str(current.get("title") or ""),
                            "reason": "duplicate_opportunity",
                            "profile_id": str(profile.id),
                        }
                    )

        opportunities = sorted(
            grouped.values(),
            key=lambda row: (
                _safe_float(row.get("readiness"), 0.0),
                _safe_float(row.get("confidence"), 0.0),
                _safe_float(row.get("novelty"), 0.0),
            ),
            reverse=True,
        )[:30]

        _emit(
            60, "planning", f"Evaluating {len(opportunities)} portfolio opportunities"
        )
        await db.commit()

        auto_launch_decisions: list[dict[str, Any]] = []
        skipped_opportunities: list[dict[str, Any]] = []
        blocked_structural_opportunities: list[dict[str, Any]] = []
        eligible_opportunities: list[dict[str, Any]] = []
        cooldown_opportunities: list[dict[str, Any]] = []
        completed_waiting_change_opportunities: list[dict[str, Any]] = []
        recent_reentry_decisions: list[dict[str, Any]] = []
        child_job_ids = [
            str(v) for v in (portfolio.child_job_ids or []) if str(v).strip()
        ]
        created_plan_ids: list[str] = []
        linked_validation_run_ids = [
            str(v)
            for v in (portfolio.latest_validation_run_ids or [])
            if str(v).strip()
        ]
        launched_validation_job_ids: list[str] = []
        launched_follow_up_ids: list[str] = []
        transient_skip_reason_codes = {"concurrency_limit", "backoff_cooldown"}
        now = datetime.utcnow()
        portfolio_config_revision = compute_research_portfolio_config_revision(
            portfolio_profile,
            portfolio_policy,
            sandbox_profile_id,
        )

        def _parse_dt(value: Any) -> Optional[datetime]:
            text = str(value or "").strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except Exception:
                return None

        def _record_reentry(
            row: dict[str, Any],
            *,
            state: str,
            decision_type: str,
            reason_code: Optional[str] = None,
            next_eligible_at: Optional[datetime] = None,
        ) -> None:
            row["autonomy_state"] = state
            row["last_evaluated_at"] = now.isoformat()
            row["last_decision_type"] = decision_type
            row["last_decision_reason_code"] = reason_code
            row["portfolio_config_revision"] = portfolio_config_revision
            row["next_eligible_at"] = (
                next_eligible_at.isoformat() if next_eligible_at else None
            )
            recent_reentry_decisions.append(
                {
                    "opportunity_id": row.get("opportunity_id"),
                    "canonical_key": row.get("canonical_key"),
                    "title": row.get("title"),
                    "autonomy_state": state,
                    "decision_type": decision_type,
                    "reason_code": reason_code,
                    "next_eligible_at": row.get("next_eligible_at"),
                }
            )

        follow_up_review_mode = (
            str(portfolio_policy.get("follow_up_review_mode") or "auto_launch_safe")
            .strip()
            .lower()
        )
        if follow_up_review_mode not in {
            "auto_launch_safe",
            "queue_for_approval",
            "manual_only",
        }:
            follow_up_review_mode = "auto_launch_safe"
        pending_follow_up_approvals: list[dict[str, Any]] = []
        manual_follow_up_recommendations: list[dict[str, Any]] = []
        suppressed_relaunches: list[dict[str, Any]] = []

        def _record_suppressed_relaunch(
            row: dict[str, Any],
            *,
            reason_code: str,
            category: str,
        ) -> None:
            suppressed_relaunches.append(
                {
                    "opportunity_id": row.get("opportunity_id"),
                    "canonical_key": row.get("canonical_key"),
                    "title": row.get("title"),
                    "reason_code": reason_code,
                    "category": category,
                    "follow_up_review_status": row.get("follow_up_review_status"),
                }
            )

        def _current_follow_up_review_applies(
            row: dict[str, Any], *, evidence_revision: str
        ) -> bool:
            return (
                str(row.get("follow_up_review_evidence_revision") or "").strip()
                == evidence_revision
            )

        scheduler_state = (
            ((job.results or {}).get("execution_strategy") or {}).get("scheduler_state")
            if isinstance((job.results or {}).get("execution_strategy"), dict)
            else {}
        )
        scheduler_state = scheduler_state if isinstance(scheduler_state, dict) else {}

        for row in opportunities:
            key = str(row.get("canonical_key") or "").strip()
            confidence = _safe_float(row.get("confidence"), 0.0)
            readiness = _safe_float(row.get("readiness"), 0.0)
            novelty = _safe_float(row.get("novelty"), 0.0)
            prior_row = prior_by_key.get(key) or {}
            prior_evidence_revision = str(
                prior_row.get("evidence_revision") or ""
            ).strip()
            current_evidence_revision = compute_research_opportunity_evidence_revision(
                row
            )
            row["evidence_revision"] = current_evidence_revision
            row["portfolio_config_revision"] = portfolio_config_revision
            row["last_evaluated_at"] = now.isoformat()
            if prior_evidence_revision != current_evidence_revision:
                row["last_material_change_at"] = now.isoformat()
                for field in (
                    "follow_up_review_status",
                    "follow_up_reviewed_at",
                    "follow_up_reviewed_by_user_id",
                    "follow_up_review_note",
                    "follow_up_review_evidence_revision",
                ):
                    row[field] = None
            else:
                row["last_material_change_at"] = (
                    str(prior_row.get("last_material_change_at") or "").strip()
                    or now.isoformat()
                )
            if str(row.get("decision_state") or "").strip().lower() == "suppressed":
                row["stage"] = "suppressed"
                _record_reentry(
                    row,
                    state="blocked_structural",
                    decision_type="suppressed",
                    reason_code="operator_suppressed",
                )
                continue
            existing_plan_ids = [
                str(v)
                for v in (row.get("linked_experiment_plan_ids") or [])
                if str(v).strip()
            ]
            existing_run_ids = [
                str(v)
                for v in (row.get("linked_validation_run_ids") or [])
                if str(v).strip()
            ]
            existing_child_job_ids = [
                str(v) for v in (row.get("child_job_ids") or []) if str(v).strip()
            ]
            prior_state = str(prior_row.get("autonomy_state") or "").strip()
            prior_stage = str(prior_row.get("stage") or "").strip()
            evidence_unchanged = (
                bool(prior_evidence_revision)
                and prior_evidence_revision == current_evidence_revision
            )
            prior_config_revision = str(
                prior_row.get("portfolio_config_revision") or ""
            ).strip()
            if not evidence_unchanged and prior_stage in {"completed", "blocked"}:
                existing_run_ids = []
                existing_child_job_ids = []
                row["linked_validation_run_ids"] = []
                row["child_job_ids"] = []
            if prior_stage == "completed" and evidence_unchanged:
                row["stage"] = "completed"
                _record_reentry(
                    row,
                    state="completed_waiting_change",
                    decision_type="completed_hold",
                    reason_code="completed_current_evidence",
                )
                completed_waiting_change_opportunities.append(
                    {
                        "opportunity_id": row.get("opportunity_id"),
                        "canonical_key": key,
                        "title": row.get("title"),
                        "reason_code": "completed_current_evidence",
                    }
                )
                skipped_opportunities.append(completed_waiting_change_opportunities[-1])
                continue
            if prior_state == "cooldown" and evidence_unchanged:
                next_eligible_at = _parse_dt(prior_row.get("next_eligible_at"))
                if next_eligible_at and next_eligible_at > now:
                    row["stage"] = "planned" if existing_plan_ids else "accepted"
                    row["last_skip_reason_code"] = str(
                        prior_row.get("last_decision_reason_code")
                        or prior_row.get("last_skip_reason_code")
                        or "backoff_cooldown"
                    )
                    _record_reentry(
                        row,
                        state="cooldown",
                        decision_type="cooldown_hold",
                        reason_code=row["last_skip_reason_code"],
                        next_eligible_at=next_eligible_at,
                    )
                    cooldown_opportunities.append(
                        {
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            "title": row.get("title"),
                            "reason_code": row["last_skip_reason_code"],
                            "next_eligible_at": row.get("next_eligible_at"),
                        }
                    )
                    skipped_opportunities.append(cooldown_opportunities[-1])
                    continue
            if (
                prior_state == "blocked_structural"
                and evidence_unchanged
                and prior_config_revision == portfolio_config_revision
            ):
                row["stage"] = "blocked"
                row["last_blocked_reason_code"] = str(
                    prior_row.get("last_blocked_reason_code")
                    or prior_row.get("last_decision_reason_code")
                    or "structural_block"
                )
                _record_reentry(
                    row,
                    state="blocked_structural",
                    decision_type="structural_hold",
                    reason_code=row["last_blocked_reason_code"],
                )
                blocked_structural_opportunities.append(
                    {
                        "opportunity_id": row.get("opportunity_id"),
                        "canonical_key": key,
                        "title": row.get("title"),
                        "last_blocked_reason_code": row.get("last_blocked_reason_code"),
                    }
                )
                continue
            if existing_run_ids:
                row["stage"] = "validating"
                _record_reentry(
                    row,
                    state="active",
                    decision_type="active_hold",
                    reason_code="active_validation_exists",
                )
                skipped_opportunities.append(
                    {
                        "opportunity_id": row.get("opportunity_id"),
                        "canonical_key": key,
                        "title": row.get("title"),
                        "reason_code": "active_validation_exists",
                    }
                )
                continue
            if existing_child_job_ids:
                row["stage"] = "validating"
                _record_reentry(
                    row,
                    state="active",
                    decision_type="active_hold",
                    reason_code="active_follow_up_exists",
                )
                skipped_opportunities.append(
                    {
                        "opportunity_id": row.get("opportunity_id"),
                        "canonical_key": key,
                        "title": row.get("title"),
                        "reason_code": "active_follow_up_exists",
                    }
                )
                _record_suppressed_relaunch(
                    row,
                    reason_code="active_follow_up_exists",
                    category="active_follow_up",
                )
                continue
            if existing_plan_ids:
                row["stage"] = "planned"
            elif key in recent_plan_keys:
                row["decision_state"] = "auto_accepted"
                row["stage"] = "planned"
                row["last_skip_reason_code"] = "recent_plan_window"
                _record_reentry(
                    row,
                    state="eligible",
                    decision_type="duplicate_hold",
                    reason_code="recent_plan_window",
                )
                skipped_opportunities.append(
                    {
                        "opportunity_id": row.get("opportunity_id"),
                        "canonical_key": key,
                        "title": row.get("title"),
                        "reason_code": "recent_plan_window",
                    }
                )
                suppressed_duplicates.append(
                    {
                        "canonical_key": key,
                        "suppressed_title": str(row.get("title") or ""),
                        "kept_title": str(row.get("title") or ""),
                        "reason": "recent_plan_window",
                    }
                )
                continue
            _record_reentry(
                row, state="eligible", decision_type="evaluated", reason_code=None
            )
            eligible_opportunities.append(
                {
                    "opportunity_id": row.get("opportunity_id"),
                    "canonical_key": key,
                    "title": row.get("title"),
                }
            )
            if (
                portfolio_policy["auto_create_experiment_plans"]
                and not existing_plan_ids
                and confidence >= portfolio_policy["confidence_threshold"]
            ):
                if key in recent_plan_keys:
                    continue
                note_id = str((row.get("source_note_ids") or [None])[0] or "").strip()
                note = None
                if note_id:
                    try:
                        note = await db.get(ResearchNote, UUID(note_id))
                    except Exception:
                        note = None
                if note is not None and note.user_id == job.user_id:
                    plan = ExperimentPlan(
                        user_id=job.user_id,
                        research_note_id=note.id,
                        title=f"Experiment Plan: {str(row.get('title') or key)[:460]}",
                        hypothesis_text=str(
                            row.get("hypothesis") or row.get("title") or ""
                        ).strip(),
                        plan={
                            "portfolio_title": portfolio.title,
                            "objective": portfolio.objective,
                            "opportunity_title": str(row.get("title") or ""),
                            "supporting_evidence": row.get("supporting_evidence")
                            if isinstance(row.get("supporting_evidence"), list)
                            else [],
                            "recommended_experiments": [
                                f"Validate {str(row.get('title') or 'the opportunity')} against internal baselines",
                                "Define metrics, datasets, and failure criteria",
                                "Record whether the opportunity is ready for an implementation or experiment run",
                            ],
                        },
                        generator="research_fleet_orchestrator",
                        generator_details={
                            "origin": "research_portfolio",
                            "portfolio_id": str(portfolio.id),
                            "opportunity_id": str(row.get("opportunity_id") or ""),
                            "idea_title": str(row.get("title") or ""),
                            "confidence": confidence,
                            "novelty": novelty,
                            "readiness": readiness,
                            "source_profile_ids": row.get("source_profile_ids")
                            if isinstance(row.get("source_profile_ids"), list)
                            else [],
                            "source_job_ids": row.get("source_job_ids")
                            if isinstance(row.get("source_job_ids"), list)
                            else [],
                            "source_note_ids": row.get("source_note_ids")
                            if isinstance(row.get("source_note_ids"), list)
                            else [],
                            "created_at": datetime.utcnow().isoformat(),
                        },
                    )
                    db.add(plan)
                    await db.flush()
                    plan_id = str(plan.id)
                    row["linked_experiment_plan_ids"] = [plan_id]
                    row["decision_state"] = "auto_accepted"
                    row["stage"] = "planned"
                    _record_reentry(
                        row,
                        state="active",
                        decision_type="experiment_plan_created",
                        reason_code=None,
                    )
                    created_plan_ids.append(plan_id)
                    if plan_id not in linked_plan_ids:
                        linked_plan_ids.append(plan_id)
                    recent_plan_keys.add(key)
                    auto_launch_decisions.append(
                        {
                            "type": "experiment_plan_created",
                            "opportunity_id": row.get("opportunity_id"),
                            "plan_id": plan_id,
                        }
                    )
            if (
                portfolio_policy["auto_launch_experiment_runs"]
                and row.get("linked_experiment_plan_ids")
                and not existing_run_ids
                and confidence >= portfolio_policy["confidence_threshold"]
                and readiness >= portfolio_policy["experiment_readiness_threshold"]
            ):
                plan_id = str(
                    (row.get("linked_experiment_plan_ids") or [None])[0] or ""
                ).strip()
                plan = None
                if plan_id:
                    try:
                        plan = await db.get(ExperimentPlan, UUID(plan_id))
                    except Exception:
                        plan = None
                if plan is not None and plan.user_id == job.user_id:
                    source_profile_id = str(
                        (row.get("source_profile_ids") or [None])[0] or ""
                    ).strip()
                    source_repo_ids = [
                        str(v).strip()
                        for v in (row.get("source_repo_ids") or [])
                        if str(v).strip()
                    ]
                    source_profile = None
                    if source_profile_id:
                        try:
                            source_profile = await db.get(
                                DomainResearchProfile, UUID(source_profile_id)
                            )
                        except Exception:
                            source_profile = None
                    decision = await executor._create_scientific_validation_run(
                        db=db,
                        parent_job=job,
                        experiment_plan=plan,
                        track_type=str(
                            row.get("track_type")
                            or (
                                source_profile.track_type
                                if source_profile
                                else "generic"
                            )
                        ),
                        objective=portfolio.objective,
                        hypothesis_title=str(row.get("title") or ""),
                        hypothesis_text=str(
                            row.get("hypothesis") or row.get("title") or ""
                        ),
                        validation_policy=portfolio_policy,
                        sandbox_profile_id=sandbox_profile_id
                        or (
                            source_profile.sandbox_profile_id
                            if source_profile
                            else None
                        ),
                        repo_source_ids=source_repo_ids
                        or (
                            [str(v) for v in (source_profile.repo_source_ids or [])]
                            if source_profile
                            else []
                        ),
                        benchmark_queries=(
                            [
                                str(v).strip()
                                for v in (source_profile.benchmark_queries or [])
                                if str(v).strip()
                            ]
                            if source_profile
                            and isinstance(source_profile.benchmark_queries, list)
                            else []
                        ),
                        supporting_evidence=(
                            row.get("supporting_evidence")
                            if isinstance(row.get("supporting_evidence"), list)
                            else []
                        ),
                        supporting_sources=[],
                        profile_id=source_profile_id or None,
                        portfolio_id=str(portfolio.id),
                        hypothesis_id=str(row.get("opportunity_id") or "").strip()
                        or None,
                        originating_job_id=str(job.id),
                    )
                    if decision.get("run_id"):
                        run_id = str(decision["run_id"])
                        if (
                            str(decision.get("reason_code") or "").strip()
                            not in transient_skip_reason_codes
                        ):
                            if run_id not in linked_validation_run_ids:
                                linked_validation_run_ids.append(run_id)
                            row["linked_validation_run_ids"] = sorted(
                                {*(row.get("linked_validation_run_ids") or []), run_id}
                            )[:8]
                    if decision.get("status") == "blocked":
                        reason_code = (
                            str(decision.get("reason_code") or "").strip() or None
                        )
                        if reason_code in transient_skip_reason_codes:
                            row["stage"] = (
                                "planned"
                                if row.get("linked_experiment_plan_ids")
                                else "accepted"
                            )
                            row["last_skip_reason_code"] = reason_code
                            cooldown_until = now + timedelta(
                                minutes=int(
                                    (
                                        (
                                            portfolio_policy.get(
                                                "validation_backoff_policy"
                                            )
                                            or {}
                                        ).get("cooldown_minutes")
                                        or 180
                                    )
                                )
                            )
                            _record_reentry(
                                row,
                                state="cooldown",
                                decision_type="validation_skip",
                                reason_code=reason_code,
                                next_eligible_at=cooldown_until,
                            )
                            skipped_opportunities.append(
                                {
                                    "opportunity_id": row.get("opportunity_id"),
                                    "canonical_key": key,
                                    "title": row.get("title"),
                                    "reason_code": reason_code,
                                    "next_eligible_at": row.get("next_eligible_at"),
                                }
                            )
                        else:
                            row["stage"] = "blocked"
                            row["last_blocked_reason_code"] = reason_code
                            _record_reentry(
                                row,
                                state="blocked_structural",
                                decision_type="validation_blocked",
                                reason_code=reason_code,
                            )
                            blocked_structural_opportunities.append(
                                {
                                    "opportunity_id": row.get("opportunity_id"),
                                    "canonical_key": key,
                                    "title": row.get("title"),
                                    "last_blocked_reason_code": row.get(
                                        "last_blocked_reason_code"
                                    ),
                                }
                            )
                    elif decision.get("run_id"):
                        row["stage"] = "validating"
                        _record_reentry(
                            row,
                            state="active",
                            decision_type="validation_run_queued",
                            reason_code=None,
                        )
                    if decision.get("job_id"):
                        launched_validation_job_ids.append(str(decision["job_id"]))
                        child_job_ids.append(str(decision["job_id"]))
                    row["decision_state"] = "auto_accepted"
                    auto_launch_decisions.append(
                        {
                            "type": "validation_run_"
                            + ("queued" if decision.get("job_id") else "blocked"),
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            **decision,
                        }
                    )
            if (
                portfolio_policy["auto_launch_follow_up"]
                and not existing_child_job_ids
                and len(launched_follow_up_ids)
                < int(portfolio_policy["max_auto_follow_up_launches"])
                and confidence >= portfolio_policy["confidence_threshold"]
                and readiness >= portfolio_policy["experiment_readiness_threshold"]
            ):
                if key in prior_keys and str(row.get("stage") or "") in {
                    "planned",
                    "validating",
                }:
                    _record_suppressed_relaunch(
                        row,
                        reason_code="active_or_planned_hold",
                        category="planned_hold",
                    )
                    continue
                review_status = (
                    str(row.get("follow_up_review_status") or "").strip().lower()
                )
                review_matches_current_evidence = _current_follow_up_review_applies(
                    row, evidence_revision=current_evidence_revision
                )
                if (
                    review_status == "pending_approval"
                    and review_matches_current_evidence
                ):
                    row["stage"] = (
                        "planned"
                        if row.get("linked_experiment_plan_ids")
                        else "accepted"
                    )
                    _record_reentry(
                        row,
                        state="eligible",
                        decision_type="follow_up_pending_approval",
                        reason_code="follow_up_pending_approval",
                    )
                    pending_follow_up_approvals.append(
                        {
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            "title": row.get("title"),
                            "reason_code": "follow_up_pending_approval",
                        }
                    )
                    _record_suppressed_relaunch(
                        row,
                        reason_code="follow_up_pending_approval",
                        category="pending_approval",
                    )
                    continue
                if review_status == "rejected" and review_matches_current_evidence:
                    row["stage"] = (
                        "planned"
                        if row.get("linked_experiment_plan_ids")
                        else "accepted"
                    )
                    _record_reentry(
                        row,
                        state="eligible",
                        decision_type="follow_up_rejected_hold",
                        reason_code="operator_rejected_follow_up",
                    )
                    _record_suppressed_relaunch(
                        row,
                        reason_code="operator_rejected_follow_up",
                        category="rejected_follow_up",
                    )
                    continue
                if follow_up_review_mode == "queue_for_approval":
                    row["follow_up_review_status"] = "pending_approval"
                    row[
                        "follow_up_review_evidence_revision"
                    ] = current_evidence_revision
                    row["stage"] = (
                        "planned"
                        if row.get("linked_experiment_plan_ids")
                        else "accepted"
                    )
                    _record_reentry(
                        row,
                        state="eligible",
                        decision_type="follow_up_queued_for_approval",
                        reason_code="follow_up_pending_approval",
                    )
                    pending_follow_up_approvals.append(
                        {
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            "title": row.get("title"),
                            "reason_code": "follow_up_pending_approval",
                        }
                    )
                    auto_launch_decisions.append(
                        {
                            "type": "follow_up_queued_for_approval",
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                        }
                    )
                    _record_suppressed_relaunch(
                        row,
                        reason_code="follow_up_pending_approval",
                        category="pending_approval",
                    )
                    continue
                if follow_up_review_mode == "manual_only":
                    row["follow_up_review_status"] = "manual_recommendation"
                    row[
                        "follow_up_review_evidence_revision"
                    ] = current_evidence_revision
                    row["stage"] = (
                        "planned"
                        if row.get("linked_experiment_plan_ids")
                        else "accepted"
                    )
                    _record_reentry(
                        row,
                        state="eligible",
                        decision_type="follow_up_manual_recommendation",
                        reason_code="manual_follow_up_recommendation",
                    )
                    manual_follow_up_recommendations.append(
                        {
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            "title": row.get("title"),
                            "reason_code": "manual_follow_up_recommendation",
                        }
                    )
                    auto_launch_decisions.append(
                        {
                            "type": "follow_up_manual_recommendation",
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                        }
                    )
                    _record_suppressed_relaunch(
                        row,
                        reason_code="manual_follow_up_recommendation",
                        category="manual_recommendation",
                    )
                    continue
                child_job = await executor._create_domain_research_follow_up_job(
                    db=db,
                    job=job,
                    domain=str(row.get("title") or portfolio.title),
                    objective=portfolio.objective,
                    customer_context="research_portfolio",
                    track_type=str(row.get("track_type") or "generic"),
                    source_scope="kb_plus_arxiv_plus_repo"
                    if row.get("source_repo_ids")
                    else "kb_plus_arxiv",
                    top_idea={
                        "title": str(row.get("title") or ""),
                        "hypothesis": str(row.get("hypothesis") or ""),
                        "confidence": confidence,
                        "next_steps": row.get("next_steps")
                        if isinstance(row.get("next_steps"), list)
                        else [],
                    },
                    docs=[],
                    repo_documents=[],
                    papers=[],
                    repo_source_ids=[
                        str(v).strip()
                        for v in (row.get("source_repo_ids") or [])
                        if str(v).strip()
                    ],
                    benchmark_queries=[],
                    automation_profile=portfolio.automation_profile,
                    automation_policy=portfolio.automation_policy
                    if isinstance(portfolio.automation_policy, dict)
                    else {},
                    sandbox_profile_id=portfolio.sandbox_profile_id,
                )
                if child_job is not None:
                    child_id = str(child_job.id)
                    launched_follow_up_ids.append(child_id)
                    child_job_ids.append(child_id)
                    row["child_job_ids"] = [child_id]
                    row["decision_state"] = "auto_accepted"
                    row["stage"] = "validating"
                    row["follow_up_review_status"] = "approved_launch"
                    row[
                        "follow_up_review_evidence_revision"
                    ] = current_evidence_revision
                    _record_reentry(
                        row,
                        state="active",
                        decision_type="follow_up_launched",
                        reason_code=None,
                    )
                    auto_launch_decisions.append(
                        {
                            "type": "follow_up_launched",
                            "opportunity_id": row.get("opportunity_id"),
                            "canonical_key": key,
                            "job_id": child_id,
                        }
                    )

        opportunities = [normalize_research_opportunity(row) for row in opportunities]
        linked_ids = collect_research_opportunity_linked_ids(opportunities)
        stage_counts = summarize_research_opportunity_stages(opportunities)
        autonomy_state_counts = summarize_research_opportunity_autonomy_states(
            opportunities
        )
        blocked_opportunities = [
            row
            for row in opportunities
            if str(row.get("stage") or "").strip() == "blocked"
        ][:12]
        blocked_structural_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "blocked_structural"
        ][:12]
        cooldown_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "cooldown"
        ][:12]
        completed_waiting_change_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip()
            == "completed_waiting_change"
        ][:12]
        eligible_opportunities = [
            row
            for row in opportunities
            if str(row.get("autonomy_state") or "").strip() == "eligible"
        ][:12]
        follow_up_review_counts = {
            "pending_approval": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "pending_approval"
            ),
            "manual_recommendation": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "manual_recommendation"
            ),
            "rejected": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip() == "rejected"
            ),
            "approved_launch": sum(
                1
                for row in opportunities
                if str(row.get("follow_up_review_status") or "").strip()
                == "approved_launch"
            ),
        }
        scheduler_summary = {
            "schedule_type": str(job.schedule_type or "").strip() or None,
            "next_run_at": job.next_run_at.isoformat()
            if isinstance(job.next_run_at, datetime)
            else None,
            "last_evaluated_at": now.isoformat(),
            "last_scheduled_at": str(
                scheduler_state.get("last_scheduled_at") or ""
            ).strip()
            or None,
            "last_dispatched_at": str(
                scheduler_state.get("last_dispatched_at") or ""
            ).strip()
            or None,
            "last_run_status": str(scheduler_state.get("last_run_status") or "").strip()
            or None,
            "pending_follow_up_approvals_count": len(pending_follow_up_approvals),
            "manual_follow_up_recommendations_count": len(
                manual_follow_up_recommendations
            ),
            "suppressed_relaunches_count": len(suppressed_relaunches),
            "launched_follow_up_job_count": len(launched_follow_up_ids),
        }
        autonomy_summary = {
            "autonomy_mode": portfolio_profile,
            "suppressed_duplicates_count": len(suppressed_duplicates),
            "blocked_opportunities_count": len(blocked_opportunities),
            "skipped_opportunities_count": len(skipped_opportunities),
            "eligible_opportunities_count": len(eligible_opportunities),
            "cooldown_opportunities_count": len(cooldown_opportunities),
            "completed_waiting_change_count": len(
                completed_waiting_change_opportunities
            ),
            "structural_blocked_opportunities_count": len(
                blocked_structural_opportunities
            ),
            "created_experiment_plan_count": len(created_plan_ids),
            "launched_validation_run_count": len(linked_validation_run_ids),
            "launched_follow_up_job_count": len(launched_follow_up_ids),
        }
        latest_summary = {
            "portfolio_title": portfolio.title,
            "objective": portfolio.objective,
            "autonomy_mode": portfolio_profile,
            "autonomy_summary": autonomy_summary,
            "effective_policy": portfolio_policy,
            "portfolio_config_revision": portfolio_config_revision,
            "scheduler_summary": scheduler_summary,
            "linked_profile_count": len(profiles),
            "opportunity_count": len(opportunities),
            "stage_counts": stage_counts,
            "autonomy_state_counts": autonomy_state_counts,
            "pending_follow_up_approvals": pending_follow_up_approvals[:12],
            "manual_follow_up_recommendations": manual_follow_up_recommendations[:12],
            "suppressed_relaunches": suppressed_relaunches[:20],
            "follow_up_review_counts": follow_up_review_counts,
            "suppressed_duplicates_count": len(suppressed_duplicates),
            "suppressed_duplicates": suppressed_duplicates[:20],
            "blocked_opportunities": blocked_opportunities,
            "structural_blocked_opportunities": blocked_structural_opportunities,
            "skipped_opportunities": skipped_opportunities[:20],
            "eligible_opportunities": eligible_opportunities,
            "cooldown_opportunities": cooldown_opportunities,
            "completed_waiting_change_opportunities": completed_waiting_change_opportunities,
            "recent_reentry_decisions": recent_reentry_decisions[:20],
            "auto_launch_decisions": auto_launch_decisions[:20],
            "last_autonomous_decisions": auto_launch_decisions[:20],
            "created_experiment_plan_ids": created_plan_ids[:30],
            "launched_validation_run_ids": linked_validation_run_ids[:30],
            "launched_follow_up_job_ids": launched_follow_up_ids[:30],
            "latest_validation_run_ids": linked_validation_run_ids[:30],
            "validation_runs": [
                row
                for row in auto_launch_decisions
                if isinstance(row, dict)
                and str(row.get("type") or "").startswith("validation_run_")
            ][:20],
            "top_opportunities": opportunities[:8],
            "opportunities": opportunities,
        }

        portfolio.opportunities = opportunities
        portfolio.latest_summary = latest_summary
        portfolio.latest_note_ids = list(
            dict.fromkeys([*linked_note_ids, *linked_ids["note_ids"]])
        )[:30]
        portfolio.latest_experiment_plan_ids = list(
            dict.fromkeys([*linked_plan_ids, *linked_ids["plan_ids"]])
        )[:30]
        portfolio.latest_validation_run_ids = list(
            dict.fromkeys([*linked_validation_run_ids, *linked_ids["run_ids"]])
        )[:30]
        portfolio.child_job_ids = list(
            dict.fromkeys([*child_job_ids, *linked_ids["child_job_ids"]])
        )[:50]
        portfolio.latest_run_job_id = job.id
        portfolio.last_run_at = datetime.utcnow()
        if str(portfolio.status or "").strip().lower() != "paused":
            portfolio.status = (
                "running"
                if str(job.schedule_type or "").strip().lower() == "continuous"
                else "completed"
            )

        _emit(90, "persisting", "Persisting portfolio opportunities and launches")
        job.results = {
            "research_portfolio": {
                "portfolio_id": str(portfolio.id),
                "title": portfolio.title,
                "objective": portfolio.objective,
                "automation_profile": portfolio_profile,
                "automation_policy": portfolio_policy,
                "opportunities": opportunities,
                "suppressed_duplicates": suppressed_duplicates[:20],
                "auto_launch_decisions": auto_launch_decisions[:20],
                "linked_profile_ids": linked_ids,
                "linked_experiment_plan_ids": linked_plan_ids[:30],
                "linked_experiment_run_ids": linked_validation_run_ids[:30],
                "portfolio_summary": latest_summary,
            },
            "summary": f"Research fleet processed {len(opportunities)} opportunities from {len(profiles)} profiles.",
            "goal_progress": 100,
        }
        job.output_artifacts = (
            [
                {
                    "type": "experiment_plan",
                    "id": plan_id,
                    "title": f"Experiment Plan {plan_id[:8]}",
                }
                for plan_id in created_plan_ids[:20]
            ]
            + [
                {
                    "type": "experiment_run",
                    "id": run_id,
                    "title": f"Scientific Validation {run_id[:8]}",
                }
                for run_id in linked_validation_run_ids[:20]
            ]
            + [
                {
                    "type": "agent_job",
                    "id": child_id,
                    "title": "Research Fleet Follow-up",
                }
                for child_id in launched_follow_up_ids[:20]
            ]
        )[:50]
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        job.progress = 100
        job.current_phase = "completed"
        job.phase_details = str(job.results.get("summary") or "")
        await db.commit()

        for child_id in launched_follow_up_ids:
            try:
                from app.tasks.agent_job_tasks import execute_agent_job_task

                execute_agent_job_task.delay(child_id, str(job.user_id))
            except Exception:
                continue
        for child_id in launched_validation_job_ids:
            try:
                from app.tasks.agent_job_tasks import execute_agent_job_task

                execute_agent_job_task.delay(child_id, str(job.user_id))
            except Exception:
                continue

        return {"status": "completed", "results": job.results}

    async def run_research_engineer_paper_update(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: append implementation notes into a LaTeX project based on parent code patch results.

        Expects:
          - job.config.latex_project_id (UUID)
          - job.config.inherited_data.parent_results.code_patch (from code_patch_proposer)
        """
        from uuid import UUID as _UUID

        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "research_engineer_paper_update",
                    "result": details,
                }
            )

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = source or ""
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        config = job.config if isinstance(job.config, dict) else {}
        latex_project_id = (config or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing job.config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            latex_project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (
            (config or {}).get("inherited_data") if isinstance(config, dict) else None
        )
        parent_results = (
            inherited.get("parent_results") if isinstance(inherited, dict) else None
        )
        code_patch = (
            parent_results.get("code_patch")
            if isinstance(parent_results, dict)
            else None
        )
        if not isinstance(code_patch, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing inherited code_patch results"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(40, "writing", "Updating LaTeX project with implementation notes")
        await db.commit()

        title = str(code_patch.get("title") or "Code Patch Proposal").strip()
        summary = str(code_patch.get("summary") or "").strip()
        risks = (
            code_patch.get("risks") if isinstance(code_patch.get("risks"), list) else []
        )
        tests = (
            code_patch.get("tests_to_run")
            if isinstance(code_patch.get("tests_to_run"), list)
            else []
        )
        proposal_id = str(code_patch.get("proposal_id") or "").strip()

        bullets = []
        if summary:
            bullets.append(f"\\item Summary: {summary}")
        if risks:
            bullets.append(
                "\\item Risks: "
                + "; ".join([str(r).strip() for r in risks if str(r).strip()][:8])
            )
        if tests:
            bullets.append(
                "\\item Tests: "
                + "; ".join([str(t).strip() for t in tests if str(t).strip()][:8])
            )
        if proposal_id:
            bullets.append(f"\\item Proposal ID: \\texttt{{{proposal_id}}}")
        exp = (
            parent_results.get("experiment_run")
            if isinstance(parent_results, dict)
            and isinstance(parent_results.get("experiment_run"), dict)
            else None
        )
        if isinstance(exp, dict):
            runs = exp.get("runs") if isinstance(exp.get("runs"), list) else []
            ok = exp.get("ok")
            if ok is None:
                bullets.append(
                    "\\item Experiments: skipped (unsafe execution disabled)"
                )
            elif ok:
                bullets.append("\\item Experiments: PASS")
            else:
                failed_cmds = []
                for r in runs:
                    if isinstance(r, dict) and not bool(r.get("ok")):
                        failed_cmds.append(str(r.get("command") or "")[:120])
                if failed_cmds:
                    bullets.append(
                        "\\item Experiments: FAIL (" + "; ".join(failed_cmds[:3]) + ")"
                    )
                else:
                    bullets.append("\\item Experiments: FAIL")

        kb_apply = (
            parent_results.get("code_patch_kb_apply")
            if isinstance(parent_results, dict)
            and isinstance(parent_results.get("code_patch_kb_apply"), dict)
            else None
        )
        if isinstance(kb_apply, dict) and kb_apply.get("enabled") is True:
            if kb_apply.get("dry_run") is True:
                ok = kb_apply.get("ok")
                bullets.append(f"\\item KB apply: dry-run ({'OK' if ok else 'errors'})")
            else:
                bullets.append(
                    "\\item KB apply: "
                    + ("APPLIED" if kb_apply.get("did_apply") else "not applied")
                )

        section = "\\section{Implementation Notes}\n"
        section += f"\\subsection{{{title}}}\n"
        section += (
            "\\begin{itemize}\n"
            + ("\n".join(bullets) if bullets else "\\item (No details available)")
            + "\n\\end{itemize}\n"
        )

        project = await db.get(LatexProject, latex_project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project.tex_source = _insert_before_end_document(
            project.tex_source or "", section
        )
        await db.commit()

        job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
        job.results["research_engineer_paper_update"] = {
            "latex_project_id": str(latex_project_uuid),
            "code_patch_proposal_id": proposal_id or None,
        }

        _emit(100, "completed", "Paper updated with implementation notes")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        return {"status": "completed", "results": job.results}

    async def run_swarm_fan_in_aggregate(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Deterministic runner: aggregate swarm sibling outputs into a strict merged schema."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "swarm_fan_in_aggregate", "result": details}
            )

        inherited = (
            cfg.get("inherited_data")
            if isinstance(cfg.get("inherited_data"), dict)
            else {}
        )
        parent_results = (
            inherited.get("parent_results")
            if isinstance(inherited.get("parent_results"), dict)
            else {}
        )
        swarm_payload = (
            inherited.get("swarm") if isinstance(inherited.get("swarm"), dict) else {}
        )
        if cfg.get("coding_swarm_enabled") and isinstance(swarm_payload, dict):
            swarm_payload = {
                **swarm_payload,
                "coding_swarm_enabled": True,
                "coding_swarm_profile": str(cfg.get("coding_swarm_profile") or "")
                .strip()
                .lower(),
                "coding_harness_enabled": bool(
                    cfg.get("coding_harness_enabled", False)
                ),
                "coding_harness_version": str(
                    cfg.get("coding_harness_version") or ""
                ).strip(),
                "coding_swarm_confidence_threshold": cfg.get(
                    "coding_swarm_confidence_threshold"
                ),
                "coding_swarm_tiebreaker_threshold": cfg.get(
                    "coding_swarm_tiebreaker_threshold"
                ),
                "tie_breaker_attempted": bool(cfg.get("tie_breaker_attempted")),
                "tie_breaker_job_id": str(cfg.get("tie_breaker_job_id") or ""),
                "tie_breaker_source_job_id": str(
                    cfg.get("tie_breaker_source_job_id") or ""
                ),
                "file_paths": cfg.get("file_paths")
                if isinstance(cfg.get("file_paths"), list)
                else [],
                "commands": cfg.get("commands")
                if isinstance(cfg.get("commands"), list)
                else [],
            }
        sibling_jobs = (
            swarm_payload.get("sibling_jobs")
            if isinstance(swarm_payload.get("sibling_jobs"), list)
            else []
        )
        fan_in_group_id = str(
            cfg.get("swarm_fan_in_group_id")
            or swarm_payload.get("swarm_fan_in_group_id")
            or ""
        ).strip()

        if not sibling_jobs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing inherited swarm sibling data"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(
            30,
            "aggregating",
            f"Aggregating outputs from {len(sibling_jobs)} swarm siblings",
        )
        await db.commit()

        merged = executor._build_swarm_fan_in_result(
            swarm_payload,
            fan_in_group_id=fan_in_group_id,
        )
        consensus_rows = (
            merged.get("consensus_findings")
            if isinstance(merged.get("consensus_findings"), list)
            else []
        )
        findings: List[Dict[str, Any]] = []
        for row in consensus_rows[:12]:
            if not isinstance(row, dict):
                continue
            findings.append(
                {
                    "type": "insight",
                    "category": "swarm_consensus",
                    "title": str(row.get("finding") or "")[:280],
                    "support_count": int(row.get("support_count", 0) or 0),
                    "roles": row.get("supporting_roles", [])[:10]
                    if isinstance(row.get("supporting_roles"), list)
                    else [],
                }
            )

        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}
        base_results["swarm_fan_in"] = merged
        base_results["findings"] = findings
        base_results["summary"] = (
            f"Swarm fan-in complete: {int(merged.get('received_siblings', 0) or 0)}/"
            f"{int(merged.get('expected_siblings', 0) or 0)} siblings aggregated, "
            f"{len(consensus_rows)} consensus findings."
        )
        auto_repair_job = None
        auto_tie_breaker_job = None
        auto_backlog_item = None
        if cfg.get("coding_swarm_enabled") and bool(
            cfg.get("coding_swarm_auto_launch_repair_chain", True)
        ):
            confidence = float(((merged.get("confidence") or {}).get("overall") or 0.0))
            threshold = float(cfg.get("coding_swarm_confidence_threshold") or 0.70)
            tiebreaker_threshold = float(
                cfg.get("coding_swarm_tiebreaker_threshold") or 0.50
            )
            tie_breaker_attempted = bool(
                cfg.get("tie_breaker_attempted")
                or swarm_payload.get("tie_breaker_attempted")
            )
            guardrails_met = bool(merged.get("file_converged")) and bool(
                merged.get("command_converged")
            )
            if bool(cfg.get("coding_harness_enabled")):
                guardrails_met = guardrails_met and bool(
                    merged.get("verification_guardrail_met")
                )
            if (
                confidence >= threshold
                and str(merged.get("winning_slice_id") or "").strip()
                and guardrails_met
            ):
                auto_repair_job = await executor._launch_bug_triage_swarm_repair_job(
                    fan_in_job=job,
                    db=db,
                    merged=merged,
                )
                if auto_repair_job is not None:
                    merged["repair_chain_job_id"] = str(auto_repair_job.id)
                    merged["promotion_reason"] = str(
                        merged.get("promotion_reason")
                        or "Auto-promoted winning coding slice."
                    )
                    base_results["swarm_fan_in"] = merged
                    base_results[
                        "summary"
                    ] = f"{base_results['summary']} Auto-launched repair chain {str(auto_repair_job.id)[:8]}."
            elif (
                confidence >= tiebreaker_threshold
                and not tie_breaker_attempted
                and (
                    not bool(cfg.get("coding_harness_enabled"))
                    or bool(str(merged.get("winning_slice_id") or "").strip())
                )
            ):
                auto_tie_breaker_job = (
                    await executor._launch_bug_triage_swarm_tie_breaker_job(
                        fan_in_job=job,
                        db=db,
                        merged=merged,
                        swarm_payload=swarm_payload,
                    )
                )
                if auto_tie_breaker_job is not None:
                    merged["review_state"] = "tie_break_running"
                    merged["review_required"] = False
                    merged["review_reason"] = str(
                        merged.get("review_reason") or "Verifier tie-breaker running."
                    )
                    merged["tie_breaker_job_id"] = str(auto_tie_breaker_job.id)
                    merged["tie_breaker_attempted"] = True
                    base_results["swarm_fan_in"] = merged
                    base_results[
                        "summary"
                    ] = f"{base_results['summary']} Auto-launched verifier tie-breaker {str(auto_tie_breaker_job.id)[:8]}."
        review_state = str(merged.get("review_state") or "").strip().lower()
        if (
            cfg.get("coding_swarm_enabled")
            and auto_repair_job is None
            and auto_tie_breaker_job is None
            and not str(merged.get("repair_chain_job_id") or "").strip()
            and review_state
            in {"needs_review", "insufficient_swarm_consensus", "consensus_failed"}
        ):
            auto_backlog_item = await executor._auto_route_swarm_to_backlog(
                fan_in_job=job,
                db=db,
                merged=merged,
            )
            if auto_backlog_item is not None:
                summary_suffix = (
                    f" Auto-routing suppressed because backlog {str(auto_backlog_item.id)[:8]} is already linked."
                    if str(
                        merged.get("backlog_auto_route_suppressed_reason") or ""
                    ).strip()
                    else f" Auto-routed to backlog {str(auto_backlog_item.id)[:8]}."
                )
                base_results["swarm_fan_in"] = merged
                base_results["summary"] = f"{base_results['summary']}{summary_suffix}"
        job.results = base_results

        if auto_tie_breaker_job is not None:
            _emit(
                100, "completed", "Swarm fan-in complete; verifier tie-breaker running"
            )
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
        elif review_state in {
            "consensus_failed",
            "insufficient_swarm_consensus",
            "needs_review",
        }:
            _emit(
                100,
                "paused",
                str(merged.get("review_reason") or "Paused for operator review"),
            )
            job.status = AgentJobStatus.PAUSED.value
            job.completed_at = None
            job.current_phase = "needs_review"
            job.phase_details = str(
                merged.get("review_reason") or "Paused for operator review"
            )[:280]
        else:
            _emit(100, "completed", "Swarm fan-in aggregation complete")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
        await db.commit()
        if auto_repair_job is not None:
            try:
                from app.tasks.agent_job_tasks import execute_agent_job_task

                execute_agent_job_task.delay(str(auto_repair_job.id), str(job.user_id))
            except Exception:
                logger.exception("Failed to queue bug triage swarm repair handoff job")
        if auto_tie_breaker_job is not None:
            try:
                from app.tasks.agent_job_tasks import execute_agent_job_task

                execute_agent_job_task.delay(
                    str(auto_tie_breaker_job.id), str(job.user_id)
                )
            except Exception:
                logger.exception("Failed to queue bug triage swarm tie-breaker job")
        return {"status": "completed", "results": job.results}
