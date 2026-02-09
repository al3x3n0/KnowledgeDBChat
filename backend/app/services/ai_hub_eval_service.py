"""
AI Hub evaluation service.

Evaluation templates are treated as "plugins": JSON files on disk that define
test cases + a rubric. This makes the feature configurable per customer without
changing product focus (AI Hub remains training + registry + eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from app.core.config import settings


@dataclass(frozen=True)
class EvalTemplate:
    id: str
    name: str
    description: str
    version: int
    judge_preamble: str
    rubric: Dict[str, Any]
    cases: List[Dict[str, Any]]


class AIHubEvalService:
    def __init__(self) -> None:
        self._templates_dir = (
            Path(settings.AI_HUB_EVAL_TEMPLATES_DIR)
            if getattr(settings, "AI_HUB_EVAL_TEMPLATES_DIR", None)
            else Path(__file__).resolve().parents[1] / "plugins" / "ai_hub" / "eval_templates"
        )

    def list_templates(self) -> List[EvalTemplate]:
        templates: List[EvalTemplate] = []
        if not self._templates_dir.exists():
            return templates

        for path in sorted(self._templates_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                templates.append(
                    EvalTemplate(
                        id=data["id"],
                        name=data.get("name", data["id"]),
                        description=data.get("description", ""),
                        version=int(data.get("version", 1)),
                        judge_preamble=data.get("judge_preamble", "") or "",
                        rubric=data.get("rubric", {}) or {},
                        cases=data.get("cases", []) or [],
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to load eval template {path}: {exc}")
        return templates

    def get_template(self, template_id: str) -> Optional[EvalTemplate]:
        for t in self.list_templates():
            if t.id == template_id:
                return t
        return None

    async def _ollama_generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "options": {"num_predict": max_tokens, "temperature": 0.2},
                    "stream": False,
                },
                timeout=120.0,
            )
            if resp.status_code != 200:
                raise ValueError(f"Ollama generate failed: {resp.text}")
            data = resp.json()
            return data.get("response", "") or ""

    async def run_eval(
        self,
        *,
        template: EvalTemplate,
        base_model: str,
        candidate_model: str,
        judge_model: str,
    ) -> Dict[str, Any]:
        """
        Run evaluation by generating responses for base + candidate, then asking a judge model
        to rate the candidate on a 1-5 scale (and optionally compare against base).
        """
        results: List[Dict[str, Any]] = []
        scores: List[int] = []

        rubric = template.rubric or {}
        criteria = rubric.get("criteria") or []
        judge_preamble = template.judge_preamble.strip() or "You are an evaluator for a research assistant."

        for case in template.cases:
            case_id = case.get("id") or "case"
            prompt = case.get("prompt") or ""
            if not prompt:
                continue

            base_resp = await self._ollama_generate(base_model, prompt, max_tokens=512)
            cand_resp = await self._ollama_generate(candidate_model, prompt, max_tokens=512)

            judge_prompt = (
                f"{judge_preamble}\n"
                "Rate the CANDIDATE answer on a 1-5 scale using the rubric.\n"
                "Return ONLY a JSON object: {\"score\": <int 1-5>, \"notes\": \"...\"}.\n\n"
                f"Rubric criteria:\n- " + "\n- ".join([str(x) for x in criteria]) + "\n\n"
                f"Question:\n{prompt}\n\n"
                f"BASE ANSWER:\n{base_resp}\n\n"
                f"CANDIDATE ANSWER:\n{cand_resp}\n"
            )

            judge_raw = await self._ollama_generate(judge_model, judge_prompt, max_tokens=256)
            score = None
            notes = None
            try:
                # best-effort JSON parsing
                start = judge_raw.find("{")
                end = judge_raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    payload = json.loads(judge_raw[start : end + 1])
                    score = int(payload.get("score"))
                    notes = payload.get("notes")
            except Exception:
                pass

            if score is None or score < 1 or score > 5:
                score = 3
                notes = (notes or "Judge output not parseable; defaulted to 3.")[:500]

            scores.append(score)
            results.append(
                {
                    "case_id": case_id,
                    "prompt": prompt,
                    "base_model": base_model,
                    "candidate_model": candidate_model,
                    "base_response": base_resp,
                    "candidate_response": cand_resp,
                    "judge_model": judge_model,
                    "score": score,
                    "notes": notes,
                }
            )

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "template_id": template.id,
            "template_version": template.version,
            "base_model": base_model,
            "candidate_model": candidate_model,
            "judge_model": judge_model,
            "avg_score": avg,
            "num_cases": len(results),
            "results": results,
        }


ai_hub_eval_service = AIHubEvalService()
