"""
Schemas for AI Hub eval templates and runs.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalTemplateInfo(BaseModel):
    id: str
    name: str
    description: str
    version: int
    rubric: Dict[str, Any] = Field(default_factory=dict)
    case_count: int = 0


class EvalTemplatesResponse(BaseModel):
    templates: List[EvalTemplateInfo]


class RunEvalRequest(BaseModel):
    adapter_id: str
    template_id: str
    judge_model: Optional[str] = None


class RunEvalResponse(BaseModel):
    template_id: str
    template_version: int
    base_model: str
    candidate_model: str
    judge_model: str
    avg_score: float
    num_cases: int
    results: List[Dict[str, Any]]

