"""Pure recommendation policy for autonomous-job templates."""

from typing import Optional

from app.schemas.agent_job import AgentJobTemplateResponse


def score_template_recommendation(
    template: AgentJobTemplateResponse,
    *,
    category: Optional[str],
    recommend_goal: Optional[str],
    recommend_scope: Optional[str],
) -> tuple[int, list[str]]:
    """Rank a template against lightweight operator intent signals."""

    score = 0
    reasons: list[str] = []
    name = str(template.name or "").strip().lower()
    display_name = str(template.display_name or "").strip().lower()
    template_category = str(template.category or "").strip().lower()
    config = (
        template.default_config if isinstance(template.default_config, dict) else {}
    )
    runner = str(config.get("deterministic_runner") or "").strip().lower()

    if (
        category
        and template_category
        and template_category == str(category).strip().lower()
    ):
        score += 10
        reasons.append("matches_category")

    goal_text = str(recommend_goal or "").strip().lower()
    scope_text = str(recommend_scope or "").strip().lower()
    context = f"{goal_text} {scope_text}".strip()
    backend_context = any(
        signal in context
        for signal in (
            "backend",
            "api",
            "server",
            "fastapi",
            "flask",
            "django",
            "pytest",
        )
    )
    code_context = any(
        signal in context
        for signal in (
            "code",
            "patch",
            "refactor",
            "test",
            "bug",
            "fix",
            "implementation",
        )
    )
    latex_context = any(
        signal in context for signal in ("latex", "paper", "citation", "bibtex")
    )

    if backend_context and name == "claude_code_backend":
        score += 80
        reasons.append("backend_loop_specialized")
    if code_context and template_category == "code":
        score += 20
        reasons.append("code_category_fit")
    if backend_context and runner in {"code_patch_proposer", "experiment_runner"}:
        score += 15
        reasons.append("backend_code_runner_fit")
    if backend_context and (
        "backend" in display_name
        or "backend" in str(template.default_goal or "").lower()
    ):
        score += 10
        reasons.append("backend_goal_fit")
    if latex_context and template_category == "latex":
        score += 30
        reasons.append("latex_category_fit")
    if not category and not context and template.is_system:
        score += 2
        reasons.append("system_default")

    return score, reasons[:4]
