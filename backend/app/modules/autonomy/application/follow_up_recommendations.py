"""Build and rank bounded follow-up recommendations for research inbox items."""

import re
from datetime import datetime
from typing import Any

from app.models.research_inbox import ResearchInboxItem
from app.schemas.agent_job import AgentCheckpointQueueActionResponse
from app.services.agent_job_chain_templates import (
    ARXIV_REPO_CODE_PATCH_CHAIN_ID,
    CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID,
)

FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN = "deep_dive_chain"
FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB = "single_research_job"
FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN = "repo_patch_chain"

_LEARNING_STOP_WORDS = {
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


def tokenize_learning_text(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower())
    tokens: list[str] = []
    for token in raw:
        token = token.strip("_-")
        if len(token) < 3 or token in _LEARNING_STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def score_follow_up_action(
    item: ResearchInboxItem,
    action: AgentCheckpointQueueActionResponse,
    *,
    learning_profile: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    recommendation_key = str(action.recommendation_key or "").strip()
    item_type = str(item.item_type or "").strip().lower()
    text = f"{str(item.title or '').strip()} {str(item.summary or '').strip()}".strip()
    tokens = tokenize_learning_text(text)
    phrases = [
        f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)
    ]

    profile = learning_profile if isinstance(learning_profile, dict) else {}
    token_scores = profile.get("token_scores") or {}
    phrase_scores = profile.get("phrase_scores") or {}
    recommendation_scores = profile.get("recommendation_scores") or {}
    source_type_scores = profile.get("source_type_scores") or {}

    if recommendation_key and recommendation_key in recommendation_scores:
        delta = int(recommendation_scores.get(recommendation_key) or 0)
        score += delta * 10
        reasons.append(f"learned_recommendation:{delta}")
    if item_type and item_type in source_type_scores:
        delta = int(source_type_scores.get(item_type) or 0)
        score += delta * 6
        reasons.append(f"source_type:{item_type}:{delta}")

    token_delta = sum(int(token_scores.get(token) or 0) for token in tokens[:10])
    if token_delta:
        score += token_delta
        reasons.append("token_bias")
    phrase_delta = sum(int(phrase_scores.get(phrase) or 0) for phrase in phrases[:6])
    if phrase_delta:
        score += phrase_delta * 2
        reasons.append("phrase_bias")

    if recommendation_key == FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN:
        score += 24
        reasons.append("deep_dive_default")
        if item_type == "arxiv":
            score += 4
            reasons.append("paper_deep_dive_fit")
    elif recommendation_key == FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB:
        score += 18
        reasons.append("single_job_default")
        if item_type == "document":
            score += 3
            reasons.append("document_single_job_fit")
    elif recommendation_key == FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN:
        score += 8
        reasons.append("repo_patch_specialized")
        metadata = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        repos = metadata.get("repos") if isinstance(metadata.get("repos"), list) else []
        if repos:
            score += 14
            reasons.append("repos_present")
        if item_type == "arxiv":
            score += 5
            reasons.append("paper_repo_fit")

    if str(action.autonomy_eligibility or "").strip().lower() == "auto_launchable":
        score += 5
        reasons.append("safe_autonomy_eligible")
    return int(score), reasons[:5]


def build_follow_up_actions(
    item: ResearchInboxItem,
    *,
    learning_profile: dict[str, Any] | None = None,
) -> list[AgentCheckpointQueueActionResponse]:
    """Return bounded follow-up launch recommendations for an accepted inbox item."""
    title = str(item.title or "").strip() or "Selected research signal"
    summary = str(item.summary or "").strip()
    customer = str(item.customer or "").strip()
    customer_hint = f"Customer: {customer}" if customer else ""
    item_type = str(item.item_type or "").strip().lower()
    inbox_item_payload = {
        "id": str(item.id),
        "item_type": str(item.item_type or "").strip(),
        "item_key": str(item.item_key or "").strip(),
        "title": title,
        "url": str(item.url or "").strip() or None,
        "summary": summary or None,
        "customer": customer or None,
    }
    source = {
        "id": str(item.item_key),
        "title": title,
        "url": str(item.url or "").strip() or None,
        "score": None,
        "source": "inbox",
    }
    top_papers = [source] if item_type == "arxiv" else []
    top_documents = [] if top_papers else [source]
    inherited_data = {
        "parent_results": {
            "summary": f"Seeded from accepted Research Inbox item: {title}",
            "research_bundle": {
                "top_documents": top_documents,
                "top_papers": top_papers,
                "insights": [],
                "next_steps": [],
                "artifacts": [],
            },
            "inbox_items": [inbox_item_payload],
        },
        "parent_findings": [
            {
                "type": "paper" if top_papers else "document",
                "title": title,
                "id": str(item.item_key),
                "url": str(item.url or "").strip() or None,
                "snippet": summary or None,
            }
        ],
    }
    today = datetime.utcnow().strftime("%Y-%m-%d")
    actions = [
        AgentCheckpointQueueActionResponse(
            kind="launch_chain",
            label="Launch Deep Dive",
            description="Start the recommended scout-to-deep-dive chain with this accepted signal preloaded.",
            recommended=True,
            launch_label="Deep Dive Chain",
            recommendation_key=FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN,
            autonomy_eligibility="auto_launchable",
            chain_create_payload={
                "chain_definition_id": str(CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID),
                "name_prefix": f"Inbox Research - {today}",
                "variables": {
                    "goal": f"Deep-dive on {title} and propose concrete next steps."
                },
                "config_overrides": {
                    "customer_context": customer_hint,
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 12,
                    "max_papers": 8,
                    "persist_artifacts": False,
                    "reading_list_name": "Customer Research",
                    "inherited_data": inherited_data,
                },
                "start_immediately": True,
            },
        ),
        AgentCheckpointQueueActionResponse(
            kind="launch_job",
            label="Launch Single Research Job",
            description="Create one bounded research job instead of a full chain.",
            launch_label="Single Research Job",
            recommendation_key=FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB,
            autonomy_eligibility="auto_launchable",
            job_create_payload={
                "name": f"Inbox Research - {today}",
                "job_type": "research",
                "goal": f"Deep-dive on {title} and propose concrete next steps.",
                "config": {
                    "customer_context": customer_hint,
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 12,
                    "max_papers": 8,
                    "persist_artifacts": False,
                    "reading_list_name": "Customer Research",
                    "inherited_data": inherited_data,
                },
                "start_immediately": True,
            },
        ),
    ]
    metadata = item.item_metadata if isinstance(item.item_metadata, dict) else {}
    repos = metadata.get("repos") if isinstance(metadata.get("repos"), list) else []
    if item_type == "arxiv" and repos:
        actions.append(
            AgentCheckpointQueueActionResponse(
                kind="launch_chain",
                label="Launch Repo -> Patch Chain",
                description="Use extracted repository links to move from paper to repo ingest and a patch proposal.",
                launch_label="Repo -> Patch Chain",
                recommendation_key=FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN,
                autonomy_eligibility="manual_only",
                chain_create_payload={
                    "chain_definition_id": str(ARXIV_REPO_CODE_PATCH_CHAIN_ID),
                    "name_prefix": f"Paper Repo Patch - {today}",
                    "variables": {
                        "goal": f"Implement the most relevant change suggested by {title}"
                    },
                    "config_overrides": {
                        "inbox_item_id": str(item.id),
                        "customer_context": customer_hint,
                    },
                    "start_immediately": True,
                },
            )
        )

    for action in actions:
        score, reasons = score_follow_up_action(
            item, action, learning_profile=learning_profile
        )
        action.recommendation_score = score
        action.recommendation_reasons = reasons
        action.recommended = False
    actions.sort(
        key=lambda action: (
            1
            if str(action.autonomy_eligibility or "").strip().lower() == "manual_only"
            else 0,
            -int(action.recommendation_score or 0),
            0
            if str(action.recommendation_key or "")
            == FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN
            else 1,
        )
    )
    if actions:
        actions[0].recommended = True
    return actions
