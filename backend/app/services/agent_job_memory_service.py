"""
Agent Job Memory Service.

Integrates the memory system with autonomous agent jobs, enabling:
- Memory extraction from completed job results
- Memory injection into new jobs for context
- Cross-job memory sharing
- Memory sharing between jobs and chat sessions
"""

import json
import re
from collections import Counter
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from loguru import logger

from app.models.memory import ConversationMemory, UserPreferences
from app.models.agent_job import AgentJob
from app.services.llm_service import LLMService, UserLLMSettings


# Prompts for LLM-based memory operations
EXTRACT_MEMORIES_PROMPT = """You are an expert analyst extracting valuable insights from an autonomous agent job's results.

Analyze the job details below and extract key memories that would be valuable for future jobs.

Job Name: {job_name}
Job Type: {job_type}
Goal: {goal}
Status: {status}

Results Summary:
{results_summary}

Key Findings:
{findings}

Actions Taken:
{actions}

Errors Encountered:
{errors}

Extract memories in the following categories. For each memory, provide:
- TYPE: finding | insight | pattern | lesson
- CONTENT: The memory content (1-2 sentences, factual and actionable)
- IMPORTANCE: 0.0 to 1.0 (higher = more important)
- TAGS: Comma-separated relevant tags

Categories:
1. FINDING: Key factual discoveries from the job
2. INSIGHT: Strategic insights derived from the results
3. PATTERN: Recurring patterns identified
4. LESSON: Lessons learned from the execution (especially from errors)

Output format (one per line):
TYPE: <type> | CONTENT: <content> | IMPORTANCE: <score> | TAGS: <tags>

Extract 3-8 high-value memories. Focus on actionable, reusable knowledge."""


RANK_MEMORIES_PROMPT = """You are evaluating the relevance of memories for an autonomous agent job.

Job Goal: {goal}
Job Type: {job_type}

Rank these memories by relevance to this job (most relevant first).
Return ONLY the memory IDs in order, one per line.

Memories:
{memories}

Output the IDs in order of relevance (most relevant first), one per line:"""


class AgentJobMemoryService:
    """Service for managing memories related to autonomous agent jobs."""

    # Memory types specific to agent jobs
    JOB_MEMORY_TYPES = ["finding", "insight", "pattern", "lesson"]
    _GRAPH_STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
        "are", "was", "were", "will", "can", "could", "should", "would", "have", "has",
        "had", "your", "you", "our", "their", "they", "them", "job", "jobs", "agent",
        "agents", "result", "results", "finding", "findings", "analysis", "research",
        "lesson", "insight", "pattern", "data", "model", "models", "tool", "tools",
    }

    def __init__(self):
        self.llm_service = LLMService()

    def _extract_project_scope(self, job: AgentJob) -> str:
        """Resolve a stable project/customer scope marker from job config."""
        cfg = job.config if isinstance(job.config, dict) else {}
        for key in ["project_id", "project", "project_name", "customer", "team", "workspace", "repo", "repository", "domain"]:
            val = str(cfg.get(key) or "").strip()
            if val:
                return val[:120]
        return ""

    def _resolve_job_role(self, job: AgentJob) -> str:
        """Best-effort role extraction for swarm/specialized jobs."""
        cfg = job.config if isinstance(job.config, dict) else {}
        role = str(cfg.get("agent_role") or cfg.get("swarm_role") or cfg.get("role") or "").strip().lower()
        role = role.replace("-", "_").replace(" ", "_")
        return role[:80]

    def _execution_outcome(self, job: AgentJob) -> str:
        """Map job status to a compact outcome label."""
        status = str(job.status or "").strip().lower()
        if status == "completed":
            return "success"
        if status == "failed":
            return "failure"
        if status == "cancelled":
            return "cancelled"
        return "partial"

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize memory text for lightweight graph similarity scoring."""
        raw = re.findall(r"[a-zA-Z0-9_\\-]+", str(text or "").lower())
        out: set[str] = set()
        for tok in raw:
            tok = tok.strip("_-")
            if len(tok) < 3 or tok in self._GRAPH_STOPWORDS:
                continue
            out.add(tok[:40])
        return out

    def _normalize_tags(self, tags: Any) -> set[str]:
        """Normalize memory tags into a lowercase set."""
        if not isinstance(tags, list):
            return set()
        out: set[str] = set()
        for t in tags:
            s = str(t or "").strip().lower()
            if s:
                out.add(s[:80])
        return out

    def _score_memory_link(
        self,
        left: ConversationMemory,
        right: ConversationMemory,
    ) -> tuple[float, list[str]]:
        """Compute weighted link score between two memories."""
        if str(left.id) == str(right.id):
            return 0.0, []

        left_tags = self._normalize_tags(left.tags)
        right_tags = self._normalize_tags(right.tags)
        shared_tags = sorted(list(left_tags & right_tags))

        score = 0.0
        reasons: list[str] = []
        if shared_tags:
            score += min(3.0, 0.75 * len(shared_tags))
            reasons.append(f"shared_tags:{','.join(shared_tags[:3])}")

        left_tokens = self._tokenize(left.content)
        right_tokens = self._tokenize(right.content)
        if left_tokens and right_tokens:
            overlap = left_tokens & right_tokens
            union = left_tokens | right_tokens
            if union:
                jaccard = float(len(overlap)) / float(len(union))
                if jaccard >= 0.10:
                    score += min(2.5, jaccard * 4.0)
                    if overlap:
                        reasons.append(f"topic_overlap:{','.join(sorted(list(overlap))[:3])}")

        left_ctx = left.context if isinstance(left.context, dict) else {}
        right_ctx = right.context if isinstance(right.context, dict) else {}
        left_scope = str(left_ctx.get("project_scope") or "").strip().lower()
        right_scope = str(right_ctx.get("project_scope") or "").strip().lower()
        if left_scope and right_scope and left_scope == right_scope:
            score += 0.9
            reasons.append("same_scope")

        left_outcome = str(left_ctx.get("execution_outcome") or "").strip().lower()
        right_outcome = str(right_ctx.get("execution_outcome") or "").strip().lower()
        if left_outcome and right_outcome and left_outcome == right_outcome:
            score += 0.5
            reasons.append(f"same_outcome:{left_outcome}")

        if str(left.job_id or "") and str(left.job_id or "") == str(right.job_id or ""):
            score += 0.8
            reasons.append("same_job")

        if score < 1.0:
            return 0.0, []
        return round(score, 4), reasons[:4]

    async def link_memories_into_task_graph(
        self,
        new_memories: List[ConversationMemory],
        user_id: str,
        db: AsyncSession,
        *,
        max_links_per_memory: int = 5,
        candidate_limit: int = 180,
        min_link_score: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Build and persist graph links for new memories.

        Links are stored in each memory's `context.task_graph_links`.
        """
        if not new_memories:
            return {"linked_memories": 0, "links_created": 0}

        memory_ids = {str(m.id) for m in new_memories}
        result = await db.execute(
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.is_active == True,
                )
            )
            .order_by(desc(ConversationMemory.importance_score), desc(ConversationMemory.created_at))
            .limit(max(50, min(int(candidate_limit or 180), 500)))
        )
        candidates = [m for m in result.scalars().all() if str(m.id) not in memory_ids]
        links_created = 0
        now_iso = datetime.utcnow().isoformat()

        for memory in new_memories:
            scored: list[tuple[ConversationMemory, float, list[str]]] = []
            pool = [m for m in new_memories if str(m.id) != str(memory.id)] + candidates
            for other in pool:
                score, reasons = self._score_memory_link(memory, other)
                if score >= float(min_link_score):
                    scored.append((other, score, reasons))
            scored.sort(key=lambda row: (-float(row[1]), str(row[0].id)))
            best = scored[: max(1, min(int(max_links_per_memory or 5), 20))]

            context = memory.context if isinstance(memory.context, dict) else {}
            context["task_graph_links"] = [
                {
                    "memory_id": str(other.id),
                    "score": float(score),
                    "reasons": reasons,
                    "linked_at": now_iso,
                }
                for other, score, reasons in best
            ]
            context["task_graph_degree"] = len(context["task_graph_links"])
            memory.context = context
            links_created += len(best)

        await db.commit()
        return {
            "linked_memories": len(new_memories),
            "links_created": int(links_created),
        }

    def _connected_components_count(self, node_ids: list[str], edges: list[dict]) -> int:
        """Estimate connected component count for graph stats."""
        if not node_ids:
            return 0
        adjacency: Dict[str, set[str]] = {nid: set() for nid in node_ids}
        for e in edges:
            s = str(e.get("source") or "").strip()
            t = str(e.get("target") or "").strip()
            if s in adjacency and t in adjacency:
                adjacency[s].add(t)
                adjacency[t].add(s)

        seen: set[str] = set()
        components = 0
        for node in node_ids:
            if node in seen:
                continue
            components += 1
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend([n for n in adjacency.get(cur, set()) if n not in seen])
        return components

    async def get_task_memory_graph(
        self,
        user_id: str,
        db: AsyncSession,
        *,
        limit: int = 120,
        min_link_score: float = 1.0,
        max_edges: int = 800,
    ) -> Dict[str, Any]:
        """Build a lightweight task-memory graph across jobs/projects."""
        lim = max(20, min(int(limit or 120), 300))
        edge_cap = max(50, min(int(max_edges or 800), 3000))
        threshold = max(0.2, min(float(min_link_score or 1.0), 10.0))

        result = await db.execute(
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.is_active == True,
                )
            )
            .order_by(desc(ConversationMemory.importance_score), desc(ConversationMemory.created_at))
            .limit(lim)
        )
        memories = list(result.scalars().all())
        nodes: list[dict[str, Any]] = []
        for m in memories:
            context = m.context if isinstance(m.context, dict) else {}
            nodes.append(
                {
                    "id": str(m.id),
                    "type": str(m.memory_type or ""),
                    "content": str(m.content or "")[:280],
                    "importance_score": float(m.importance_score or 0.0),
                    "tags": m.tags if isinstance(m.tags, list) else [],
                    "job_id": str(m.job_id) if m.job_id else None,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "project_scope": str(context.get("project_scope") or "") or None,
                    "execution_outcome": str(context.get("execution_outcome") or "") or None,
                    "strategy_signal": str(context.get("strategy_signal") or "") or None,
                    "access_count": int(m.access_count or 0),
                }
            )

        edges: list[dict[str, Any]] = []
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                score, reasons = self._score_memory_link(memories[i], memories[j])
                if score < threshold:
                    continue
                edges.append(
                    {
                        "source": str(memories[i].id),
                        "target": str(memories[j].id),
                        "weight": float(score),
                        "reasons": reasons,
                    }
                )
        edges.sort(key=lambda e: (-float(e.get("weight", 0.0)), str(e.get("source")), str(e.get("target"))))
        edges = edges[:edge_cap]

        node_ids = [str(n.get("id")) for n in nodes if str(n.get("id"))]
        components = self._connected_components_count(node_ids, edges)
        avg_degree = 0.0
        if nodes:
            avg_degree = (2.0 * float(len(edges))) / float(len(nodes))

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "memory_count": len(nodes),
                "edge_count": len(edges),
                "connected_components": int(components),
                "average_degree": round(avg_degree, 4),
                "min_link_score": threshold,
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def extract_feedback_learning_signals(
        self,
        memories: List[ConversationMemory],
        *,
        job_type: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Derive tool and prompt biases from human feedback memories."""
        if not memories:
            return {}

        preferred = Counter()
        discouraged = Counter()
        notes: list[str] = []
        ratings: list[int] = []
        feedback_count = 0

        role_filter = str(role or "").strip().lower().replace("-", "_")
        job_type_filter = str(job_type or "").strip().lower()

        for memory in memories:
            tags = self._normalize_tags(memory.tags)
            context = memory.context if isinstance(memory.context, dict) else {}
            is_human_feedback = ("human_feedback" in tags) or str(context.get("feedback_type") or "").strip().lower() == "human"
            if not is_human_feedback:
                continue

            ctx_job_type = str(context.get("job_type") or "").strip().lower()
            if job_type_filter and ctx_job_type and ctx_job_type != job_type_filter:
                continue
            ctx_role = str(context.get("agent_role") or "").strip().lower().replace("-", "_")
            if role_filter and ctx_role and ctx_role != role_filter:
                continue

            feedback_count += 1
            try:
                rating = int(context.get("rating", 0) or 0)
            except Exception:
                rating = 0
            if rating > 0:
                ratings.append(max(1, min(rating, 5)))

            prefer_tools = context.get("preferred_tools") if isinstance(context.get("preferred_tools"), list) else []
            avoid_tools = context.get("discouraged_tools") if isinstance(context.get("discouraged_tools"), list) else []

            for t in list(tags):
                if t.startswith("prefer_tool:"):
                    prefer_tools.append(t.split(":", 1)[1])
                if t.startswith("avoid_tool:"):
                    avoid_tools.append(t.split(":", 1)[1])

            for tool in prefer_tools:
                name = str(tool or "").strip()
                if not name:
                    continue
                preferred[name] += max(1, rating - 2) if rating > 0 else 1
            for tool in avoid_tools:
                name = str(tool or "").strip()
                if not name:
                    continue
                discouraged[name] += max(1, 4 - rating) if rating > 0 else 1

            note = str(context.get("feedback_text") or memory.content or "").strip()
            if note:
                notes.append(note[:220])

        tool_bias_raw: Dict[str, float] = {}
        for tool in set(list(preferred.keys()) + list(discouraged.keys())):
            pref = float(preferred.get(tool, 0) or 0)
            avoid = float(discouraged.get(tool, 0) or 0)
            score = pref - (1.15 * avoid)
            if score == 0:
                continue
            tool_bias_raw[tool] = score

        max_abs = max([abs(v) for v in tool_bias_raw.values()], default=1.0)
        tool_bias = {
            tool: round(max(-1.0, min(1.0, val / max_abs)), 4)
            for tool, val in tool_bias_raw.items()
        }

        preferred_tools = [k for k, _v in preferred.most_common(8)]
        discouraged_tools = [k for k, _v in discouraged.most_common(8)]
        avg_rating = round(sum(ratings) / float(len(ratings)), 3) if ratings else None

        return {
            "feedback_count": feedback_count,
            "avg_rating": avg_rating,
            "preferred_tools": preferred_tools,
            "discouraged_tools": discouraged_tools,
            "tool_bias": tool_bias,
            "highlights": notes[:5],
        }

    async def get_user_preferences(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> UserPreferences:
        """Get or create user preferences."""
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        prefs = result.scalar_one_or_none()

        if not prefs:
            prefs = UserPreferences(
                user_id=user_id,
                agent_job_memory_types=self.JOB_MEMORY_TYPES,
                max_job_memories=10,
                auto_extract_job_memories=True,
                share_memories_with_chat=True,
            )
            db.add(prefs)
            await db.commit()
            await db.refresh(prefs)

        return prefs

    async def extract_memories_from_job(
        self,
        job: AgentJob,
        user_id: str,
        db: AsyncSession
    ) -> List[ConversationMemory]:
        """
        Extract valuable memories from a completed agent job.

        Uses LLM to analyze job results and extract:
        - Key findings
        - Strategic insights
        - Patterns identified
        - Lessons learned

        Args:
            job: The completed AgentJob
            user_id: User ID string
            db: Database session

        Returns:
            List of created ConversationMemory objects
        """
        prefs = await self.get_user_preferences(UUID(user_id), db)

        if not prefs.auto_extract_job_memories:
            logger.info(f"Auto-extract disabled for user {user_id}, skipping memory extraction")
            return []

        # Build job context for LLM
        results_summary = ""
        findings = ""
        actions = ""
        errors = ""

        if job.results:
            results_summary = job.results.get("summary", "No summary available")
            if job.results.get("findings"):
                findings = "\n".join([
                    f"- {f.get('title', 'Finding')}: {f.get('content', '')}"
                    for f in job.results.get("findings", [])[:10]
                ])
            if job.results.get("actions"):
                actions = "\n".join([
                    f"- {a.get('action', 'Action')}: {a.get('result', '')}"
                    for a in job.results.get("actions", [])[:10]
                ])

        if job.error:
            errors = job.error
        elif job.execution_log:
            error_entries = [
                entry for entry in job.execution_log
                if entry.get("error")
            ][-5:]
            if error_entries:
                errors = "\n".join([e.get("error", "") for e in error_entries])

        if not findings:
            findings = "No specific findings recorded"
        if not actions:
            actions = "No specific actions recorded"
        if not errors:
            errors = "No errors encountered"

        prompt = EXTRACT_MEMORIES_PROMPT.format(
            job_name=job.name,
            job_type=job.job_type,
            goal=job.goal,
            status=job.status,
            results_summary=results_summary,
            findings=findings,
            actions=actions,
            errors=errors,
        )

        try:
            # Get user LLM settings
            llm_settings = UserLLMSettings.from_preferences(prefs) if prefs else None

            response = await self.llm_service.generate_response(
                prompt=prompt,
                system_prompt="You are an expert analyst extracting valuable memories from job results.",
                user_id=user_id,
                user_llm_settings=llm_settings,
            )

            memories = self._parse_extracted_memories(response, job, user_id)

            # Store memories
            created_memories = []
            allowed_types = prefs.agent_job_memory_types or self.JOB_MEMORY_TYPES

            outcome = self._execution_outcome(job)
            project_scope = self._extract_project_scope(job)
            agent_role = self._resolve_job_role(job)

            for memory_data in memories:
                if memory_data["type"] not in allowed_types:
                    continue

                memory_tags = list(memory_data["tags"] or [])
                memory_tags.append(f"outcome:{outcome}")
                if project_scope:
                    memory_tags.append(f"scope:{project_scope.lower()[:60]}")
                if agent_role:
                    memory_tags.append(f"role:{agent_role}")
                if outcome == "success" and memory_data["type"] in {"insight", "pattern", "lesson"}:
                    memory_tags.append("successful_strategy")
                if outcome == "failure" and memory_data["type"] in {"pattern", "lesson"}:
                    memory_tags.append("failed_path")

                strategy_signal = "neutral"
                if "successful_strategy" in memory_tags:
                    strategy_signal = "successful_strategy"
                elif "failed_path" in memory_tags:
                    strategy_signal = "failed_path"

                memory = ConversationMemory(
                    user_id=UUID(user_id),
                    job_id=job.id,
                    memory_type=memory_data["type"],
                    content=memory_data["content"],
                    importance_score=memory_data["importance"],
                    tags=list(set(memory_tags)),
                    context={
                        "job_name": job.name,
                        "job_type": job.job_type,
                        "job_status": job.status,
                        "execution_outcome": outcome,
                        "project_scope": project_scope,
                        "agent_role": agent_role,
                        "strategy_signal": strategy_signal,
                        "extracted_at": datetime.utcnow().isoformat(),
                    },
                )
                db.add(memory)
                created_memories.append(memory)

            # Update job memory count
            job.memories_created_count = len(created_memories)
            await db.commit()

            for memory in created_memories:
                await db.refresh(memory)

            try:
                await self.link_memories_into_task_graph(
                    new_memories=created_memories,
                    user_id=user_id,
                    db=db,
                )
            except Exception as graph_exc:
                logger.warning(f"Failed to link extracted memories in task graph for job {job.id}: {graph_exc}")

            logger.info(f"Extracted {len(created_memories)} memories from job {job.id}")
            return created_memories

        except Exception as e:
            logger.error(f"Failed to extract memories from job {job.id}: {e}")
            await db.rollback()
            return []

    def _parse_extracted_memories(
        self,
        llm_response: str,
        job: AgentJob,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into memory data."""
        memories = []

        for line in llm_response.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("TYPE:"):
                continue

            try:
                parts = {}
                for part in line.split("|"):
                    part = part.strip()
                    if ":" in part:
                        key, value = part.split(":", 1)
                        parts[key.strip().lower()] = value.strip()

                if "type" in parts and "content" in parts:
                    memory_type = parts["type"].lower()
                    if memory_type not in self.JOB_MEMORY_TYPES:
                        continue

                    importance = 0.5
                    try:
                        importance = float(parts.get("importance", "0.5"))
                        importance = max(0.0, min(1.0, importance))
                    except ValueError:
                        pass

                    tags = []
                    if "tags" in parts:
                        tags = [t.strip() for t in parts["tags"].split(",") if t.strip()]

                    # Add job-related tags
                    tags.extend([job.job_type, f"job:{job.name[:30]}"])

                    memories.append({
                        "type": memory_type,
                        "content": parts["content"],
                        "importance": importance,
                        "tags": list(set(tags)),
                    })

            except Exception as e:
                logger.warning(f"Failed to parse memory line: {line} - {e}")
                continue

        return memories

    async def get_relevant_memories_for_job(
        self,
        job: AgentJob,
        user_id: str,
        db: AsyncSession,
        limit: Optional[int] = None
    ) -> List[ConversationMemory]:
        """
        Get memories relevant to a job's goal.

        Retrieves and ranks memories based on relevance to the job.

        Args:
            job: The AgentJob to find memories for
            user_id: User ID string
            db: Database session
            limit: Max memories to return (uses user pref if not specified)

        Returns:
            List of relevant ConversationMemory objects
        """
        prefs = await self.get_user_preferences(UUID(user_id), db)

        if not prefs.enable_agent_memory:
            return []

        max_memories = limit or prefs.max_job_memories or 10

        # Get candidate memories
        # Include both job-specific types and general types if sharing enabled
        memory_types = list(prefs.agent_job_memory_types or self.JOB_MEMORY_TYPES)
        if prefs.share_memories_with_chat:
            memory_types.extend(["fact", "preference", "context"])

        query = (
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.is_active == True,
                    ConversationMemory.memory_type.in_(memory_types),
                )
            )
            .order_by(desc(ConversationMemory.importance_score))
            .limit(50)  # Get more than needed for ranking
        )

        result = await db.execute(query)
        candidates = list(result.scalars().all())

        if not candidates:
            return []

        if len(candidates) <= max_memories:
            # Update access tracking
            await self._update_memory_access(candidates, db)
            return candidates

        # Use LLM to rank by relevance
        try:
            ranked = await self._rank_memories_by_relevance(
                job, candidates, user_id, prefs, db
            )
            selected = ranked[:max_memories]
            await self._update_memory_access(selected, db)
            return selected
        except Exception as e:
            logger.warning(f"Failed to rank memories, using importance: {e}")
            selected = candidates[:max_memories]
            await self._update_memory_access(selected, db)
            return selected

    async def _rank_memories_by_relevance(
        self,
        job: AgentJob,
        memories: List[ConversationMemory],
        user_id: str,
        prefs: UserPreferences,
        db: AsyncSession
    ) -> List[ConversationMemory]:
        """Use LLM to rank memories by relevance to job."""
        memory_texts = "\n".join([
            f"ID: {m.id} | TYPE: {m.memory_type} | CONTENT: {m.content[:200]}"
            for m in memories
        ])

        prompt = RANK_MEMORIES_PROMPT.format(
            goal=job.goal,
            job_type=job.job_type,
            memories=memory_texts,
        )

        llm_settings = UserLLMSettings.from_preferences(prefs) if prefs else None

        response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt="You are ranking memories by relevance.",
            user_id=user_id,
            user_llm_settings=llm_settings,
        )

        # Parse ranked IDs
        memory_map = {str(m.id): m for m in memories}
        ranked = []

        for line in response.strip().split("\n"):
            line = line.strip()
            # Extract UUID from line
            for mid in memory_map.keys():
                if mid in line:
                    if mid not in [str(m.id) for m in ranked]:
                        ranked.append(memory_map[mid])
                    break

        # Add any missing memories at the end
        for memory in memories:
            if memory not in ranked:
                ranked.append(memory)

        return ranked

    async def _update_memory_access(
        self,
        memories: List[ConversationMemory],
        db: AsyncSession
    ):
        """Update access count and timestamp for memories."""
        now = datetime.utcnow()
        for memory in memories:
            memory.access_count += 1
            memory.last_accessed_at = now
        await db.commit()

    def format_memories_for_job_context(
        self,
        memories: List[ConversationMemory],
        include_metadata: bool = False
    ) -> str:
        """
        Format memories for injection into job context.

        Args:
            memories: List of memories to format
            include_metadata: Include importance scores and tags

        Returns:
            Formatted string for job context
        """
        if not memories:
            return ""

        lines = [
            "## Relevant Memories from Past Jobs",
            "",
            "Use these memories to inform your approach:",
            ""
        ]

        for i, memory in enumerate(memories, 1):
            type_label = memory.memory_type.upper()
            if include_metadata:
                lines.append(
                    f"{i}. [{type_label}] {memory.content} "
                    f"(importance: {memory.importance_score:.1f})"
                )
                if memory.tags:
                    lines.append(f"   Tags: {', '.join(memory.tags)}")
            else:
                lines.append(f"{i}. [{type_label}] {memory.content}")

        lines.append("")
        lines.append(
            "Consider these insights when planning and executing your tasks."
        )

        return "\n".join(lines)

    async def get_job_memories(
        self,
        job_id: UUID,
        user_id: str,
        db: AsyncSession
    ) -> List[ConversationMemory]:
        """
        Get all memories created from a specific job.

        Args:
            job_id: The job ID
            user_id: User ID string
            db: Database session

        Returns:
            List of ConversationMemory objects from this job
        """
        query = (
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.job_id == job_id,
                    ConversationMemory.is_active == True,
                )
            )
            .order_by(desc(ConversationMemory.importance_score))
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_memories_by_type(
        self,
        user_id: str,
        memory_type: str,
        db: AsyncSession,
        limit: int = 20
    ) -> List[ConversationMemory]:
        """
        Get memories of a specific type.

        Args:
            user_id: User ID string
            memory_type: Memory type to filter by
            db: Database session
            limit: Max memories to return

        Returns:
            List of ConversationMemory objects
        """
        query = (
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.memory_type == memory_type,
                    ConversationMemory.is_active == True,
                )
            )
            .order_by(desc(ConversationMemory.importance_score))
            .limit(limit)
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    async def create_memory_from_job(
        self,
        job: AgentJob,
        memory_type: str,
        content: str,
        user_id: str,
        db: AsyncSession,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> ConversationMemory:
        """
        Manually create a memory from a job.

        Args:
            job: Source job
            memory_type: Type of memory
            content: Memory content
            user_id: User ID string
            db: Database session
            importance: Importance score (0.0-1.0)
            tags: Optional tags

        Returns:
            Created ConversationMemory
        """
        outcome = self._execution_outcome(job)
        project_scope = self._extract_project_scope(job)
        agent_role = self._resolve_job_role(job)
        memory = ConversationMemory(
            user_id=UUID(user_id),
            job_id=job.id,
            memory_type=memory_type,
            content=content,
            importance_score=max(0.0, min(1.0, importance)),
            tags=list(
                set(
                    (tags or [job.job_type])
                    + [f"outcome:{outcome}"]
                    + ([f"scope:{project_scope.lower()[:60]}"] if project_scope else [])
                )
            ),
            context={
                "job_name": job.name,
                "job_type": job.job_type,
                "execution_outcome": outcome,
                "project_scope": project_scope,
                "agent_role": agent_role,
                "created_manually": True,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

        db.add(memory)
        await db.commit()
        await db.refresh(memory)

        # Update job memory count
        job.memories_created_count = (job.memories_created_count or 0) + 1
        await db.commit()

        try:
            await self.link_memories_into_task_graph(
                new_memories=[memory],
                user_id=user_id,
                db=db,
            )
            await db.refresh(memory)
        except Exception as graph_exc:
            logger.warning(f"Failed to link manual memory {memory.id} into task graph: {graph_exc}")

        logger.info(f"Created manual memory {memory.id} from job {job.id}")
        return memory

    async def delete_job_memories(
        self,
        job_id: UUID,
        user_id: str,
        db: AsyncSession
    ) -> int:
        """
        Soft delete all memories from a job.

        Args:
            job_id: Job ID
            user_id: User ID string
            db: Database session

        Returns:
            Number of memories deleted
        """
        query = (
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == UUID(user_id),
                    ConversationMemory.job_id == job_id,
                    ConversationMemory.is_active == True,
                )
            )
        )

        result = await db.execute(query)
        memories = list(result.scalars().all())

        for memory in memories:
            memory.is_active = False

        await db.commit()

        logger.info(f"Soft deleted {len(memories)} memories from job {job_id}")
        return len(memories)

    async def get_memory_stats_for_user(
        self,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get memory statistics for a user.

        Args:
            user_id: User ID string
            db: Database session

        Returns:
            Dictionary with memory statistics
        """
        uid = UUID(user_id)

        # Total memories
        total_query = (
            select(ConversationMemory)
            .where(
                and_(
                    ConversationMemory.user_id == uid,
                    ConversationMemory.is_active == True,
                )
            )
        )
        total_result = await db.execute(total_query)
        all_memories = list(total_result.scalars().all())

        # Count by type
        type_counts = {}
        job_sourced = 0
        chat_sourced = 0

        for memory in all_memories:
            type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1
            if memory.job_id:
                job_sourced += 1
            elif memory.session_id:
                chat_sourced += 1

        # Most accessed
        most_accessed = sorted(
            all_memories,
            key=lambda m: m.access_count,
            reverse=True
        )[:5]

        # Most important
        most_important = sorted(
            all_memories,
            key=lambda m: m.importance_score,
            reverse=True
        )[:5]

        return {
            "total_memories": len(all_memories),
            "by_type": type_counts,
            "job_sourced": job_sourced,
            "chat_sourced": chat_sourced,
            "manual": len(all_memories) - job_sourced - chat_sourced,
            "most_accessed": [
                {
                    "id": str(m.id),
                    "type": m.memory_type,
                    "content": m.content[:100],
                    "access_count": m.access_count,
                }
                for m in most_accessed
            ],
            "most_important": [
                {
                    "id": str(m.id),
                    "type": m.memory_type,
                    "content": m.content[:100],
                    "importance": m.importance_score,
                }
                for m in most_important
            ],
        }


# Singleton instance
agent_job_memory_service = AgentJobMemoryService()
