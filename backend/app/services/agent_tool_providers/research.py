"""Autonomous-job tools: the ``research`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import select

from app.services import llm_structured
from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.agent_tool_providers.common import (
    _load_documents_for_analysis,
    _tool_snapshot_context,
)


def _document_excerpts(documents: list[Any], *, per_doc_chars: int = 4000) -> str:
    """Render documents for a prompt, truncated per document rather than overall
    so a long first document cannot crowd the others out entirely."""
    blocks = []
    for doc in documents:
        content = (getattr(doc, "content", "") or "")[:per_doc_chars]
        blocks.append(
            f"### {getattr(doc, 'title', 'Untitled')} (id={getattr(doc, 'id', '')})\n{content}"
        )
    return "\n\n".join(blocks)


_CLUSTER_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "common_themes": {"type": "array", "items": {"type": "string"}},
        "differences": {"type": "array", "items": {"type": "string"}},
        "patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}


_METHODOLOGY_COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "finding": {"type": "string"},
                },
                "required": ["aspect", "finding"],
            },
        },
        "shared_approaches": {"type": "array", "items": {"type": "string"}},
        "notable_differences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}


_RESEARCH_GAPS_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps_identified": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gap": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "supporting_evidence": {"type": "string"},
                },
                "required": ["gap"],
            },
        },
        "opportunities": {"type": "array", "items": {"type": "string"}},
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": ["gaps_identified"],
}


#: How long to wait for a queued arXiv ingestion to produce a readable
#: document. Ingestion runs in a Celery worker, so the tool that started it
#: cannot know it finished without looking. Long enough for one paper to be
#: fetched and stored; short enough that a dead worker is reported as a failure
#: within an iteration rather than hanging the run.
INGEST_WAIT_SECONDS = 90


_INGEST_POLL_SECONDS = 3


async def _wait_for_ingested_documents(source_id: str) -> List[str]:
    """Document ids that actually landed for this source, or an empty list.

    Empty is a real answer and the caller must treat it as failure. The whole
    reason this exists is that "ingestion started" and "the paper is readable"
    are different facts, and only the second one lets the next stage do its
    job.

    Polls on a session of its OWN, never the caller's, for two reasons that
    each burned a day in this project already:

    * `await db.rollback()` on a borrowed AsyncSession expires every ORM object
      in it -- `expire_on_commit=False` does not cover rollbacks -- so the
      executor's next attribute read becomes IO and raises MissingGreenlet from
      somewhere unrelated. A rollback in a loop that then continues is the
      exact dangerous shape.
    * Holding the caller's session in an open transaction for up to a minute
      and a half is what left the executor idle-in-transaction on the
      `agent_jobs` row while the lease heartbeat's UPDATE blocked behind it.

    Neither is needed anyway: the connection runs at READ COMMITTED, so each
    statement takes a fresh snapshot and sees rows the ingestion worker
    committed after this loop began.
    """
    import asyncio as _asyncio

    from app.core.database import create_celery_session
    from app.models.document import Document

    loop = _asyncio.get_event_loop()
    deadline = loop.time() + INGEST_WAIT_SECONDS
    session_factory = create_celery_session()
    # That builds a fresh engine, and unless CELERY_DB_USE_NULLPOOL is set it
    # is a QueuePool holding real connections. A Celery task creates one per
    # invocation and lives with it; a TOOL can be called many times inside one
    # job, so an undisposed engine per call would walk the worker into
    # connection exhaustion.
    engine = getattr(session_factory, "kw", {}).get("bind")
    try:
        while True:
            try:
                async with session_factory() as poll_db:
                    rows = await poll_db.execute(
                        select(Document.id).where(Document.source_id == source_id)
                    )
                    found = [str(value) for value in rows.scalars().all()]
            except Exception:  # pragma: no cover - defensive
                # An unreadable poll is not an ingestion that succeeded.
                # Returning empty makes the caller report failure, which is the
                # honest reading of "I could not tell".
                return []
            if found:
                return found
            if loop.time() >= deadline:
                return []
            await _asyncio.sleep(_INGEST_POLL_SECONDS)
    finally:
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:  # pragma: no cover - defensive
                pass


def build_autonomous_research_provider(executor: Any) -> FunctionToolProvider:
    """Research-family tools for AutonomousAgentExecutor."""

    async def _arxiv_search(**kwargs: Any) -> List[Dict[str, Any]]:
        """Run an arXiv search and return its entries as plain dicts.

        ``ArxivSearchService.search`` returns an ``ArxivSearchResult`` dataclass;
        every caller here wants the entry list.
        """
        result = await executor.arxiv_service.search(**kwargs)
        items = getattr(result, "items", result) or []
        return [item for item in items if isinstance(item, dict)]

    async def _search_arxiv(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        papers = await _arxiv_search(
            query=params.get("query", job.goal[:100] if job else ""),
            max_results=params.get("max_results", 10),
        )
        return {
            "success": True,
            "data": papers,
            "findings": [
                {
                    "type": "paper",
                    "title": paper.get("title"),
                    "id": paper.get("id"),
                    "arxiv_id": paper.get("id"),
                    "summary": paper.get("summary", "")[:500],
                    "authors": paper.get("authors", []),
                    "published": paper.get("published"),
                }
                for paper in papers[:10]
            ],
        }

    async def _save_research_finding(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        job = ctx.job
        source_scope_id = str(params.get("source_id") or "").strip() or None
        metrics = params.get("metrics")
        finding = {
            "id": str(uuid.uuid4()),
            "title": params.get("title"),
            "content": params.get("content"),
            "category": params.get("category"),
            # A conclusion with no type and no readable number is a conclusion
            # no contract can check. The validity predicates bound fields on
            # typed findings, and until this tool could carry either, the only
            # findings they could police were the ones tools emitted -- never
            # the claim a run actually drew from them.
            "type": str(params.get("finding_type") or "").strip() or None,
            "metrics": dict(metrics) if isinstance(metrics, dict) else {},
            "source_document_ids": params.get("source_document_ids", []),
            "source_id": source_scope_id,
            "confidence": params.get("confidence", 0.8),
            "tags": params.get("tags", []),
            "created_at": datetime.utcnow().isoformat(),
        }

        job_id_str = str(job.id)
        if job_id_str not in executor._job_findings:
            executor._job_findings[job_id_str] = []
        executor._job_findings[job_id_str].append(finding)

        # Deliberately not appended to state["findings"] here. The executor
        # extends that from the "findings" this returns, as it does for every
        # other tool that produces them -- doing both recorded each finding
        # twice, which is why every derived result in a run appeared as an
        # identical pair.
        return {
            "success": True,
            "data": {"finding_id": finding["id"]},
            "findings": [finding],
        }

    async def _get_research_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        findings = list(executor._job_findings.get(str(job.id), []))
        category = params.get("category")
        if category:
            findings = [
                finding for finding in findings if finding.get("category") == category
            ]
        min_confidence = params.get("min_confidence")
        if min_confidence:
            findings = [
                finding
                for finding in findings
                if finding.get("confidence", 0) >= min_confidence
            ]
        findings = findings[: params.get("limit", 50)]
        return {
            "success": True,
            "data": {"findings": findings, "total": len(findings)},
        }

    async def _literature_review_arxiv(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """The chat tool, as the job's owner.

        Its spec names research, monitor and knowledge_expansion jobs, and it
        is the producer of `literature_review` -- yet only chat had a handler,
        so a contract requiring that evidence could not be met by any run.
        """
        user_id = ctx.user_id or getattr(ctx.job, "user_id", None)
        if user_id is None:
            return {"error": "literature_review_arxiv needs the job's owner"}
        from app.services.agent_service import AgentService

        try:
            return await AgentService()._tool_literature_review_arxiv(
                params, user_id, ctx.db
            )
        except ValueError as exc:
            return {"error": str(exc)}

    async def _ingest_paper_by_id(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Put the paper in the corpus, then say so -- in that order.

        This used to run an arXiv search, return the metadata, store nothing,
        and emit a `papers_ingested` finding anyway. Measured live: a
        reproduction pipeline's first stage completed at 100% with its contract
        satisfied while `research_papers` held zero rows and no document
        existed. The next stage then searched the corpus for the paper, found a
        DIFFERENT paper by the same author left over from earlier work, and was
        one call away from writing a specification for the wrong algorithm --
        which the stages after it would have implemented, measured and scored
        against a claim it was never about.

        So it delegates to the real ingestion path and waits for the documents
        to land. Waiting is the point: `papers_ingested` has to mean the paper
        is readable, because the next stage's first act is to read it. A
        finding that means "ingestion was queued" is one a downstream stage can
        satisfy its own contract against while the corpus is still empty.
        """
        arxiv_id = str(params.get("arxiv_id") or "").strip()
        if not arxiv_id:
            return {"error": "Missing required parameter: arxiv_id"}

        papers = await _arxiv_search(query=f"id:{arxiv_id}", max_results=1)
        if not papers:
            return {"error": f"Paper {arxiv_id} not found"}
        paper = papers[0]

        # This builder is given the executor, not the AgentService that owns
        # the real ingestion path, so it is constructed here. Imported inside
        # the function: agent_service imports this module's siblings, and a
        # module-level import closes the cycle.
        from app.services.agent_service import AgentService

        started = await AgentService()._tool_ingest_arxiv_papers(
            {
                "name": f"arXiv {arxiv_id}",
                "paper_ids": [arxiv_id],
                "max_results": 1,
                "auto_sync": True,
                # The caller's word for it. Ignored entirely before now, along
                # with add_to_reading_list -- both accepted, neither used.
                "auto_summarize": bool(params.get("extract_insights", True)),
            },
            ctx.user_id,
            ctx.db,
        )
        source_id = (started or {}).get("source_id")
        if not source_id:
            return {
                "error": (
                    f"Could not start ingestion for {arxiv_id}: no document "
                    "source was created."
                )
            }

        landed = await _wait_for_ingested_documents(source_id)
        if not landed:
            # Not a success with a caveat. A stage whose contract is
            # `papers_ingested` must not pass on a paper that is not there.
            return {
                "success": False,
                "error": (
                    f"Ingestion of {arxiv_id} was started (source {source_id}) "
                    f"but no document had appeared after "
                    f"{INGEST_WAIT_SECONDS}s. The paper is not readable yet, so "
                    "nothing downstream can read it. Retry, or check the "
                    "ingestion worker."
                ),
                "data": {"source_id": source_id, "arxiv_id": arxiv_id},
            }

        return {
            "success": True,
            "data": {**paper, "source_id": source_id, "documents": landed},
            "findings": [
                {
                    "type": "papers_ingested",
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title"),
                    # So a later stage reads THIS paper rather than whatever
                    # a corpus search surfaces. The substitution that made this
                    # necessary was silent precisely because the finding named
                    # no document.
                    "document_ids": landed,
                    "source_id": source_id,
                }
            ],
        }

    async def _batch_ingest_papers(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid

        job = ctx.job
        arxiv_ids = [
            x.strip()
            for x in (params.get("arxiv_ids") or [])
            if isinstance(x, str) and x.strip()
        ]
        search_queries = [
            x.strip()
            for x in (params.get("search_queries") or [])
            if isinstance(x, str) and x.strip()
        ]
        categories = [
            x.strip()
            for x in (params.get("categories") or [])
            if isinstance(x, str) and x.strip()
        ]
        max_results = max(1, min(int(params.get("max_results") or 25), 200))
        if not arxiv_ids and not search_queries and not categories:
            return {
                "error": "Provide at least one of: arxiv_ids, search_queries, categories"
            }

        display = params.get("display") or "Autonomous job import"
        source_name = f"ArXiv Import (Job {str(job.id)[:8]}) #{uuid.uuid4().hex[:6]}"
        cfg = {
            "paper_ids": arxiv_ids,
            "search_queries": search_queries,
            "categories": categories,
            "max_results": max_results,
            "requested_by_user_id": str(job.user_id),
            "requested_by": str(job.user_id),
            "display": display,
        }
        source = await executor.document_service.create_document_source(
            name=source_name,
            source_type="arxiv",
            config=cfg,
            db=ctx.db,
        )

        queued = False
        try:
            from app.services.job_dispatch import enqueue

            enqueue(
                ctx.db, "app.tasks.ingestion_tasks.ingest_from_source", str(source.id)
            )
            queued = True
        except Exception:
            queued = False

        return {
            "success": True,
            "data": {
                "source_id": str(source.id),
                "source_name": source.name,
                "queued": queued,
                "paper_ids_count": len(arxiv_ids),
                "search_queries_count": len(search_queries),
                "categories_count": len(categories),
                "max_results": max_results,
            },
            "findings": [
                {
                    "type": "arxiv_ingest_requested",
                    "source_id": str(source.id),
                    "queued": queued,
                }
            ],
            "artifacts": [
                {
                    "type": "document_source",
                    "id": str(source.id),
                    "name": source.name,
                    "source_type": "arxiv",
                },
                {
                    "type": "arxiv_ingest_requested",
                    "source_id": str(source.id),
                    "queued": queued,
                },
            ],
        }

    async def _monitor_arxiv_topic(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timedelta, timezone

        from app.services.config_values import bounded_int

        topic = str(params.get("topic") or "").strip()
        query = str(params.get("query") or "").strip()
        if not topic and not query:
            # Without either this searched arXiv for the literal "all:None".
            return {"error": "topic (or query) is required"}
        query = query or f"all:{topic}"

        # `categories` and `since_days` were declared and never read.
        categories = params.get("categories")
        if isinstance(categories, str):
            categories = [categories]
        categories = [str(c).strip() for c in (categories or []) if str(c).strip()]
        if categories:
            query = (
                f"({query}) AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
            )

        papers = await _arxiv_search(
            query=query,
            max_results=bounded_int(params.get("max_results"), 20, 1, 100),
            sort_by="submittedDate",
            sort_order="descending",
        )

        since_days = params.get("since_days")
        if since_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(
                days=bounded_int(since_days, 7, 1, 3650)
            )

            def _recent(paper: Dict[str, Any]) -> bool:
                try:
                    published = datetime.fromisoformat(
                        str(paper.get("published")).replace("Z", "+00:00")
                    )
                except ValueError:
                    return True  # undated: not provably old
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                return published >= cutoff

            papers = [p for p in papers if _recent(p)]

        return {
            "success": True,
            "data": papers,
            "findings": [
                {
                    "type": "new_paper",
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("id"),
                    "published": paper.get("published"),
                }
                for paper in papers[:10]
            ],
        }

    async def _find_related_papers(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.document import Document
        from app.services.config_values import bounded_int, parse_uuid

        relation = str(params.get("relation_type") or "semantic").strip()
        if relation not in {
            "semantic",
            "citations",
            "shared_authors",
            "shared_topics",
            "all",
        }:
            return {"error": f"Unknown relation_type: {relation}"}
        limit = bounded_int(params.get("limit"), 10, 1, 50)

        title, author, reference_arxiv_id, reference_doc_id = "", "", "", None
        doc_id = params.get("document_id")
        arxiv_id = str(params.get("arxiv_id") or "").strip()
        if doc_id:
            doc_uuid = parse_uuid(str(doc_id))
            if doc_uuid is None:
                return {"error": f"Invalid document_id: {doc_id}"}
            doc = (
                await ctx.db.execute(select(Document).where(Document.id == doc_uuid))
            ).scalar_one_or_none()
            if doc:
                title, author, reference_doc_id = (
                    doc.title or "",
                    doc.author or "",
                    doc.id,
                )
        elif arxiv_id:
            papers = await _arxiv_search(query=f"id:{arxiv_id}", max_results=1)
            if papers:
                title = papers[0].get("title", "")
                reference_arxiv_id = str(papers[0].get("id") or arxiv_id)
                authors = papers[0].get("authors") or []
                author = str(authors[0]) if authors else ""

        if not title:
            return {
                "error": "No query could be built: the reference paper was not found"
            }

        # What "related" means decides what is asked. It was declared and
        # ignored: every relation ran the same title search.
        if relation == "shared_authors":
            if not author:
                return {"error": "The reference paper has no author to match on"}
            query = f'au:"{author}"'
        elif relation == "citations":
            query = f'all:"{title}"'
        else:
            query = title

        if not params.get("search_external", True):
            # The knowledge base, not arXiv. This branch used to answer "No
            # query could be built" although one had been.
            # By the words of the title, in the database: no index and no
            # network are needed to say what else here is about the same thing.
            import re as _re

            from sqlalchemy import or_ as _or

            from app.services.config_values import like_literal

            words = [w for w in _re.findall(r"[A-Za-z0-9]{4,}", title)][:8]
            related = []
            if words:
                rows = await ctx.db.execute(
                    select(Document)
                    .where(
                        Document.id != reference_doc_id,
                        _or(
                            *[
                                Document.title.ilike(
                                    f"%{like_literal(w)}%", escape="\\"
                                )
                                for w in words
                            ]
                        ),
                    )
                    .order_by(Document.updated_at.desc())
                    .limit(limit)
                )
                related = [
                    {"id": str(d.id), "title": d.title, "source": "knowledge_base"}
                    for d in rows.scalars().all()
                ]
            return {
                "success": True,
                "data": related,
                "findings": [
                    {
                        "type": "related_paper_set",
                        "title": paper.get("title"),
                        "document_id": paper.get("id"),
                    }
                    for paper in related
                ],
            }

        found = await _arxiv_search(query=query, max_results=limit + 1)
        # A paper is not related to itself.
        related = [
            paper
            for paper in found
            if not (
                reference_arxiv_id
                and str(paper.get("id") or "").split("v")[0]
                == reference_arxiv_id.split("v")[0]
            )
            and str(paper.get("title") or "").strip().lower() != title.strip().lower()
        ][:limit]
        return {
            "success": True,
            "data": related,
            "findings": [
                {
                    "type": "related_paper_set",
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("id"),
                }
                for paper in related
            ],
        }

    async def _extract_paper_insights(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from uuid import UUID

        from app.models.document import Document

        job = ctx.job
        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}

        prompt = f"""Extract key insights from this research paper.
Focus on: {', '.join(params.get('focus_areas', ['methodology', 'results', 'contributions']))}

Paper Title: {doc.title}
Content: {doc.content[:8000]}

Provide structured insights in JSON format:
{{
    "methodology": "...",
    "key_findings": ["..."],
    "contributions": ["..."],
    "limitations": ["..."],
    "future_work": ["..."]
}}"""
        try:
            response = await executor.llm_service.generate_response(
                system_prompt="You are a research paper analyst. Extract structured insights.",
                user_message=prompt,
                routing=executor._llm_routing_from_job_config(job.config),
                task_type="summarization",
                user_id=job.user_id,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "extract_paper_insights"),
            )
            insights = json.loads(response)
            return {
                "success": True,
                "data": insights,
                "findings": [
                    {
                        "type": "paper_insights",
                        "document_id": doc_id,
                        "insights": insights,
                        "source_id": str(doc.source_id)
                        if getattr(doc, "source_id", None)
                        else None,
                    }
                ],
            }
        except Exception as exc:
            raw = response if "response" in locals() else str(exc)
            return {"success": True, "data": {"raw_analysis": raw}}

    async def _extract_algorithm_spec(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Read a paper into something implementable, with its claims separated.

        The claims are pulled out as their own list, with the conditions each
        was measured under, because that is what makes them testable later. A
        claimed number buried in prose gets remembered approximately and
        compared against generously.
        """
        import json
        from uuid import UUID

        from app.models.document import Document

        job = getattr(ctx, "job", None)
        # `executor` is the closure argument of this provider builder, not a
        # field on the context -- AgentToolExecutionContext has no such
        # attribute, and reading one raised AttributeError on the first real
        # call. The sibling paper tools rely on the same closure.
        doc_id = str(params.get("document_id") or "").strip()
        if not doc_id:
            return {"error": "document_id is required"}

        # The id is a UUID column; comparing it against a bare string is the
        # sibling tool's mistake to avoid, and the import has to be local
        # because this closure has no module-level Document in scope.
        try:
            doc_uuid = UUID(doc_id)
        except (ValueError, AttributeError, TypeError):
            return {"error": f"document_id is not a UUID: {doc_id!r}"}

        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}

        wanted = str(params.get("algorithm_name") or "").strip()
        focus = (
            f"Focus on the algorithm called {wanted!r}."
            if wanted
            else "If the paper describes several algorithms, take the main one."
        )
        # Sized from settings, and the truncation is SAID rather than done
        # quietly. "No worked examples in this paper" and "none in the third of
        # it I was given" are different facts, and a specification that cannot
        # tell them apart sends the implement stage looking for cases that were
        # there all along, further down.
        from app.core.config import settings as _settings

        window = int(getattr(_settings, "SPEC_EXTRACTION_MAX_CHARS", 60000))
        content = doc.content or ""
        truncated = len(content) > window
        seen = content[:window]
        truncation_note = (
            (
                f"\n\nNOTE: this is the FIRST {window} characters of a "
                f"{len(content)}-character paper, not all of it. Anything you "
                "do not find may be in the part you were not given -- say so in "
                "`unstated` rather than concluding the paper omits it."
            )
            if truncated
            else ""
        )

        prompt = f"""Read this paper into a specification precise enough to implement.
{focus}

Paper Title: {doc.title}
Content: {seen}{truncation_note}

Return JSON:
{{
  "algorithm_name": "...",
  "inputs": ["what it takes, with types and shapes"],
  "outputs": ["what it produces"],
  "steps": ["ordered steps, precise enough to code from"],
  "parameters": {{"name": "value or range the paper used"}},
  "complexity": "the paper's stated complexity, or null",
  "reference_cases": [
    {{"name": "...", "input": "...", "expected_output": "...",
      "source": "where in the paper this worked example comes from"}}
  ],
  "properties": [
    {{"name": "...", "statement": "what must hold for ANY valid run",
      "why": "the sentence or step in the paper it follows from"}}
  ],
  "claims": [
    {{"metric": "speedup", "value": 3.0, "unit": "x",
      "conditions": {{"hardware": "...", "input_size": "...", "baseline": "..."}},
      "quote": "the sentence the number comes from"}}
  ],
  "unstated": ["anything the paper leaves unspecified that an implementer must choose"]
}}

Rules: put a number in "claims" only if the paper states it -- do not
estimate one. Leave "reference_cases" empty rather than inventing examples;
a case you made up checks nothing.

"properties" is NOT the same thing and is not an exception to that rule. A
worked example is a specific input the paper says produces a specific output.
A property is something that must hold for EVERY run because the paper's own
description says so -- an output range, an invariant preserved by a step, a
distribution the method is defined to produce, an equivalence with the
baseline it replaces. Deriving those from the text is reading; a made-up
input/output pair is inventing. Most algorithm papers give no worked examples
at all, and an implementation with no way to be checked is one nobody may
time, so state the properties the paper does give you.

"unstated" is important: papers routinely omit initialisation, tie-breaking
and precision, and an implementer who does not know what they are choosing
cannot say why their number differs."""
        response = ""
        try:
            response = await executor.llm_service.generate_response(
                system_prompt=(
                    "You extract implementable algorithm specifications from "
                    "papers. You never invent a number or a worked example."
                ),
                user_message=prompt,
                routing=executor._llm_routing_from_job_config(job.config),
                task_type="analysis",
                user_id=job.user_id,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "extract_algorithm_spec"),
            )
            spec = json.loads(response)
        except Exception as exc:
            # A response cut off by the output budget is a correct PREFIX, and
            # closing its open braces recovers the fields the model had already
            # written -- the same recovery the decision parser does, for the
            # same reason: asking again spends another call re-deriving the
            # answer on the budget that just proved too small. This extraction
            # is long (steps, claims with quotes, properties) and hit it as
            # soon as the input grew from an abstract to a paper.
            spec = None
            if response:
                from app.services import llm_truncation

                closed = llm_truncation.repair_truncated_json(response)
                if closed:
                    try:
                        spec = json.loads(closed)
                    except Exception:
                        spec = None
            if spec is None:
                return {
                    "error": (
                        "Could not read a specification out of this paper: "
                        f"{exc}. Raw response kept for inspection."
                    ),
                    "raw": (response or str(exc))[:2000],
                }
            # The recovery is recorded, not hidden. A specification closed from
            # a truncated response is missing whatever came after the cut, and
            # a run that cannot tell will read an absent field as the paper
            # lacking it -- the same confusion the truncation note above exists
            # to prevent on the input side.
            spec["_recovered_from_truncated_response"] = True

        cases = spec.get("reference_cases")
        claims = spec.get("claims")
        properties = spec.get("properties")
        case_count = len(cases) if isinstance(cases, list) else 0
        property_count = len(properties) if isinstance(properties, list) else 0
        return {
            "success": True,
            "data": spec,
            # Said in the tool's own reply, because the next stage decides what
            # to do from this. A run that reads "no worked examples" and does
            # not read "but here are four properties" goes looking for cases
            # that do not exist -- measured: six iterations of searching, no
            # code written, and the stage ended unverified.
            "note": (
                f"{case_count} worked example(s) from the paper and "
                f"{property_count} propert(ies) it states. "
                + (
                    "With no worked examples, check the implementation against "
                    "the properties: a program that asserts them and prints a "
                    "single pass line is a real check. What is forbidden is a "
                    "case invented to match what you wrote."
                    if case_count == 0 and property_count
                    else ""
                )
                + (
                    " The paper was TRUNCATED for this extraction, so anything "
                    "absent here may simply be further down it."
                    if truncated
                    else ""
                )
            ).strip(),
            "findings": [
                {
                    "type": "algorithm_spec",
                    "document_id": doc_id,
                    "algorithm_name": spec.get("algorithm_name") or wanted,
                    "spec": spec,
                    # Surfaced separately because the two downstream tools each
                    # need one of them, and a run should be able to see it has
                    # a spec with no testable claim before it starts coding.
                    "reference_case_count": case_count,
                    "property_count": property_count,
                    "claim_count": len(claims) if isinstance(claims, list) else 0,
                    # An empty `reference_cases` means two different things and
                    # only this tells them apart: the paper gives no worked
                    # examples, or the extractor was not shown the part that
                    # does.
                    "paper_truncated": bool(truncated),
                    "paper_chars_read": len(seen),
                    "paper_chars_total": len(content),
                }
            ],
        }

    async def _create_synthesis_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid

        job = ctx.job
        title = params.get("title")
        topic = params.get("topic")
        document_ids = params.get("document_ids", [])
        persist = bool(params.get("persist")) or bool(
            (job.config or {}).get("persist_artifacts", False)
        )
        scoped_source_id = str(params.get("source_id") or "").strip()

        findings = list(executor._job_findings.get(str(job.id), []))
        if scoped_source_id:
            findings = [
                finding
                for finding in findings
                if not isinstance(finding, dict)
                or not str(finding.get("source_id") or "").strip()
                or str(finding.get("source_id") or "").strip() == scoped_source_id
            ]

        synthesis_content = f"# {title}\n\n## Research Topic\n{topic}\n\n"
        if findings:
            synthesis_content += "## Key Findings\n"
            for i, finding in enumerate(findings[:20], 1):
                synthesis_content += f"\n### {i}. {finding.get('title', 'Finding')}\n"
                synthesis_content += f"{finding.get('content', '')}\n"
                if finding.get("category"):
                    synthesis_content += f"*Category: {finding['category']}*\n"

        result = {
            "success": True,
            "data": {
                "title": title,
                "content": synthesis_content,
                "findings_included": len(findings),
            },
            "artifacts": [
                {
                    "type": "synthesis_document",
                    "title": title,
                    "content": synthesis_content,
                }
            ],
            # The same thing again as a *finding*, because a goal contract
            # counts finding types and not artifact types. This tool's spec
            # declares produces=("synthesis_document",) and it emitted the name
            # only as an artifact, so a contract requiring synthesis_document
            # could not be satisfied by the one tool meant to satisfy it:
            # measured on a live pipeline whose writeup stage called this once,
            # succeeded, and still ended `completed_contract_unmet` on
            # finding_type:synthesis_document -- and in 364 jobs no finding of
            # that type had ever been recorded. Recorded here rather than in
            # the persist branch below, because the synthesis exists whether or
            # not it was also saved as a document.
            "findings": [
                {
                    "type": "synthesis_document",
                    "title": title,
                    "topic": topic,
                    "findings_included": len(findings),
                }
            ],
        }

        if persist and title and synthesis_content.strip():
            try:
                from app.models.document import Document

                notes_source = (
                    await executor.document_service._get_or_create_agent_notes_source(
                        ctx.db
                    )
                )
                content_hash = hashlib.sha256(
                    synthesis_content.encode("utf-8")
                ).hexdigest()
                doc = Document(
                    title=str(title).strip(),
                    content=synthesis_content,
                    content_hash=content_hash,
                    url=None,
                    file_path=None,
                    file_type="text/markdown",
                    file_size=len(synthesis_content.encode("utf-8")),
                    source_id=notes_source.id,
                    source_identifier=f"agent_synthesis:{uuid.uuid4().hex}",
                    author=None,
                    tags=["autonomous_job", "research"],
                    extra_metadata={
                        "origin": "autonomous_job",
                        "job_id": str(job.id),
                        "job_type": job.job_type,
                        "topic": topic,
                        "document_ids": document_ids,
                        "source_scope_id": scoped_source_id or None,
                    },
                    is_processed=False,
                )
                ctx.db.add(doc)
                await ctx.db.commit()
                await ctx.db.refresh(doc)
                try:
                    await executor.document_service.reprocess_document(
                        doc.id, ctx.db, user_id=job.user_id
                    )
                except Exception:
                    pass
                result["data"]["document_id"] = str(doc.id)
                result["artifacts"].append(
                    {"type": "document", "id": str(doc.id), "title": doc.title}
                )
                # Point the evidence at the saved document, so a later stage
                # can read what this one wrote.
                result["findings"][0]["document_id"] = str(doc.id)
            except Exception:
                pass

        return result

    async def _compare_methodologies(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Compare the methodologies described across several papers."""
        documents, error = await _load_documents_for_analysis(
            ctx, params.get("document_ids")
        )
        if error:
            return {"error": error}

        aspects = params.get("comparison_aspects")
        if not isinstance(aspects, list) or not aspects:
            aspects = ["approach", "results"]

        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_METHODOLOGY_COMPARISON_SCHEMA,
            user_message=(
                "Compare the methodologies in these papers. For each aspect, say "
                "how the papers differ and what that implies. Return JSON with "
                "comparisons, shared_approaches, notable_differences and summary.\n"
                f"ASPECTS: {', '.join(str(a) for a in aspects)}\n\n"
                + _document_excerpts(documents)
            ),
            task_type="methodology_comparison",
            temperature=0.2,
            max_tokens=1800,
            user_id=ctx.user_id or getattr(ctx.job, "user_id", None),
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable comparison"}
        payload["documents_compared"] = [str(d.id) for d in documents]
        return {
            "success": True,
            "data": payload,
            "findings": [{"type": "methodology_comparison", **(payload or {})}],
        }

    async def _identify_research_gaps(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Identify research gaps from the job's findings and named documents."""
        job = ctx.job
        findings = list(executor._job_findings.get(str(getattr(job, "id", "")), []))
        documents: list[Any] = []
        if isinstance(params.get("document_ids"), list) and params["document_ids"]:
            documents, error = await _load_documents_for_analysis(
                ctx, params.get("document_ids")
            )
            if error:
                return {"error": error}

        if not findings and not documents:
            return {
                "error": (
                    "No evidence to analyse: pass document_ids, or record "
                    "findings first with save_research_finding"
                )
            }

        topic = str(params.get("topic") or getattr(job, "goal", "") or "").strip()
        import json as _json

        evidence = _json.dumps(findings[:40], ensure_ascii=False, default=str)
        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_RESEARCH_GAPS_SCHEMA,
            user_message=(
                "Identify research gaps and opportunities from the evidence "
                "below. A gap must be supported by what is present; say so "
                "explicitly when the evidence is too thin to support any.\n"
                f"TOPIC: {topic}\n\nFINDINGS:\n{evidence}\n\n"
                + (_document_excerpts(documents) if documents else "")
            ),
            task_type="research_gap_analysis",
            temperature=0.3,
            max_tokens=1500,
            user_id=ctx.user_id or getattr(ctx.job, "user_id", None),
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable gap analysis"}
        payload["findings_analyzed"] = len(findings)
        payload["topic"] = topic
        return {
            "success": True,
            "data": payload,
            "findings": [{"type": "research_gap", **(payload or {})}],
        }

    async def _add_to_reading_list(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from sqlalchemy import func
        from sqlalchemy.exc import IntegrityError

        from app.models.document import Document
        from app.models.reading_list import ReadingList, ReadingListItem

        job = ctx.job
        list_name = (params.get("list_name") or "").strip()
        items = params.get("items", []) or []
        scoped_source_id = str(params.get("source_id") or "").strip()
        scoped_source_uuid = None
        if scoped_source_id:
            try:
                scoped_source_uuid = UUID(scoped_source_id)
            except Exception:
                scoped_source_uuid = None
        if not list_name:
            return {"error": "Missing required parameter: list_name"}
        if not isinstance(items, list) or not items:
            return {"error": "Missing required parameter: items"}

        rl_res = await ctx.db.execute(
            select(ReadingList).where(
                ReadingList.user_id == job.user_id, ReadingList.name == list_name
            )
        )
        rl = rl_res.scalar_one_or_none()
        if not rl:
            rl = ReadingList(
                user_id=job.user_id,
                name=list_name,
                description=None,
                source_id=scoped_source_uuid,
            )
            ctx.db.add(rl)
            await ctx.db.flush()

        max_pos = int(
            (
                await ctx.db.execute(
                    select(func.max(ReadingListItem.position)).where(
                        ReadingListItem.reading_list_id == rl.id
                    )
                )
            ).scalar()
            or 0
        )
        added = 0
        skipped = 0
        warnings: list[str] = []

        for raw in items:
            if not isinstance(raw, dict):
                skipped += 1
                continue
            doc_id = raw.get("document_id")
            arxiv_id = raw.get("arxiv_id")
            notes = raw.get("notes")
            priority = int(raw.get("priority", 3) or 3)

            doc = None
            if doc_id:
                try:
                    doc = await ctx.db.get(Document, UUID(str(doc_id)))
                except Exception:
                    doc = None
            elif arxiv_id:
                arxiv_id = str(arxiv_id).strip()
                if arxiv_id.startswith("arxiv:"):
                    arxiv_id = arxiv_id.split("arxiv:", 1)[1].strip()
                if arxiv_id:
                    doc_res = await ctx.db.execute(
                        select(Document)
                        .where(Document.source_identifier == arxiv_id)
                        .limit(1)
                    )
                    doc = doc_res.scalar_one_or_none()

            if not doc:
                skipped += 1
                if arxiv_id:
                    warnings.append(f"Document not found for arXiv id: {arxiv_id}")
                elif doc_id:
                    warnings.append(f"Document not found for id: {doc_id}")
                continue
            if (
                scoped_source_uuid
                and getattr(doc, "source_id", None) != scoped_source_uuid
            ):
                skipped += 1
                warnings.append(
                    f"Document {doc.id} is outside scoped source {scoped_source_id}"
                )
                continue

            exists = await ctx.db.execute(
                select(func.count())
                .select_from(ReadingListItem)
                .where(
                    ReadingListItem.reading_list_id == rl.id,
                    ReadingListItem.document_id == doc.id,
                )
            )
            if int(exists.scalar() or 0) > 0:
                skipped += 1
                continue

            item = ReadingListItem(
                reading_list_id=rl.id,
                document_id=doc.id,
                status="to-read",
                priority=max(0, min(priority, 5)),
                position=max_pos + 1,
                notes=str(notes).strip()[:2000] if notes else None,
            )
            ctx.db.add(item)
            try:
                await ctx.db.flush()
            except IntegrityError:
                await ctx.db.rollback()
                skipped += 1
                continue
            max_pos += 1
            added += 1

        await ctx.db.commit()
        return {
            "success": True,
            "data": {
                "reading_list_id": str(rl.id),
                "list_name": rl.name,
                "items_added": added,
                "items_skipped": skipped,
                "warnings": warnings[:25],
            },
            "artifacts": [
                {
                    "type": "reading_list",
                    "id": str(rl.id),
                    "name": rl.name,
                    "items_added": added,
                }
            ],
        }

    async def _get_reading_lists(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from sqlalchemy import desc

        from app.models.document import Document
        from app.models.reading_list import ReadingList, ReadingListItem

        job = ctx.job
        list_name = (params.get("list_name") or "").strip()
        include_items = bool(params.get("include_items", True))
        scoped_source_id = str(params.get("source_id") or "").strip()
        scoped_source_uuid = None
        if scoped_source_id:
            try:
                scoped_source_uuid = UUID(scoped_source_id)
            except Exception:
                scoped_source_uuid = None

        q = (
            select(ReadingList)
            .where(ReadingList.user_id == job.user_id)
            .order_by(desc(ReadingList.updated_at))
        )
        if list_name:
            q = q.where(ReadingList.name == list_name)
        if scoped_source_uuid:
            q = q.where(ReadingList.source_id == scoped_source_uuid)

        lists = (await ctx.db.execute(q.limit(100))).scalars().all()
        payload = []
        for rl in lists:
            entry: dict[str, Any] = {
                "id": str(rl.id),
                "name": rl.name,
                "description": rl.description,
                "created_at": rl.created_at.isoformat() if rl.created_at else None,
                "updated_at": rl.updated_at.isoformat() if rl.updated_at else None,
            }
            if include_items:
                items_res = await ctx.db.execute(
                    select(ReadingListItem, Document.title)
                    .join(Document, Document.id == ReadingListItem.document_id)
                    .where(ReadingListItem.reading_list_id == rl.id)
                    .order_by(
                        ReadingListItem.position.asc(), ReadingListItem.created_at.asc()
                    )
                )
                entry["items"] = [
                    {
                        "id": str(item.id),
                        "document_id": str(item.document_id),
                        "document_title": title,
                        "status": item.status,
                        "priority": item.priority,
                        "position": item.position,
                        "notes": item.notes,
                        "created_at": item.created_at.isoformat()
                        if item.created_at
                        else None,
                    }
                    for item, title in items_res.all()
                ]
            payload.append(entry)
        return {
            "success": True,
            "data": {"reading_lists": payload, "total": len(payload)},
        }

    async def _write_progress_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        report = {
            "summary": params.get("summary"),
            "completed_tasks": params.get("completed_tasks", []),
            "pending_tasks": params.get("pending_tasks", []),
            "key_findings": params.get("key_findings", []),
            "blockers": params.get("blockers", []),
            "next_steps": params.get("next_steps", []),
            "iteration": job.iteration,
            "progress": job.progress,
            "timestamp": datetime.utcnow().isoformat(),
        }
        reports = state.get("progress_reports")
        if not isinstance(reports, list):
            reports = []
            state["progress_reports"] = reports
        reports.append(report)
        return {
            "success": True,
            "data": report,
            "artifacts": [{"type": "progress_report", "report": report}],
        }

    async def _suggest_next_action(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        prompt = f"""Given the current research goal and progress, suggest the best next action.

Goal: {params.get('current_goal', job.goal)}
Progress so far: {params.get('progress_so_far', '')}
Findings count: {len(state.get('findings', []))}
Iteration: {job.iteration}/{job.max_iterations}

Available actions:
- Search for more papers on arXiv
- Analyze existing documents
- Synthesize findings
- Create a report
- Monitor for new papers

Suggest the single best next action and explain why."""
        try:
            suggestion = await executor.llm_service.generate_response(
                system_prompt="You are a research planning assistant.",
                user_message=prompt,
                routing=executor._llm_routing_from_job_config(job.config),
                task_type="research_engineer_scientist",
                user_id=job.user_id,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "suggest_next_action"),
            )
            return {"success": True, "data": {"suggestion": suggestion}}
        except Exception as exc:
            return {"error": str(exc)}

    async def _generate_research_presentation(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Queue a real presentation job.

        Previously reported presentation_queued=True without queueing anything.
        """
        from app.models.presentation import PresentationJob
        from app.services.job_dispatch import enqueue

        title = str(params.get("title") or "").strip()
        topic = str(params.get("topic") or "").strip()
        if not title or not topic:
            return {"error": "title and topic are required"}

        user_id = ctx.user_id or getattr(ctx.job, "user_id", None)
        if user_id is None:
            return {"error": "no user context for the presentation job"}

        try:
            slide_count = int(params.get("slide_count", 12) or 12)
        except (TypeError, ValueError):
            slide_count = 12
        slide_count = max(1, min(slide_count, 60))

        document_ids = params.get("source_document_ids")
        job_record = PresentationJob(
            user_id=user_id,
            title=title[:500],
            topic=topic[:500],
            source_document_ids=(
                [str(d) for d in document_ids] if isinstance(document_ids, list) else []
            ),
            slide_count=slide_count,
            style=str(params.get("style") or "professional"),
            include_diagrams=1 if params.get("include_diagrams", True) else 0,
            status="pending",
            progress=0,
        )
        ctx.db.add(job_record)
        await ctx.db.commit()
        await ctx.db.refresh(job_record)
        enqueue(
            ctx.db,
            "app.tasks.presentation_tasks.generate_presentation_task",
            str(job_record.id),
            str(user_id),
        )

        return {
            "success": True,
            "data": {
                "presentation_queued": True,
                "presentation_job_id": str(job_record.id),
                "title": job_record.title,
                "topic": job_record.topic,
                "slides": slide_count,
            },
            "findings": [
                {
                    "type": "research_presentation",
                    "presentation_job_id": str(job_record.id),
                    "title": job_record.title,
                    "slides": slide_count,
                }
            ],
        }

    async def _analyze_document_cluster(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Find common themes, differences and patterns across documents."""
        documents, error = await _load_documents_for_analysis(
            ctx, params.get("document_ids")
        )
        if error:
            return {"error": error}

        analysis_type = str(params.get("analysis_type") or "comprehensive").strip()
        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_CLUSTER_ANALYSIS_SCHEMA,
            user_message=(
                "Analyse this set of documents as a cluster. Return JSON with "
                "common_themes, differences, patterns and a short summary.\n"
                f"ANALYSIS TYPE: {analysis_type}\n\n" + _document_excerpts(documents)
            ),
            task_type="document_cluster_analysis",
            temperature=0.2,
            max_tokens=1500,
            user_id=ctx.user_id or getattr(ctx.job, "user_id", None),
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable cluster analysis"}
        payload["documents_analyzed"] = [str(d.id) for d in documents]
        return {
            "success": True,
            "data": payload,
            "findings": [{"type": "document_cluster", **(payload or {})}],
        }

    return FunctionToolProvider(
        name="autonomous_research_tools",
        modes={"autonomous"},
        handlers={
            "search_arxiv": _search_arxiv,
            "save_research_finding": _save_research_finding,
            "get_research_findings": _get_research_findings,
            "ingest_paper_by_id": _ingest_paper_by_id,
            "literature_review_arxiv": _literature_review_arxiv,
            "batch_ingest_papers": _batch_ingest_papers,
            "monitor_arxiv_topic": _monitor_arxiv_topic,
            "find_related_papers": _find_related_papers,
            "extract_paper_insights": _extract_paper_insights,
            "extract_algorithm_spec": _extract_algorithm_spec,
            "create_synthesis_document": _create_synthesis_document,
            "compare_methodologies": _compare_methodologies,
            "identify_research_gaps": _identify_research_gaps,
            "add_to_reading_list": _add_to_reading_list,
            "get_reading_lists": _get_reading_lists,
            "write_progress_report": _write_progress_report,
            "suggest_next_action": _suggest_next_action,
            "generate_research_presentation": _generate_research_presentation,
            "analyze_document_cluster": _analyze_document_cluster,
        },
    )
