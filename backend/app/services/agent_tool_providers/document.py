"""Autonomous-job tools: the ``document`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)

#: merge_documents combines at most this many documents.
MERGE_MAX_DOCUMENTS = 20


def build_autonomous_document_provider(executor: Any) -> FunctionToolProvider:
    """Document-domain tools for AutonomousAgentExecutor."""

    async def _search_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        db = ctx.db
        job = ctx.job
        query = params.get("query", job.goal[:100] if job else "")
        limit = params.get("limit", 10)
        source_id = str(params.get("source_id") or "").strip() or None
        page_size = max(1, min(int(limit or 10), 100))
        results, _total, _took = await executor.search_service.search(
            query=query,
            mode="smart",
            page=1,
            page_size=page_size,
            source_id=source_id,
            db=db,
        )
        return {
            "success": True,
            "data": results,
            "findings": [
                {
                    "type": "document",
                    "title": row.get("title"),
                    "id": row.get("id"),
                    "score": row.get("relevance_score", row.get("score")),
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                }
                for row in results[:10]
            ],
        }

    async def _search_with_filters(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        query = params.get("query", "")
        limit = params.get("limit", 20)
        source_id = str(params.get("source_id") or "").strip() or None
        file_type = str(params.get("file_type") or "").strip() or None
        mode = str(params.get("mode") or "smart").strip().lower() or "smart"
        page_size = max(1, min(int(limit or 20), 100))
        results, _total, _took = await executor.search_service.search(
            query=query,
            mode=mode,
            page=1,
            page_size=page_size,
            source_id=source_id,
            file_type=file_type,
            db=ctx.db,
        )
        return {"success": True, "data": results}

    async def _web_scrape(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from urllib.parse import urlparse

        from app.models.document import DocumentSource
        from app.services.web_scraper_service import WebScraperService

        url = params.get("url", "")
        if not url:
            return {"error": "Missing required parameter: url"}

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allow_private = False
        if host:
            src_res = await ctx.db.execute(
                select(DocumentSource).where(
                    DocumentSource.source_type == "web",
                    DocumentSource.is_active.is_(True),
                )
            )
            sources = src_res.scalars().all()

            def host_matches(allowed: str) -> bool:
                allowed = (allowed or "").strip().lower()
                if not allowed:
                    return False
                return host == allowed or host.endswith("." + allowed)

            for source in sources:
                cfg = source.config or {}
                for domain in cfg.get("allowed_domains") or []:
                    if host_matches(domain):
                        allow_private = True
                        break
                if allow_private:
                    break
                for base in cfg.get("base_urls") or []:
                    try:
                        base_host = (urlparse(str(base)).hostname or "").lower()
                    except Exception:
                        base_host = ""
                    if base_host and host_matches(base_host):
                        allow_private = True
                        break
                if allow_private:
                    break

        scraper = WebScraperService(enforce_network_safety=True)
        try:
            scrape_result = await scraper.scrape(
                url,
                follow_links=bool(params.get("follow_links", False)),
                max_pages=int(params.get("max_pages", 1)),
                max_depth=int(params.get("max_depth", 0)),
                same_domain_only=bool(params.get("same_domain_only", True)),
                include_links=bool(params.get("include_links", True)),
                allow_private_networks=allow_private,
                max_content_chars=int(params.get("max_content_chars", 50_000)),
            )
            payload: Dict[str, Any] = {"success": True, "data": scrape_result}
            pages = (
                scrape_result.get("pages", [])
                if isinstance(scrape_result, dict)
                else []
            )
            if pages:
                payload["findings"] = [
                    {
                        "type": "web_page",
                        "title": page.get("title"),
                        "url": page.get("url"),
                        "content_preview": (page.get("content") or "")[:500],
                    }
                    for page in pages[:5]
                ]
            return payload
        finally:
            await scraper.aclose()

    async def _ingest_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.user import User
        from app.services.url_ingestion_service import UrlIngestionService

        job = ctx.job
        url = (params.get("url") or "").strip()
        if not url:
            return {"error": "Missing required parameter: url"}
        user_res = await ctx.db.execute(select(User).where(User.id == job.user_id))
        user = user_res.scalar_one_or_none()
        if not user:
            return {"error": "User not found"}
        service = UrlIngestionService()
        ingest = await service.ingest_url(
            db=ctx.db,
            user=user,
            url=url,
            title=params.get("title"),
            tags=params.get("tags"),
            follow_links=bool(params.get("follow_links", False)),
            max_pages=int(params.get("max_pages", 1)),
            max_depth=int(params.get("max_depth", 0)),
            same_domain_only=bool(params.get("same_domain_only", True)),
            one_document_per_page=bool(params.get("one_document_per_page", False)),
            allow_private_networks=bool(params.get("allow_private_networks", False)),
            max_content_chars=int(params.get("max_content_chars", 50_000)),
        )
        if ingest.get("error"):
            return {"error": ingest["error"]}
        return {"success": True, "data": ingest}

    async def _get_document_details(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}
        return {
            "success": True,
            "data": {
                "id": str(doc.id),
                "title": doc.title,
                "source": doc.source,
                "file_type": doc.file_type,
                "author": doc.author,
                "summary": doc.summary,
                "has_content": bool(doc.content),
            },
        }

    async def _read_document_content(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        max_length = params.get("max_length", 10000)
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}
        return {
            "success": True,
            "data": {
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content[:max_length],
                "truncated": len(doc.content) > max_length,
            },
        }

    async def _summarize_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}
        if doc.summary:
            return {
                "success": True,
                "data": {"summary": doc.summary},
                "findings": [
                    {
                        "type": "document_summary",
                        "document_id": doc_id,
                        "content": doc.summary[:500],
                        "source_id": str(doc.source_id)
                        if getattr(doc, "source_id", None)
                        else None,
                    }
                ],
            }
        return {"success": True, "data": {"status": "summarization_queued"}}

    async def _find_similar_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        limit = params.get("limit", 5)
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}
        similar, _total, _took = await executor.search_service.search(
            query=doc.title + " " + (doc.summary or doc.content[:500]),
            mode="smart",
            page=1,
            page_size=max(1, min(int(limit or 5) + 1, 100)),
            source_id=str(doc.source_id) if getattr(doc, "source_id", None) else None,
            db=ctx.db,
        )
        similar = [row for row in similar if str(row.get("id")) != doc_id][:limit]
        return {
            "success": True,
            "data": similar,
            "findings": [
                {
                    "type": "similar_document",
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "source_id": row.get("source_id"),
                }
                for row in similar
            ],
        }

    async def _get_knowledge_base_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from collections import Counter

        from sqlalchemy import desc, func

        from app.models.document import Document, DocumentSource
        from app.services.config_values import bounded_int, parse_uuid
        from app.services.document_tags import clean_tags

        limit = bounded_int(params.get("recent_limit"), 25, 1, 100)
        source_id_raw = str(params.get("source_id") or "").strip()
        source_uuid = None
        if source_id_raw:
            # Refused, not ignored: a source id that could not be read gave
            # whole-knowledge-base figures presented as that source's.
            source_uuid = parse_uuid(source_id_raw)
            if source_uuid is None:
                return {"error": f"Invalid source_id: {source_id_raw}"}
            if await ctx.db.get(DocumentSource, source_uuid) is None:
                return {"error": f"Source not found: {source_id_raw}"}

        def scoped(query):
            return (
                query.where(Document.source_id == source_uuid) if source_uuid else query
            )

        async def scalar(query) -> int:
            return int((await ctx.db.execute(scoped(query))).scalar() or 0)

        total_docs = await scalar(select(func.count(Document.id)))
        processed_docs = await scalar(
            select(func.count(Document.id)).where(Document.is_processed.is_(True))
        )
        total_size = await scalar(
            select(func.coalesce(func.sum(Document.file_size), 0))
        )
        total_sources = (
            1
            if source_uuid
            else int(
                (
                    await ctx.db.execute(
                        select(func.count()).select_from(DocumentSource)
                    )
                ).scalar()
                or 0
            )
        )

        recent_query = scoped(
            select(Document.id, Document.title, Document.created_at)
            .order_by(desc(Document.created_at))
            .limit(limit)
        )
        rows = (await ctx.db.execute(recent_query)).all()

        # Over every document, like the totals beside it. Counted over the
        # `recent_limit` newest rows only, "top tags" was the top tags of the
        # last twenty-five documents.
        tag_counter: Counter[str] = Counter()
        tag_rows = await ctx.db.execute(
            scoped(select(Document.tags).where(Document.tags.isnot(None)))
        )
        for (tags,) in tag_rows.all():
            tag_counter.update(list(dict.fromkeys(t.lower() for t in clean_tags(tags))))

        return {
            "success": True,
            "data": {
                "documents_total": total_docs,
                "processed_documents": processed_docs,
                "pending_processing": total_docs - processed_docs,
                "total_storage_bytes": total_size,
                "sources_total": total_sources,
                "source_id": str(source_uuid) if source_uuid else None,
                "recent_documents": [
                    {"id": str(doc_id), "title": title, "created_at": str(created_at)}
                    for doc_id, title, created_at in rows
                ],
                "top_tags": [
                    {"tag": tag, "count": count}
                    for tag, count in tag_counter.most_common(10)
                ],
            },
        }

    async def _create_document_from_text(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid

        from app.models.document import Document

        job = ctx.job
        title = (params.get("title") or "").strip()
        content = (params.get("content") or "").strip()
        tags = params.get("tags") or []
        source_scope_id = str(params.get("source_id") or "").strip() or None

        if not title:
            return {"error": "Title is required"}
        if not content:
            return {"error": "Content is required"}

        notes_source = (
            await executor.document_service._get_or_create_agent_notes_source(ctx.db)
        )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        doc = Document(
            title=title,
            content=content,
            content_hash=content_hash,
            url=None,
            file_path=None,
            file_type="text/plain",
            file_size=len(content.encode("utf-8")),
            source_id=notes_source.id,
            source_identifier=f"agent_note:{uuid.uuid4().hex}",
            author=None,
            tags=tags if isinstance(tags, list) else None,
            extra_metadata={
                "origin": "autonomous_job",
                "job_id": str(job.id),
                "job_type": job.job_type,
                "source_scope_id": source_scope_id,
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

        return {
            "success": True,
            "data": {
                "document_id": str(doc.id),
                "title": doc.title,
                "source_scope_id": source_scope_id,
            },
            "artifacts": [
                {
                    "type": "document",
                    "id": str(doc.id),
                    "title": doc.title,
                    "source_scope_id": source_scope_id,
                }
            ],
            # An artifact is what the run produced; a finding is what the run
            # established. Only the second satisfies a goal contract, and this
            # tool emitted the artifact alone -- so a stage asking for
            # documents_ingested planned this tool, ran it successfully, and
            # was never any closer to its contract.
            "findings": [
                {
                    "type": "documents_ingested",
                    "document_id": str(doc.id),
                    "title": doc.title,
                    "source_scope_id": source_scope_id,
                }
            ],
        }

    async def _list_documents_by_tag(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        tags_param = params.get("tags")
        if not tags_param or not isinstance(tags_param, list) or not tags_param:
            return {"error": "tags is required and must be a non-empty array"}
        match_all = bool(params.get("match_all", False))
        try:
            limit = min(int(params.get("limit", 20) or 20), 100)
        except (TypeError, ValueError):
            return {"error": "limit must be a number"}
        if limit < 1:
            return {"error": "limit must be at least 1"}
        tags_set = set(str(tag).strip() for tag in tags_param if str(tag).strip())
        if not tags_set:
            # The empty set is a subset of every document's tags, so with
            # match_all a list of blank tags listed everything.
            return {"error": "tags is required and must be a non-empty array"}

        from app.services.document_tags import documents_with_tags

        matched = await documents_with_tags(
            ctx.db,
            list(tags_set),
            match_all=match_all,
            limit=limit,
            newest_by="created_at",
        )
        return {
            "success": True,
            "data": {
                "tags": list(tags_set),
                "match_all": match_all,
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "tags": doc.tags or [],
                        "file_type": doc.file_type,
                        "summary": (doc.summary or "")[:200],
                        "created_at": doc.created_at.isoformat()
                        if doc.created_at
                        else None,
                    }
                    for doc in matched
                ],
                "count": len(matched),
            },
        }

    async def _merge_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid
        from uuid import UUID as _UUID

        from loguru import logger

        from app.models.document import Document

        job = ctx.job
        doc_ids = params.get("document_ids")
        merge_title = str(params.get("title", "")).strip()
        if not doc_ids or not isinstance(doc_ids, list) or not doc_ids:
            return {"error": "document_ids is required and must be a non-empty array"}
        if not merge_title:
            return {"error": "title is required"}

        separator = params.get("separator")
        separator = "\n\n---\n\n" if separator is None else str(separator)
        merge_tags = params.get("tags") if isinstance(params.get("tags"), list) else []
        # Every id that is not merged is named with its reason. They used to
        # be dropped in silence -- an id that was not one, a document that
        # did not exist, anything past the 20th -- and an id given twice was
        # merged twice.
        sections = []
        source_ids: List[str] = []
        skipped: List[Dict[str, str]] = []
        for doc_id_str in doc_ids:
            raw_id = str(doc_id_str).strip()
            try:
                doc_uuid = _UUID(raw_id)
            except (ValueError, AttributeError, TypeError):
                skipped.append({"id": raw_id, "reason": "not a document id"})
                continue
            if str(doc_uuid) in source_ids:
                skipped.append({"id": raw_id, "reason": "listed more than once"})
                continue
            if len(source_ids) >= MERGE_MAX_DOCUMENTS:
                skipped.append(
                    {"id": raw_id, "reason": f"over the {MERGE_MAX_DOCUMENTS} limit"}
                )
                continue
            doc_obj = await ctx.db.get(Document, doc_uuid)
            if doc_obj is None:
                skipped.append({"id": raw_id, "reason": "no such document"})
            elif not doc_obj.content:
                skipped.append({"id": raw_id, "reason": "document has no content"})
            else:
                sections.append(f"# {doc_obj.title}\n\n{doc_obj.content}")
                source_ids.append(str(doc_obj.id))
        if not sections:
            return {
                "error": "No valid documents with content found",
                "skipped": skipped,
            }

        merged_content = separator.join(sections)
        if len(merged_content.encode("utf-8")) > 2_000_000:
            return {"error": "Merged content exceeds 2MB limit"}

        content_hash = hashlib.sha256(merged_content.encode("utf-8")).hexdigest()
        notes_source = (
            await executor.document_service._get_or_create_agent_notes_source(ctx.db)
        )
        new_doc = Document(
            title=merge_title,
            content=merged_content,
            content_hash=content_hash,
            file_type="text/plain",
            file_size=len(merged_content.encode("utf-8")),
            source_id=notes_source.id,
            source_identifier=f"agent_merge:{uuid.uuid4().hex}",
            tags=merge_tags,
            extra_metadata={
                "origin": "agent_merge",
                "source_document_ids": source_ids,
                "job_id": str(job.id),
            },
            is_processed=False,
        )
        ctx.db.add(new_doc)
        await ctx.db.commit()
        await ctx.db.refresh(new_doc)

        try:
            indexed = await executor.document_service.reprocess_document(
                new_doc.id, ctx.db, user_id=job.user_id
            )
        except Exception as exc:  # noqa: BLE001 - the merge is kept either way
            logger.warning(f"merge_documents: indexing {new_doc.id} failed: {exc}")
            indexed = False

        data = {
            "document_id": str(new_doc.id),
            "title": new_doc.title,
            "source_count": len(source_ids),
            "content_length": len(merged_content),
        }
        if skipped:
            data["skipped"] = skipped
        if not indexed:
            # Saved but not searchable: a search for it would find nothing.
            data["warning"] = (
                "The merged document was saved but could not be indexed, so "
                "search will not find it until it is reprocessed."
            )
        return {
            "success": True,
            "data": data,
            "artifacts": [
                {"type": "document", "id": str(new_doc.id), "title": new_doc.title}
            ],
        }

    return FunctionToolProvider(
        name="autonomous_document_tools",
        modes={"autonomous"},
        handlers={
            "search_documents": _search_documents,
            "search_with_filters": _search_with_filters,
            "web_scrape": _web_scrape,
            "ingest_url": _ingest_url,
            "get_document_details": _get_document_details,
            "read_document_content": _read_document_content,
            "summarize_document": _summarize_document,
            "find_similar_documents": _find_similar_documents,
            "get_knowledge_base_stats": _get_knowledge_base_stats,
            "create_document_from_text": _create_document_from_text,
            "list_documents_by_tag": _list_documents_by_tag,
            "merge_documents": _merge_documents,
        },
    )
