"""
Knowledge extraction service: identifies entities and relations from text
and stores them as a lightweight knowledge graph with provenance.

Supports both rule-based (fast, lightweight) and LLM-based (accurate, comprehensive)
extraction methods.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.knowledge_graph import Entity, EntityMention, Relationship
from app.models.document import Document, DocumentChunk
from app.core.config import settings

if TYPE_CHECKING:
    from app.services.llm_service import LLMService, UserLLMSettings


def _sanitize_type(s: str, default: str, max_len: int = 64) -> str:
    """
    Normalize an arbitrary type label to a compact snake_case token.

    We intentionally avoid a hardcoded taxonomy here; the LLM can introduce
    new types as needed. We only enforce formatting and a safe fallback.
    """
    try:
        s = (s or "").strip().lower()
    except Exception:
        s = ""
    if not s:
        return default
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return default
    return s[:max_len]


@dataclass
class ExtractedEntity:
    text: str
    entity_type: str
    start: Optional[int] = None
    end: Optional[int] = None
    sentence: Optional[str] = None


@dataclass
class ExtractedRelation:
    head_text: str
    tail_text: str
    relation_type: str
    confidence: float
    sentence: Optional[str] = None


class KnowledgeExtractor:
    """Rule-based extractor with optional simple NER patterns.

    This is intentionally lightweight to avoid heavyweight model dependencies.
    It captures common entities (emails, URLs, capitalized names, orgs) and a
    few relation patterns (works_for, mentions, references).
    """

    def __init__(self) -> None:
        # Basic regex patterns
        self.email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        self.url_re = re.compile(r"https?://[\w./?#&%=-]+", re.IGNORECASE)
        # Naive person: Firstname Lastname (Title Case)
        self.person_re = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")
        # Naive org: Words ending with Inc.|LLC|Ltd.|JSC|Corp.|Company
        self.org_re = re.compile(r"\b([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)*\s+(?:Inc\.|LLC|Ltd\.|JSC|Corp\.|Company))\b")

        # Relation patterns: "X at Y", "X from Y"
        self.works_for_patterns = [
            re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b\s+(?:at|@|from)\s+\b([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)*\b)")
        ]

    def _sentences(self, text: str) -> List[str]:
        # Simple sentence splitter; avoids heavy tokenizers
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        # Emails and URLs first (precise)
        for m in self.email_re.finditer(text):
            entities.append(ExtractedEntity(m.group(0), "email", m.start(), m.end()))
        for m in self.url_re.finditer(text):
            entities.append(ExtractedEntity(m.group(0), "url", m.start(), m.end()))

        # Sentence-based for persons/orgs to attach sentence
        for sent in self._sentences(text):
            base_offset = text.find(sent)
            if base_offset < 0:
                base_offset = None
            for m in self.org_re.finditer(sent):
                start = (base_offset + m.start()) if base_offset is not None else None
                end = (base_offset + m.end()) if base_offset is not None else None
                entities.append(ExtractedEntity(m.group(1), "org", start, end, sent))
            for m in self.person_re.finditer(sent):
                start = (base_offset + m.start()) if base_offset is not None else None
                end = (base_offset + m.end()) if base_offset is not None else None
                entities.append(ExtractedEntity(m.group(1), "person", start, end, sent))

        return entities

    def extract_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        relations: List[ExtractedRelation] = []
        # Build quick lookup for person/org in each sentence
        for sent in self._sentences(text):
            for pat in self.works_for_patterns:
                for m in pat.finditer(sent):
                    head, tail = m.group(1), m.group(2)
                    relations.append(ExtractedRelation(head, tail, "works_for", 0.7, sent))
        # Mentions: email/url mentioned by org/person in same sentence
        # Keep basic for now; can be extended
        return relations

    async def _get_or_create_entity(self, db: AsyncSession, name: str, etype: str) -> Entity:
        q = await db.execute(select(Entity).where(Entity.canonical_name == name, Entity.entity_type == etype))
        ent = q.scalar_one_or_none()
        if ent:
            return ent
        ent = Entity(canonical_name=name, entity_type=etype)
        db.add(ent)
        await db.flush()
        return ent

    async def index_chunk(self, db: AsyncSession, document: Document, chunk: DocumentChunk) -> Tuple[int, int]:
        """Extract entities and relations from a chunk and persist.

        Returns: (entities_created_or_linked, relations_created)
        """
        try:
            text = chunk.content or ""
            if not text.strip():
                return (0, 0)

            ents = self.extract_entities(text)
            rels = self.extract_relations(text, ents)

            # Deduplicate by text+type within this chunk
            seen = set()
            created_mentions = 0
            for e in ents:
                key = (e.text, e.entity_type)
                if key in seen:
                    continue
                seen.add(key)
                ent = await self._get_or_create_entity(db, e.text[:512], e.entity_type)
                mention = EntityMention(
                    entity_id=ent.id,
                    document_id=document.id,
                    chunk_id=chunk.id,
                    text=e.text[:512],
                    start_pos=e.start,
                    end_pos=e.end,
                    sentence=e.sentence,
                )
                db.add(mention)
                created_mentions += 1

            created_rels = 0
            # Map by canonical name to entity id for quick linking
            # We only link relations if both ends exist as entities
            ent_map: Dict[str, Entity] = {}
            for e in ents:
                ent_map.setdefault(e.text, None)
            if ent_map:
                q = await db.execute(select(Entity).where(Entity.canonical_name.in_(list(ent_map.keys()))))
                for ent in q.scalars().all():
                    ent_map[ent.canonical_name] = ent

            for r in rels:
                head = ent_map.get(r.head_text)
                tail = ent_map.get(r.tail_text)
                if not head or not tail:
                    # Try to create lazily if missing
                    if not head:
                        head = await self._get_or_create_entity(db, r.head_text[:512], "person")
                    if not tail:
                        tail = await self._get_or_create_entity(db, r.tail_text[:512], "org")

                # Upsert-like: rely on unique constraint per doc
                rel = Relationship(
                    relation_type=r.relation_type,
                    confidence=r.confidence,
                    source_entity_id=head.id,
                    target_entity_id=tail.id,
                    document_id=document.id,
                    chunk_id=chunk.id,
                    evidence=r.sentence,
                )
                try:
                    db.add(rel)
                    await db.flush()  # Flush to trigger unique constraint check
                    created_rels += 1
                except IntegrityError:
                    # Duplicate relationship due to unique constraint; skip
                    await db.rollback()
                    logger.debug("Duplicate relationship skipped")
                except Exception as e:
                    await db.rollback()
                    logger.warning(f"Failed to add relationship: {e}")

            return (created_mentions, created_rels)
        except Exception as e:
            logger.warning(f"KG extraction failed for chunk {chunk.id}: {e}")
            return (0, 0)


class LLMKnowledgeExtractor:
    """LLM-powered entity and relationship extraction.

    Uses the LLM service to extract structured knowledge from text,
    providing better accuracy than rule-based patterns, especially for:
    - Complex entity names and titles
    - Diverse relationship types
    - Context-aware entity typing
    - Abstract concepts and technologies
    """

    EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Text:
{text}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"text": "entity name", "type": "short_snake_case_type", "description": "brief context (optional)"}}
  ],
  "relationships": [
    {{"source": "entity A name", "target": "entity B name", "type": "short_snake_case_relation", "confidence": 0.9, "evidence": "supporting text snippet"}}
  ]
}}

Known entity types (prefer these when they fit):
{known_entity_types}

Known relationship types (prefer these when they fit):
{known_relation_types}

Guidelines:
1. Extract ALL named entities (people, companies, places, products, technologies)
2. Extract concepts and key terms that are important to the text
3. Identify relationships between entities - be specific about the relationship type
4. Use high confidence (0.8+) only for clearly stated relationships
5. Include evidence snippets that support each relationship
6. Keep entity names as they appear in the text (preserve capitalization)
7. When a known type fits, use it. If none fits, use "other" for entities and "related_to" for relationships.
8. Keep your type vocabulary small and consistent across extractions; avoid near-duplicate synonyms.

Return ONLY valid JSON, no markdown code blocks or explanation."""

    TYPE_RESOLUTION_PROMPT = """You are a normalization step for a knowledge graph extractor.

You will be given:
1) an "open list" of allowed entity types
2) an "open list" of allowed relationship types
3) an extraction JSON with entities/relationships

Your task:
- Replace every entity "type" with the single best match from the allowed entity types.
- Replace every relationship "type" with the single best match from the allowed relationship types.
- Keep all other fields the same.
- If nothing fits well, use "other" for entity types and "related_to" for relationship types.

Allowed entity types:
{allowed_entity_types}

Allowed relationship types:
{allowed_relation_types}

Extraction JSON:
{extraction_json}

Return ONLY valid JSON matching the original structure (keys: entities, relationships)."""

    def __init__(self) -> None:
        self._llm_service: Optional["LLMService"] = None
        self._known_types_cache: Optional[Dict[str, Any]] = None

    def _get_llm_service(self) -> "LLMService":
        """Lazy-load LLM service to avoid circular imports."""
        if self._llm_service is None:
            from app.services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    async def _get_known_types(self, db: AsyncSession) -> Tuple[List[str], List[str]]:
        # Small in-process TTL cache to avoid a DISTINCT scan per chunk.
        try:
            from time import time as _time
            now = _time()
            ttl_s = 300.0
            cached = self._known_types_cache or {}
            if cached.get("ts") and (now - float(cached["ts"])) < ttl_s:
                return cached.get("entity_types", []) or [], cached.get("relation_types", []) or []

            et = (await db.execute(select(Entity.entity_type).distinct().order_by(Entity.entity_type))).fetchall()
            rt = (await db.execute(select(Relationship.relation_type).distinct().order_by(Relationship.relation_type))).fetchall()
            entity_types = [r[0] for r in et if r and r[0]]
            relation_types = [r[0] for r in rt if r and r[0]]

            # Prevent prompt bloat.
            entity_types = entity_types[:100]
            relation_types = relation_types[:100]

            self._known_types_cache = {"ts": now, "entity_types": entity_types, "relation_types": relation_types}
            return entity_types, relation_types
        except Exception:
            return [], []

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown blocks."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: try to parse the whole thing
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM extraction response: {e}")
            return {"entities": [], "relationships": []}

    async def extract_from_text(
        self,
        text: str,
        user_settings: Optional["UserLLMSettings"] = None,
        known_entity_types: Optional[List[str]] = None,
        known_relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract entities and relationships from text using LLM.

        Args:
            text: The text to extract from (will be truncated if too long)
            user_settings: Optional user LLM settings for provider preference

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        # Truncate text if too long
        max_len = settings.KG_EXTRACTION_MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len] + "..."

        prompt = self.EXTRACTION_PROMPT.format(
            text=text,
            known_entity_types=json.dumps(known_entity_types or []),
            known_relation_types=json.dumps(known_relation_types or []),
        )

        try:
            llm = self._get_llm_service()
            response = await llm.generate_response(
                query=prompt,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000,
                user_settings=user_settings,
                task_type="knowledge_extraction",
                # System default for KG extraction; user_settings can still override.
                model=getattr(settings, "KG_EXTRACTION_MODEL", None) or None,
            )
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def _normalize_entity_type(self, etype: str) -> str:
        """Normalize entity type to a safe snake_case token, without a fixed taxonomy."""
        return _sanitize_type(etype, default="other")

    def _normalize_relation_type(self, rtype: str) -> str:
        """Normalize relationship type to a safe snake_case token, without a fixed taxonomy."""
        return _sanitize_type(rtype, default="related_to")

    def _coerce_allowed(self, t: str, allowed: List[str], fallback: str) -> str:
        """Ensure the normalized type is in the allowed open-list when provided."""
        t = _sanitize_type(t, default=fallback)
        if not allowed:
            return t
        if t in allowed:
            return t
        return fallback if fallback in allowed else t

    async def _resolve_types_from_open_list(
        self,
        extraction: Dict[str, Any],
        *,
        allowed_entity_types: List[str],
        allowed_relation_types: List[str],
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> Dict[str, Any]:
        """Resolve extracted entity/relation types to the nearest type from the open-list using the LLM.

        This reduces taxonomy drift without hardcoding a fixed set of types.
        """
        if not extraction or (not extraction.get("entities") and not extraction.get("relationships")):
            return extraction

        ent_allowed = [t for t in (allowed_entity_types or []) if isinstance(t, str) and t.strip()]
        rel_allowed = [t for t in (allowed_relation_types or []) if isinstance(t, str) and t.strip()]
        if not ent_allowed and not rel_allowed:
            return extraction

        # Ensure fallbacks exist in the open list we provide to the model.
        if ent_allowed and "other" not in ent_allowed:
            ent_allowed = ent_allowed + ["other"]
        if rel_allowed and "related_to" not in rel_allowed:
            rel_allowed = rel_allowed + ["related_to"]

        # Keep prompt bounded.
        ent_allowed = ent_allowed[:120]
        rel_allowed = rel_allowed[:160]

        # The extraction payload might include non-serializable values; keep only what we use.
        safe = {
            "entities": extraction.get("entities", []) or [],
            "relationships": extraction.get("relationships", []) or [],
        }

        prompt = self.TYPE_RESOLUTION_PROMPT.format(
            allowed_entity_types=json.dumps(ent_allowed),
            allowed_relation_types=json.dumps(rel_allowed),
            extraction_json=json.dumps(safe, ensure_ascii=True)[:12000],
        )

        try:
            llm = self._get_llm_service()
            resp = await llm.generate_response(
                query=prompt,
                temperature=0.0,
                max_tokens=1200,
                user_settings=user_settings,
                task_type="knowledge_extraction",
                model=getattr(settings, "KG_EXTRACTION_MODEL", None) or None,
            )
            resolved = self._parse_json_response(resp)
        except Exception as e:
            logger.warning(f"Type resolution failed; using raw types. Error: {e}")
            return extraction

        # Post-enforce membership + normalization.
        out_ents: List[Any] = []
        for e in (resolved.get("entities") or []):
            if not isinstance(e, dict) or "text" not in e:
                continue
            if ent_allowed:
                e["type"] = self._coerce_allowed(e.get("type", "other"), ent_allowed, "other")
            else:
                e["type"] = self._normalize_entity_type(e.get("type", "other"))
            out_ents.append(e)

        out_rels: List[Any] = []
        for r in (resolved.get("relationships") or []):
            if not isinstance(r, dict) or "source" not in r or "target" not in r:
                continue
            if rel_allowed:
                r["type"] = self._coerce_allowed(r.get("type", "related_to"), rel_allowed, "related_to")
            else:
                r["type"] = self._normalize_relation_type(r.get("type", "related_to"))
            out_rels.append(r)

        return {"entities": out_ents, "relationships": out_rels}

    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using LLM."""
        result = await self.extract_from_text(text)
        entities = []

        for e in result.get("entities", []):
            if not isinstance(e, dict) or "text" not in e:
                continue
            etype = self._normalize_entity_type(e.get("type", "other"))
            entities.append(ExtractedEntity(
                text=e["text"][:512],
                entity_type=etype,
                sentence=e.get("description"),
            ))

        return entities

    async def extract_relations(self, text: str) -> List[ExtractedRelation]:
        """Extract relationships from text using LLM."""
        result = await self.extract_from_text(text)
        relations = []

        for r in result.get("relationships", []):
            if not isinstance(r, dict):
                continue
            if "source" not in r or "target" not in r:
                continue

            rtype = self._normalize_relation_type(r.get("type", "related_to"))
            confidence = float(r.get("confidence", 0.7))
            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            relations.append(ExtractedRelation(
                head_text=r["source"][:512],
                tail_text=r["target"][:512],
                relation_type=rtype,
                confidence=confidence,
                sentence=r.get("evidence"),
            ))

        return relations

    async def index_chunk(
        self,
        db: AsyncSession,
        document: Document,
        chunk: DocumentChunk,
        rule_extractor: Optional[KnowledgeExtractor] = None,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> Tuple[int, int]:
        """Extract entities and relations from a chunk using LLM and persist.

        Falls back to rule-based extraction on failure.

        Returns: (entities_created_or_linked, relations_created)
        """
        try:
            text = chunk.content or ""
            if not text.strip():
                return (0, 0)

            # Extract using LLM
            known_entity_types, known_relation_types = await self._get_known_types(db)
            extraction_result = await self.extract_from_text(
                text,
                user_settings=user_settings,
                known_entity_types=known_entity_types,
                known_relation_types=known_relation_types,
            )

            # Resolve entity/relation types to the closest from the open-list to reduce drift.
            extraction_result = await self._resolve_types_from_open_list(
                extraction_result,
                allowed_entity_types=known_entity_types,
                allowed_relation_types=known_relation_types,
                user_settings=user_settings,
            )

            raw_entities = extraction_result.get("entities", [])
            raw_relations = extraction_result.get("relationships", [])

            # If LLM returned nothing, fall back to rule-based
            if not raw_entities and not raw_relations:
                if rule_extractor:
                    return await self._index_with_rule_extractor(
                        db, document, chunk, rule_extractor
                    )
                return (0, 0)

            # Process entities
            seen = set()
            created_mentions = 0
            ent_map: Dict[str, Entity] = {}

            for e in raw_entities:
                if not isinstance(e, dict) or "text" not in e:
                    continue

                etype = self._normalize_entity_type(e.get("type", "other"))
                name = e["text"][:512]
                key = (name.lower(), etype)

                if key in seen:
                    continue
                seen.add(key)

                # Get or create entity
                ent = await self._get_or_create_entity(db, name, etype)
                ent_map[name.lower()] = ent

                # Create mention
                mention = EntityMention(
                    entity_id=ent.id,
                    document_id=document.id,
                    chunk_id=chunk.id,
                    text=name,
                    sentence=e.get("description"),
                )
                db.add(mention)
                created_mentions += 1

            # Process relationships
            created_rels = 0
            for r in raw_relations:
                if not isinstance(r, dict):
                    continue
                if "source" not in r or "target" not in r:
                    continue

                head_name = r["source"][:512]
                tail_name = r["target"][:512]

                # Look up or create entities
                head = ent_map.get(head_name.lower())
                tail = ent_map.get(tail_name.lower())

                if not head:
                    head = await self._get_or_create_entity(db, head_name, "other")
                    ent_map[head_name.lower()] = head
                if not tail:
                    tail = await self._get_or_create_entity(db, tail_name, "other")
                    ent_map[tail_name.lower()] = tail

                rtype = self._normalize_relation_type(r.get("type", "related_to"))
                confidence = float(r.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))

                # Create relationship
                rel = Relationship(
                    relation_type=rtype,
                    confidence=confidence,
                    source_entity_id=head.id,
                    target_entity_id=tail.id,
                    document_id=document.id,
                    chunk_id=chunk.id,
                    evidence=r.get("evidence"),
                )
                try:
                    db.add(rel)
                    await db.flush()
                    created_rels += 1
                except IntegrityError:
                    await db.rollback()
                    logger.debug("Duplicate relationship skipped")
                except Exception as e:
                    await db.rollback()
                    logger.warning(f"Failed to add relationship: {e}")

            return (created_mentions, created_rels)

        except Exception as e:
            logger.warning(f"LLM KG extraction failed for chunk {chunk.id}: {e}")
            # Fall back to rule-based extraction
            if rule_extractor:
                return await self._index_with_rule_extractor(
                    db, document, chunk, rule_extractor
                )
            return (0, 0)

    async def _get_or_create_entity(self, db: AsyncSession, name: str, etype: str) -> Entity:
        """Get existing entity or create new one."""
        q = await db.execute(
            select(Entity).where(Entity.canonical_name == name, Entity.entity_type == etype)
        )
        ent = q.scalar_one_or_none()
        if ent:
            return ent
        ent = Entity(canonical_name=name, entity_type=etype)
        db.add(ent)
        await db.flush()
        return ent

    async def _index_with_rule_extractor(
        self,
        db: AsyncSession,
        document: Document,
        chunk: DocumentChunk,
        rule_extractor: KnowledgeExtractor
    ) -> Tuple[int, int]:
        """Fallback to rule-based extraction."""
        return await rule_extractor.index_chunk(db, document, chunk)


# Global instances
extractor = KnowledgeExtractor()
llm_extractor = LLMKnowledgeExtractor()
