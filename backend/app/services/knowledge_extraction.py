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
    from app.services.llm_service import LLMService


# Extended entity types for LLM extraction
ENTITY_TYPES = ("person", "org", "location", "product", "email", "url", "concept", "technology", "event", "other")

# Extended relationship types for LLM extraction
RELATION_TYPES = (
    "works_for", "manages", "reports_to", "collaborates_with",
    "owns", "uses", "implements", "part_of", "located_in",
    "related_to", "mentions", "references", "created_by"
)


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
    {{"text": "entity name", "type": "person|org|location|product|concept|technology|event", "description": "brief context (optional)"}}
  ],
  "relationships": [
    {{"source": "entity A name", "target": "entity B name", "type": "relationship_type", "confidence": 0.9, "evidence": "supporting text snippet"}}
  ]
}}

Entity types: person, org, location, product, concept, technology, event, email, url
Relationship types: works_for, manages, reports_to, collaborates_with, owns, uses, implements, part_of, located_in, related_to, mentions, references, created_by

Guidelines:
1. Extract ALL named entities (people, companies, places, products, technologies)
2. Extract concepts and key terms that are important to the text
3. Identify relationships between entities - be specific about the relationship type
4. Use high confidence (0.8+) only for clearly stated relationships
5. Include evidence snippets that support each relationship
6. Keep entity names as they appear in the text (preserve capitalization)

Return ONLY valid JSON, no markdown code blocks or explanation."""

    def __init__(self) -> None:
        self._llm_service: Optional["LLMService"] = None

    def _get_llm_service(self) -> "LLMService":
        """Lazy-load LLM service to avoid circular imports."""
        if self._llm_service is None:
            from app.services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service

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

    async def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text using LLM.

        Args:
            text: The text to extract from (will be truncated if too long)

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        # Truncate text if too long
        max_len = settings.KG_EXTRACTION_MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len] + "..."

        prompt = self.EXTRACTION_PROMPT.format(text=text)

        try:
            llm = self._get_llm_service()
            response = await llm.generate_response(
                query=prompt,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000,
            )
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def _normalize_entity_type(self, etype: str) -> str:
        """Normalize entity type to valid type."""
        etype = etype.lower().strip()
        if etype in ENTITY_TYPES:
            return etype
        # Map common variations
        type_map = {
            "organization": "org",
            "company": "org",
            "corporation": "org",
            "place": "location",
            "city": "location",
            "country": "location",
            "tool": "technology",
            "framework": "technology",
            "library": "technology",
            "language": "technology",
            "idea": "concept",
            "topic": "concept",
        }
        return type_map.get(etype, "other")

    def _normalize_relation_type(self, rtype: str) -> str:
        """Normalize relationship type to valid type."""
        rtype = rtype.lower().strip().replace(" ", "_").replace("-", "_")
        if rtype in RELATION_TYPES:
            return rtype
        # Map common variations
        type_map = {
            "employed_by": "works_for",
            "works_at": "works_for",
            "employee_of": "works_for",
            "leads": "manages",
            "supervises": "manages",
            "belongs_to": "part_of",
            "member_of": "part_of",
            "in": "located_in",
            "based_in": "located_in",
            "utilizes": "uses",
            "employs": "uses",
            "develops": "created_by",
            "made_by": "created_by",
            "associated_with": "related_to",
            "connected_to": "related_to",
        }
        return type_map.get(rtype, "related_to")

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
        rule_extractor: Optional[KnowledgeExtractor] = None
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
            extraction_result = await self.extract_from_text(text)

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

