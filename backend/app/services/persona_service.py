"""
Persona service utilities for linking documents, users, and detections.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional, Dict, Any, Iterable
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.persona import Persona, DocumentPersonaDetection
from app.models.document import Document
from app.models.user import User


class PersonaService:
    """Provides helpers for creating personas and recording detections."""

    def __init__(self) -> None:
        self._slug_re = re.compile(r"[^a-z0-9]+")

    def _slugify(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        slug = self._slug_re.sub("-", value.strip().lower())
        slug = slug.strip("-")
        return slug or None

    def _scoped_platform_id(self, scope: Optional[str], identifier: Optional[str]) -> Optional[str]:
        if not identifier:
            return None
        ident = identifier.strip()
        return f"{scope}:{ident}" if scope else ident

    async def _get_by_platform_id(self, db: AsyncSession, platform_id: str) -> Optional[Persona]:
        result = await db.execute(select(Persona).where(Persona.platform_id == platform_id))
        return result.scalar_one_or_none()

    async def _get_by_name(self, db: AsyncSession, name: str) -> Optional[Persona]:
        result = await db.execute(select(Persona).where(func.lower(Persona.name) == name.lower()))
        return result.scalar_one_or_none()

    async def ensure_persona(
        self,
        db: AsyncSession,
        *,
        name: Optional[str],
        platform_id: Optional[str],
        defaults: Optional[Dict[str, Any]] = None,
        link_user_id: Optional[UUID] = None,
        is_system: bool = False,
    ) -> Persona:
        """Return an existing persona or create a new one."""
        if not name and not platform_id:
            raise ValueError("Persona requires at least a name or platform_id")

        persona: Optional[Persona] = None
        if platform_id:
            persona = await self._get_by_platform_id(db, platform_id)
        if not persona and name:
            persona = await self._get_by_name(db, name.strip())

        if persona:
            updated = False
            if link_user_id and persona.user_id != link_user_id:
                persona.user_id = link_user_id
                updated = True
            if defaults:
                for key, value in defaults.items():
                    if value is None:
                        continue
                    current = getattr(persona, key, None)
                    if current != value:
                        setattr(persona, key, value)
                        updated = True
            if updated:
                await db.flush()
            return persona

        persona = Persona(
            name=(name or platform_id or f"Persona-{uuid.uuid4().hex[:6]}")[:255],
            platform_id=platform_id,
            user_id=link_user_id,
            is_system=is_system,
        )
        if defaults:
            for key, value in defaults.items():
                if value is not None and hasattr(persona, key):
                    setattr(persona, key, value)
        db.add(persona)
        await db.flush()
        return persona

    async def ensure_user_persona(self, db: AsyncSession, user: User) -> Persona:
        """Return persona linked to a platform user."""
        display_name = user.full_name or user.username or user.email or f"User {user.id}"
        platform_id = f"user:{user.id}"
        defaults = {
            "avatar_url": user.avatar_url,
            "description": user.full_name or user.username,
            "is_active": user.is_active,
            "extra_metadata": {"username": user.username, "email": user.email},
        }
        persona = await self.ensure_persona(
            db,
            name=display_name,
            platform_id=platform_id,
            defaults=defaults,
            link_user_id=user.id,
            is_system=False,
        )
        if persona.is_active != user.is_active:
            persona.is_active = user.is_active
            await db.flush()
        return persona

    async def assign_document_owner(
        self,
        db: AsyncSession,
        document: Document,
        *,
        user: Optional[User] = None,
        author_name: Optional[str] = None,
        platform_scope: Optional[str] = None,
        platform_identifier: Optional[str] = None,
        is_system: bool = True,
    ) -> Optional[Persona]:
        """Attach a persona as the owner of a document."""
        if user:
            persona = await self.ensure_user_persona(db, user)
            author_name = author_name or persona.name
            platform_scope = platform_scope or "user"
            platform_identifier = str(user.id)
        else:
            cleaned_name = (author_name or document.author or "").strip()
            if not cleaned_name and not platform_identifier:
                return None
            slug = platform_identifier or self._slugify(cleaned_name)
            platform_id = self._scoped_platform_id(platform_scope, slug)
            persona = await self.ensure_persona(
                db,
                name=cleaned_name or platform_id,
                platform_id=platform_id,
                defaults={"is_system": is_system},
                is_system=is_system,
            )

        if not persona:
            return None

        document.owner_persona_id = persona.id
        if author_name and not document.author:
            document.author = author_name
        await db.flush()

        await self.record_detection(
            db,
            document_id=document.id,
            persona=persona,
            role="owner",
            detection_type="owner",
            details={"scope": platform_scope} if platform_scope else None,
        )
        return persona

    async def clear_detections(
        self,
        db: AsyncSession,
        document_id: UUID,
        *,
        role: Optional[str] = None,
        detection_type: Optional[str] = None,
    ) -> None:
        """Remove persona detections for a document."""
        stmt = delete(DocumentPersonaDetection).where(DocumentPersonaDetection.document_id == document_id)
        if role:
            stmt = stmt.where(DocumentPersonaDetection.role == role)
        if detection_type:
            stmt = stmt.where(DocumentPersonaDetection.detection_type == detection_type)
        await db.execute(stmt)
        await db.flush()

    async def record_detection(
        self,
        db: AsyncSession,
        *,
        document_id: UUID,
        persona: Persona,
        role: str,
        detection_type: Optional[str],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        confidence: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> DocumentPersonaDetection:
        """Create or update a persona detection for a document."""
        filters = [
            DocumentPersonaDetection.document_id == document_id,
            DocumentPersonaDetection.persona_id == persona.id,
            DocumentPersonaDetection.role == role,
        ]
        if start_time is None:
            filters.append(DocumentPersonaDetection.start_time.is_(None))
        else:
            filters.append(DocumentPersonaDetection.start_time == float(start_time))
        if end_time is None:
            filters.append(DocumentPersonaDetection.end_time.is_(None))
        else:
            filters.append(DocumentPersonaDetection.end_time == float(end_time))

        result = await db.execute(select(DocumentPersonaDetection).where(and_(*filters)))
        detection = result.scalar_one_or_none()
        if detection:
            updated = False
            if detection_type and detection.detection_type != detection_type:
                detection.detection_type = detection_type
                updated = True
            if confidence is not None and detection.confidence != confidence:
                detection.confidence = confidence
                updated = True
            if details:
                merged = dict(detection.details or {})
                merged.update(details)
                detection.details = merged
                updated = True
            if updated:
                await db.flush()
            return detection

        detection = DocumentPersonaDetection(
            document_id=document_id,
            persona_id=persona.id,
            role=role,
            detection_type=detection_type,
            confidence=confidence,
            start_time=float(start_time) if start_time is not None else None,
            end_time=float(end_time) if end_time is not None else None,
            details=details,
        )
        db.add(detection)
        await db.flush()
        return detection

    async def record_sentence_speakers(
        self,
        db: AsyncSession,
        *,
        document: Document,
        sentence_segments: Optional[Iterable[Dict[str, Any]]],
        base_document_id: Optional[UUID] = None,
    ) -> None:
        """Persist diarized speakers from transcription metadata."""
        await self.clear_detections(db, document.id, role="speaker", detection_type="diarization")
        if not sentence_segments:
            return

        scope_id = base_document_id or document.id
        scope = f"doc:{scope_id}:speaker"
        persona_cache: Dict[str, Persona] = {}

        for segment in sentence_segments:
            speaker_label = (segment.get("speaker") or "").strip()
            if not speaker_label:
                continue
            persona = persona_cache.get(speaker_label)
            if not persona:
                slug = self._slugify(speaker_label) or uuid.uuid4().hex[:6]
                platform_id = self._scoped_platform_id(scope, slug)
                persona = await self.ensure_persona(
                    db,
                    name=speaker_label,
                    platform_id=platform_id,
                    defaults={
                        "description": f"Speaker detected via diarization for document {scope_id}",
                        "is_system": True,
                    },
                    is_system=True,
                )
                persona_cache[speaker_label] = persona

            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text")
            try:
                start_val = float(start) if start is not None else None
            except Exception:
                start_val = None
            try:
                end_val = float(end) if end is not None else start_val
            except Exception:
                end_val = start_val

            await self.record_detection(
                db,
                document_id=document.id,
                persona=persona,
                role="speaker",
                detection_type="diarization",
                start_time=start_val,
                end_time=end_val,
                details={"text": text} if text else None,
            )


persona_service = PersonaService()
