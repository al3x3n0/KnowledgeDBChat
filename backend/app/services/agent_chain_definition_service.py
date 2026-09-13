"""Persistence and serialization for reusable agent-job chain definitions."""

from datetime import datetime
from typing import Iterable
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJobChainDefinition
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionResponse,
    AgentJobChainDefinitionUpdate,
)
from app.services.agent_job_chain_templates import (
    get_builtin_agent_job_chain_definition,
    list_builtin_agent_job_chain_definitions,
)
from app.services.agent_scope_service import normalize_scope_keys_deep


class AgentChainDefinitionError(RuntimeError):
    """Domain error translated to an HTTP response by the API layer."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class AgentChainDefinitionService:
    """Own chain-definition visibility, normalization, and persistence rules."""

    @staticmethod
    def to_response(
        chain: AgentJobChainDefinition | object,
    ) -> AgentJobChainDefinitionResponse:
        raw_steps = getattr(chain, "chain_steps", None)
        chain_steps: list[dict] = []
        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                item = dict(step)
                if isinstance(item.get("config"), dict):
                    item["config"] = normalize_scope_keys_deep(item["config"])
                chain_steps.append(item)

        return AgentJobChainDefinitionResponse(
            id=getattr(chain, "id"),
            name=getattr(chain, "name"),
            display_name=getattr(chain, "display_name"),
            description=getattr(chain, "description", None),
            chain_steps=chain_steps,
            default_settings=normalize_scope_keys_deep(
                getattr(chain, "default_settings", None)
            ),
            owner_user_id=getattr(chain, "owner_user_id", None),
            is_system=bool(getattr(chain, "is_system", False)),
            is_active=bool(getattr(chain, "is_active", True)),
            created_at=getattr(chain, "created_at"),
            updated_at=getattr(chain, "updated_at"),
        )

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        db: AsyncSession,
    ) -> list[AgentJobChainDefinition | object]:
        rows = list(
            (
                await db.execute(
                    select(AgentJobChainDefinition)
                    .where(
                        and_(
                            AgentJobChainDefinition.is_active.is_(True),
                            or_(
                                AgentJobChainDefinition.is_system.is_(True),
                                AgentJobChainDefinition.owner_user_id == user_id,
                            ),
                        )
                    )
                    .order_by(
                        AgentJobChainDefinition.is_system.desc(),
                        AgentJobChainDefinition.name,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [*rows, *list_builtin_agent_job_chain_definitions()]

    async def create(
        self,
        *,
        request: AgentJobChainDefinitionCreate,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJobChainDefinition:
        await self._ensure_name_available(request.name, db=db)
        chain = AgentJobChainDefinition(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            chain_steps=self._normalized_steps(request.chain_steps),
            default_settings=normalize_scope_keys_deep(request.default_settings),
            owner_user_id=user_id,
            is_system=False,
            is_active=True,
        )
        db.add(chain)
        await db.commit()
        await db.refresh(chain)
        return chain

    async def get_visible(
        self,
        *,
        chain_id: UUID,
        user_id: UUID,
        db: AsyncSession,
        require_active: bool = False,
    ) -> AgentJobChainDefinition | object:
        builtin = get_builtin_agent_job_chain_definition(chain_id)
        if builtin is not None:
            return builtin

        conditions = [
            AgentJobChainDefinition.id == chain_id,
            or_(
                AgentJobChainDefinition.is_system.is_(True),
                AgentJobChainDefinition.owner_user_id == user_id,
            ),
        ]
        if require_active:
            conditions.append(AgentJobChainDefinition.is_active.is_(True))
        chain = (
            await db.execute(select(AgentJobChainDefinition).where(and_(*conditions)))
        ).scalar_one_or_none()
        if chain is None:
            suffix = " or not active" if require_active else ""
            raise AgentChainDefinitionError(
                f"Chain definition not found{suffix}",
                status_code=404,
            )
        return chain

    async def update_owned(
        self,
        *,
        chain_id: UUID,
        request: AgentJobChainDefinitionUpdate,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJobChainDefinition:
        chain = await self._get_owned_editable(
            chain_id=chain_id,
            user_id=user_id,
            db=db,
            action="editable",
        )
        update_data = request.model_dump(exclude_unset=True)
        if "chain_steps" in update_data and request.chain_steps:
            update_data["chain_steps"] = self._normalized_steps(request.chain_steps)
        if "default_settings" in update_data:
            update_data["default_settings"] = normalize_scope_keys_deep(
                update_data["default_settings"]
            )
        for field, value in update_data.items():
            setattr(chain, field, value)
        chain.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(chain)
        return chain

    async def delete_owned(
        self,
        *,
        chain_id: UUID,
        user_id: UUID,
        db: AsyncSession,
    ) -> None:
        chain = await self._get_owned_editable(
            chain_id=chain_id,
            user_id=user_id,
            db=db,
            action="deletable",
        )
        await db.delete(chain)
        await db.commit()

    async def _get_owned_editable(
        self,
        *,
        chain_id: UUID,
        user_id: UUID,
        db: AsyncSession,
        action: str,
    ) -> AgentJobChainDefinition:
        chain = (
            await db.execute(
                select(AgentJobChainDefinition).where(
                    and_(
                        AgentJobChainDefinition.id == chain_id,
                        AgentJobChainDefinition.owner_user_id == user_id,
                        AgentJobChainDefinition.is_system.is_(False),
                    )
                )
            )
        ).scalar_one_or_none()
        if chain is None:
            raise AgentChainDefinitionError(
                f"Chain definition not found or not {action}",
                status_code=404,
            )
        return chain

    @staticmethod
    async def _ensure_name_available(
        name: str,
        *,
        db: AsyncSession,
    ) -> None:
        existing = (
            await db.execute(
                select(AgentJobChainDefinition).where(
                    AgentJobChainDefinition.name == name
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise AgentChainDefinitionError(
                "Chain definition with this name already exists",
                status_code=400,
            )

    @staticmethod
    def _normalized_steps(steps: Iterable[object]) -> list[dict]:
        normalized: list[dict] = []
        for step in steps:
            item = step.model_dump() if hasattr(step, "model_dump") else dict(step)
            if isinstance(item.get("config"), dict):
                item["config"] = normalize_scope_keys_deep(item["config"])
            normalized.append(item)
        return normalized


agent_chain_definition_service = AgentChainDefinitionService()
