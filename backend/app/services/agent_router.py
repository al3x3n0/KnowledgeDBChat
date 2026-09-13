"""Compatibility shim for the extracted router with DB loading helpers."""

from typing import Dict, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core.routing import CAPABILITY_KEYWORDS
from app.agent_core.routing import AgentRouter as CoreAgentRouter
from app.models.agent_definition import AgentDefinition
from app.services.llm_service import LLMService


class AgentRouter(CoreAgentRouter):
    def __init__(self, llm_service: Optional[LLMService] = None):
        super().__init__(llm_service=llm_service or LLMService())

    async def load_agents(self, db: AsyncSession) -> Dict[str, AgentDefinition]:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.is_active.is_(True))
        )
        agents = result.scalars().all()
        cache = {agent.name: agent for agent in agents}
        self.set_agents(cache)
        logger.info(f"Loaded {len(cache)} active agents")
        return cache

    async def get_agent_by_name(
        self,
        name: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[AgentDefinition]:
        if name in self._agent_cache:
            return self._agent_cache.get(name)
        if db is None:
            return None
        result = await db.execute(
            select(AgentDefinition).where(
                AgentDefinition.name == name,
                AgentDefinition.is_active.is_(True),
            )
        )
        agent = result.scalar_one_or_none()
        if agent is not None:
            self._agent_cache[name] = agent
        return agent

    async def get_generalist_agent(
        self,
        db: Optional[AsyncSession] = None,
    ) -> Optional[AgentDefinition]:
        return await self.get_agent_by_name("generalist", db)


__all__ = ["AgentRouter", "CAPABILITY_KEYWORDS"]
