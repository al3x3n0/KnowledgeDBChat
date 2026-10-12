"""
Model Registry Service for AI Hub.

Manages trained model adapters: listing, deployment to Ollama, and inference.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.model_registry import AdapterStatus, ModelAdapter
from app.schemas.training import DeployAdapterRequest, ModelAdapterUpdate
from app.services.storage_service import storage_service

# What Ollama accepts as a model name: a name and an optional tag.
_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")


class DeploymentFailed(Exception):
    """Ollama did not create the model. Not the caller's request at fault."""


def _one_line(value: Any) -> str:
    """A value safe to put on one line of a Modelfile.

    The file is line-oriented: a display name containing a line break would
    add a directive of its author's choosing.
    """
    return " ".join(str(value or "").split())


def _model_keys(name: str) -> set:
    """The names one Ollama model answers to: `x` and `x:latest` are one."""
    name = str(name or "").strip()
    return {name, f"{name}:latest"} if ":" not in name else {name}


class ModelRegistryService:
    """Service for managing model adapters and deployments."""

    def __init__(self):
        self.storage = storage_service
        self.ollama_url = settings.OLLAMA_BASE_URL

    async def get_adapter(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> Optional[ModelAdapter]:
        """Get an adapter by ID."""
        query = select(ModelAdapter).where(ModelAdapter.id == adapter_id)

        if user_id:
            query = query.where(
                (ModelAdapter.user_id == user_id) | (ModelAdapter.is_public.is_(True))
            )

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_adapters(
        self,
        db: AsyncSession,
        user_id: UUID,
        include_public: bool = True,
        status: Optional[str] = None,
        base_model: Optional[str] = None,
        is_deployed: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[ModelAdapter], int]:
        """List adapters for a user."""
        # Build base query
        if include_public:
            base_filter = (ModelAdapter.user_id == user_id) | (
                ModelAdapter.is_public.is_(True)
            )
        else:
            base_filter = ModelAdapter.user_id == user_id

        query = select(ModelAdapter).where(base_filter)

        # Apply filters
        if status:
            query = query.where(ModelAdapter.status == status)
        if base_model:
            query = query.where(ModelAdapter.base_model == base_model)
        if is_deployed is not None:
            query = query.where(ModelAdapter.is_deployed == is_deployed)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Apply pagination
        query = query.order_by(ModelAdapter.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        adapters = list(result.scalars().all())

        return adapters, total

    async def update_adapter(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
        data: ModelAdapterUpdate,
    ) -> Optional[ModelAdapter]:
        """Update an adapter."""
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter or adapter.user_id != user_id:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(adapter, field, value)

        adapter.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(adapter)

        return adapter

    async def delete_adapter(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Delete an adapter."""
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter or adapter.user_id != user_id:
            return False

        # Undeploy if deployed
        if adapter.is_deployed:
            await self.undeploy_from_ollama(db, adapter_id, user_id)

        # Delete adapter files from storage
        if adapter.adapter_path:
            try:
                await self.storage.delete_file(adapter.adapter_path)
            except Exception as e:
                logger.warning(f"Failed to delete adapter files: {e}")

        await db.delete(adapter)
        await db.commit()

        logger.info(f"Deleted adapter {adapter_id}")
        return True

    async def _owned_adapter(
        self, db: AsyncSession, adapter_id: UUID, user_id: UUID
    ) -> ModelAdapter:
        """An adapter the caller may change, or ValueError.

        `get_adapter` also returns other people's public adapters, which is
        right for reading and testing. Deploying and undeploying went through
        it too, so anyone could do either to an adapter that was merely
        shared with them.
        """
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter or adapter.user_id != user_id:
            raise ValueError(f"Adapter {adapter_id} not found")
        return adapter

    async def _name_is_taken(
        self, db: AsyncSession, adapter: ModelAdapter, model_name: str
    ) -> Optional[str]:
        """Why this name cannot be used, or None.

        `ollama create` replaces a model of the same name and undeploying
        deletes by name, so a deployment called after the installation's own
        chat model would overwrite it and later remove it.
        """
        wanted = _model_keys(model_name)
        others = await db.execute(
            select(ModelAdapter).where(
                ModelAdapter.is_deployed.is_(True), ModelAdapter.id != adapter.id
            )
        )
        for other in others.scalars().all():
            if wanted & _model_keys(other.get_ollama_model_name() or ""):
                return "another adapter is deployed under that name"
        for existing in await self.get_deployed_models():
            if wanted & _model_keys(existing.get("name") or ""):
                return "Ollama already has a model of that name"
        return None

    async def deploy_to_ollama(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
        request: Optional[DeployAdapterRequest] = None,
    ) -> ModelAdapter:
        """
        Deploy an adapter to Ollama as a custom model.

        Creates a Modelfile and registers the model with Ollama. Raises
        ValueError for a request that cannot be honoured and DeploymentFailed
        when Ollama did not create the model.
        """
        adapter = await self._owned_adapter(db, adapter_id, user_id)

        if not adapter.can_deploy():
            raise ValueError(f"Adapter cannot be deployed (status: {adapter.status})")

        # Determine model name
        model_name = (
            request.ollama_model_name
            if request and request.ollama_model_name
            else adapter.name
        ).strip()
        if not _MODEL_NAME.match(model_name):
            raise ValueError(
                "Model name must be lower-case letters, digits, '.', '_' or "
                "'-', with an optional ':tag'"
            )
        taken = await self._name_is_taken(db, adapter, model_name)
        if taken:
            raise ValueError(f"Cannot deploy as '{model_name}': {taken}")

        adapter.status = AdapterStatus.DEPLOYING.value
        await db.commit()

        try:
            await self._create_ollama_model(adapter, model_name)
        except Exception as e:
            # Back to READY, not FAILED: `can_deploy` requires READY, so one
            # outage used to leave the adapter undeployable for good.
            adapter.status = AdapterStatus.READY.value
            adapter.deployment_config = {"last_error": str(e)[:500]}
            await db.commit()
            raise DeploymentFailed(f"Failed to deploy adapter: {e}") from e

        adapter.is_deployed = True
        adapter.status = AdapterStatus.DEPLOYED.value
        adapter.deployment_config = {
            "ollama_model_name": model_name,
            "deployed_at": datetime.utcnow().isoformat(),
        }
        adapter.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(adapter)

        logger.info(f"Deployed adapter {adapter_id} as Ollama model '{model_name}'")
        return adapter

    async def _create_ollama_model(
        self,
        adapter: ModelAdapter,
        model_name: str,
    ):
        """
        Create a custom model in Ollama, or raise.

        The Modelfile names the base model only: **the adapter's weights are
        not applied**, so what is created answers as the base model under a
        new name. Merging them is not built.

        Every failure used to be logged and ignored ("continue anyway for
        demo purposes"), so an adapter was recorded as deployed with Ollama
        unreachable or refusing the request.
        """
        metrics = adapter.training_metrics or {}
        modelfile = f"""FROM {_one_line(adapter.base_model)}

# Custom adapter: {_one_line(adapter.display_name)}
# Trained on: {_one_line(metrics.get('training_date', 'unknown'))}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/create",
                    json={
                        "name": model_name,
                        "modelfile": modelfile,
                        "stream": False,
                    },
                    timeout=300.0,  # Model creation can take time
                )
            except httpx.TimeoutException as e:
                raise RuntimeError("Ollama did not answer in time") from e
            except httpx.HTTPError as e:
                raise RuntimeError(f"Ollama is not reachable: {e}") from e

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Ollama answered {response.status_code}: {response.text[:300]}"
            )

    async def undeploy_from_ollama(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
    ) -> ModelAdapter:
        """Undeploy an adapter from Ollama."""
        adapter = await self._owned_adapter(db, adapter_id, user_id)

        if not adapter.is_deployed:
            raise ValueError("Adapter is not deployed")

        # Get model name from deployment config
        model_name = adapter.get_ollama_model_name()

        if model_name:
            # `AsyncClient.delete` takes no body, so `delete(..., json=...)`
            # raised TypeError into the `except` that used to be here: no
            # undeploy ever removed its model, and the adapter was marked
            # undeployed all the same.
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        "DELETE",
                        f"{self.ollama_url}/api/delete",
                        json={"name": model_name},
                        timeout=30.0,
                    )
            except httpx.HTTPError as e:
                raise DeploymentFailed(f"Ollama is not reachable: {e}") from e
            # 404: already gone, which is what was asked for.
            if response.status_code not in (200, 404):
                raise DeploymentFailed(
                    f"Ollama answered {response.status_code}: {response.text[:300]}"
                )

        # Update adapter record
        adapter.is_deployed = False
        adapter.status = AdapterStatus.READY.value
        adapter.deployment_config = None
        adapter.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(adapter)

        logger.info(f"Undeployed adapter {adapter_id}")
        return adapter

    async def test_adapter(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Test an adapter with a prompt.

        For deployed adapters, uses the Ollama model.
        For non-deployed adapters, this is not available.
        """
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter:
            raise ValueError(f"Adapter {adapter_id} not found")

        if not adapter.is_deployed:
            raise ValueError("Adapter must be deployed to test. Deploy first.")

        model_name = adapter.get_ollama_model_name()
        if not model_name:
            raise ValueError("No Ollama model name configured")

        import time

        start_time = time.time()

        # Generate with Ollama
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                    "stream": False,
                },
                timeout=60.0,
            )

            if response.status_code != 200:
                raise ValueError(f"Ollama generate failed: {response.text}")

            data = response.json()
            generated_text = data.get("response", "")
            generation_time = int((time.time() - start_time) * 1000)

            # Update usage count
            adapter.increment_usage()
            await db.commit()

            return {
                "prompt": prompt,
                "response": generated_text,
                "tokens_generated": len(generated_text.split()),  # Rough estimate
                "generation_time_ms": generation_time,
            }

    async def get_deployed_models(self) -> List[Dict[str, Any]]:
        """Get list of deployed custom models from Ollama."""
        models = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        models.append(
                            {
                                "name": model["name"],
                                "size": model.get("size", 0),
                                "modified_at": model.get("modified_at"),
                            }
                        )

        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")

        return models

    async def get_stats(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> Dict[str, Any]:
        """Get adapter statistics for a user."""
        result = await db.execute(
            select(
                func.count(ModelAdapter.id).label("total"),
                func.count(ModelAdapter.id)
                .filter(ModelAdapter.is_deployed.is_(True))
                .label("deployed"),
                func.sum(ModelAdapter.usage_count).label("total_usage"),
            ).where(ModelAdapter.user_id == user_id)
        )
        row = result.one()

        return {
            "total_adapters": row.total or 0,
            "deployed_adapters": row.deployed or 0,
            "total_usage": row.total_usage or 0,
        }


# Global service instance
model_registry_service = ModelRegistryService()
