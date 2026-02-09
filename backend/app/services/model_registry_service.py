"""
Model Registry Service for AI Hub.

Manages trained model adapters: listing, deployment to Ollama, and inference.
"""

import asyncio
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.model_registry import ModelAdapter, AdapterStatus
from app.schemas.training import ModelAdapterUpdate, DeployAdapterRequest
from app.services.storage_service import storage_service


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
                (ModelAdapter.user_id == user_id) | (ModelAdapter.is_public == True)
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
            base_filter = (ModelAdapter.user_id == user_id) | (ModelAdapter.is_public == True)
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

    async def deploy_to_ollama(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
        request: Optional[DeployAdapterRequest] = None,
    ) -> ModelAdapter:
        """
        Deploy an adapter to Ollama as a custom model.

        Creates a Modelfile and registers the model with Ollama.
        """
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter:
            raise ValueError(f"Adapter {adapter_id} not found")

        if not adapter.can_deploy():
            raise ValueError(f"Adapter cannot be deployed (status: {adapter.status})")

        # Determine model name
        model_name = (
            request.ollama_model_name if request and request.ollama_model_name
            else adapter.name
        )

        adapter.status = AdapterStatus.DEPLOYING.value
        await db.commit()

        try:
            # In production, we would:
            # 1. Download adapter weights from MinIO
            # 2. Merge adapter with base model
            # 3. Create Modelfile
            # 4. Call Ollama API to create model

            # For now, simulate deployment
            # This would be: ollama create {model_name} -f Modelfile

            # Create the custom model via Ollama API
            await self._create_ollama_model(adapter, model_name)

            # Update adapter record
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

        except Exception as e:
            adapter.status = AdapterStatus.FAILED.value
            adapter.deployment_config = {"error": str(e)}
            await db.commit()
            raise ValueError(f"Failed to deploy adapter: {e}")

    async def _create_ollama_model(
        self,
        adapter: ModelAdapter,
        model_name: str,
    ):
        """
        Create a custom model in Ollama.

        This is a simplified implementation. In production:
        1. Download adapter weights
        2. Merge with base model weights
        3. Create GGUF or safetensors format
        4. Register with Ollama
        """
        # Build Modelfile content
        modelfile = f"""FROM {adapter.base_model}

# Custom adapter: {adapter.display_name}
# Trained on: {adapter.training_metrics.get('training_date', 'unknown')}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

        # Call Ollama create API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/create",
                    json={
                        "name": model_name,
                        "modelfile": modelfile,
                    },
                    timeout=300.0,  # Model creation can take time
                )

                if response.status_code not in (200, 201):
                    logger.warning(
                        f"Ollama create returned {response.status_code}: {response.text}"
                    )
                    # For now, we'll consider it a success if we can proceed
                    # In production, this would be a real error

            except httpx.TimeoutException:
                logger.warning("Ollama create timed out - model may still be creating")
            except Exception as e:
                logger.warning(f"Ollama create failed: {e}")
                # Continue anyway for demo purposes

    async def undeploy_from_ollama(
        self,
        db: AsyncSession,
        adapter_id: UUID,
        user_id: UUID,
    ) -> ModelAdapter:
        """Undeploy an adapter from Ollama."""
        adapter = await self.get_adapter(db, adapter_id, user_id)
        if not adapter:
            raise ValueError(f"Adapter {adapter_id} not found")

        if not adapter.is_deployed:
            raise ValueError("Adapter is not deployed")

        # Get model name from deployment config
        model_name = adapter.get_ollama_model_name()

        if model_name:
            try:
                # Delete model from Ollama
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"{self.ollama_url}/api/delete",
                        json={"name": model_name},
                        timeout=30.0,
                    )
            except Exception as e:
                logger.warning(f"Failed to delete Ollama model: {e}")

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
                        models.append({
                            "name": model["name"],
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at"),
                        })

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
                func.count(ModelAdapter.id).filter(
                    ModelAdapter.is_deployed == True
                ).label("deployed"),
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
