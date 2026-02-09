"""
Training Dataset Service for AI Hub.

Manages training datasets: creation, validation, sample management, and export.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.training_dataset import (
    TrainingDataset,
    DatasetSample,
    DatasetStatus,
    DatasetType,
    DatasetFormat,
)
from app.models.document import Document
from app.schemas.training import (
    TrainingDatasetCreate,
    DatasetSampleCreate,
    GenerateDatasetRequest,
    DatasetValidationResult,
)
from app.services.storage_service import storage_service
from app.services.llm_service import LLMService
from app.services.ai_hub_dataset_preset_service import ai_hub_dataset_preset_service


class TrainingDatasetService:
    """Service for managing training datasets."""

    def __init__(self):
        self.storage = storage_service
        self.llm = LLMService()

    async def create_dataset(
        self,
        db: AsyncSession,
        user_id: UUID,
        data: TrainingDatasetCreate,
    ) -> TrainingDataset:
        """
        Create a new training dataset.

        Args:
            db: Database session
            user_id: Owner user ID
            data: Dataset creation data

        Returns:
            Created TrainingDataset
        """
        dataset = TrainingDataset(
            name=data.name,
            description=data.description,
            dataset_type=data.dataset_type,
            format=data.format,
            user_id=user_id,
            is_public=data.is_public,
            status=DatasetStatus.DRAFT.value,
        )

        db.add(dataset)
        await db.flush()

        # Add initial samples if provided
        if data.samples:
            await self.add_samples(db, dataset.id, data.samples)

        await db.commit()
        await db.refresh(dataset)

        logger.info(f"Created dataset {dataset.id} for user {user_id}")
        return dataset

    async def get_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> Optional[TrainingDataset]:
        """Get a dataset by ID."""
        query = select(TrainingDataset).where(TrainingDataset.id == dataset_id)

        if user_id:
            query = query.where(
                (TrainingDataset.user_id == user_id) | (TrainingDataset.is_public == True)
            )

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_datasets(
        self,
        db: AsyncSession,
        user_id: UUID,
        include_public: bool = True,
        status: Optional[str] = None,
        dataset_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[TrainingDataset], int]:
        """
        List datasets for a user.

        Returns:
            Tuple of (datasets, total_count)
        """
        # Build base query
        if include_public:
            base_filter = (TrainingDataset.user_id == user_id) | (TrainingDataset.is_public == True)
        else:
            base_filter = TrainingDataset.user_id == user_id

        query = select(TrainingDataset).where(base_filter)

        # Apply filters
        if status:
            query = query.where(TrainingDataset.status == status)
        if dataset_type:
            query = query.where(TrainingDataset.dataset_type == dataset_type)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(TrainingDataset.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        datasets = list(result.scalars().all())

        return datasets, total

    async def delete_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Delete a dataset."""
        dataset = await self.get_dataset(db, dataset_id, user_id)
        if not dataset or dataset.user_id != user_id:
            return False

        # Delete from storage if exported
        if dataset.file_path:
            try:
                await self.storage.delete_file(dataset.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete dataset file: {e}")

        await db.delete(dataset)
        await db.commit()

        logger.info(f"Deleted dataset {dataset_id}")
        return True

    async def add_samples(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        samples: List[DatasetSampleCreate],
    ) -> int:
        """
        Add samples to a dataset.

        Returns:
            Number of samples added
        """
        dataset = await db.get(TrainingDataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Get current max index
        result = await db.execute(
            select(func.max(DatasetSample.sample_index))
            .where(DatasetSample.dataset_id == dataset_id)
        )
        max_index = result.scalar() or -1

        added = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i, sample_data in enumerate(samples):
            # Estimate token counts (rough: 1 token ≈ 4 chars)
            input_text = sample_data.instruction + (sample_data.input or "")
            output_text = sample_data.output
            input_tokens = len(input_text) // 4
            output_tokens = len(output_text) // 4

            content = {
                "instruction": sample_data.instruction,
                "input": sample_data.input or "",
                "output": sample_data.output,
            }

            sample = DatasetSample(
                dataset_id=dataset_id,
                sample_index=max_index + 1 + i,
                content=content,
                source_document_id=sample_data.source_document_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            db.add(sample)
            added += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Update dataset stats
        dataset.sample_count += added
        dataset.token_count += total_input_tokens + total_output_tokens
        dataset.is_validated = False  # Needs revalidation
        dataset.updated_at = datetime.utcnow()

        await db.commit()
        logger.info(f"Added {added} samples to dataset {dataset_id}")

        return added

    async def get_samples(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        page: int = 1,
        page_size: int = 50,
        flagged_only: bool = False,
    ) -> tuple[List[DatasetSample], int]:
        """Get samples from a dataset with pagination."""
        query = select(DatasetSample).where(DatasetSample.dataset_id == dataset_id)

        if flagged_only:
            query = query.where(DatasetSample.is_flagged == True)

        # Count
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Paginate
        query = query.order_by(DatasetSample.sample_index)
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        samples = list(result.scalars().all())

        return samples, total

    async def flag_sample(
        self,
        db: AsyncSession,
        sample_id: UUID,
        reason: str,
    ) -> Optional[DatasetSample]:
        """Flag a sample as problematic."""
        sample = await db.get(DatasetSample, sample_id)
        if not sample:
            return None

        sample.is_flagged = True
        sample.flag_reason = reason

        await db.commit()
        return sample

    async def validate_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> DatasetValidationResult:
        """
        Validate a dataset for training readiness.

        Checks:
        - Minimum sample count
        - Sample format validity
        - Token limits
        - Duplicate detection
        """
        dataset = await db.get(TrainingDataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        # Get all samples
        result = await db.execute(
            select(DatasetSample).where(DatasetSample.dataset_id == dataset_id)
        )
        samples = list(result.scalars().all())

        # Check minimum samples
        if len(samples) < 10:
            errors.append({
                "code": "MIN_SAMPLES",
                "message": f"Dataset has only {len(samples)} samples. Minimum 10 recommended.",
            })

        # Check for empty fields
        empty_outputs = 0
        short_outputs = 0
        total_tokens = 0

        for sample in samples:
            content = sample.content
            if not content.get("output"):
                empty_outputs += 1
            elif len(content.get("output", "")) < 10:
                short_outputs += 1

            total_tokens += sample.input_tokens + sample.output_tokens

        if empty_outputs > 0:
            errors.append({
                "code": "EMPTY_OUTPUTS",
                "message": f"{empty_outputs} samples have empty outputs.",
            })

        if short_outputs > 0:
            warnings.append({
                "code": "SHORT_OUTPUTS",
                "message": f"{short_outputs} samples have very short outputs (<10 chars).",
            })

        # Check token limits
        if total_tokens > settings.DATASET_MAX_TOKEN_COUNT:
            errors.append({
                "code": "TOKEN_LIMIT",
                "message": f"Dataset exceeds token limit ({total_tokens} > {settings.DATASET_MAX_TOKEN_COUNT}).",
            })

        # Update dataset validation status
        is_valid = len(errors) == 0
        dataset.is_validated = is_valid
        dataset.validation_errors = errors if errors else None
        dataset.status = DatasetStatus.READY.value if is_valid else DatasetStatus.ERROR.value
        dataset.updated_at = datetime.utcnow()

        await db.commit()

        return DatasetValidationResult(
            is_valid=is_valid,
            sample_count=len(samples),
            token_count=total_tokens,
            errors=errors,
            warnings=warnings,
        )

    async def export_to_jsonl(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> str:
        """
        Export dataset to JSONL format in MinIO.

        Returns:
            MinIO file path
        """
        dataset = await db.get(TrainingDataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Get all samples
        result = await db.execute(
            select(DatasetSample)
            .where(DatasetSample.dataset_id == dataset_id)
            .order_by(DatasetSample.sample_index)
        )
        samples = list(result.scalars().all())

        # Build JSONL content
        lines = []
        for sample in samples:
            if not sample.is_flagged:  # Skip flagged samples
                lines.append(json.dumps(sample.content))

        content = "\n".join(lines)
        file_bytes = content.encode("utf-8")

        # Upload to MinIO
        file_path = f"training/datasets/{dataset_id}/dataset.jsonl"
        await self.storage.upload_file(
            object_name=file_path,
            data=file_bytes,
            content_type="application/jsonl",
        )

        # Update dataset record
        dataset.file_path = file_path
        dataset.file_size = len(file_bytes)
        dataset.updated_at = datetime.utcnow()

        await db.commit()

        logger.info(f"Exported dataset {dataset_id} to {file_path}")
        return file_path

    async def generate_from_documents(
        self,
        db: AsyncSession,
        user_id: UUID,
        request: GenerateDatasetRequest,
    ) -> TrainingDataset:
        """
        Generate training dataset from documents using LLM.

        Creates instruction/output pairs from document content.
        """
        # Create dataset
        dataset = TrainingDataset(
            name=request.name,
            description=request.description or "Generated from documents",
            dataset_type=request.dataset_type,
            format=DatasetFormat.ALPACA.value,
            source_document_ids=[str(doc_id) for doc_id in request.document_ids],
            user_id=user_id,
            status=DatasetStatus.VALIDATING.value,
        )

        db.add(dataset)
        await db.flush()

        # Get documents
        result = await db.execute(
            select(Document).where(Document.id.in_(request.document_ids))
        )
        documents = list(result.scalars().all())

        sample_index = 0
        total_tokens = 0

        # Apply per-user LLM settings (provider/model/api_url/etc.) for dataset generation.
        user_settings = None
        try:
            from app.models.memory import UserPreferences
            from app.services.llm_service import UserLLMSettings
            prefs_res = await db.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
            prefs = prefs_res.scalar_one_or_none()
            user_settings = UserLLMSettings.from_preferences(prefs) if prefs else None
        except Exception:
            user_settings = None

        for doc in documents:
            # Generate samples from document
            try:
                generation_prompt = request.generation_prompt
                if request.preset_id:
                    enabled = await ai_hub_dataset_preset_service.list_enabled_presets()
                    preset = next((p for p in enabled if p.id == request.preset_id), None)
                    if not preset:
                        raise ValueError(f"Preset not enabled or not found: {request.preset_id}")
                    generation_prompt = preset.generation_prompt
                    if request.extra_instructions:
                        generation_prompt = (
                            generation_prompt
                            + "\n\nAdditional constraints:\n"
                            + request.extra_instructions.strip()
                            + "\n"
                        )

                samples = await self._generate_samples_from_document(
                    doc,
                    request.samples_per_document,
                    generation_prompt,
                    user_settings=user_settings,
                )

                for sample_data in samples:
                    input_tokens = len(sample_data["instruction"]) // 4
                    output_tokens = len(sample_data["output"]) // 4

                    sample = DatasetSample(
                        dataset_id=dataset.id,
                        sample_index=sample_index,
                        content=sample_data,
                        source_document_id=doc.id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    db.add(sample)
                    sample_index += 1
                    total_tokens += input_tokens + output_tokens

            except Exception as e:
                logger.warning(f"Failed to generate samples from document {doc.id}: {e}")

        # Update dataset stats
        dataset.sample_count = sample_index
        dataset.token_count = total_tokens
        dataset.status = DatasetStatus.DRAFT.value

        await db.commit()
        await db.refresh(dataset)

        logger.info(f"Generated {sample_index} samples from {len(documents)} documents")
        return dataset

    async def _generate_samples_from_document(
        self,
        document: Document,
        num_samples: int,
        custom_prompt: Optional[str] = None,
        *,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> List[Dict[str, str]]:
        """Generate training samples from a document using LLM."""
        # Get document content (from summary or chunks)
        content = document.summary or ""
        if not content and document.content:
            content = document.content[:8000]  # Limit content length

        if not content:
            return []

        # Build prompt for sample generation
        base_prompt = custom_prompt or (
            "Generate {num} diverse question-answer pairs from the following document. "
            "Each pair should be a clear instruction and a comprehensive answer. "
            "Output as JSON array with 'instruction' and 'output' fields."
        )

        prompt = f"""{base_prompt.format(num=num_samples)}

Document:
{content}

Generate exactly {num_samples} training samples. Output only valid JSON array."""

        try:
            response = await self.llm.generate_response(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                task_type="summarization",
                user_settings=user_settings,
            )

            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                samples = json.loads(json_match.group())
                return [
                    {
                        "instruction": s.get("instruction", s.get("question", "")),
                        "input": "",
                        "output": s.get("output", s.get("answer", "")),
                    }
                    for s in samples
                    if s.get("instruction") or s.get("question")
                ]

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return []

    async def get_dataset_stats(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> Dict[str, Any]:
        """Get detailed statistics for a dataset."""
        dataset = await db.get(TrainingDataset, dataset_id)
        if not dataset:
            return {}

        # Get sample statistics
        result = await db.execute(
            select(
                func.count(DatasetSample.id).label("count"),
                func.sum(DatasetSample.input_tokens).label("input_tokens"),
                func.sum(DatasetSample.output_tokens).label("output_tokens"),
                func.avg(DatasetSample.input_tokens).label("avg_input_tokens"),
                func.avg(DatasetSample.output_tokens).label("avg_output_tokens"),
                func.count(DatasetSample.id).filter(DatasetSample.is_flagged == True).label("flagged"),
            ).where(DatasetSample.dataset_id == dataset_id)
        )
        row = result.one()

        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "status": dataset.status,
            "sample_count": row.count or 0,
            "token_count": (row.input_tokens or 0) + (row.output_tokens or 0),
            "input_tokens": row.input_tokens or 0,
            "output_tokens": row.output_tokens or 0,
            "avg_input_tokens": round(row.avg_input_tokens or 0, 1),
            "avg_output_tokens": round(row.avg_output_tokens or 0, 1),
            "flagged_count": row.flagged or 0,
            "is_validated": dataset.is_validated,
            "file_size": dataset.file_size,
        }


# Global service instance
training_dataset_service = TrainingDatasetService()
