"""
API endpoints for training datasets.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.training import (
    TrainingDatasetCreate,
    TrainingDatasetUpdate,
    TrainingDatasetResponse,
    TrainingDatasetListResponse,
    DatasetSampleCreate,
    DatasetSampleResponse,
    AddSamplesResponse,
    DatasetValidationResult,
    GenerateDatasetRequest,
)
from app.services.training_dataset_service import training_dataset_service
from app.schemas.ai_hub_dataset_presets import DatasetPresetsResponse, DatasetPresetInfo
from app.services.ai_hub_dataset_preset_service import ai_hub_dataset_preset_service

router = APIRouter()


def _dataset_to_response(dataset) -> TrainingDatasetResponse:
    """Convert dataset model to response schema."""
    return TrainingDatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        format=dataset.format,
        source_document_ids=dataset.source_document_ids,
        file_path=dataset.file_path,
        file_size=dataset.file_size,
        sample_count=dataset.sample_count,
        token_count=dataset.token_count,
        is_validated=dataset.is_validated,
        validation_errors=dataset.validation_errors,
        version=dataset.version,
        parent_dataset_id=dataset.parent_dataset_id,
        user_id=dataset.user_id,
        is_public=dataset.is_public,
        status=dataset.status,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


@router.post(
    "",
    response_model=TrainingDatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    data: TrainingDatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new training dataset."""
    dataset = await training_dataset_service.create_dataset(
        db=db,
        user_id=current_user.id,
        data=data,
    )
    return _dataset_to_response(dataset)


@router.get("", response_model=TrainingDatasetListResponse)
async def list_datasets(
    status: Optional[str] = Query(None, description="Filter by status"),
    dataset_type: Optional[str] = Query(None, description="Filter by type"),
    include_public: bool = Query(True, description="Include public datasets"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List training datasets for the current user."""
    datasets, total = await training_dataset_service.list_datasets(
        db=db,
        user_id=current_user.id,
        include_public=include_public,
        status=status,
        dataset_type=dataset_type,
        page=page,
        page_size=page_size,
    )

    return TrainingDatasetListResponse(
        datasets=[_dataset_to_response(d) for d in datasets],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/{dataset_id}", response_model=TrainingDatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a training dataset by ID."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    return _dataset_to_response(dataset)


@router.patch("/{dataset_id}", response_model=TrainingDatasetResponse)
async def update_dataset(
    dataset_id: UUID,
    data: TrainingDatasetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a training dataset."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this dataset",
        )

    # Update fields
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)

    await db.commit()
    await db.refresh(dataset)

    return _dataset_to_response(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a training dataset."""
    success = await training_dataset_service.delete_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or not authorized",
        )


@router.post("/{dataset_id}/samples", response_model=AddSamplesResponse)
async def add_samples(
    dataset_id: UUID,
    samples: list[DatasetSampleCreate],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Add samples to a dataset."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this dataset",
        )

    added = await training_dataset_service.add_samples(
        db=db,
        dataset_id=dataset_id,
        samples=samples,
    )

    # Refresh to get updated counts
    await db.refresh(dataset)

    return AddSamplesResponse(
        added_count=added,
        total_count=dataset.sample_count,
        token_count=dataset.token_count,
    )


@router.get("/{dataset_id}/samples")
async def get_samples(
    dataset_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    flagged_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get samples from a dataset."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    samples, total = await training_dataset_service.get_samples(
        db=db,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
        flagged_only=flagged_only,
    )

    return {
        "samples": [
            DatasetSampleResponse(
                id=s.id,
                dataset_id=s.dataset_id,
                sample_index=s.sample_index,
                content=s.content,
                source_document_id=s.source_document_id,
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                is_flagged=s.is_flagged,
                flag_reason=s.flag_reason,
                created_at=s.created_at,
            )
            for s in samples
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": (page * page_size) < total,
    }


@router.post("/{dataset_id}/validate", response_model=DatasetValidationResult)
async def validate_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Validate a dataset for training readiness."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    result = await training_dataset_service.validate_dataset(
        db=db,
        dataset_id=dataset_id,
    )

    return result


@router.post("/{dataset_id}/export")
async def export_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Export dataset to JSONL file in storage."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    file_path = await training_dataset_service.export_to_jsonl(
        db=db,
        dataset_id=dataset_id,
    )

    return {
        "file_path": file_path,
        "file_size": dataset.file_size,
    }


@router.get("/{dataset_id}/stats")
async def get_dataset_stats(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get detailed statistics for a dataset."""
    dataset = await training_dataset_service.get_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    stats = await training_dataset_service.get_dataset_stats(
        db=db,
        dataset_id=dataset_id,
    )

    return stats


@router.post("/generate-from-documents", response_model=TrainingDatasetResponse)
async def generate_from_documents(
    request: GenerateDatasetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate training dataset from documents using LLM."""
    dataset = await training_dataset_service.generate_from_documents(
        db=db,
        user_id=current_user.id,
        request=request,
    )

    return _dataset_to_response(dataset)


@router.get("/presets", response_model=DatasetPresetsResponse)
async def list_dataset_presets(
    current_user: User = Depends(get_current_active_user),
):
    presets = ai_hub_dataset_preset_service.list_presets()
    return DatasetPresetsResponse(
        presets=[
            DatasetPresetInfo(
                id=p.id,
                name=p.name,
                description=p.description,
                dataset_type=p.dataset_type,
            )
            for p in presets
        ]
    )


@router.get("/presets/enabled", response_model=DatasetPresetsResponse)
async def list_enabled_dataset_presets(
    current_user: User = Depends(get_current_active_user),
):
    presets = await ai_hub_dataset_preset_service.list_enabled_presets()
    return DatasetPresetsResponse(
        presets=[
            DatasetPresetInfo(
                id=p.id,
                name=p.name,
                description=p.description,
                dataset_type=p.dataset_type,
            )
            for p in presets
        ]
    )


@router.post("/{sample_id}/flag")
async def flag_sample(
    sample_id: UUID,
    reason: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Flag a sample as problematic."""
    sample = await training_dataset_service.flag_sample(
        db=db,
        sample_id=sample_id,
        reason=reason,
    )

    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample not found",
        )

    return {"success": True, "sample_id": str(sample.id)}
