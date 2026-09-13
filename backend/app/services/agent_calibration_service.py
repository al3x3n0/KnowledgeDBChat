"""Score an agent's methodology by how well it predicts what is measured.

Two rules give the score its meaning, and both are about order rather than
arithmetic:

* a prediction cannot be created already settled, so a row is always a claim
  made before its outcome was known;
* a prediction cannot be edited once it has been measured, so a claim cannot be
  quietly moved to where the result landed.

Without those, the error column measures nothing: an agent that writes its
prediction after seeing the answer scores perfectly and has learned nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_prediction import AgentPrediction


class CalibrationError(ValueError):
    """Raised when an operation would make the error column meaningless."""


def relative_error(predicted: float, measured: float) -> Optional[float]:
    """Error as a fraction of what was measured.

    Undefined when the measurement is zero: dividing by it would report an
    infinite error for a prediction that may have been nearly right, and a
    number that cannot be compared is worse than no number.
    """
    if measured == 0:
        return None
    return abs(predicted - measured) / abs(measured)


async def record_prediction(
    db: AsyncSession,
    *,
    subject: str,
    metric: str,
    predicted_value: float,
    methodology: str,
    prediction_basis: str = "",
    methodology_tags: Optional[List[str]] = None,
    job_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    notes: str = "",
) -> AgentPrediction:
    """Register a claim, before its outcome is known."""
    if not str(subject).strip():
        raise CalibrationError("subject is required: a prediction about nothing")
    if not str(metric).strip():
        raise CalibrationError(
            "metric is required: errors on different quantities cannot be compared"
        )
    if not str(methodology).strip():
        raise CalibrationError(
            "methodology is required: the point of the record is to score how the "
            "number was reached, not just whether it was right"
        )

    prediction = AgentPrediction(
        subject=str(subject)[:300],
        metric=str(metric)[:120],
        predicted_value=float(predicted_value),
        methodology=str(methodology),
        prediction_basis=str(prediction_basis) or None,
        methodology_tags=list(methodology_tags or []) or None,
        job_id=job_id,
        user_id=user_id,
        notes=str(notes) or None,
        predicted_at=datetime.utcnow(),
    )
    db.add(prediction)
    await db.flush()
    return prediction


async def record_measurement(
    db: AsyncSession,
    *,
    prediction_id: UUID,
    measured_value: float,
    measurement_source: str,
    notes: str = "",
) -> AgentPrediction:
    """Settle a claim with what was actually measured."""
    if not str(measurement_source).strip():
        raise CalibrationError(
            "measurement_source is required: a number without the referee that "
            "produced it cannot be compared with another"
        )

    prediction = (
        await db.execute(
            select(AgentPrediction).where(AgentPrediction.id == prediction_id)
        )
    ).scalar_one_or_none()
    if prediction is None:
        raise CalibrationError(f"No prediction {prediction_id}")
    if prediction.measured_value is not None:
        raise CalibrationError(
            "This prediction is already settled. Recording a second measurement "
            "would let the better one be kept."
        )

    prediction.measured_value = float(measured_value)
    prediction.measurement_source = str(measurement_source)[:300]
    prediction.measured_at = datetime.utcnow()
    prediction.error_absolute = abs(
        float(measured_value) - float(prediction.predicted_value)
    )
    prediction.error_relative = relative_error(
        float(prediction.predicted_value), float(measured_value)
    )
    if notes:
        prediction.notes = f"{prediction.notes}\n{notes}" if prediction.notes else notes
    await db.flush()
    return prediction


def summarize(predictions: List[AgentPrediction]) -> Dict[str, Any]:
    """Describe how well a set of predictions held up.

    Open predictions are counted, not dropped: a methodology that predicts
    often and is measured rarely looks accurate only because nobody checked.
    """
    settled = [p for p in predictions if p.measured_value is not None]
    errors = [p.error_relative for p in settled if p.error_relative is not None]
    by_tag: Dict[str, List[float]] = {}
    for prediction in settled:
        if prediction.error_relative is None:
            continue
        for tag in prediction.methodology_tags or []:
            by_tag.setdefault(str(tag), []).append(prediction.error_relative)

    return {
        "total": len(predictions),
        "settled": len(settled),
        "open": len(predictions) - len(settled),
        "mean_relative_error": (sum(errors) / len(errors)) if errors else None,
        "worst_relative_error": max(errors) if errors else None,
        "best_relative_error": min(errors) if errors else None,
        "by_methodology_tag": {
            tag: {
                "settled": len(values),
                "mean_relative_error": sum(values) / len(values),
            }
            for tag, values in sorted(by_tag.items())
        },
    }


async def calibration_report(
    db: AsyncSession,
    *,
    metric: Optional[str] = None,
    subject: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """What has this methodology predicted, and how did it do?"""
    query = select(AgentPrediction).order_by(AgentPrediction.predicted_at.desc())
    if metric:
        query = query.where(AgentPrediction.metric == metric)
    if subject:
        query = query.where(AgentPrediction.subject == subject)
    rows = list((await db.execute(query.limit(limit))).scalars().all())

    return {
        "filter": {"metric": metric, "subject": subject},
        "summary": summarize(rows),
        "predictions": [
            {
                "id": str(row.id),
                "subject": row.subject,
                "metric": row.metric,
                "predicted": row.predicted_value,
                "measured": row.measured_value,
                "relative_error": row.error_relative,
                "measurement_source": row.measurement_source,
                "methodology_tags": row.methodology_tags or [],
                "predicted_at": row.predicted_at.isoformat()
                if row.predicted_at
                else None,
            }
            for row in rows
        ],
    }
