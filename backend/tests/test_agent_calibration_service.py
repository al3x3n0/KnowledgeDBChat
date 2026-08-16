"""Scoring a methodology by prediction error.

The order is the whole point. A prediction written after its outcome is
unfalsifiable, so an error column computed from one measures nothing: an agent
that records its claim once it has seen the answer scores perfectly and has
learned nothing.
"""

import pytest

from app.services import agent_calibration_service as calibration
from app.services.agent_calibration_service import CalibrationError


async def _predict(db, **overrides):
    kwargs = dict(
        subject="fused ldp+fmla",
        metric="speedup",
        predicted_value=1.25,
        methodology="mca delta on the isolated loop, scaled by dynamic block count",
        methodology_tags=["mca", "static-scaling"],
    )
    kwargs.update(overrides)
    return await calibration.record_prediction(db, **kwargs)


@pytest.mark.asyncio
async def test_a_measurement_settles_a_prediction_and_scores_it(db_session):
    prediction = await _predict(db_session)

    settled = await calibration.record_measurement(
        db_session,
        prediction_id=prediction.id,
        measured_value=1.0,
        measurement_source="gem5 O3 neoverse-n1",
    )

    assert settled.measured_value == 1.0
    assert settled.error_absolute == pytest.approx(0.25)
    assert settled.error_relative == pytest.approx(0.25)
    assert settled.measurement_source == "gem5 O3 neoverse-n1"
    assert settled.measured_at is not None


@pytest.mark.asyncio
async def test_a_settled_prediction_cannot_be_measured_again(db_session):
    """Otherwise the flattering measurement can be kept."""
    prediction = await _predict(db_session)
    await calibration.record_measurement(
        db_session,
        prediction_id=prediction.id,
        measured_value=1.0,
        measurement_source="gem5",
    )

    with pytest.raises(CalibrationError) as error:
        await calibration.record_measurement(
            db_session,
            prediction_id=prediction.id,
            measured_value=1.24,
            measurement_source="gem5 again",
        )

    assert "already settled" in str(error.value)


@pytest.mark.asyncio
async def test_a_measurement_must_say_what_produced_it(db_session):
    prediction = await _predict(db_session)

    with pytest.raises(CalibrationError) as error:
        await calibration.record_measurement(
            db_session,
            prediction_id=prediction.id,
            measured_value=1.0,
            measurement_source="  ",
        )

    assert "measurement_source is required" in str(error.value)


@pytest.mark.asyncio
async def test_a_prediction_must_say_how_it_was_reached(db_session):
    with pytest.raises(CalibrationError) as error:
        await _predict(db_session, methodology="")

    assert "methodology is required" in str(error.value)


@pytest.mark.asyncio
async def test_a_prediction_must_name_its_metric(db_session):
    """Errors on different quantities cannot be averaged together."""
    with pytest.raises(CalibrationError):
        await _predict(db_session, metric=" ")


def test_relative_error_is_undefined_against_a_zero_measurement():
    """Infinite error for a nearly-right prediction is worse than no number."""
    assert calibration.relative_error(0.1, 0.0) is None
    assert calibration.relative_error(1.5, 1.0) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_open_predictions_are_counted_not_dropped(db_session):
    """A methodology measured rarely must not look accurate for that reason."""
    settled = await _predict(db_session)
    await calibration.record_measurement(
        db_session,
        prediction_id=settled.id,
        measured_value=1.25,
        measurement_source="gem5",
    )
    await _predict(db_session, subject="never checked")

    report = await calibration.calibration_report(db_session, metric="speedup")

    assert report["summary"]["total"] == 2
    assert report["summary"]["settled"] == 1
    assert report["summary"]["open"] == 1
    assert report["summary"]["mean_relative_error"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_errors_group_by_methodology_tag(db_session):
    """So a later run can ask which approach has been predicting well."""
    good = await _predict(db_session, methodology_tags=["gem5-sampled"])
    await calibration.record_measurement(
        db_session,
        prediction_id=good.id,
        measured_value=1.25,
        measurement_source="gem5",
    )
    poor = await _predict(
        db_session, predicted_value=2.0, methodology_tags=["mca-only"]
    )
    await calibration.record_measurement(
        db_session, prediction_id=poor.id, measured_value=1.0, measurement_source="gem5"
    )

    report = await calibration.calibration_report(db_session)
    by_tag = report["summary"]["by_methodology_tag"]

    assert by_tag["gem5-sampled"]["mean_relative_error"] == pytest.approx(0.0)
    assert by_tag["mca-only"]["mean_relative_error"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_the_tools_round_trip_a_prediction_through_dispatch(db_session):
    """The agent-facing path: predict, then settle, then read the history."""
    from types import SimpleNamespace

    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    job = SimpleNamespace(id=None, user_id=None, config={})
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=db_session, service=None, user_id=None, job=job, state={}
    )
    provider = executor.tool_registry.resolve("record_prediction", ctx)
    assert provider is not None, "record_prediction is not registered"

    recorded = await provider.execute(
        "record_prediction",
        {
            "subject": "fused rsqrt",
            "metric": "speedup",
            "predicted_value": 1.4,
            "methodology": "mca delta scaled by dynamic block count",
            "methodology_tags": ["mca"],
        },
        ctx,
    )
    assert recorded["success"] is True
    prediction_id = recorded["data"]["prediction_id"]

    settled = await provider.execute(
        "record_measurement",
        {
            "prediction_id": prediction_id,
            "measured_value": 1.0,
            "measurement_source": "gem5 O3 neoverse-n1",
        },
        ctx,
    )
    assert settled["success"] is True
    assert settled["data"]["relative_error"] == pytest.approx(0.4)

    report = await provider.execute("calibration_report", {"metric": "speedup"}, ctx)
    assert report["data"]["summary"]["settled"] == 1
    assert report["data"]["summary"]["by_methodology_tag"]["mca"][
        "mean_relative_error"
    ] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_a_non_uuid_prediction_id_is_explained_not_raised(db_session):
    from types import SimpleNamespace

    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db_session,
        service=None,
        user_id=None,
        job=SimpleNamespace(id=None, user_id=None, config={}),
        state={},
    )
    provider = executor.tool_registry.resolve("record_measurement", ctx)

    result = await provider.execute(
        "record_measurement",
        {
            "prediction_id": "not-a-uuid",
            "measured_value": 1.0,
            "measurement_source": "gem5",
        },
        ctx,
    )

    assert "should be a UUID" in result["error"]
