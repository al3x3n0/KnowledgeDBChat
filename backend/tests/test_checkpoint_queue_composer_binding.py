"""Tests for endpoint-facing checkpoint queue dependency binding."""

from app.modules.autonomy.application import checkpoint_queue_composer


def test_bound_checkpoint_queue_composer_forwards_runtime_inputs(monkeypatch):
    def dependency(*_args, **_kwargs):
        return None

    dependencies = checkpoint_queue_composer.CheckpointQueueCompositionDependencies(
        extract_approval_checkpoint=dependency,
        extract_scheduler_state=dependency,
        queue_customer_for_job=dependency,
        present_job=dependency,
        queue_priority_fields=dependency,
        queue_evidence_summary_for_job=dependency,
        queue_reason_label=dependency,
        parse_optional_datetime=dependency,
        extract_launch_mode=dependency,
        build_policy_compat_fields=dependency,
        safe_autonomy_recommendations=("deep_dive_chain",),
        build_follow_up_actions=dependency,
        customer_profile_key=dependency,
        build_portfolio_summary=dependency,
        build_profile_summary=dependency,
        classify_operator_review=dependency,
        build_operator_context=dependency,
        clean_text_list=dependency,
    )
    captured = {}
    expected = [object()]

    def fake_compose(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        checkpoint_queue_composer,
        "compose_checkpoint_queue",
        fake_compose,
    )
    builder = checkpoint_queue_composer.bind_checkpoint_queue_composer(
        deps=dependencies
    )
    jobs = [object()]
    inbox_items = [object()]
    learning_profiles = {"acme": {"token_scores": {}}}

    result = builder(
        jobs,
        inbox_items,
        learning_profiles=learning_profiles,
        monitor_health_rows=[{"customer": "Acme"}],
    )

    assert result is expected
    assert captured["args"] == (jobs, inbox_items)
    assert captured["kwargs"]["deps"] is dependencies
    assert captured["kwargs"]["learning_profiles"] is learning_profiles
    assert captured["kwargs"]["monitor_health_rows"] == [{"customer": "Acme"}]


def test_bound_checkpoint_queue_composer_can_resolve_fresh_dependencies(monkeypatch):
    dependencies = object()
    calls = []

    def dependencies_factory():
        calls.append(True)
        return dependencies

    monkeypatch.setattr(
        checkpoint_queue_composer,
        "compose_checkpoint_queue",
        lambda *_args, **kwargs: kwargs["deps"],
    )
    builder = checkpoint_queue_composer.bind_checkpoint_queue_composer(
        dependencies_factory=dependencies_factory
    )

    assert builder([], []) is dependencies
    assert builder([], []) is dependencies
    assert calls == [True, True]
