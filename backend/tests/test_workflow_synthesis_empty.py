"""A synthesized workflow with no steps must not look like a success.

The normalizer completes a draft by adding a start and an end node. When the
model returned no steps at all, that produced a start -> end workflow whose
only warnings were "Added missing start node." and "Generated linear edges" --
which read as tidy-up, not as "this workflow does nothing". It was then saved
and exposed as a runnable tool.
"""

from app.services.workflow_synthesis_service import WorkflowSynthesisService


class _Catalog:
    builtin_tools = [{"name": "compile_c_snippet"}, {"name": "search_documents"}]
    custom_tools: list = []


def _normalize(data):
    return WorkflowSynthesisService()._normalize_workflow(
        data,
        _Catalog(),
        fallback_name="probe",
        fallback_description="d",
        fallback_trigger={"type": "manual"},
        fallback_is_active=True,
        synthesize_custom_tools=False,
        preferred_tool_type=None,
        expose_workflow_as_tool=False,
        workflow_tool_name=None,
    )


def test_a_draft_with_no_steps_is_flagged_loudly():
    _, warnings = _normalize({"name": "empty", "nodes": [], "edges": []})

    assert any("PRODUCED NO STEPS" in w for w in warnings), warnings


def test_a_draft_that_omits_nodes_entirely_is_flagged():
    _, warnings = _normalize({"name": "empty"})

    assert any("PRODUCED NO STEPS" in w for w in warnings), warnings


def test_a_draft_with_real_steps_is_not_flagged():
    data = {
        "name": "real",
        "nodes": [
            {
                "node_id": "compile",
                "node_type": "tool",
                "builtin_tool": "compile_c_snippet",
                "config": {},
            }
        ],
        "edges": [],
    }

    normalized, warnings = _normalize(data)

    assert not any("PRODUCED NO STEPS" in w for w in warnings), warnings
    assert any(
        n.get("builtin_tool") == "compile_c_snippet" for n in normalized["nodes"]
    )


def test_the_prompt_demands_at_least_one_tool_node():
    """The model occasionally returned no steps; the rules now forbid it."""
    from app.services.workflow_synthesis_service import WorkflowSynthesisService

    prompt = WorkflowSynthesisService()._build_prompt(
        description="compile a kernel and compare flags",
        name="probe",
        trigger_config={"type": "manual"},
        catalog=_Catalog(),
        synthesize_custom_tools=False,
        preferred_tool_type=None,
        expose_workflow_as_tool=False,
        workflow_tool_name=None,
    )

    assert "MUST contain at least one node" in prompt


def test_work_node_detection_ignores_start_and_end():
    from app.services.workflow_synthesis_service import WorkflowSynthesisService

    svc = WorkflowSynthesisService()

    assert not svc._has_work_nodes(
        {"nodes": [{"node_type": "start"}, {"node_type": "end"}]}
    )
    assert svc._has_work_nodes(
        {"nodes": [{"node_type": "start"}, {"node_type": "tool"}]}
    )
    assert not svc._has_work_nodes({})


def test_the_retry_instruction_names_the_actual_problem():
    from app.services.workflow_synthesis_service import RETRY_SUFFIX_NO_STEPS

    assert "no tool nodes" in RETRY_SUFFIX_NO_STEPS
    assert "would do nothing" in RETRY_SUFFIX_NO_STEPS
