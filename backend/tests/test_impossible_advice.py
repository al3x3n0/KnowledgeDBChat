"""Two ways a run was told to do something it could not do.

Both came out of reading a live agent's chain-of-thought. It was advised to
retry a compile "with the allowlisted image explicitly set", worked out from
the tool schema that no such parameter existed, tried anyway, failed the same
way, and spent two of its five iterations on it.
"""

from app.services import agent_sandbox_runtime as runtime
from app.services.autonomous_agent_executor import _tools_with_params


class TestTheRefusalDoesNotSuggestTheImpossible:
    def test_it_says_the_caller_cannot_choose_the_image(self, monkeypatch):
        monkeypatch.setattr(runtime, "allowed_images", lambda: ["ghcr.io/ok:latest"])
        message = runtime.image_not_allowlisted("ghcr.io/blocked:latest")

        assert "chosen by the server, not by the caller" in message
        assert "retrying will fail the same way" in message

    def test_it_still_says_what_is_allowed(self, monkeypatch):
        monkeypatch.setattr(runtime, "allowed_images", lambda: ["ghcr.io/ok:latest"])
        assert "ghcr.io/ok:latest" in runtime.image_not_allowlisted("other")

    def test_it_handles_nothing_being_allowlisted(self, monkeypatch):
        monkeypatch.setattr(runtime, "allowed_images", lambda: [])
        message = runtime.image_not_allowlisted("anything")

        assert "No images are allowlisted at all" in message
        assert "ask an operator" in message

    def test_both_sandboxes_use_it(self):
        """The wording was duplicated, so a fix to one left the other lying."""
        import inspect

        from app.services import agent_compiler_sandbox, agent_gem5_sandbox

        for module in (agent_compiler_sandbox, agent_gem5_sandbox):
            source = inspect.getsource(module)
            assert "image_not_allowlisted(" in source, module.__name__
            assert (
                "is not allowlisted. Allowed:" not in source
            ), f"{module.__name__} still carries the old wording"


class TestTheCriticSeesWhatToolsAccept:
    def test_parameters_are_rendered_beside_each_tool(self):
        rendered = _tools_with_params(["compile_c_snippet"])
        assert rendered.startswith("compile_c_snippet(")
        assert "code" in rendered and "flags" in rendered

    def test_a_parameter_a_tool_lacks_is_visibly_absent(self):
        """The specific advice that cost two iterations."""
        assert "image" not in _tools_with_params(["compile_c_snippet"])

    def test_an_unknown_tool_renders_without_inventing_parameters(self):
        assert _tools_with_params(["no_such_tool"]) == "no_such_tool()"

    def test_a_long_tool_list_is_truncated_rather_than_flooding_the_prompt(self):
        names = [f"tool_{i}" for i in range(500)]
        rendered = _tools_with_params(names, limit=200)

        assert len(rendered) < 400
        assert "more" in rendered, "truncation must say how much it dropped"

    def test_the_critic_is_told_not_to_invent_parameters(self):
        import inspect

        from app.services import autonomous_agent_executor as mod

        source = inspect.getsource(mod)
        assert "do not advise " in source
        assert "setting a parameter a tool does not have" in source


class TestBoundedConclusionsAreActionable:
    """A bound the run has no way to satisfy is worse than no bound.

    Validity bounds name a finding type and a numeric field. For a type a tool
    emits, calling the tool is the mechanism. For a conclusion the run draws
    itself, nothing emitted it and nothing said how -- so the counting half of
    the contract passed while the claim went unchecked, which is how a latency
    four times the real figure was reported with nothing able to object.
    """

    def _how(self, validity):
        from app.services.autonomous_agent_executor import (
            _how_to_record_bounded_findings,
        )

        return _how_to_record_bounded_findings(validity)

    def test_a_type_no_tool_emits_gets_the_mechanism(self):
        text = self._how(
            {"bounds": {"latency_measurement": {"field": "cycles_per_multiply"}}}
        )
        assert "save_research_finding(" in text
        assert "finding_type='latency_measurement'" in text
        assert "'cycles_per_multiply'" in text

    def test_a_type_a_tool_emits_is_left_alone(self):
        """simulate_c_workload produces simulated_measurement; calling it is
        the mechanism and repeating that would be noise."""
        assert (
            self._how({"bounds": {"simulated_measurement": {"field": "instructions"}}})
            == ""
        )

    def test_it_says_the_number_must_be_in_metrics(self):
        text = self._how({"bounds": {"x_measurement": {"field": "y"}}})
        assert "prose is not readable" in text

    def test_a_contract_without_bounds_says_nothing(self):
        assert self._how({}) == ""
        assert self._how({"bounds": {}}) == ""
        assert self._how(None) == ""

    def test_a_malformed_bound_is_skipped_rather_than_crashing(self):
        assert self._how({"bounds": {"t": "not-a-mapping"}}) == ""
        assert self._how({"bounds": {"t": {"min": 1}}}) == ""  # no field named
