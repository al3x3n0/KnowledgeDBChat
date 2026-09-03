"""Reaching the pipeline checker from outside the backend.

`agent_pipeline_spec` and `agent_pipeline_binding` were written, tested, and
then unreachable: no endpoint, no client method, nothing in the UI. These
tests are about the part that matters once they are reachable — that a bad
pipeline is refused *here*, before anything expensive starts, and that the
refusal says enough for the author to fix it.
"""


CHECK = "/api/v1/agent-pipelines/check"
BIND = "/api/v1/agent-pipelines/bind"


# Finding types the evidence map actually has a tool for. A fixture built on
# an invented type would trip the "no tool produces this" check and never reach
# the behaviour under test.
def _stage(stage_id, types, **over):
    spec = {
        "id": stage_id,
        "goal": f"do {stage_id}",
        "contract": {"required_finding_types": list(types)},
    }
    spec.update(over)
    return spec


def _spec(*stages, name="study"):
    return {"name": name, "stages": list(stages)}


class TestCheckingBeforeAnythingRuns:
    def test_a_workable_pipeline_comes_back_valid_and_planned(
        self, client, auth_headers
    ):
        payload = {
            "spec": _spec(
                _stage("measure", ["benchmark_measurement"]),
                _stage(
                    "explain",
                    ["bottleneck_attribution"],
                    depends_on=["measure"],
                    assumes=["benchmark_measurement"],
                ),
            )
        }
        response = client.post(CHECK, json=payload, headers=auth_headers)
        assert response.status_code == 200
        body = response.json()

        assert body["valid"] is True
        assert body["problems"] == []
        assert body["expressible"] is True
        # The plan is the point of asking: an order, and what it will cost.
        assert body["plan"]["order"] == ["measure", "explain"]
        assert body["plan"]["total_seconds"] > 0
        assert body["description"]

    def test_a_stage_reading_evidence_nothing_upstream_produces_is_refused(
        self, client, auth_headers
    ):
        # The failure this whole module exists to prevent: a run that goes the
        # distance and produces something unusable because stage two needed a
        # measurement stage one never took.
        payload = {
            "spec": _spec(
                _stage("measure", ["benchmark_measurement"]),
                _stage(
                    "explain",
                    ["bottleneck_attribution"],
                    depends_on=["measure"],
                    assumes=["counter_trace"],
                ),
            )
        }
        response = client.post(CHECK, json=payload, headers=auth_headers)
        assert response.status_code == 200
        body = response.json()

        assert body["valid"] is False
        assert body["problems"]
        # It must name the stage and the missing evidence, or the author
        # cannot act on it.
        joined = " ".join(body["problems"])
        assert "explain" in joined
        assert "counter_trace" in joined

    def test_a_cycle_is_refused(self, client, auth_headers):
        payload = {
            "spec": _spec(
                _stage("a", ["bottleneck_attribution"], depends_on=["b"]),
                _stage("b", ["bottleneck_attribution"], depends_on=["a"]),
            )
        }
        body = (client.post(CHECK, json=payload, headers=auth_headers)).json()

        assert body["valid"] is False
        assert any("cycle" in p.lower() for p in body["problems"])
        # No plan for something that cannot be ordered — reporting one would be
        # inventing an order that does not exist.
        assert body["plan"] is None

    def test_every_problem_is_reported_at_once(self, client, auth_headers):
        # Fixing one problem only to be told about the next is the slow way to
        # discover a spec is unusable.
        payload = {
            "spec": _spec(
                _stage("a", ["bottleneck_attribution"], depends_on=["missing"]),
                _stage("b", ["bottleneck_attribution"], assumes=["never_produced"]),
            )
        }
        body = (client.post(CHECK, json=payload, headers=auth_headers)).json()

        assert body["valid"] is False
        assert len(body["problems"]) >= 2

    def test_a_budget_is_a_separate_answer_from_validity(self, client, auth_headers):
        # A pipeline can be perfectly well formed and still unaffordable; one
        # boolean covering both would tell the author the wrong thing.
        spec = _spec(_stage("measure", ["benchmark_measurement"]))

        generous = (
            client.post(
                CHECK,
                json={"spec": spec, "budget_seconds": 10_000_000},
                headers=auth_headers,
            )
        ).json()
        assert generous["valid"] is True
        assert generous["budget"]["affordable"] is True

        stingy = (
            client.post(
                CHECK, json={"spec": spec, "budget_seconds": 1}, headers=auth_headers
            )
        ).json()
        assert stingy["valid"] is True
        assert stingy["budget"]["affordable"] is False
        assert stingy["budget"]["estimated_seconds"] > 1

    def test_a_spec_that_is_not_a_pipeline_is_a_400(self, client, auth_headers):
        response = client.post(
            CHECK,
            json={"spec": {"name": "x", "stages": "not a list"}},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "pipeline spec" in response.json()["detail"].lower()

    def test_it_needs_a_caller(self, client):
        response = client.post(CHECK, json={"spec": _spec()})
        assert response.status_code in (401, 403)


class TestCompilingToAChain:
    def test_it_returns_the_chain_without_launching_it(self, client, auth_headers):
        payload = {
            "spec": _spec(
                _stage("measure", ["benchmark_measurement"]),
                _stage(
                    "explain",
                    ["bottleneck_attribution"],
                    depends_on=["measure"],
                    assumes=["benchmark_measurement"],
                ),
                name="int8-study",
            )
        }
        response = client.post(BIND, json=payload, headers=auth_headers)
        assert response.status_code == 200
        body = response.json()

        assert body["name"] == "int8-study"
        assert body["chain_config"]["roots"]
        assert body["description"]

    def test_an_invalid_pipeline_is_refused_rather_than_compiled(
        self, client, auth_headers
    ):
        payload = {
            "spec": _spec(
                _stage("a", ["bottleneck_attribution"], depends_on=["b"]),
                _stage("b", ["bottleneck_attribution"], depends_on=["a"]),
            )
        }
        response = client.post(BIND, json=payload, headers=auth_headers)
        # 422, not 400: the request was well formed and the pipeline is not,
        # which the caller handles differently.
        assert response.status_code == 422

    def test_a_checkpoint_is_reported_so_it_is_not_a_surprise(
        self, client, auth_headers
    ):
        payload = {
            "spec": _spec(
                _stage("measure", ["benchmark_measurement"], checkpoint=True),
                _stage(
                    "explain",
                    ["bottleneck_attribution"],
                    depends_on=["measure"],
                    assumes=["benchmark_measurement"],
                ),
            )
        }
        body = (client.post(BIND, json=payload, headers=auth_headers)).json()

        # A pipeline that stops for a person should say so before it is run,
        # not when it stops.
        assert "measure" in body["checkpoints"]
