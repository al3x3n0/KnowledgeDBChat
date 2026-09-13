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


def _coding_stage(stage_id, types, **over):
    """A stage that may actually call the coding tools.

    Every coding tool is restricted to the `analysis` and `coding` job types,
    so a coding stage left at the default `research` is planned with tools the
    runtime then hides from it. Watched live: such a stage spent eight
    iterations calling search_documents while its plan named
    clone_and_index_repo, apply_patch and run_repo_tests.
    """
    over.setdefault("job_type", "coding")
    return _stage(stage_id, types, **over)


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


LAUNCH = "/api/v1/agent-pipelines/launch"


class TestLaunchingOne:
    """The first endpoint here that spends anything.

    Every check `/check` performs runs again at launch, and none of them are
    advisory any more. These tests are about the refusals: a launch that should
    not have happened costs a whole run to discover.
    """

    def _good(self):
        return _spec(
            _stage("measure", ["benchmark_measurement"]),
            _stage(
                "attribute",
                ["bottleneck_attribution"],
                depends_on=["measure"],
                assumes=["benchmark_measurement"],
            ),
            name="int8-study",
        )

    def test_it_starts_a_job_and_says_what_it_started(self, client, auth_headers):
        response = client.post(
            LAUNCH, json={"spec": self._good()}, headers=auth_headers
        )
        assert response.status_code == 201
        body = response.json()

        assert body["job_id"]
        assert body["stages"] == ["measure", "attribute"]
        assert body["estimated_seconds"] > 0

    def test_an_invalid_pipeline_is_not_started(self, client, auth_headers):
        payload = {
            "spec": _spec(
                _stage(
                    "attribute",
                    ["bottleneck_attribution"],
                    assumes=["counter_trace"],
                )
            )
        }
        response = client.post(LAUNCH, json=payload, headers=auth_headers)
        assert response.status_code == 422
        assert "counter_trace" in response.json()["detail"]

    def test_a_budget_is_a_refusal_here_rather_than_a_note(self, client, auth_headers):
        # On /check a budget is information. At launch it has to stop the run,
        # or pricing a pipeline beforehand bought nothing.
        response = client.post(
            LAUNCH,
            json={"spec": self._good(), "budget_seconds": 1},
            headers=auth_headers,
        )
        assert response.status_code == 422
        assert "budget" in response.json()["detail"].lower()

    def test_a_spec_edited_since_it_was_priced_is_refused(self, client, auth_headers):
        # The ordinary way someone starts a run they never actually saw priced:
        # check one spec, edit it, launch. The estimate the caller agreed to
        # must be the estimate about to be spent.
        response = client.post(
            LAUNCH,
            json={"spec": self._good(), "acknowledged_seconds": 7},
            headers=auth_headers,
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "7" in detail and "again" in detail.lower()

    def test_the_matching_estimate_is_accepted(self, client, auth_headers):
        checked = client.post(
            CHECK, json={"spec": self._good()}, headers=auth_headers
        ).json()
        total = checked["plan"]["total_seconds"]

        response = client.post(
            LAUNCH,
            json={"spec": self._good(), "acknowledged_seconds": total},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_several_independent_roots_are_refused_rather_than_half_run(
        self, client, auth_headers
    ):
        # A chain runs from one head. Launching only the first root would
        # silently run part of what was asked for.
        payload = {
            "spec": _spec(
                _stage("one", ["benchmark_measurement"]),
                _stage("two", ["dynamic_profile"]),
            )
        }
        response = client.post(LAUNCH, json=payload, headers=auth_headers)
        assert response.status_code == 422
        assert "starting stages" in response.json()["detail"]

    def test_it_needs_a_caller(self, client):
        response = client.post(LAUNCH, json={"spec": self._good()})
        assert response.status_code in (401, 403)


class TestResearchWorkIsExpressible:
    """Literature and synthesis work as a pipeline, not just microarchitecture.

    The evidence map began as measurement tools only — compile, profile,
    benchmark, gem5 — so a pipeline could express a cycle-accurate study and
    could not express reading twenty papers and writing up what they missed.
    The research tools existed the whole time; they simply declared no
    evidence, so nothing could plan with them.
    """

    def test_a_literature_survey_compiles_from_its_contracts(
        self, client, auth_headers
    ):
        payload = {
            "spec": _spec(
                _stage("gather", ["papers_ingested"]),
                _stage(
                    "read",
                    ["paper_insights"],
                    depends_on=["gather"],
                    assumes=["papers_ingested"],
                    loop={"max_iterations": 8},
                ),
                _stage(
                    "compare",
                    ["methodology_comparison"],
                    depends_on=["read"],
                    assumes=["paper_insights"],
                ),
                _stage(
                    "writeup",
                    ["synthesis_document"],
                    depends_on=["compare"],
                    assumes=["methodology_comparison"],
                    checkpoint=True,
                ),
                name="attention-survey",
            )
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is True, body["problems"]
        assert body["expressible"] is True
        assert body["plan"]["order"] == ["gather", "read", "compare", "writeup"]

        # The tools are deduced from what each stage must be true of, the same
        # way a measurement pipeline's are.
        tools = {s["stage_id"]: s["tools"] for s in body["plan"]["stages"]}
        assert "extract_paper_insights" in tools["read"]
        assert "compare_methodologies" in tools["compare"]
        assert "create_synthesis_document" in tools["writeup"]
        # And a survey that stops for a person says so beforehand.
        assert body["plan"]["checkpoints"] == ["writeup"]

    def test_reading_is_priced_per_iteration(self, client, auth_headers):
        # Eight papers cost eight extractions. A loop that priced as one pass
        # would make every survey look affordable.
        once = client.post(
            CHECK,
            json={"spec": _spec(_stage("read", ["paper_insights"]))},
            headers=auth_headers,
        ).json()
        eight = client.post(
            CHECK,
            json={
                "spec": _spec(
                    _stage("read", ["paper_insights"], loop={"max_iterations": 8})
                )
            },
            headers=auth_headers,
        ).json()

        assert eight["plan"]["total_seconds"] > once["plan"]["total_seconds"]

    def test_synthesising_from_evidence_nothing_gathered_is_refused(
        self, client, auth_headers
    ):
        # The research-side version of the failure the whole module exists for:
        # a write-up stage resting on papers no stage ever read.
        payload = {
            "spec": _spec(
                _stage(
                    "writeup",
                    ["synthesis_document"],
                    assumes=["paper_insights"],
                )
            )
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is False
        assert "paper_insights" in " ".join(body["problems"])

    def test_a_contract_no_research_tool_can_satisfy_is_refused(
        self, client, auth_headers
    ):
        payload = {"spec": _spec(_stage("review", ["peer_review"]))}
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is False
        assert "peer_review" in " ".join(body["problems"])


SAVED = "/api/v1/agent-pipelines"


class TestSavingOne:
    """A pipeline that outlives the browser tab it was written in.

    Until this existed a spec lived in one browser's localStorage: no library,
    no reuse, and no record of which spec produced which run. Authoring a
    five-stage survey and launching it left nothing behind to run again.
    """

    def _spec_ok(self):
        return _spec(
            _stage("measure", ["benchmark_measurement"]),
            _stage(
                "attribute",
                ["bottleneck_attribution"],
                depends_on=["measure"],
                assumes=["benchmark_measurement"],
            ),
            name="int8-study",
        )

    def test_it_saves_and_comes_back(self, client, auth_headers):
        created = client.post(
            SAVED,
            json={"name": "INT8 study", "spec": self._spec_ok()},
            headers=auth_headers,
        )
        assert created.status_code == 201
        body = created.json()
        assert body["name"] == "INT8 study"
        # The verdict is recorded alongside, so a list can say which of twenty
        # pipelines are not ready without re-planning all twenty.
        assert body["last_check_valid"] == "valid"
        assert body["last_estimated_seconds"] > 0

        listed = client.get(SAVED, headers=auth_headers).json()
        assert [p["name"] for p in listed] == ["INT8 study"]

    def test_a_half_written_pipeline_is_still_savable(self, client, auth_headers):
        # Refusing to save an invalid spec would mean the only way to keep work
        # in progress is to leave the tab open.
        response = client.post(
            SAVED,
            json={
                "name": "wip",
                "spec": _spec(_stage("a", ["telepathy"])),
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        assert response.json()["last_check_valid"] == "invalid"

    def test_a_spec_that_is_barely_a_pipeline_is_still_savable(
        self, client, auth_headers
    ):
        # `normalize` is lenient — `stages: "oops"` becomes no stages — so this
        # records as invalid rather than unknown. Either way it saves: the
        # point is that nothing about a spec prevents keeping it.
        response = client.post(
            SAVED,
            json={"name": "rubble", "spec": {"stages": "not a list"}},
            headers=auth_headers,
        )
        assert response.status_code == 201
        assert response.json()["last_check_valid"] in ("invalid", "unknown")

    def test_two_pipelines_cannot_share_a_name(self, client, auth_headers):
        payload = {"name": "same", "spec": self._spec_ok()}
        assert client.post(SAVED, json=payload, headers=auth_headers).status_code == 201
        clash = client.post(SAVED, json=payload, headers=auth_headers)
        assert clash.status_code == 409

    def test_editing_the_spec_rechecks_it(self, client, auth_headers):
        created = client.post(
            SAVED,
            json={"name": "evolving", "spec": self._spec_ok()},
            headers=auth_headers,
        ).json()
        assert created["last_check_valid"] == "valid"

        # The cached verdict must never be older than the spec it describes.
        updated = client.patch(
            f"{SAVED}/{created['id']}",
            json={"spec": _spec(_stage("a", ["telepathy"]))},
            headers=auth_headers,
        ).json()
        assert updated["last_check_valid"] == "invalid"

    def test_renaming_leaves_the_spec_alone(self, client, auth_headers):
        created = client.post(
            SAVED,
            json={"name": "before", "spec": self._spec_ok()},
            headers=auth_headers,
        ).json()

        renamed = client.patch(
            f"{SAVED}/{created['id']}", json={"name": "after"}, headers=auth_headers
        ).json()
        assert renamed["name"] == "after"
        assert renamed["spec"] == created["spec"]

    def test_someone_elses_pipeline_is_not_found(self, client, auth_headers):
        import uuid as _uuid

        response = client.get(f"{SAVED}/{_uuid.uuid4()}", headers=auth_headers)
        assert response.status_code == 404

    def test_launching_a_saved_pipeline_records_it_on_both_sides(
        self, client, auth_headers
    ):
        saved = client.post(
            SAVED,
            json={"name": "provenance", "spec": self._spec_ok()},
            headers=auth_headers,
        ).json()

        launched = client.post(
            LAUNCH,
            json={"spec": self._spec_ok(), "pipeline_id": saved["id"]},
            headers=auth_headers,
        )
        assert launched.status_code == 201
        assert launched.json()["pipeline_id"] == saved["id"]

        # And the pipeline knows what it produced. Without this a launched spec
        # is anonymous the moment the editor closes.
        after = client.get(f"{SAVED}/{saved['id']}", headers=auth_headers).json()
        assert after["launch_count"] == 1
        assert after["last_job_id"] == launched.json()["job_id"]
        assert after["last_launched_at"]

    def test_deleting_a_pipeline_leaves_its_runs(self, client, auth_headers):
        saved = client.post(
            SAVED,
            json={"name": "temporary", "spec": self._spec_ok()},
            headers=auth_headers,
        ).json()
        launched = client.post(
            LAUNCH,
            json={"spec": self._spec_ok(), "pipeline_id": saved["id"]},
            headers=auth_headers,
        ).json()

        assert (
            client.delete(f"{SAVED}/{saved['id']}", headers=auth_headers).status_code
            == 200
        )
        # The run outlives the recipe: deleting a pipeline must not delete the
        # work it produced.
        job = client.get(
            f"/api/v1/agent-jobs/{launched['job_id']}", headers=auth_headers
        )
        assert job.status_code == 200


class TestCodingWorkIsExpressible:
    """A coding agent's workflow as a pipeline.

    The coding tools existed and declared no evidence, so the planner could not
    reach them: a run could clone a repo and patch it, but only as a black box
    a deterministic runner drove. Nothing could state "clone, locate, test,
    patch, verify" and be told beforehand that it would not work.

    `requires` is real on this side, unlike the research tools. Every coding
    tool takes a workspace_id and only the clone produces one, so an ordering
    mistake here is a stage that cannot run at all.
    """

    def test_a_fix_workflow_compiles_from_its_contracts(self, client, auth_headers):
        payload = {
            "spec": _spec(
                _coding_stage("clone", ["repo_workspace"]),
                _coding_stage(
                    "locate",
                    ["symbol_index"],
                    depends_on=["clone"],
                    assumes=["repo_workspace"],
                ),
                _coding_stage(
                    "tests",
                    ["test_targets"],
                    depends_on=["locate"],
                    assumes=["symbol_index"],
                ),
                _coding_stage(
                    "patch",
                    ["patch_applied"],
                    depends_on=["tests"],
                    assumes=["test_targets"],
                    checkpoint=True,
                ),
                _coding_stage(
                    "verify",
                    ["command_result"],
                    depends_on=["patch"],
                    assumes=["patch_applied"],
                    loop={"max_iterations": 3},
                ),
                name="fix-the-off-by-one",
            )
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is True, body["problems"]
        assert body["expressible"] is True
        tools = {s["stage_id"]: s["tools"] for s in body["plan"]["stages"]}
        assert "clone_and_index_repo" in tools["clone"]
        assert "retrieve_repo_symbols" in tools["locate"]
        assert "apply_patch" in tools["patch"]
        # A patch nobody has looked at should not reach a repository.
        assert body["plan"]["checkpoints"] == ["patch"]

    def test_patching_a_repo_nothing_cloned_is_refused(self, client, auth_headers):
        # The coding version of the failure the checker exists for, and a hard
        # one: apply_patch takes a workspace_id that only the clone returns, so
        # this stage could not run however well the model behaved.
        payload = {
            "spec": _spec(
                _coding_stage("patch", ["patch_applied"], assumes=["repo_workspace"])
            )
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is False
        assert "repo_workspace" in " ".join(body["problems"])

    def test_a_coding_stage_at_the_default_job_type_is_refused(
        self, client, auth_headers
    ):
        """The failure that made this class's pipelines a fiction.

        Every coding tool is restricted to `analysis` and `coding`, and the
        default job type is `research`. Such a stage validated, planned as
        `clone_and_index_repo, apply_patch, run_repo_tests`, started -- and
        then could not see one of them. Watched live: eight iterations of
        search_documents and save_research_finding while the plan promised a
        repository fix.

        A plan naming tools the runtime forbids is the same failure as
        evidence declared but never emitted: a statement about the system that
        nothing checked against the system.
        """
        payload = {
            "spec": _spec(_stage("clone", ["repo_workspace"]))  # default research
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is False
        problems = " ".join(body["problems"])
        assert "clone_and_index_repo" in problems
        # And says what to change, not merely that something is wrong.
        assert "job_type" in problems and "coding" in problems

    def test_the_toolchain_carries_its_own_prerequisites(self, client, auth_headers):
        # Finding tests for a symbol needs the symbol index, which needs the
        # clone. The planner derives that chain rather than the author stating
        # it, which is the point of deriving tools from contracts at all.
        payload = {
            "spec": _spec(
                _coding_stage("clone", ["repo_workspace"]),
                _coding_stage(
                    "tests",
                    ["test_targets"],
                    depends_on=["clone"],
                    assumes=["repo_workspace"],
                ),
            )
        }
        body = client.post(CHECK, json=payload, headers=auth_headers).json()

        assert body["valid"] is True, body["problems"]
        tools = {s["stage_id"]: s["tools"] for s in body["plan"]["stages"]}
        assert "retrieve_repo_symbols" in tools["tests"]
        assert "find_tests_for_symbol" in tools["tests"]
