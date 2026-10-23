"""propose_code_patch, verify_run_bundle, literature_review_arxiv,
generate_gitlab_architecture, update_document_tags and batch_summarize_documents,
called through their real handlers.

Each test runs the handler the tool registry would run, against the in-memory
database with real rows. Only edges that leave the process are replaced, and
each replacement binds its arguments against the real callee, so a call the
real thing would refuse fails the test:

* `git diff` -- `asyncio.create_subprocess_exec`, answering as git does
  (tracked modifications only; exit 129 outside a repository);
* arXiv -- `httpx.MockTransport` behind the real `ArxivSearchService`;
* Celery -- `.delay` bound against the task function;
* GitLab -- the architecture service, bound against its real method;
* the evidence bundle is real, written under a temporary root.

literature_review_arxiv is declared callable by research jobs, and had no
autonomous handler until these tests found it. Three of the others are declared
chat-only (`job_types=()`), and the tests check that the declaration and the
registry agree.

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from app.agent_core.tool_specs import spec_for
from app.core.config import settings
from app.models.agent_job import AgentJob
from app.models.code_patch_proposal import CodePatchProposal
from app.models.document import Document, DocumentSource
from app.services import agent_evidence_bundle as bundle
from app.services import gitlab_architecture_service as gitlab_module
from app.services.agent_service import AgentService
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    AgentToolRegistry,
    FunctionToolProvider,
    build_agent_service_analytics_content_provider,
    build_agent_service_document_provider,
    build_agent_service_research_provider,
    build_autonomous_workspace_mutation_provider,
)
from app.services.auth_service import AuthService
from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
)
from app.services.document_service import DocumentService
from app.services.gitlab_architecture_service import GitLabArchitectureService
from app.services.text_processor import TextProcessor
from app.services.vector_store import vector_store_service
from app.tasks.ingestion_tasks import ingest_from_source
from app.tasks.summarization_tasks import summarize_document

pytestmark = pytest.mark.unit

_SUBPROCESS_SIGNATURE = inspect.signature(asyncio.create_subprocess_exec)


# --------------------------------------------------------------------------
# Rows and contexts
# --------------------------------------------------------------------------


async def _job(db, user, **overrides):
    fields = {
        "name": "Calling job",
        "goal": "Fix the off-by-one in the parser",
        "job_type": "coding",
        "user_id": user.id,
        "status": "running",
        "config": {},
    }
    fields.update(overrides)
    job = AgentJob(**fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _other_user(db):
    return await AuthService().create_user(
        username="someone_else",
        email="else@example.com",
        password="otherpassword123",
        full_name="Someone Else",
        db=db,
    )


def _ctx(db, job, state=None, mode="autonomous", user_id=None):
    return AgentToolExecutionContext(
        mode=mode,
        db=db,
        service=None,
        user_id=user_id,
        job=job,
        state={} if state is None else state,
    )


def _chat_ctx(db, user):
    return AgentToolExecutionContext(mode="chat", db=db, service=None, user_id=user.id)


def _autonomous_executor():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    return AutonomousAgentExecutor()


# --------------------------------------------------------------------------
# propose_code_patch
# --------------------------------------------------------------------------


class _Proc:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self._out = stdout
        self._err = stderr

    async def communicate(self, input=None):
        return self._out, self._err


class _Git:
    """`git diff` as git answers it.

    Only changes to *tracked* files appear; a directory with no `.git` gets
    exit 129 and usage text on stderr, with nothing on stdout.
    """

    def __init__(self):
        self.calls = []
        self.diffs = {}

    async def exec(self, *args, **kwargs):
        _SUBPROCESS_SIGNATURE.bind(*args, **kwargs)
        cwd = kwargs.get("cwd")
        self.calls.append({"argv": list(args), "cwd": cwd})
        if not (Path(cwd) / ".git").is_dir():
            return _Proc(
                129,
                b"",
                b"warning: Not a git repository. Use --no-index to compare two "
                b"paths outside a working tree\nusage: git diff --no-index "
                b"[<options>] <path> <path>\n",
            )
        return _Proc(0, self.diffs.get(cwd, "").encode(), b"")


@pytest.fixture
def git(monkeypatch):
    rec = _Git()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", rec.exec)
    return rec


def _workspace(manager, root, job, *, files=None, with_git=True):
    """A workspace whose original files are `files`, as clone_and_index makes."""
    base = root / f"ws-{uuid4().hex[:8]}"
    base.mkdir(parents=True)
    if with_git:
        (base / ".git").mkdir()
    hashes = {}
    for rel, text in (files or {}).items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        hashes[rel] = hashlib.sha256(text.encode()).hexdigest()
    ws = CodingWorkspace(
        workspace_id=str(uuid4()),
        base_path=base,
        owner_job_id=str(job.id),
        original_hashes=hashes,
    )
    manager._workspaces[ws.workspace_id] = ws
    return ws


MODIFY_DIFF = (
    "diff --git a/src/parser.py b/src/parser.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/parser.py\n"
    "+++ b/src/parser.py\n"
    "@@ -1,2 +1,3 @@\n"
    "-end = len(items) - 1\n"
    "+end = len(items)\n"
    "+assert end >= 0\n"
    " return items[:end]\n"
)

DELETE_DIFF = (
    "diff --git a/src/legacy.py b/src/legacy.py\n"
    "deleted file mode 100644\n"
    "index 3333333..0000000\n"
    "--- a/src/legacy.py\n"
    "+++ /dev/null\n"
    "@@ -1,2 +0,0 @@\n"
    "-def old():\n"
    "-    return 1\n"
)


@pytest.fixture
def patching(tmp_path, git):
    manager = CodingWorkspaceManager()
    provider = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(workspace_manager=manager)
    )
    return SimpleNamespace(
        manager=manager,
        handler=provider._handlers["propose_code_patch"],
        root=tmp_path,
        git=git,
    )


async def test_propose_refuses_without_a_workspace(patching, db_session, test_user):
    job = await _job(db_session, test_user)

    result = await patching.handler({"title": "Fix it"}, _ctx(db_session, job))

    assert result == {"error": "No active coding workspace"}
    assert patching.git.calls == []


async def test_propose_refuses_a_workspace_id_it_does_not_know(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    _workspace(patching.manager, patching.root, job)

    result = await patching.handler(
        {"title": "Fix it", "workspace_id": str(uuid4())}, _ctx(db_session, job)
    )

    assert result == {"error": "No active coding workspace"}
    assert patching.git.calls == []


@pytest.mark.parametrize("params", [{}, {"title": ""}, {"title": "   "}])
async def test_propose_requires_a_title(patching, db_session, test_user, params):
    job = await _job(db_session, test_user)
    ws = _workspace(patching.manager, patching.root, job)

    result = await patching.handler(
        params, _ctx(db_session, job, {"coding_workspace_id": ws.workspace_id})
    )

    assert result == {"error": "title is required"}
    assert patching.git.calls == []


async def test_propose_records_the_diff_and_produces_its_declared_evidence(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    ws = _workspace(
        patching.manager,
        patching.root,
        job,
        files={"src/parser.py": "end = len(items) - 1\nreturn items[:end]\n"},
    )
    (ws.base_path / "src/parser.py").write_text(
        "end = len(items)\nassert end >= 0\nreturn items[:end]\n"
    )
    patching.git.diffs[str(ws.base_path)] = MODIFY_DIFF
    state = {"coding_workspace_id": ws.workspace_id}

    result = await patching.handler(
        {"title": "  Fix the off-by-one  ", "rationale": "test_slice fails"},
        _ctx(db_session, job, state),
    )

    assert result["success"] is True
    assert patching.git.calls == [{"argv": ["git", "diff"], "cwd": str(ws.base_path)}]
    finding_types = {f["type"] for f in result["findings"]}
    assert set(spec_for("propose_code_patch").produces) <= finding_types
    (finding,) = result["findings"]
    assert finding == {
        "type": "code_patch_proposal",
        "title": "Fix the off-by-one",
        "files": ["src/parser.py"],
        "lines_added": 2,
        "lines_removed": 1,
    }
    assert "diff" not in result["data"]
    assert result["data"]["rationale"] == "test_slice fails"
    assert result["data"]["workspace_id"] == ws.workspace_id
    assert state["code_patch_proposal"]["diff"] == MODIFY_DIFF


async def test_propose_reads_the_named_workspace_over_the_states_default(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    default = _workspace(patching.manager, patching.root, job)
    named = _workspace(patching.manager, patching.root, job)
    patching.git.diffs[str(named.base_path)] = MODIFY_DIFF

    result = await patching.handler(
        {"title": "Fix", "workspace_id": named.workspace_id},
        _ctx(db_session, job, {"coding_workspace_id": default.workspace_id}),
    )

    assert result["success"] is True
    assert patching.git.calls[0]["cwd"] == str(named.base_path)


async def test_propose_refuses_a_workspace_with_nothing_changed(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    ws = _workspace(
        patching.manager, patching.root, job, files={"src/parser.py": "x = 1\n"}
    )
    state = {"coding_workspace_id": ws.workspace_id}

    result = await patching.handler({"title": "Fix"}, _ctx(db_session, job, state))

    assert "nothing to propose" in result["error"]
    assert "findings" not in result
    assert "code_patch_proposal" not in state


async def test_propose_stores_a_reviewable_proposal_row(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    ws = _workspace(patching.manager, patching.root, job)
    patching.git.diffs[str(ws.base_path)] = MODIFY_DIFF

    result = await patching.handler(
        {"title": "Fix the off-by-one", "rationale": "test_slice fails"},
        _ctx(db_session, job, {"coding_workspace_id": ws.workspace_id}),
    )
    assert result["success"] is True

    rows = (
        (
            await db_session.execute(
                select(CodePatchProposal).where(CodePatchProposal.job_id == job.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.user_id == test_user.id
    assert row.title == "Fix the off-by-one"
    assert row.diff_unified == MODIFY_DIFF
    assert row.status == "proposed"
    assert (row.proposal_metadata or {}).get("files_touched") == ["src/parser.py"]


async def test_propose_includes_a_file_the_run_created(patching, db_session, test_user):
    job = await _job(db_session, test_user)
    ws = _workspace(
        patching.manager, patching.root, job, files={"src/parser.py": "x = 1\n"}
    )
    (ws.base_path / "tests").mkdir()
    (ws.base_path / "tests/test_parser.py").write_text("def test_x():\n    pass\n")
    assert patching.manager.get_status(ws)["added"] == ["tests/test_parser.py"]

    result = await patching.handler(
        {"title": "Add a regression test"},
        _ctx(db_session, job, {"coding_workspace_id": ws.workspace_id}),
    )

    assert result.get("success") is True, result
    assert "tests/test_parser.py" in result["findings"][0]["files"]


async def test_propose_reports_a_failed_diff_as_a_failed_diff(
    patching, db_session, test_user
):
    job = await _job(db_session, test_user)
    ws = _workspace(
        patching.manager,
        patching.root,
        job,
        files={"src/parser.py": "x = 1\n"},
        with_git=False,
    )
    (ws.base_path / "src/parser.py").write_text("x = 2\n")
    assert patching.manager.get_status(ws)["changes_count"] == 1

    result = await patching.handler(
        {"title": "Fix"},
        _ctx(db_session, job, {"coding_workspace_id": ws.workspace_id}),
    )

    assert "error" in result
    assert "no uncommitted changes" not in result["error"]


async def test_propose_lists_a_deleted_file(patching, db_session, test_user):
    job = await _job(db_session, test_user)
    ws = _workspace(patching.manager, patching.root, job)
    patching.git.diffs[str(ws.base_path)] = DELETE_DIFF

    result = await patching.handler(
        {"title": "Remove legacy helper"},
        _ctx(db_session, job, {"coding_workspace_id": ws.workspace_id}),
    )

    assert result["success"] is True
    assert result["findings"][0]["lines_removed"] == 2
    assert result["findings"][0]["files"] == ["src/legacy.py"]


async def test_propose_refuses_another_jobs_workspace(patching, db_session, test_user):
    other = await _other_user(db_session)
    theirs = await _job(db_session, other, name="Their job")
    mine = await _job(db_session, test_user)
    ws = _workspace(patching.manager, patching.root, theirs)
    patching.git.diffs[str(ws.base_path)] = MODIFY_DIFF

    result = await patching.handler(
        {"title": "Fix", "workspace_id": ws.workspace_id}, _ctx(db_session, mine)
    )

    assert "error" in result
    assert patching.git.calls == []


# --------------------------------------------------------------------------
# verify_run_bundle
# --------------------------------------------------------------------------


class _Compiler:
    """A deterministic stand-in for compile_c_snippet, one of EVIDENCE_TOOLS."""

    def __init__(self):
        self.calls = []
        self.assembly = "add x0, x0, x1"

    async def handle(self, params, ctx):
        self.calls.append(dict(params))
        return {
            "success": True,
            "data": {"assembly": self.assembly, "flags": params.get("flags")},
        }


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    root = tmp_path / "bundles"
    monkeypatch.setattr(settings, "AGENT_BUNDLE_ROOT", str(root), raising=False)
    compiler = _Compiler()
    registry = AgentToolRegistry(
        [
            FunctionToolProvider(
                name="fake_compiler",
                modes={"autonomous"},
                handlers={"compile_c_snippet": compiler.handle},
            )
        ]
    )
    provider = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(tool_registry=registry, workspace_manager=None)
    )

    async def measure(db, job, code):
        handled, result = await registry.try_execute(
            "compile_c_snippet",
            {"code": code, "flags": "-O2"},
            _ctx(db, job),
        )
        assert handled and result["success"]
        return result

    return SimpleNamespace(
        root=root,
        compiler=compiler,
        registry=registry,
        handler=provider._handlers["verify_run_bundle"],
        measure=measure,
    )


async def test_verify_refuses_without_a_job(bundles, db_session):
    result = await bundles.handler({}, _ctx(db_session, None))
    assert "No job in context" in result["error"]


async def test_verify_refuses_a_run_that_recorded_nothing(
    bundles, db_session, test_user
):
    job = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler({}, _ctx(db_session, job))

    assert "recorded no evidence" in result["error"]
    assert "findings" not in result


@pytest.mark.parametrize("bad", ["not-a-uuid", "12345"])
async def test_verify_refuses_a_job_id_that_is_not_one(
    bundles, db_session, test_user, bad
):
    job = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler({"job_id": bad}, _ctx(db_session, job))

    assert "is not a job id" in result["error"]


async def test_verify_refuses_an_unknown_job(bundles, db_session, test_user):
    job = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler({"job_id": str(uuid4())}, _ctx(db_session, job))

    assert "belonging to this user" in result["error"]


async def test_verify_of_this_run_checks_what_the_registry_recorded(
    bundles, db_session, test_user
):
    job = await _job(db_session, test_user, job_type="analysis")
    await bundles.measure(db_session, job, "int f(int a,int b){return a+b;}")
    await bundles.measure(db_session, job, "int g(int a){return a*2;}")

    result = await bundles.handler({}, _ctx(db_session, job))

    assert result["success"] is True
    assert result["verified_job_id"] == str(job.id)
    assert result["verified_own_run"] is True
    assert result["data"]["integrity"]["intact"] is True
    assert result["data"]["integrity"]["artifacts_checked"] == 4
    assert result["data"]["bundle"]["tools"] == {"compile_c_snippet": 2}
    assert result["data"]["bundle"]["path"] == str(bundles.root / str(job.id))
    assert result["data"]["replay"] == {}
    (finding,) = result["findings"]
    assert finding["type"] == "bundle_verified"
    assert finding["entries"] == 2
    assert finding["intact"] is True
    assert finding["replay_verdict"] == "not replayed"
    assert finding["verified_job_id"] == str(job.id)
    assert finding["replay_summary"] is None
    # No replay was asked for, so nothing was run again.
    assert len(bundles.compiler.calls) == 2


async def test_verify_reports_a_tampered_artifact(bundles, db_session, test_user):
    job = await _job(db_session, test_user, job_type="analysis")
    await bundles.measure(db_session, job, "int f(void){return 1;}")
    entry = bundle.read_manifest(str(job.id))[0]
    result_file = bundles.root / str(job.id) / entry["artifact_dir"] / "result.json"
    tampered = json.loads(result_file.read_text())
    tampered["data"]["assembly"] = "mov x0, #42"
    result_file.write_text(json.dumps(tampered))

    result = await bundles.handler({}, _ctx(db_session, job))

    assert result["data"]["integrity"]["intact"] is False
    assert result["data"]["integrity"]["changed"] == ["1/result.json"]
    assert result["findings"][0]["intact"] is False
    assert "BROKEN" in result["findings"][0]["title"]


async def test_verify_checks_an_earlier_run_of_the_same_owner(
    bundles, db_session, test_user
):
    earlier = await _job(db_session, test_user, name="Earlier", status="completed")
    await bundles.measure(db_session, earlier, "int f(void){return 1;}")
    current = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler(
        {"job_id": f"  {earlier.id}  "}, _ctx(db_session, current)
    )

    assert result["success"] is True
    assert result["verified_job_id"] == str(earlier.id)
    assert result["verified_own_run"] is False
    assert result["findings"][0]["entries"] == 1


async def test_verify_refuses_another_users_run_even_with_a_bundle(
    bundles, db_session, test_user
):
    other = await _other_user(db_session)
    theirs = await _job(db_session, other, name="Theirs", status="completed")
    await bundles.measure(db_session, theirs, "int secret(void){return 7;}")
    mine = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler({"job_id": str(theirs.id)}, _ctx(db_session, mine))

    assert "belonging to this user" in result["error"]
    assert "data" not in result
    assert "secret" not in json.dumps(result)


async def test_verify_with_replay_reproduces_a_deterministic_run(
    bundles, db_session, test_user
):
    earlier = await _job(db_session, test_user, name="Earlier", status="completed")
    await bundles.measure(db_session, earlier, "int f(void){return 1;}")
    current = await _job(db_session, test_user, job_type="analysis")

    result = await bundles.handler(
        {"job_id": str(earlier.id), "replay": True}, _ctx(db_session, current)
    )

    assert result["data"]["replay"]["verdict"] == "reproduced"
    assert result["findings"][0]["replay_verdict"] == "reproduced"
    # The replay ran the recorded call with the recorded parameters.
    assert bundles.compiler.calls[-1] == {
        "code": "int f(void){return 1;}",
        "flags": "-O2",
    }


@pytest.mark.parametrize("finding_type", ["bundle_verified", "bundle_reproduced"])
async def test_saved_verification_labels_cannot_satisfy_runtime_proof_contracts(
    db_session, test_user, finding_type
):
    from app.services.agent_goal_contract_service import AgentGoalContractService
    from app.services.agent_tool_providers.research import (
        build_autonomous_research_provider,
    )
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = await _job(
        db_session,
        test_user,
        job_type="analysis",
        config={
            "goal_contract": {
                "enabled": True,
                "required_finding_types": [finding_type],
            }
        },
    )
    save = build_autonomous_research_provider(
        SimpleNamespace(_job_findings={})
    )._handlers["save_research_finding"]
    result = await save(
        {
            "title": "A claimed verification",
            "finding_type": finding_type,
            "metrics": {
                "intact": True,
                "verified": True,
                "replay_verdict": "reproduced",
            },
        },
        _ctx(db_session, job),
    )
    evaluation = AgentGoalContractService().evaluate_goal_contract(
        AutonomousAgentExecutor.__new__(AutonomousAgentExecutor),
        job,
        {"goal_progress": 100, "findings": result["findings"]},
    )
    assert evaluation["satisfied"] is False


@pytest.mark.parametrize(
    "mode",
    [
        "integrity_only",
        "matched",
        "different",
        "failed",
        "empty",
        "missing",
        "tampered",
        "timing_only",
        "legacy_unhashed",
        "partial_legacy",
        "mixed_timing",
        "original_failure",
    ],
)
async def test_reproduction_contract_requires_intact_nonempty_successful_replay(
    bundles, db_session, test_user, monkeypatch, mode
):
    from app.services.agent_goal_contract_service import AgentGoalContractService
    from app.services.agent_tool_providers.research import (
        build_autonomous_research_provider,
    )
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = await _job(db_session, test_user, job_type="analysis")
    if mode == "timing_only":
        bundle.record_entry(
            job_id=str(job.id),
            tool="benchmark_c_snippet",
            params={},
            result={"success": True, "data": {"fastest_ms": 1}},
        )
    else:
        await bundles.measure(db_session, job, "int f(void){return 1;}")
    if mode == "partial_legacy":
        await bundles.measure(db_session, job, "int g(void){return 2;}")
    if mode in {"mixed_timing", "original_failure"}:
        bundle.record_entry(
            job_id=str(job.id),
            tool="benchmark_c_snippet"
            if mode == "mixed_timing"
            else "compile_c_snippet",
            params={},
            result={"success": mode == "mixed_timing"},
        )
    if mode == "different":
        bundles.compiler.assembly = "sub x0, x0, x1"
    if mode in {"failed", "empty", "missing"}:

        async def fail(params, ctx):
            return (
                None
                if mode == "empty"
                else {"success": False, "error": "sandbox unavailable"}
            )

        providers = (
            []
            if mode == "missing"
            else [
                FunctionToolProvider(
                    name="failed",
                    modes={"autonomous"},
                    handlers={"compile_c_snippet": fail},
                )
            ]
        )
        monkeypatch.setattr(bundles.registry, "_providers", providers)
    if mode == "tampered":
        entry = bundle.read_manifest(str(job.id))[0]
        path = bundle.bundle_dir(str(job.id)) / entry["artifact_dir"] / "result.json"
        payload = json.loads(path.read_text())
        payload["data"]["assembly"] = "corrupted"
        path.write_text(json.dumps(payload))
    if mode in {"legacy_unhashed", "partial_legacy"}:
        entries = bundle.read_manifest(str(job.id))
        entries[0].pop("canonical_sha256")
        (bundle.bundle_dir(str(job.id)) / bundle.MANIFEST_NAME).write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries)
        )
    result = await bundles.handler(
        {"replay": mode != "integrity_only"}, _ctx(db_session, job)
    )
    findings = result["findings"]
    assert findings[0]["type"] == "bundle_verified"
    assert findings[0]["verified"] is (mode != "tampered")
    proofs = [f for f in findings if f["type"] == "bundle_reproduced"]
    replay_passes = mode in {"matched", "mixed_timing", "original_failure"}
    assert bool(proofs) is replay_passes
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    for finding_type, expected in (
        ("bundle_verified", mode != "tampered"),
        ("bundle_reproduced", replay_passes),
    ):
        job.config = {
            "goal_contract": {"enabled": True, "required_finding_types": [finding_type]}
        }
        evaluation = AgentGoalContractService().evaluate_goal_contract(
            executor, job, {"goal_progress": 100, "findings": findings}
        )
        assert evaluation["satisfied"] is expected, evaluation
    if mode == "partial_legacy":
        assert result["data"]["replay"]["verdict"] == "reproduced"
        assert result["data"]["replay"]["judged"] == 1
        assert result["data"]["replay"]["expected_comparisons"] == 2
    if mode == "matched":
        proof = proofs[0]
        assert proof["id"] != findings[0]["id"]
        assert proof["replay_summary"]["judged"] == 1
        job.results = {"findings": findings}
        job.status = "completed"
        await db_session.commit()
        child = await _job(
            db_session,
            test_user,
            name="Consumer",
            job_type="analysis",
            parent_job_id=job.id,
            config={"pipeline_assumes": ["bundle_reproduced"]},
        )
        state = {"findings": []}
        await AutonomousAgentExecutor._inherit_assumed_findings(
            None, child, state, db_session
        )
        reader = build_autonomous_research_provider(
            SimpleNamespace(_job_findings={})
        )._handlers["get_research_findings"]
        read = await reader({}, _ctx(db_session, child, state))
        (inherited,) = read["data"]["findings"]
        assert inherited["type"] == "bundle_reproduced"
        assert inherited["verified_job_id"] == str(job.id)
        assert inherited["id"] == proof["id"]


async def test_verify_with_replay_reports_a_result_that_changed(
    bundles, db_session, test_user
):
    earlier = await _job(db_session, test_user, name="Earlier", status="completed")
    await bundles.measure(db_session, earlier, "int f(void){return 1;}")
    current = await _job(db_session, test_user, job_type="analysis")
    bundles.compiler.assembly = "sub x0, x0, x1"

    result = await bundles.handler(
        {"job_id": str(earlier.id), "replay": True}, _ctx(db_session, current)
    )

    assert result["data"]["replay"]["verdict"] == "differed"
    assert result["findings"][0]["replay_verdict"] == "differed"


@pytest.mark.parametrize("mode", ["missing", "failed", "empty"])
async def test_verify_replay_reports_execution_failure_without_a_changed_measurement(
    bundles, db_session, test_user, monkeypatch, mode
):
    job = await _job(db_session, test_user, job_type="analysis")
    await bundles.measure(db_session, job, "int f(void){return 1;}")
    manifest = bundle.bundle_dir(str(job.id)) / bundle.MANIFEST_NAME
    before = manifest.read_bytes()

    async def fail(params, ctx):
        return (
            None
            if mode == "empty"
            else {"success": False, "error": "sandbox is unavailable"}
        )

    providers = (
        []
        if mode == "missing"
        else [
            FunctionToolProvider(
                name="failing_compiler",
                modes={"autonomous"},
                handlers={"compile_c_snippet": fail},
            )
        ]
    )
    monkeypatch.setattr(bundles.registry, "_providers", providers)
    result = await bundles.handler({"replay": True}, _ctx(db_session, job))
    replay = result["data"]["replay"]
    assert replay["verdict"] == "inconclusive"
    assert replay["judged"] == 0 and replay["differed"] == []
    summary = result["findings"][0]["replay_summary"]
    assert summary["judged"] == 0 and summary["error_count"] == 1
    assert summary["errors"] == replay["errored"]
    assert len(replay["errored"]) == 1
    assert "unavailable" in replay["errored"][0]["why"] or mode == "empty"
    assert result["findings"][0]["replay_verdict"] == "inconclusive"
    assert result["data"]["integrity"]["intact"]
    assert manifest.read_bytes() == before


async def test_unsampled_program_changes_survive_while_identical_replays_deduplicate(
    bundles, db_session, test_user, monkeypatch
):
    from dataclasses import replace

    from app.agent_core import research_programs as programs
    from app.agent_core.tool_specs.spec import ToolSpec
    from app.services import agent_loop_policy
    from app.services.agent_tool_providers.research import (
        build_autonomous_research_provider,
    )
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    def lab(name, tool):
        return programs.ResearchProgram(
            name=name,
            version="1.0",
            tool_specs=(ToolSpec(name=tool, description="Measure", parameters={}),),
            evidence_tools=(tool,),
        )

    alpha = lab("alpha_lab", "fixture_alpha")
    beta = lab("beta_lab", "fixture_beta")
    monkeypatch.setattr(programs, "_REPORT", programs.LoadReport(loaded=[alpha, beta]))
    source = await _job(db_session, test_user, name="Experiment", status="completed")
    payload = {"success": True, "data": {"cycles": 42}}
    for index in range(12):
        bundle.record_entry(
            job_id=str(source.id),
            tool="fixture_alpha" if index < 11 else "fixture_beta",
            params={"case": index},
            result=payload,
        )

    async def measure(params, ctx):
        return dict(payload)

    bundles.registry.register(
        FunctionToolProvider(
            name="fixtures",
            modes={"autonomous"},
            handlers={"fixture_alpha": measure, "fixture_beta": measure},
        )
    )
    verifier = await _job(db_session, test_user, name="Verifier", job_type="analysis")
    findings = []
    for version in ("2.0", "3.0", "3.0"):
        monkeypatch.setattr(
            programs,
            "_REPORT",
            programs.LoadReport(
                loaded=[
                    replace(alpha, version="2.0"),
                    replace(beta, version=version),
                ]
            ),
        )
        result = await bundles.handler(
            {"job_id": str(source.id), "replay": True}, _ctx(db_session, verifier)
        )
        findings.extend(f for f in result["findings"] if f["type"] == "bundle_verified")
    first, second, repeat = findings
    assert first["title"] == second["title"] == repeat["title"]
    assert (
        first["replay_summary"] == second["replay_summary"] == repeat["replay_summary"]
    )
    assert first["id"] != second["id"] == repeat["id"]
    assert first["id"].startswith("bundle-verification:")
    assert agent_loop_policy.distinct_findings(findings) == 2
    verifier.results = {"findings": findings}
    await db_session.commit()
    await db_session.refresh(verifier)
    child = await _job(
        db_session,
        test_user,
        name="Consumer",
        job_type="analysis",
        parent_job_id=verifier.id,
        root_job_id=verifier.id,
        config={"pipeline_assumes": ["bundle_verified"]},
    )
    state = {"findings": []}
    await AutonomousAgentExecutor._inherit_assumed_findings(
        None, child, state, db_session
    )
    inherited = state["findings"]
    assert {f["id"] for f in inherited} == {first["id"], second["id"]}
    assert len(inherited) == 2
    reader = build_autonomous_research_provider(
        SimpleNamespace(_job_findings={})
    )._handlers["get_research_findings"]
    read = await reader({}, _ctx(db_session, child, state))
    assert len(read["data"]["findings"]) == 2
    assert all(f["verified_job_id"] == str(source.id) for f in read["data"]["findings"])
    assert bundle.verify_integrity(str(source.id))["intact"]
    assert len(bundle.read_manifest(str(source.id))) == 12


async def test_changing_only_replay_error_wording_does_not_create_progress(
    bundles, db_session, test_user, monkeypatch
):
    from app.services import agent_loop_policy

    job = await _job(db_session, test_user, job_type="analysis")
    await bundles.measure(db_session, job, "int f(void){return 1;}")
    calls = []

    async def fail(params, ctx):
        calls.append(params)
        return {
            "success": False,
            "error": f"temporary sandbox /tmp/replay-{len(calls)} failed",
        }

    monkeypatch.setattr(
        bundles.registry,
        "_providers",
        [
            FunctionToolProvider(
                name="failing_compiler",
                modes={"autonomous"},
                handlers={"compile_c_snippet": fail},
            )
        ],
    )
    first = (await bundles.handler({"replay": True}, _ctx(db_session, job)))[
        "findings"
    ][0]
    second = (await bundles.handler({"replay": True}, _ctx(db_session, job)))[
        "findings"
    ][0]
    assert first["replay_summary"]["errors"] != second["replay_summary"]["errors"]
    assert first["id"] == second["id"]
    assert agent_loop_policy.distinct_findings([first, second]) == 1
    assert first["replay_verdict"] == second["replay_verdict"] == "inconclusive"


async def test_different_bundle_ids_survive_inheritance_with_identical_summaries(
    bundles, db_session, test_user
):
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    verifier = await _job(db_session, test_user, name="Verifier", job_type="analysis")
    findings = []
    source_ids = set()
    for index in range(2):
        source = await _job(
            db_session, test_user, name=f"Source {index}", status="completed"
        )
        await bundles.measure(db_session, source, "int f(void){return 1;}")
        source_ids.add(str(source.id))
        result = await bundles.handler(
            {"job_id": str(source.id)}, _ctx(db_session, verifier)
        )
        findings.extend(result["findings"])
    verifier.results = {"findings": findings}
    await db_session.commit()
    child = await _job(
        db_session,
        test_user,
        name="Consumer",
        job_type="analysis",
        parent_job_id=verifier.id,
        root_job_id=verifier.id,
        config={"pipeline_assumes": ["bundle_verified"]},
    )
    state = {"findings": []}
    await AutonomousAgentExecutor._inherit_assumed_findings(
        None, child, state, db_session
    )
    inherited = state["findings"]
    assert len(inherited) == 2
    assert {f["verified_job_id"] for f in inherited} == source_ids
    assert all(f["inherited_from_job_id"] == str(verifier.id) for f in inherited)


@pytest.mark.parametrize("mode", ["matched", "upgraded", "unavailable"])
async def test_replay_context_survives_persistence_inheritance_and_research_read(
    bundles, db_session, test_user, monkeypatch, mode
):
    from dataclasses import replace

    from app.agent_core import research_programs as programs
    from app.agent_core.tool_specs.spec import ToolSpec
    from app.services.agent_tool_providers.research import (
        build_autonomous_research_provider,
    )
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    lab = programs.ResearchProgram(
        name="fixture_lab",
        version="1.0",
        tool_specs=(
            ToolSpec(name="fixture_measure", description="Measure", parameters={}),
        ),
        evidence_tools=("fixture_measure",),
    )
    monkeypatch.setattr(programs, "_REPORT", programs.LoadReport(loaded=[lab]))
    earlier = await _job(
        db_session, test_user, name="Original experiment", status="completed"
    )
    payload = {"success": True, "data": {"cycles": 42}}
    for index in range(12):
        bundle.record_entry(
            job_id=str(earlier.id),
            tool="fixture_measure",
            params={"case": index},
            result=payload,
        )
    if mode in {"matched", "upgraded"}:
        monkeypatch.setattr(
            programs,
            "_REPORT",
            programs.LoadReport(
                loaded=[replace(lab, version="2.0" if mode == "upgraded" else "1.0")]
            ),
        )

        async def measure(params, ctx):
            return dict(payload)

        bundles.registry.register(
            FunctionToolProvider(
                name="fixture_lab",
                modes={"autonomous"},
                handlers={"fixture_measure": measure},
            )
        )
    else:
        monkeypatch.setattr(
            programs, "_REPORT", programs.LoadReport(disabled=[lab.name])
        )

    verifier = await _job(db_session, test_user, name="Verifier", job_type="analysis")
    result = await bundles.handler(
        {"job_id": str(earlier.id), "replay": True}, _ctx(db_session, verifier)
    )
    finding = result["findings"][0]
    replay = result["data"]["replay"]
    summary = finding["replay_summary"]
    proofs = [f for f in result["findings"] if f["type"] == "bundle_reproduced"]
    assert bool(proofs) is (mode != "unavailable")
    if proofs:
        assert proofs[0]["replay_summary"] == summary
        assert proofs[0]["id"] != finding["id"]
    expected_status = {
        "matched": "match",
        "upgraded": "version_mismatch",
        "unavailable": "program_unavailable",
    }[mode]
    issue_count = 0 if mode == "matched" else 12
    assert finding["verified_job_id"] == str(earlier.id)
    assert ("program issues: 12" in finding["title"]) == (issue_count > 0)
    assert summary["program_status_counts"] == {expected_status: 12}
    assert summary["program_issue_count"] == issue_count
    assert len(summary["program_issues"]) == min(issue_count, 10)
    assert len(replay["program_diagnostics"]) == 12
    assert (
        summary["judged"]
        == summary["reproduced"]
        == (0 if mode == "unavailable" else 12)
    )
    assert summary["error_count"] == (12 if mode == "unavailable" else 0)
    assert len(summary["errors"]) == (10 if mode == "unavailable" else 0)
    assert finding["replay_verdict"] == (
        "inconclusive" if mode == "unavailable" else "reproduced"
    )
    # The durable finding is independent of transient response normalization.
    replay["program_diagnostics"][0]["recorded"]["version"] = "transient mutation"
    if summary["program_issues"]:
        assert summary["program_issues"][0]["recorded"]["version"] == "1.0"
    if mode == "unavailable":
        replay["errored"][0]["why"] = "transient mutation"
        assert "unavailable" in summary["errors"][0]["why"]
    verifier.results = {"findings": [finding]}
    verifier.status = "completed"
    await db_session.commit()
    await db_session.refresh(verifier)
    assert verifier.results["findings"][0]["replay_summary"] == summary
    child = await _job(
        db_session,
        test_user,
        name="Consumer",
        job_type="analysis",
        parent_job_id=verifier.id,
        root_job_id=verifier.id,
        config={"pipeline_assumes": ["bundle_verified"]},
    )
    state = {"findings": []}
    await AutonomousAgentExecutor._inherit_assumed_findings(
        None, child, state, db_session
    )
    reader = build_autonomous_research_provider(
        SimpleNamespace(_job_findings={})
    )._handlers["get_research_findings"]
    read = await reader({}, _ctx(db_session, child, state))
    (inherited,) = read["data"]["findings"]
    assert inherited["inherited"] is True
    assert inherited["inherited_from_job_id"] == str(verifier.id)
    assert inherited["verified_job_id"] == str(earlier.id)
    assert inherited["replay_summary"] == summary
    assert bundle.verify_integrity(str(earlier.id))["intact"]
    assert len(bundle.read_manifest(str(earlier.id))) == 12
    assert not bundle.read_manifest(str(child.id))


async def test_replaying_this_run_does_not_add_to_its_bundle(
    bundles, db_session, test_user
):
    job = await _job(db_session, test_user, job_type="analysis")
    await bundles.measure(db_session, job, "int f(void){return 1;}")
    await bundles.measure(db_session, job, "int g(void){return 2;}")

    result = await bundles.handler({"replay": True}, _ctx(db_session, job))

    assert result["data"]["replay"]["verdict"] == "reproduced"
    assert result["data"]["integrity"]["entries"] == 2
    assert result["findings"][0]["entries"] == 2
    assert len(bundle.read_manifest(str(job.id))) == 2


# --------------------------------------------------------------------------
# literature_review_arxiv
# --------------------------------------------------------------------------


def _entry(arxiv_url, title):
    return (
        "<entry>"
        f"<id>{arxiv_url}</id>"
        f"<title>{title}</title>"
        "<summary>An abstract.</summary>"
        "<published>2026-01-02T00:00:00Z</published>"
        "<author><name>A. Author</name></author>"
        '<category term="cs.LG"/>'
        '<arxiv:primary_category term="cs.LG"/>'
        "</entry>"
    )


def _feed(*entries):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{len(entries)}</opensearch:totalResults>"
        + "".join(entries)
        + "</feed>"
    )


class _Arxiv:
    def __init__(self):
        self.requests = []
        self.status = 200
        self.body = _feed(
            _entry("http://arxiv.org/abs/2601.00001v1", "Sparse attention I"),
            _entry("http://arxiv.org/abs/2601.00002v2", "Sparse attention II"),
        )

    def handle(self, request):
        self.requests.append(request)
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(self.status, text=self.body)

    @property
    def last_params(self):
        return dict(self.requests[-1].url.params)


class _Queue:
    def __init__(self, task):
        self.signature = inspect.signature(task.run)
        self.calls = []
        self.fail = False

    def delay(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        if self.fail:
            raise ConnectionError("broker unreachable")
        self.calls.append(dict(bound.arguments))
        return SimpleNamespace(id=str(uuid4()))


def _document_service():
    docs = DocumentService.__new__(DocumentService)
    docs.vector_store = vector_store_service
    docs.text_processor = TextProcessor()
    docs._vector_store_initialized = False
    return docs


@pytest.fixture
def research(monkeypatch):
    arxiv = _Arxiv()
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(arxiv.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    ingest = _Queue(ingest_from_source)
    monkeypatch.setattr(ingest_from_source, "delay", ingest.delay)
    service = AgentService.__new__(AgentService)
    service.document_service = _document_service()
    provider = build_agent_service_research_provider(service)
    return SimpleNamespace(
        arxiv=arxiv,
        ingest=ingest,
        handler=provider._handlers["literature_review_arxiv"],
    )


async def _sources(db):
    return (await db.execute(select(DocumentSource))).scalars().all()


@pytest.mark.parametrize("params", [{}, {"topic": ""}, {"topic": "   "}])
async def test_review_requires_a_topic(research, db_session, test_user, params):
    with pytest.raises(ValueError, match="topic is required"):
        await research.handler(params, _chat_ctx(db_session, test_user))
    assert research.arxiv.requests == []
    assert await _sources(db_session) == []


async def test_review_searches_ingests_and_produces_its_declared_evidence(
    research, db_session, test_user
):
    result = await research.handler(
        {
            "topic": "sparse attention",
            "categories": ["cs.LG", " ", 3],
            "max_papers": 7,
            "sort_by": "submittedDate",
            "sort_order": "ascending",
        },
        _chat_ctx(db_session, test_user),
    )

    sent = research.arxiv.last_params
    assert sent["search_query"] == 'all:"sparse attention" AND (cat:cs.LG)'
    assert sent["max_results"] == "7"
    assert sent["sortBy"] == "submittedDate"
    assert sent["sortOrder"] == "ascending"

    assert set(spec_for("literature_review_arxiv").produces) <= {
        f["type"] for f in result["findings"]
    }
    (finding,) = result["findings"]
    assert finding["type"] == "literature_review"
    assert finding["paper_count"] == 2
    assert finding["paper_ids"] == ["2601.00001v1", "2601.00002v2"]

    (source,) = await _sources(db_session)
    assert source.source_type == "arxiv"
    assert source.config["paper_ids"] == ["2601.00001v1", "2601.00002v2"]
    assert source.config["requested_by"] == test_user.username
    assert source.config["requested_by_user_id"] == str(test_user.id)
    assert source.config["auto_literature_review"] is True
    assert source.config["topic"] == "sparse attention"
    assert result["ingest"]["source_id"] == str(source.id)
    assert result["ingest"]["queued"] is True
    assert research.ingest.calls == [{"source_id": str(source.id)}]


async def test_review_uses_an_explicit_query_and_caps_the_paper_count(
    research, db_session, test_user
):
    await research.handler(
        {"topic": "x", "query": "ti:prefetching", "max_papers": 400, "ingest": False},
        _chat_ctx(db_session, test_user),
    )

    sent = research.arxiv.last_params
    assert sent["search_query"] == "ti:prefetching"
    assert sent["max_results"] == "25"


async def test_review_without_ingest_creates_no_source(research, db_session, test_user):
    result = await research.handler(
        {"topic": "prefetching", "ingest": False}, _chat_ctx(db_session, test_user)
    )

    assert research.arxiv.last_params["search_query"] == "all:prefetching"
    assert result["ingest"] is None
    assert len(result["papers"]) == 2
    assert await _sources(db_session) == []
    assert research.ingest.calls == []


async def test_review_reports_a_broker_that_refused_the_ingestion(
    research, db_session, test_user
):
    research.ingest.fail = True

    result = await research.handler(
        {"topic": "prefetching"}, _chat_ctx(db_session, test_user)
    )

    assert result["ingest"]["queued"] is False
    assert result["ingest"]["task_id"] is None
    # Said where it is seen: on the result, and on the finding a contract
    # counts.
    assert "not queued" in result["warning"]
    assert result["findings"][0]["ingest_queued"] is False


async def test_review_fails_when_arxiv_answers_with_an_error(
    research, db_session, test_user
):
    research.arxiv.status = 406

    with pytest.raises(httpx.HTTPStatusError):
        await research.handler(
            {"topic": "prefetching"}, _chat_ctx(db_session, test_user)
        )
    assert await _sources(db_session) == []
    assert research.ingest.calls == []


async def test_review_with_no_papers_produces_no_review(
    research, db_session, test_user
):
    research.arxiv.body = _feed()

    result = await research.handler(
        {"topic": "nonexistent topic"}, _chat_ctx(db_session, test_user)
    )

    assert not any(
        f.get("type") == "literature_review" for f in result.get("findings") or []
    )


async def test_review_keeps_an_old_style_arxiv_id_whole(
    research, db_session, test_user
):
    research.arxiv.body = _feed(
        _entry("http://arxiv.org/abs/hep-th/9901001v1", "An old paper")
    )

    result = await research.handler(
        {"topic": "strings"}, _chat_ctx(db_session, test_user)
    )

    assert result["findings"][0]["paper_ids"] == ["hep-th/9901001v1"]


async def test_review_names_a_max_papers_it_cannot_read(
    research, db_session, test_user
):
    try:
        await research.handler(
            {"topic": "prefetching", "max_papers": "five", "ingest": False},
            _chat_ctx(db_session, test_user),
        )
    except Exception as exc:  # noqa: BLE001 - the message is what is judged
        assert "max_papers" in str(exc)


async def test_review_is_callable_by_the_job_types_it_is_declared_for(
    db_session, test_user
):
    spec = spec_for("literature_review_arxiv")
    assert "research" in spec.job_types
    job = await _job(db_session, test_user, job_type="research")

    executor = _autonomous_executor()

    assert executor.tool_registry.resolve(
        "literature_review_arxiv", _ctx(db_session, job)
    )


async def test_a_research_job_calling_it_is_told_why_it_failed(db_session, test_user):
    from app.services.agent_action_service import AgentActionService

    job = await _job(db_session, test_user, job_type="research")

    result = await AgentActionService()._act_unjournaled(
        _autonomous_executor(),
        job,
        {"tool": "no_such_research_tool", "params": {"topic": "prefetching"}},
        {},
        db_session,
    )

    assert result["success"] is False
    assert result.get("error")


# --------------------------------------------------------------------------
# generate_gitlab_architecture
# --------------------------------------------------------------------------


class _GitLab:
    signature = inspect.signature(
        GitLabArchitectureService.generate_architecture_diagram
    )

    def __init__(self):
        self.calls = []
        self.fail = None

    async def generate_architecture_diagram(self, *args, **kwargs):
        bound = self.signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        self.calls.append(arguments)
        if self.fail:
            raise self.fail
        return {
            "project": {"path": arguments["project_id"]},
            "mermaid_code": "graph TD; A-->B",
            "diagram_type": arguments["diagram_type"],
            "focus": arguments["focus"],
            "analysis_summary": "two services",
            "svg": "<svg/>",
        }


@pytest.fixture
def gitlab(monkeypatch):
    rec = _GitLab()
    monkeypatch.setattr(gitlab_module, "get_gitlab_architecture_service", lambda: rec)
    service = AgentService.__new__(AgentService)
    provider = build_agent_service_analytics_content_provider(service)
    rec.handler = provider._handlers["generate_gitlab_architecture"]
    return rec


async def _gitlab_source(db, user, *, token="tok", url="https://gitlab.example"):
    source = DocumentSource(
        name=f"gitlab-{uuid4().hex[:6]}",
        source_type="gitlab",
        is_active=True,
        config={"gitlab_url": url, "token": token, "requested_by": user.username},
    )
    db.add(source)
    await db.commit()
    return source


async def test_gitlab_requires_a_project_id(gitlab, db_session, test_user):
    await _gitlab_source(db_session, test_user)

    result = await gitlab.handler({}, _chat_ctx(db_session, test_user))

    assert result == {"error": "project_id is required"}
    assert gitlab.calls == []


async def test_gitlab_forwards_every_declared_parameter(gitlab, db_session, test_user):
    await _gitlab_source(db_session, test_user, token="mine")
    params = {
        "project_id": "group/project",
        "branch": "develop",
        "diagram_type": "c4",
        "focus": "data_flow",
        "detail_level": "high",
    }
    declared = set(spec_for("generate_gitlab_architecture").parameters["properties"])
    assert declared == set(params)

    result = await gitlab.handler(params, _chat_ctx(db_session, test_user))

    assert result["success"] is True
    assert result["mermaid_code"] == "graph TD; A-->B"
    assert result["has_svg"] is True and result["has_png"] is False
    assert gitlab.calls == [
        {
            "gitlab_url": "https://gitlab.example",
            "token": "mine",
            "project_id": "group/project",
            "branch": "develop",
            "diagram_type": "c4",
            "focus": "data_flow",
            "detail_level": "high",
        }
    ]


async def test_gitlab_applies_the_declared_defaults(gitlab, db_session, test_user):
    await _gitlab_source(db_session, test_user)

    await gitlab.handler({"project_id": "42"}, _chat_ctx(db_session, test_user))

    (call,) = gitlab.calls
    assert call["diagram_type"] == "auto"
    assert call["detail_level"] == "medium"
    assert call["branch"] is None and call["focus"] is None


async def test_gitlab_uses_the_callers_identity_not_anothers_source(
    gitlab, db_session, test_user
):
    other = await _other_user(db_session)
    await _gitlab_source(db_session, other, token="theirs")

    result = await gitlab.handler(
        {"project_id": "group/project"}, _chat_ctx(db_session, test_user)
    )

    assert "No active GitLab data source" in result["error"]
    assert gitlab.calls == []


async def test_gitlab_refuses_a_source_without_a_token(gitlab, db_session, test_user):
    await _gitlab_source(db_session, test_user, token="")

    result = await gitlab.handler(
        {"project_id": "group/project"}, _chat_ctx(db_session, test_user)
    )

    assert "missing URL or token" in result["error"]
    assert gitlab.calls == []


async def test_gitlab_reports_a_failed_generation_as_a_failure(
    gitlab, db_session, test_user
):
    await _gitlab_source(db_session, test_user)
    gitlab.fail = RuntimeError("404 Project Not Found")

    result = await gitlab.handler(
        {"project_id": "group/missing"}, _chat_ctx(db_session, test_user)
    )

    assert "success" not in result
    assert result["error"] == "404 Project Not Found"
    assert result["project_id"] == "group/missing"


# --------------------------------------------------------------------------
# update_document_tags / batch_summarize_documents: the wrappers
# --------------------------------------------------------------------------


@pytest.fixture
def documents(monkeypatch):
    queue = _Queue(summarize_document)
    monkeypatch.setattr(summarize_document, "delay", queue.delay)
    service = AgentService.__new__(AgentService)
    service.document_service = _document_service()
    provider = build_agent_service_document_provider(service)
    return SimpleNamespace(service=service, provider=provider, queue=queue)


async def _doc(db, title, **extra):
    source = DocumentSource(
        name=f"Uploads {uuid4().hex[:6]}", source_type="file", config={}
    )
    db.add(source)
    await db.commit()
    doc = Document(
        title=title,
        content="body",
        content_hash=hashlib.sha256(title.encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        **extra,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _stored_tags(db, doc_id):
    return (
        await db.execute(select(Document.tags).where(Document.id == doc_id))
    ).scalar_one()


async def test_tags_wrapper_passes_the_call_through_unchanged(
    documents, db_session, test_user
):
    via_wrapper = await _doc(db_session, "Via wrapper", tags=["a"])
    direct = await _doc(db_session, "Direct", tags=["a"])
    params = {"tags": ["b", "a"], "action": "replace"}

    wrapped = await documents.provider._handlers["update_document_tags"](
        {**params, "document_id": str(via_wrapper.id)},
        _chat_ctx(db_session, test_user),
    )
    called = await documents.service._tool_update_document_tags(
        {**params, "document_id": str(direct.id)}, db_session
    )

    assert await _stored_tags(db_session, via_wrapper.id) == ["b", "a"]
    assert await _stored_tags(db_session, direct.id) == ["b", "a"]
    for key in ("previous_tags", "current_tags", "action"):
        assert wrapped[key] == called[key]


async def test_tags_wrapper_reports_a_bad_id_as_the_service_does(
    documents, db_session, test_user
):
    result = await documents.provider._handlers["update_document_tags"](
        {"document_id": "nope", "tags": ["x"]}, _chat_ctx(db_session, test_user)
    )
    assert result == {"error": "Invalid document ID: nope"}


async def test_summarize_wrapper_queues_through_the_real_task_signature(
    documents, db_session, test_user
):
    lacking = await _doc(db_session, "No summary yet")
    summarized = await _doc(db_session, "Has one", summary="Already summarized.")

    result = await documents.provider._handlers["batch_summarize_documents"](
        {"document_ids": [str(lacking.id), str(summarized.id), "bad"]},
        _chat_ctx(db_session, test_user),
    )

    assert result["queued_count"] == 1
    assert result["skipped_count"] == 1
    assert result["invalid_ids"] == ["bad"]
    assert [(c["document_id"], c["force"]) for c in documents.queue.calls] == [
        (str(lacking.id), False)
    ]


async def test_summarize_wrapper_passes_force_regenerate(
    documents, db_session, test_user
):
    summarized = await _doc(db_session, "Has one", summary="Already summarized.")

    result = await documents.provider._handlers["batch_summarize_documents"](
        {"document_ids": [str(summarized.id)], "force_regenerate": True},
        _chat_ctx(db_session, test_user),
    )

    assert result["queued_count"] == 1
    assert documents.queue.calls[0]["force"] is True


@pytest.mark.parametrize(
    "tool",
    [
        "update_document_tags",
        "batch_summarize_documents",
        "generate_gitlab_architecture",
    ],
)
async def test_chat_only_tools_are_declared_and_registered_as_chat_only(
    db_session, test_user, tool
):
    """job_types=() says no autonomous job may call these; nothing does."""
    assert spec_for(tool).job_types == ()
    job = await _job(db_session, test_user, job_type="research")

    executor = _autonomous_executor()

    assert executor.tool_registry.resolve(tool, _ctx(db_session, job)) is None
    chat = AgentService.__new__(AgentService)
    providers = [
        build_agent_service_document_provider(chat),
        build_agent_service_analytics_content_provider(chat),
    ]
    chat_ctx = _chat_ctx(db_session, test_user)
    assert any(p.can_handle(tool, chat_ctx) for p in providers)
