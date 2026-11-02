"""The DeepSeek harness drives a job; the platform still executes and judges.

The real harness is not run here (it is an alpha runtime in its own
virtualenv and calls DeepSeek). A stand-in driver speaks the same JSON lines
and calls the platform's MCP server over HTTP exactly as the harness does, so
the server, the runner and the adapter seam are the real ones.
"""

import asyncio
import json
import sys
import textwrap
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.services import agent_harness_backbone as backbone
from app.utils.datetimes import utc_now

pytestmark = pytest.mark.unit


class TestThePatch:
    def test_every_tool_and_upload_of_its_own_is_off(self):
        patch = backbone.build_patch("http://127.0.0.1:5/mcp", "t0k")
        for row in (
            "persistent-bash",
            "terminal-bash",
            "tool-fs",
            "mcp-resources",
            "session-log-deepseek",
        ):
            assert f"- id: {row}\n  disabled: true" in patch
        assert "name: '@deepseek-ai/dsh-mcp-client'" in patch
        assert '"Bearer t0k"' in patch
        assert "failOnStartupError: true" in patch

    def test_the_driver_gets_no_secret_but_the_model_key(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/x")
        monkeypatch.setenv("SECRET_KEY", "s")
        env = backbone.driver_environment("sk-test")
        assert env["DEEPSEEK_API_KEY"] == "sk-test"
        assert "DATABASE_URL" not in env and "SECRET_KEY" not in env


TOOLS = [
    backbone.mcp_tool(
        {
            "name": "search_documents",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }
    )
]


async def _echo(name, params):
    return {"text": f"{name}:{json.dumps(params)}"}


@pytest.mark.asyncio
class TestTheProtocol:
    async def test_initialize_lists_and_calls(self):
        init = await backbone.handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            tools=TOOLS,
            call=_echo,
        )
        assert init["result"]["protocolVersion"] == "2025-06-18"
        listed = await backbone.handle_rpc(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, tools=TOOLS, call=_echo
        )
        assert [t["name"] for t in listed["result"]["tools"]] == ["search_documents"]
        called = await backbone.handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search_documents", "arguments": {"q": "x"}},
            },
            tools=TOOLS,
            call=_echo,
        )
        assert called["result"]["content"][0]["text"] == 'search_documents:{"q": "x"}'

    async def test_a_tool_the_job_may_not_call_is_not_called(self):
        answer = await backbone.handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "delete_document", "arguments": {}},
            },
            tools=TOOLS,
            call=_echo,
        )
        assert answer["error"]["code"] == -32602

    async def test_a_notification_gets_no_answer(self):
        assert (
            await backbone.handle_rpc(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                tools=TOOLS,
                call=_echo,
            )
            is None
        )


#: A stand-in for deepseek_harness_driver.py: reads the spec, finds the MCP
#: server and token in the patch, calls the first tool until told the run is
#: over (and once more, to see it refused), and reports like the real driver.
FAKE_DRIVER = textwrap.dedent(
    """
    import json, re, sys, urllib.request
    spec = json.loads(sys.stdin.read())
    patch = open(spec["patch"]).read()
    url = re.search(r'url: "([^"]+)"', patch).group(1)
    auth = re.search(r'Authorization: "([^"]+)"', patch).group(1)
    def rpc(i, method, params=None):
        body = json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}})
        req = urllib.request.Request(url, body.encode(), {"Authorization": auth,
              "Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    rpc(1, "initialize", {"protocolVersion": "2025-06-18"})
    tools = rpc(2, "tools/list")["result"]["tools"]
    texts = []
    for i in range(3, 10):
        print(json.dumps({"event": "assistant", "usage": {"inputTokens": 10}}), flush=True)
        out = rpc(i, "tools/call", {"name": tools[0]["name"], "arguments": {"query": f"q{i}"}})
        texts.append(out["result"]["content"][0]["text"])
        if "The run is over" in texts[-1] and len([t for t in texts if "The run is over" in t]) > 1:
            break
    print(json.dumps({"event": "result", "final_response": "done", "finish_reason": "completed",
                      "usage": {"inputTokens": 10 * len(texts)}, "texts": texts}), flush=True)
    """
)


class FakeAdapter:
    """The adapter seam, recording what the runner asked of it."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.acted = []
        self.completed = 0
        self.externally_driven = False
        self.state = {
            "goal_contract_last": {
                "enabled": True,
                "satisfied": False,
                "missing": ["benchmark_measurement"],
            }
        }
        self.job = SimpleNamespace(
            id="job-1",
            goal="find out",
            job_type="research",
            config={},
            llm_calls_used=0,
            log=[],
            add_log_entry=lambda e: self.job.log.append(e),
        )
        self.executor = SimpleNamespace(
            _resolve_agent_skill_profile=lambda job, state=None: {},
            _get_tools_for_job_type=lambda job_type, config, profile=None: [
                "search_documents"
            ],
        )
        self.agent_def = None
        self.events = []
        self.db = SimpleNamespace(commit=self._commit)
        self.start_time = utc_now()
        self.max_runtime = timedelta(minutes=5)

    async def _commit(self):
        self.events.append("commit")

    async def can_continue(self):
        return True

    async def on_iteration_start(self):
        pass

    async def act_phase(self, decision):
        assert self.externally_driven, "the harness's choice must be the one that runs"
        self.events.append("act")
        self.acted.append(decision["action"])
        return {
            "action": decision["action"],
            "action_result": {"success": True, "n": len(self.acted)},
        }

    async def evaluate_phase(self, decision, bundle):
        if len(self.acted) >= self.stop_after:
            self.state["goal_contract_last"] = {"enabled": True, "satisfied": True}
            return {"should_stop": True, "stop_reason": "goal contract satisfied"}
        return {"should_stop": False}

    async def on_iteration_complete(self, observation, decision, bundle, evaluation):
        self.completed += 1

    async def on_iteration_error(self, exc):
        raise exc

    async def build_run_result(self):
        return {"status": "completed", "iterations": self.completed}


def test_the_harness_chooses_and_the_contract_stops_it(tmp_path, monkeypatch):
    monkeypatch.setattr(backbone.settings, "DEEPSEEK_API_KEY", "sk-test")
    driver = tmp_path / "fake_driver.py"
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    adapter = FakeAdapter(stop_after=2)
    runner = backbone.HarnessBackboneRunner(
        driver_python=sys.executable, driver_script=str(driver)
    )

    result = asyncio.run(runner.run(adapter))

    assert result == {"status": "completed", "iterations": 2}
    assert [a["params"]["query"] for a in adapter.acted] == ["q3", "q4"]
    assert all(a["tool"] == "search_documents" for a in adapter.acted)
    (session,) = [e for e in adapter.job.log if e["phase"] == "harness_session"]
    assert session["stopped_by_platform"] == "goal contract satisfied"
    assert session["calls_after_stop"] == 1, "a call after the stop is refused, not run"
    assert session["llm_turns"] == 3
    assert (
        adapter.job.llm_calls_used == 3
    ), "the harness's model turns count against the budget"
    assert adapter.state["harness_final_response"] == "done"
    # Committed before the harness had control and after every call: an
    # uncommitted job row held the lock the lease heartbeat needs.
    assert adapter.events[:2] == ["commit", "act"]
    assert adapter.events.count("commit") >= 1 + len(adapter.acted)


HANGING_DRIVER = textwrap.dedent(
    """
    import os, sys, time
    open(sys.argv[0] + ".pid", "w").write(str(os.getpid()))
    sys.stdin.read()
    time.sleep(600)
    """
)


def test_a_run_that_ends_takes_the_harness_with_it(tmp_path, monkeypatch):
    """Abandoned for a lost lease, the job left its harness running."""
    import os

    monkeypatch.setattr(backbone.settings, "DEEPSEEK_API_KEY", "sk-test")
    driver = tmp_path / "hanging_driver.py"
    driver.write_text(HANGING_DRIVER, encoding="utf-8")
    runner = backbone.HarnessBackboneRunner(
        driver_python=sys.executable, driver_script=str(driver)
    )
    pid_file = tmp_path / "hanging_driver.py.pid"

    async def _cancel_once_started():
        task = asyncio.create_task(runner.run(FakeAdapter(stop_after=1)))
        for _ in range(400):
            if pid_file.exists() and pid_file.read_text():
                break
            await asyncio.sleep(0.025)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(pid_file.read_text())

    pid = asyncio.run(_cancel_once_started())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


ERRORING_DRIVER = textwrap.dedent(
    """
    import json, sys
    sys.stdin.read()
    print(json.dumps({"event": "result", "final_response": "", "finish_reason": "error",
                      "turn_end_reasons": [{"kind": "error", "message": "429 rate limited"}],
                      "errors": []}), flush=True)
    """
)


def test_a_session_that_errs_before_any_call_fails_the_job(tmp_path, monkeypatch):
    """Four sessions ended in error with no model turn and were finalised as
    completed runs with their contracts unmet; nothing said why."""
    monkeypatch.setattr(backbone.settings, "DEEPSEEK_API_KEY", "sk-test")
    driver = tmp_path / "erroring_driver.py"
    driver.write_text(ERRORING_DRIVER, encoding="utf-8")
    adapter = FakeAdapter(stop_after=1)
    runner = backbone.HarnessBackboneRunner(
        driver_python=sys.executable, driver_script=str(driver)
    )

    with pytest.raises(backbone.HarnessSessionFailed, match="429 rate limited"):
        asyncio.run(runner.run(adapter))
    (session,) = [e for e in adapter.job.log if e["phase"] == "harness_session"]
    assert session["turn_end_reasons"] == [
        {"kind": "error", "message": "429 rate limited"}
    ]


class TestTheDefault:
    """The harness is the default backbone (decided 2026-10-30); the loop is
    what a job falls back to when the harness would ignore its owner's choice
    of provider or cannot run on this worker."""

    @pytest.fixture
    def harness_runs_here(self, monkeypatch):
        monkeypatch.setattr(
            backbone.settings, "AGENT_DEFAULT_BACKBONE", "deepseek_harness"
        )
        monkeypatch.setattr(backbone.settings, "AGENT_DEEPSEEK_HARNESS_ENABLED", True)
        monkeypatch.setattr(
            backbone.settings, "AGENT_DEEPSEEK_HARNESS_PYTHON", sys.executable
        )
        monkeypatch.setattr(backbone.settings, "DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr(backbone.settings, "LLM_PROVIDER", "deepseek")

    @staticmethod
    def _job(**config):
        return SimpleNamespace(config=config)

    def test_a_job_naming_no_backbone_gets_the_harness(self, harness_runs_here):
        assert backbone.resolve_backbone(self._job()) == ("deepseek_harness", None)

    def test_a_job_may_still_ask_for_the_loop(self, harness_runs_here):
        assert backbone.resolve_backbone(self._job(backbone="loop")) == ("loop", None)

    def test_an_owner_who_chose_another_provider_keeps_the_loop(
        self, harness_runs_here
    ):
        chosen, why = backbone.resolve_backbone(
            self._job(), SimpleNamespace(provider="openai")
        )
        assert chosen == "loop" and "'openai'" in why

    def test_a_saved_provider_counts_as_another_provider(self, harness_runs_here):
        chosen, why = backbone.resolve_backbone(
            self._job(), SimpleNamespace(provider="custom:abc")
        )
        assert chosen == "loop" and why

    def test_a_worker_without_the_harness_falls_back_and_says_why(
        self, harness_runs_here, monkeypatch
    ):
        monkeypatch.setattr(
            backbone.settings, "AGENT_DEEPSEEK_HARNESS_PYTHON", "/nonexistent/python"
        )
        chosen, why = backbone.resolve_backbone(self._job())
        assert chosen == "loop" and "/nonexistent/python" in why

    def test_a_job_that_asked_for_the_harness_is_not_quietly_moved(
        self, harness_runs_here, monkeypatch
    ):
        monkeypatch.setattr(backbone.settings, "DEEPSEEK_API_KEY", "")
        # Resolved as asked; the runner then refuses with the reason.
        assert backbone.resolve_backbone(self._job(backbone="deepseek_harness")) == (
            "deepseek_harness",
            None,
        )

    def test_the_loop_as_the_deployment_default(self, harness_runs_here, monkeypatch):
        monkeypatch.setattr(backbone.settings, "AGENT_DEFAULT_BACKBONE", "loop")
        assert backbone.resolve_backbone(self._job()) == ("loop", None)
