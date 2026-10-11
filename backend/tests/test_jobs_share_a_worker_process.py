"""What two jobs in one worker process must not share.

A prefork worker runs one job per process, so anything a process holds was
one job's. With ``--pool=threads`` every job in the worker shares it, each on
its own event loop in its own thread. These pin the places found holding
something that belongs to one loop, one call or one parse.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from app.utils.limited_process import process_limits, with_limits

APP = Path(__file__).resolve().parent.parent / "app"


# -- resource limits without a fork hook ---------------------------------------


def test_no_subprocess_is_given_a_fork_hook():
    # `preexec_fn` runs in a forked copy of a process that has threads.
    found = []
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "preexec_fn":
                found.append(f"{path.relative_to(APP)}:{node.value.lineno}")
            if isinstance(node, ast.Constant) and node.value == "preexec_fn":
                found.append(f"{path.relative_to(APP)}:{node.lineno}")

    assert found == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_a_wrapped_command_really_runs_under_its_limits():
    report = (
        "import resource as r;"
        "print(r.getrlimit(r.RLIMIT_NOFILE)[0], r.getrlimit(r.RLIMIT_CORE)[0],"
        " r.getrlimit(r.RLIMIT_FSIZE)[0])"
    )
    limits = process_limits(
        cpu_seconds=30, memory_bytes=4 * 1024**3, file_bytes=7_000_000, open_files=64
    )

    limited = subprocess.run(
        with_limits([sys.executable, "-c", report], limits),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The control: the same program, not wrapped, is not under them.
    plain = subprocess.run(
        [sys.executable, "-c", report], capture_output=True, text=True, timeout=60
    )

    assert limited.returncode == 0, limited.stderr
    assert limited.stdout.split() == ["64", "0", "7000000"]
    assert plain.stdout.split() != limited.stdout.split()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_the_wrapped_command_keeps_its_arguments_and_exit_code():
    answer = subprocess.run(
        with_limits(
            ["/bin/sh", "-c", 'printf "%s|" "$0" "$@"; exit 7', "a b", "c"],
            process_limits(
                cpu_seconds=30,
                memory_bytes=4 * 1024**3,
                file_bytes=10**6,
                open_files=64,
            ),
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (answer.stdout, answer.returncode) == ("a b|c|", 7)


# -- clients belong to a loop --------------------------------------------------


def _on_two_loops(make):
    """`make()` called twice on one loop and once on another; the three."""

    async def twice():
        return make(), make()

    first, again = asyncio.run(twice())
    (other, _) = asyncio.run(twice())
    return first, again, other


def test_an_openai_client_is_not_handed_to_another_loop():
    from app.services.llm_providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        api_key="k", base_url="http://model.invalid/v1", default_model="m"
    )
    first, again, other = _on_two_loops(provider._get_client)

    assert first is again
    assert other is not first


def test_an_anthropic_client_is_not_handed_to_another_loop():
    from app.services.llm_providers.anthropic_sdk import AnthropicProvider

    provider = AnthropicProvider(api_key="k")
    first, again, other = _on_two_loops(provider._get_client)

    assert first is again
    assert other is not first


def test_a_gitlab_client_is_per_loop_and_still_per_token():
    from app.services.gitlab_architecture_service import GitLabArchitectureService

    service = GitLabArchitectureService()

    async def clients():
        mine = await service._get_client("http://gitlab.invalid", "token-a")
        same = await service._get_client("http://gitlab.invalid", "token-a")
        theirs = await service._get_client("http://gitlab.invalid", "token-b")
        return mine, same, theirs

    mine, same, theirs = asyncio.run(clients())
    later, _, _ = asyncio.run(clients())

    assert mine is same and theirs is not mine
    assert theirs.headers["Authorization"] == "Bearer token-b"
    assert later is not mine


# -- a splitter belongs to a call ----------------------------------------------


def test_callers_wanting_different_chunk_sizes_do_not_get_each_others():
    from app.services.text_processor import TextProcessor

    shared = TextProcessor()
    text = "word " * 4000
    sizes = [120, 900] * 4
    longest, errors = {}, []
    start = threading.Barrier(len(sizes))

    def split(slot, size):
        try:
            start.wait(timeout=30)
            for _ in range(20):
                chunks = asyncio.run(shared._fixed_chunk(text, size, 0))
                longest[slot] = max(
                    longest.get(slot, 0), max(len(chunk) for chunk in chunks)
                )
        except Exception as error:
            errors.append(repr(error))

    threads = [
        threading.Thread(target=split, args=(slot, size))
        for slot, size in enumerate(sizes)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert all(longest[slot] <= size for slot, size in enumerate(sizes)), longest
    # Nothing about one call is kept where the next would find it.
    assert not hasattr(shared, "text_splitter")


# -- a parser belongs to a parse ------------------------------------------------


def test_each_parse_gets_its_own_tree_sitter_parser():
    from app.services import repo_symbol_parsers as parsers

    first, second = parsers._ts_parser("js"), parsers._ts_parser("js")

    assert first is not None
    assert first is not second


def test_symbols_are_right_when_many_threads_parse_at_once(tmp_path):
    from app.services import repo_symbol_parsers as parsers

    sources = {}
    for n in range(6):
        js = tmp_path / f"module{n}.js"
        js.write_text(
            "\n".join(f"function js{n}_{k}(a) {{ return a + {k}; }}" for k in range(40))
        )
        c = tmp_path / f"unit{n}.c"
        c.write_text(
            "\n".join(f"int c{n}_{k}(int a) {{ return a + {k}; }}" for k in range(40))
        )
        sources[js] = {f"js{n}_{k}" for k in range(40)}
        sources[c] = {f"c{n}_{k}" for k in range(40)}

    found, errors = {}, []
    start = threading.Barrier(len(sources))

    def parse(path):
        try:
            start.wait(timeout=30)
            found[path] = {name for name, *_ in parsers.symbols_in(path, tmp_path)}
        except Exception as error:  # reported below, with the path
            errors.append((path.name, repr(error)))

    threads = [threading.Thread(target=parse, args=(path,)) for path in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert found == sources


# -- the sandbox's user can use the directory it is given -----------------------


def _others(path) -> int:
    return path.stat().st_mode & 0o007


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_work_directory_is_opened_to_the_sandbox_user_where_it_is_mounted(
    tmp_path,
):
    # As `tempfile` and a root process leave them: nobody else may enter.
    from app.services import agent_sandbox_runtime

    work = tmp_path / "work"
    nested = work / "src"
    nested.mkdir(parents=True)
    source, tool = nested / "kernel.c", work / "run.sh"
    source.write_text("int main(void) { return 0; }\n")
    tool.write_text("#!/bin/sh\n")
    for path, mode in ((source, 0o600), (tool, 0o700), (nested, 0o700), (work, 0o700)):
        path.chmod(mode)

    agent_sandbox_runtime.docker_command(image="img", workdir=str(work), script=":")

    assert (_others(work), _others(nested)) == (0o7, 0o7)
    assert _others(source) == 0o6
    # What was executable still is, for its owner; nothing is taken away.
    assert tool.stat().st_mode & 0o700 == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_read_only_mount_is_made_readable_and_no_more(tmp_path):
    from app.services import agent_sandbox_runtime

    work = tmp_path / "work"
    work.mkdir()
    source = work / "kernel.c"
    source.write_text("int x;\n")
    source.chmod(0o600)
    work.chmod(0o700)

    agent_sandbox_runtime.docker_command(
        image="img", workdir=str(work), script=":", read_only=True
    )

    assert (_others(work), _others(source)) == (0o5, 0o4)


def test_a_work_directory_that_is_not_there_is_left_to_the_run_to_report(tmp_path):
    from app.services import agent_sandbox_runtime

    command = agent_sandbox_runtime.docker_command(
        image="img", workdir=str(tmp_path / "gone"), script=":"
    )

    assert command[:2] == ["docker", "run"]


def test_as_root_the_work_directory_is_handed_to_the_sandbox_user(
    tmp_path, monkeypatch
):
    # A script may `chmod` a file this process wrote, which only its owner
    # can; opening the permission bits does not make it one.
    import os

    from app.services import agent_sandbox_runtime

    work = tmp_path / "work"
    work.mkdir()
    (work / "prog").write_bytes(b"\x7fELF")
    given = []
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        os, "chown", lambda path, uid, gid: given.append((Path(path).name, uid, gid))
    )

    agent_sandbox_runtime.docker_command(image="img", workdir=str(work), script=":")
    assert sorted(given) == [("prog", 65534, 65534), ("work", 65534, 65534)]

    # A directory mounted read-only is somebody's source tree: not given away.
    given.clear()
    agent_sandbox_runtime.docker_command(
        image="img", workdir=str(work), script=":", read_only=True
    )
    assert given == []


# -- a loop's clients are closed before it ends ---------------------------------


class _Client:
    def __init__(self):
        self.closed_on = None

    async def aclose(self):
        self.closed_on = asyncio.get_running_loop()


def test_a_finishing_loop_closes_what_it_holds_and_nothing_of_anothers():
    from app.utils.per_loop import PerLoop, close_all, close_running_loop_values

    one = PerLoop(_Client, close=lambda client: client.aclose())
    table = PerLoop(dict, close=close_all)
    plain = PerLoop(object)  # no closer: not ours to close
    other_loops, release = {}, threading.Event()

    def elsewhere():
        async def hold():
            other_loops["client"] = one.get()
            while not release.is_set():
                await asyncio.sleep(0.01)

        asyncio.run(hold())

    thread = threading.Thread(target=elsewhere)
    thread.start()

    async def job():
        mine, keyed = one.get(), table.get().setdefault("k", _Client())
        plain.get()
        while "client" not in other_loops:
            await asyncio.sleep(0.01)
        closed = await close_running_loop_values()
        return mine, keyed, closed, asyncio.get_running_loop()

    try:
        mine, keyed, closed, loop = asyncio.run(job())
    finally:
        release.set()
        thread.join(timeout=30)

    assert mine.closed_on is loop and keyed.closed_on is loop
    assert closed >= 2
    assert other_loops["client"].closed_on is None


def test_a_value_that_will_not_close_does_not_keep_the_others_open():
    from app.utils.per_loop import PerLoop, close_running_loop_values

    async def refuse(value):
        raise RuntimeError("will not close")

    stubborn = PerLoop(object, close=refuse)
    fine = PerLoop(_Client, close=lambda client: client.aclose())

    async def job():
        stubborn.get()
        client = fine.get()
        await close_running_loop_values()
        return client

    assert asyncio.run(job()).closed_on is not None


def test_an_agent_job_closes_its_clients_as_its_last_act():
    from app.tasks.agent_job_tasks import _within_wall_clock
    from app.utils.per_loop import PerLoop

    held = PerLoop(_Client, close=lambda client: client.aclose())
    seen = {}

    async def run():
        seen["client"] = held.get()
        return "done"

    async def failing():
        seen["failed"] = held.get()
        raise ValueError("the job failed")

    assert asyncio.run(_within_wall_clock(run(), 30)) == "done"
    with pytest.raises(ValueError):
        asyncio.run(_within_wall_clock(failing(), 30))

    assert seen["client"].closed_on is not None
    assert seen["failed"].closed_on is not None


def test_the_worker_loads_the_anyio_backend_before_any_job_thread_starts():
    # anyio finds its backend in `sys.modules` without waiting on the import
    # lock; jobs starting together in threads met it half initialised.
    tree = ast.parse((APP / "core" / "celery.py").read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "anyio._backends._asyncio" in imported


# -- a write waits for models that are already loading --------------------------


class _Loading:
    def __init__(self):
        self.finished = False

    def done(self):
        return self.finished


@pytest.mark.asyncio
async def test_a_write_waits_for_a_model_load_already_under_way(monkeypatch):
    from app.services.vector_store import VectorStoreService

    store, started = VectorStoreService(), []

    async def initialize(*args, **kwargs):
        started.append(kwargs)

    monkeypatch.setattr(store, "initialize", initialize)
    store._init_task = _Loading()  # another job's load, on another loop

    def finish():
        store._initialized = True
        store._init_task.finished = True

    timer = threading.Timer(0.4, finish)
    timer.start()
    try:
        await asyncio.wait_for(store._wait_for_models(), timeout=20)
    finally:
        timer.cancel()

    assert store._initialized is True
    assert started == []  # waited for that load; did not start another


@pytest.mark.asyncio
async def test_a_write_loads_the_models_when_nothing_is_loading_them(monkeypatch):
    from app.services.vector_store import VectorStoreService

    store, started = VectorStoreService(), []

    async def initialize(*args, **kwargs):
        started.append(kwargs)

    monkeypatch.setattr(store, "initialize", initialize)

    await store._wait_for_models()
    # A load that ended without the models (its job's loop closed under it)
    # is not waited on for ever either.
    store._init_task = _Loading()
    store._init_task.finished = True
    await store._wait_for_models()

    assert started == [{}, {}]  # in the foreground: no `background=True`


# -- a document's details, whatever the session has loaded ----------------------


@pytest.mark.asyncio
async def test_document_details_do_not_depend_on_what_the_session_had_loaded(
    db_session,
):
    from types import SimpleNamespace
    from uuid import uuid4

    from app.models.document import Document, DocumentSource
    from app.services.agent_tool_providers.base import AgentToolExecutionContext
    from app.services.agent_tool_providers.document import (
        build_autonomous_document_provider,
    )

    source = DocumentSource(id=uuid4(), name="Probe Wiki", source_type="web", config={})
    document = Document(
        id=uuid4(),
        title="Loop vectorisation",
        content="body text",
        content_hash="0" * 64,
        source_id=source.id,
        source_identifier="probe-1",
    )
    db_session.add_all([source, document])
    await db_session.commit()
    document_id = str(document.id)
    # As a job meets it: nothing about this document or its source in hand.
    db_session.expunge_all()

    handler = build_autonomous_document_provider(SimpleNamespace())._handlers[
        "get_document_details"
    ]
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db_session,
        service=None,
        user_id=uuid4(),
        job=SimpleNamespace(id=uuid4(), user_id=uuid4(), config={}, iteration=1),
        state={},
    )

    found = await handler({"document_id": document_id}, ctx)
    missing = await handler({"document_id": str(uuid4())}, ctx)
    malformed = await handler({"document_id": "not-an-id"}, ctx)

    assert found["success"] is True
    assert found["data"]["source"] == "Probe Wiki"  # a name, not a model object
    assert found["data"]["has_content"] is True
    assert missing == {"error": "Document not found"}
    assert "not a document id" in malformed["error"]
