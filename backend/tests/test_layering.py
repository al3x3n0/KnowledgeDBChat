"""Lower layers do not import higher ones.

Read from source, every import included -- the ones inside functions too,
because that is where these hide. 891 function-local imports let 51 modules
(services, tasks, endpoints and ``modules/autonomy``) form one import cycle
that Python never reports: no layer could be extracted or tested alone, and an
import that only runs inside an ``except Exception`` can name nothing for
months.

The rules:

- nothing outside ``app/api`` imports ``app.api``. An endpoint module is
  transport; a service that needs its helper needs that helper moved down;
- the service layer (and models, schemas, core, agent_core, utils, mcp) does
  not import a module that defines Celery tasks. Work is queued through
  ``services/job_dispatch.enqueue``, which names the task by path and sends
  only once the caller's writes are committed. (A helper under ``app/tasks``
  that defines no task, such as ``job_support``, is not the worker.)
- inside ``app/modules/<domain>``, ``application`` and ``domain`` do not
  import ``api`` (``app/modules/README.md``).

``ALLOWED`` started as the 19 violations that existed when the rule was
written, and was emptied in the same branch. It may only shrink: a new entry
is a new violation, and an entry no longer needed fails the test.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

BELOW_TASKS = {"services", "models", "schemas", "core", "agent_core", "utils", "mcp"}

#: The adapter between services and the worker.
TASK_GATEWAY = "services/job_dispatch.py"

#: (importing file, forbidden package) pairs that existed when this was written.
ALLOWED: set[tuple[str, str]] = set()


def _imported_modules(path: Path):
    """Every module ``path`` imports, ``from pkg import mod`` resolved to both."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _task_modules() -> set[str]:
    """Modules under app/tasks that define a Celery task.

    A helper living there (``job_support`` publishes to Redis) is not the
    worker; importing it drags in nothing the rule is about.
    """
    found = set()
    for path in (APP / "tasks").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "@celery_app.task" in text or "@shared_task" in text:
            rel = path.relative_to(APP.parent).with_suffix("").as_posix()
            found.add(rel.replace("/", "."))
    return found


def _forbidden(rel: str) -> list[str]:
    parts = rel.split("/")
    forbidden = []
    if parts[0] != "api" and parts[0] != "modules":
        forbidden.append("app.api")
    if parts[0] in BELOW_TASKS and rel != TASK_GATEWAY:
        forbidden.append("app.tasks")
    if (
        parts[0] == "modules"
        and len(parts) > 2
        and parts[2]
        in {
            "application",
            "domain",
        }
    ):
        forbidden.append("app.api")
        forbidden.append(f"app.modules.{parts[1]}.api")
    return forbidden


def _violates(module: str, package: str, task_modules: set[str]) -> bool:
    if package == "app.tasks":
        return module in task_modules
    return module == package or module.startswith(package + ".")


def _violations() -> set[tuple[str, str]]:
    found = set()
    task_modules = _task_modules()
    for path in APP.rglob("*.py"):
        rel = path.relative_to(APP).as_posix()
        rules = _forbidden(rel)
        if not rules:
            continue
        for module in _imported_modules(path):
            for package in rules:
                if _violates(module, package, task_modules):
                    found.add((rel, package))
    return found


def test_no_new_upward_imports():
    new = _violations() - ALLOWED
    assert not new, (
        "These modules import a layer above them. Move what they need down "
        "(a service for endpoint logic; services/job_dispatch for queueing): "
        + ", ".join(f"{rel} -> {pkg}" for rel, pkg in sorted(new))
    )


def test_the_allowlist_only_shrinks():
    stale = ALLOWED - _violations()
    assert not stale, (
        "These no longer import upward; remove them from ALLOWED so they "
        "cannot regress: " + ", ".join(f"{rel} -> {pkg}" for rel, pkg in sorted(stale))
    )


def test_the_check_sees_imports_inside_functions(tmp_path):
    # The control. A check reading top-level imports alone would pass this
    # file, and every lazy violation with it.
    source = tmp_path / "lazy.py"
    source.write_text(
        "def later():\n"
        "    from app.tasks.agent_job_tasks import execute_agent_job_task\n"
        "    from app.tasks import job_support\n"
    )
    imported = set(_imported_modules(source))
    task_modules = _task_modules()
    assert _violates("app.tasks.agent_job_tasks", "app.tasks", task_modules)
    assert "app.tasks.agent_job_tasks" in imported
    # A helper module under app/tasks that defines no task is not the worker.
    assert "app.tasks.job_support" in imported
    assert not _violates("app.tasks.job_support", "app.tasks", task_modules)


#: The largest import cycle (strongly connected component, function-local
#: imports included) when this was written. It was 51 modules spanning
#: services, tasks, endpoints and modules/autonomy; moving work down a layer
#: and importing provider types from agent_tool_providers.base rather than the
#: agent_tool_dispatch facade (which imports every provider) brought it to 7.
#: The 7 that remain are mutual recursion between peers -- a workflow runs
#: tools and a custom tool can run a workflow -- which needs an interface to
#: break, not a moved import. May only shrink.
LARGEST_CYCLE = 7


def _module_name(path: Path) -> str:
    rel = path.relative_to(APP.parent).with_suffix("").as_posix().replace("/", ".")
    return rel[: -len(".__init__")] if rel.endswith(".__init__") else rel


def _largest_cycle() -> list[str]:
    import sys

    modules = {_module_name(path): path for path in APP.rglob("*.py")}
    edges: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom) and node.module and not node.level
            ):
                continue
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                if target not in modules:
                    target = node.module
                while target and target not in modules:
                    target = target.rpartition(".")[0]
                # A package's __init__ re-exports; importing it is not a
                # dependency on every module inside it.
                if target and target != name and target != "app.services":
                    edges[name].add(target)

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    largest: list[str] = []
    counter = [0]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    def visit(node: str) -> None:
        nonlocal largest
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in edges[node]:
            if nxt not in index:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > len(largest):
                largest = component

    for module in edges:
        if module not in index:
            visit(module)
    return sorted(largest)


def test_no_import_cycle_grows():
    largest = _largest_cycle()
    assert len(largest) <= LARGEST_CYCLE, (
        f"An import cycle of {len(largest)} modules (limit {LARGEST_CYCLE}): "
        + ", ".join(largest)
        + ". Import from the module that owns a name, not from a facade "
        "that re-exports it, and move shared code down a layer."
    )


def test_the_cycle_limit_is_not_slack():
    # Lower LARGEST_CYCLE when a cycle shrinks, so the gain cannot be spent.
    assert len(_largest_cycle()) == LARGEST_CYCLE
