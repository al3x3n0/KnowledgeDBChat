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
  not import ``app.tasks``. Work is queued through ``services/job_dispatch``,
  the one module allowed to, which also sends only after the commit;
- inside ``app/modules/<domain>``, ``application`` and ``domain`` do not
  import ``api`` (``app/modules/README.md``).

``ALLOWED`` is what was already true when the rule was written. It may only
shrink: a new entry is a new violation, and an entry no longer needed fails
the test until it is removed, so progress is recorded where it happens.
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
ALLOWED = {
    # Composes the operator queue from about twenty endpoint-private pieces.
    ("tasks/monitoring_tasks.py", "app.api"),
    ("mcp/tools/generation.py", "app.tasks"),
    ("services/agent_ingestion_demo_runner_service.py", "app.tasks"),
    ("services/agent_latex_runner_service.py", "app.tasks"),
    ("services/agent_service.py", "app.tasks"),
    ("services/agent_tool_dispatch.py", "app.tasks"),
    ("services/chat_service.py", "app.tasks"),
    ("services/document_service.py", "app.tasks"),
    ("services/notification_service.py", "app.tasks"),
    ("services/training_service.py", "app.tasks"),
    ("services/workflow_engine.py", "app.tasks"),
}


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


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


def _violations() -> set[tuple[str, str]]:
    found = set()
    for path in APP.rglob("*.py"):
        rel = path.relative_to(APP).as_posix()
        rules = _forbidden(rel)
        if not rules:
            continue
        for module in _imported_modules(path):
            for package in rules:
                if module == package or module.startswith(package + "."):
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


def test_the_check_sees_imports_inside_functions():
    # The control: job_dispatch imports the task module only inside a
    # function. A check reading top-level imports alone would miss it, and
    # every lazy violation with it.
    assert "app.tasks.agent_job_tasks" in set(
        _imported_modules(APP / "services" / "job_dispatch.py")
    )
