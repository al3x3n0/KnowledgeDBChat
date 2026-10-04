"""Every Celery task is one the worker knows.

The worker imports only the modules in `celery_app`'s `include` list. A task
in any other module is registered in the API process, which calls `.delay()`
happily, and unknown to the worker, which discards the message as an
unregistered task. Nothing errors where anyone looks: the job row just stays
pending. `workflow_tasks` (since January), `export_tasks` and
`synthesis_tasks` (since February) were missing this way, so no queued
workflow, export or synthesis job ever ran in a worker.

Checked from source for the modules and against the real app for the names,
so a task defined anywhere under app/tasks is covered.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TASKS = Path(__file__).resolve().parents[1] / "app" / "tasks"


def _modules_defining_tasks():
    found = {}
    for path in sorted(TASKS.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                text = ast.unparse(decorator)
                if re.search(r"\.task\b", text):
                    explicit = re.search(r"name=['\"]([^'\"]+)", text)
                    names.append(
                        explicit.group(1)
                        if explicit
                        else f"app.tasks.{path.stem}.{node.name}"
                    )
        if names:
            found[f"app.tasks.{path.stem}"] = names
    return found


def test_the_scan_finds_task_modules():
    assert len(_modules_defining_tasks()) > 20


def test_every_task_module_is_registered():
    from app.core.celery import celery_app

    included = set(celery_app.conf.include or [])
    missing = sorted(set(_modules_defining_tasks()) - included)
    assert not missing, (
        "These modules define tasks the worker never imports, so every "
        "message sent to them is dropped:\n" + "\n".join(missing)
    )


def test_the_worker_registers_every_task_by_its_name():
    from app.core.celery import celery_app

    celery_app.loader.import_default_modules()
    unknown = sorted(
        name
        for names in _modules_defining_tasks().values()
        for name in names
        if name not in celery_app.tasks
    )
    assert not unknown, "Unregistered tasks:\n" + "\n".join(unknown)
