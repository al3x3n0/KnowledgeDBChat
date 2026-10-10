"""A call on an object built a line earlier fits the class it was built from.

`tests/test_imports_resolve.py` and `tests/test_calls_fit_signatures.py` check
methods called on imported singletons and on services a class keeps as
`self.x = X()`. An object built inside the function slipped past both:

    engine = WorkflowEngine()                  # it takes the session and user
    await engine.execute(workflow_id=...)      # it has no `execute`

    storage = StorageService()
    await storage.upload_file(..., prefix="templates")   # no such parameter

Both sat inside `except Exception`. The chat tool that runs a workflow, and
the upload of a PPTX template, had each failed every time they were used.

Read from source, and deliberately narrow: only a name assigned exactly once
in its function, to a class this codebase defines once and whose bases it also
defines, so nothing is guessed about rebinding or third-party inheritance.
"""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import UUID, uuid4

import pytest

APP = Path(__file__).resolve().parents[1] / "app"


def _modules():
    return {
        path: ast.parse(path.read_text())
        for path in sorted(APP.rglob("*.py"))
        if "alembic" not in path.parts
    }


def _plain_classes(modules):
    """Classes whose whole ancestry is defined here, once each, by name.

    A class with a base this codebase does not define could inherit anything,
    so it is left out; one whose bases are all first-party is resolved, since
    that is how `StorageService` gets `upload_file`.
    """
    seen = {}
    for tree in modules.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                seen.setdefault(node.name, []).append(node)
    unique = {name: nodes[0] for name, nodes in seen.items() if len(nodes) == 1}

    def resolvable(cls, depth=0):
        if cls.keywords or depth > 6:
            return False
        for base in cls.bases:
            if not (isinstance(base, ast.Name) and base.id in unique):
                return False
            if not resolvable(unique[base.id], depth + 1):
                return False
        return True

    _plain_classes.unique = unique
    return {name: cls for name, cls in unique.items() if resolvable(cls)}


def _ancestry(cls):
    """The class and its first-party bases, nearest first."""
    unique, order, pending = _plain_classes.unique, [], [cls]
    while pending:
        current = pending.pop(0)
        if current not in order:
            order.append(current)
            pending.extend(unique[base.id] for base in current.bases)
    return order


def _methods(cls):
    found = {}
    for owner in reversed(_ancestry(cls)):
        for item in owner.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[item.name] = item
    return found


def _own_attributes(cls):
    """Names the class gives its instances other than methods."""
    names = set()
    for owner in _ancestry(cls):
        names |= _own_attributes_of(owner)
    return names


def _own_attributes_of(cls):
    names = set()
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node in cls.body:
            for target in getattr(node, "targets", [getattr(node, "target", None)]):
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _misfit(call, function, *, bound: bool):
    """Why `call` cannot be a call of `function`, or None."""
    decorators = {
        getattr(d, "id", getattr(d, "attr", "")) for d in function.decorator_list
    }
    if decorators - {"staticmethod", "classmethod"}:
        return None  # wrapped: its real signature is the wrapper's
    params = list(function.args.posonlyargs) + list(function.args.args)
    if bound and "staticmethod" not in decorators:
        params = params[1:]
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return None  # *args or **kwargs at the call: cannot be read
    names = {p.arg for p in params} | {p.arg for p in function.args.kwonlyargs}
    if function.args.kwarg is None:
        unknown = [k.arg for k in call.keywords if k.arg not in names]
        if unknown:
            return f"no parameter named {', '.join(unknown)}"
    if function.args.vararg is None and len(call.args) > len(params):
        return f"takes {len(params)} positional, given {len(call.args)}"
    required = [p.arg for p in params[: len(params) - len(function.args.defaults)]]
    required += [
        p.arg
        for p, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if default is None
    ]
    given = {k.arg for k in call.keywords} | {p.arg for p in params[: len(call.args)]}
    missing = [name for name in required if name not in given]
    if missing:
        return f"missing {', '.join(missing)}"
    return None


def _problems(modules=None):
    modules = modules or _modules()
    classes = _plain_classes(modules)
    found, examined = [], 0
    for path, tree in modules.items():
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # The body only. A decorator belongs to the scope outside:
            # `@router.post(...)` above a function that later says
            # `router = AgentRouter(...)` is the module's router.
            inside = [node for statement in scope.body for node in ast.walk(statement)]
            # Every plain name this function binds, and how often.
            bindings = {}
            for node in inside:
                targets = []
                if isinstance(node, ast.Assign):
                    targets = [(t, node.value) for t in node.targets]
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                    targets = [(node.target, getattr(node, "value", None))]
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    targets = [(node.target, None)]
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    targets = [(i.optional_vars, None) for i in node.items]
                for target, value in targets:
                    for name in ast.walk(target) if target is not None else []:
                        if isinstance(name, ast.Name):
                            bindings.setdefault(name.id, []).append(value)
            for argument in ast.walk(scope.args):
                if isinstance(argument, ast.arg):
                    bindings.setdefault(argument.arg, []).append(None)

            for name, values in bindings.items():
                if len(values) != 1:
                    continue
                built = values[0]
                if not (
                    isinstance(built, ast.Call)
                    and isinstance(built.func, ast.Name)
                    and built.func.id in classes
                ):
                    continue
                cls = classes[built.func.id]
                methods = _methods(cls)
                where = f"{path.name}:{built.lineno}"
                if "__init__" in methods:
                    examined += 1
                    reason = _misfit(built, methods["__init__"], bound=True)
                    if reason:
                        found.append(f"{where} {cls.name}(): {reason}")
                attributes = _own_attributes(cls)
                for node in inside:
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == name
                    ):
                        continue
                    examined += 1
                    method = node.func.attr
                    here = f"{path.name}:{node.lineno} {cls.name}.{method}"
                    if method not in methods:
                        if method not in attributes:
                            found.append(f"{here}: no such method")
                        continue
                    reason = _misfit(node, methods[method], bound=True)
                    if reason:
                        found.append(f"{here}: {reason}")
    return sorted(set(found)), examined


def test_calls_on_locally_built_objects_fit_their_class():
    found, examined = _problems()

    assert examined > 200  # a scan that recognises nothing passes for ever
    assert found == [], "\n".join(found)


def test_the_scan_sees_both_mistakes_it_is_for(tmp_path):
    source = """
class Engine:
    def __init__(self, db, user):
        self.db, self.user = db, user

    async def queue_workflow(self, workflow_id, trigger_type="manual"):
        return workflow_id


class Storage:
    async def upload_file(self, document_id, filename, content):
        return filename


async def run(db, user, workflow):
    engine = Engine()
    return await engine.execute(workflow_id=workflow)


async def upload(content):
    storage = Storage()
    return await storage.upload_file("id", "a.pptx", content, prefix="templates")


async def fine(db, user, workflow):
    engine = Engine(db, user)
    return await engine.queue_workflow(workflow, trigger_type="agent")
"""
    found, _ = _problems({tmp_path / "sample.py": ast.parse(source)})

    assert found == [
        "sample.py:16 Engine(): missing db, user",
        "sample.py:17 Engine.execute: no such method",
        "sample.py:22 Storage.upload_file: no parameter named prefix",
    ]


# --- the tool this found ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_chat_tool_queues_a_workflow(db_session, test_user, monkeypatch):
    from app.models.workflow import Workflow, WorkflowExecution, WorkflowNode
    from app.services.agent_service import AgentService
    from app.tasks import workflow_tasks

    queued = []
    monkeypatch.setattr(
        workflow_tasks.execute_workflow_task,
        "delay",
        lambda execution_id, *args, **kwargs: queued.append(execution_id),
    )
    workflow = Workflow(id=uuid4(), user_id=test_user.id, name="Nightly digest")
    db_session.add(workflow)
    db_session.add(
        WorkflowNode(workflow_id=workflow.id, node_id="start", node_type="start")
    )
    await db_session.commit()

    answer = await AgentService.__new__(AgentService)._tool_run_workflow(
        {"workflow_name": "nightly DIGEST", "inputs": {"topic": "ci"}},
        test_user.id,
        db_session,
    )

    execution = await db_session.get(WorkflowExecution, UUID(answer["execution_id"]))
    assert answer["status"] == "queued" and "error" not in answer
    assert (execution.status, execution.trigger_type) == ("pending", "agent")
    assert execution.context == {"topic": "ci"}
    assert queued == [str(execution.id)]


@pytest.mark.asyncio
async def test_a_workflow_that_cannot_start_says_why(db_session, test_user):
    from app.models.workflow import Workflow
    from app.services.agent_service import AgentService

    workflow = Workflow(id=uuid4(), user_id=test_user.id, name="No start node")
    db_session.add(workflow)
    await db_session.commit()

    answer = await AgentService.__new__(AgentService)._tool_run_workflow(
        {"workflow_id": str(workflow.id)}, test_user.id, db_session
    )

    assert "exactly one start node" in answer["error"]
