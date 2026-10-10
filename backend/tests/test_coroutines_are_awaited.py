"""A coroutine function is awaited, not used as though it returned its value.

`chunks = text_processor.split_text(content)` assigns a coroutine. Iterating
it raises, and both places that did it sat inside `except Exception`, so the
only sign was a feature that never worked: neither inline arXiv ingest could
chunk a paper. Python says nothing until the line runs, and then says
"'coroutine' object is not iterable" to whoever is catching.

Read from source. The receiver has to be something this codebase defines --
a name bound to one of its classes, or a singleton imported from it -- so a
third-party object that happens to share a method name is not accused.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

#: Handing a coroutine to one of these is how it gets run.
RUNNERS = {
    "create_task",
    "ensure_future",
    "gather",
    "wait_for",
    "shield",
    "spawn",
    "run",
    "run_until_complete",
    "run_async",
    "run_coroutine_threadsafe",
    "to_thread",
}


def _modules():
    return {
        path: ast.parse(path.read_text())
        for path in sorted(APP.rglob("*.py"))
        if "alembic" not in path.parts
    }


def _classes(modules):
    """class name -> {method name: is it async} for every first-party class."""
    found = {}
    for tree in modules.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = found.setdefault(node.name, {})
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef):
                        methods[item.name] = True
                    elif isinstance(item, ast.FunctionDef):
                        methods[item.name] = False
    return found


def _instance_of(value, classes):
    """The first-party class `value` constructs, for `Name()` calls."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in classes
    ):
        return value.func.id
    return None


def _singletons(modules, classes):
    """(module dotted name, variable) -> class, for `name = Class()` at top level."""
    found = {}
    for path, tree in modules.items():
        if APP not in path.parents:
            continue
        dotted = "app." + ".".join(path.relative_to(APP).with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, built = node.targets[0], _instance_of(node.value, classes)
                if isinstance(target, ast.Name) and built:
                    found[(dotted, target.id)] = built
    return found


def _unawaited(modules=None):
    modules = modules or _modules()
    classes = _classes(modules)
    singletons = _singletons(modules, classes)
    found, examined = [], 0

    for path, tree in modules.items():
        parent = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for scope in ast.walk(tree):
            if not isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
            ):
                continue
            # What each plain name is bound to, in this scope or by import.
            bound = {}
            for node in ast.walk(scope):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        built = singletons.get((node.module, alias.name))
                        if built:
                            bound[alias.asname or alias.name] = built
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, built = node.targets[0], _instance_of(node.value, classes)
                    if isinstance(target, ast.Name) and built:
                        bound[target.id] = built
            if not bound:
                continue
            for node in ast.walk(scope):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                ):
                    continue
                cls = bound.get(node.func.value.id)
                if cls is None or classes[cls].get(node.func.attr) is not True:
                    continue
                examined += 1
                above = parent.get(node)
                if isinstance(above, (ast.Await, ast.Return, ast.withitem)):
                    continue
                if isinstance(above, (ast.Starred, ast.keyword, ast.List, ast.Tuple)):
                    continue  # collected or passed on: somebody else awaits it
                if isinstance(above, ast.Call):
                    runner = above.func
                    name = (
                        runner.attr
                        if isinstance(runner, ast.Attribute)
                        else getattr(runner, "id", "")
                    )
                    if name in RUNNERS or node in above.args:
                        continue
                found.append(
                    f"{path.name}:{node.lineno} "
                    f"{node.func.value.id}.{node.func.attr}()"
                )
    return sorted(set(found)), examined


def test_no_coroutine_is_used_as_its_own_result():
    found, examined = _unawaited()

    # A scan that recognises no calls passes for ever.
    assert examined > 200
    assert found == [], f"async method called without await: {found}"


def test_the_scan_sees_the_mistake_it_is_for(tmp_path):
    # The control: the exact shape both ingest routes had, beside the right one.
    source = """
class Splitter:
    async def split_text(self, text):
        return [text]


async def ingest(content):
    splitter = Splitter()
    right = await splitter.split_text(content)
    chunks = splitter.split_text(content)
    return right, [chunk for chunk in chunks]
"""
    path = tmp_path / "ingest.py"

    found, examined = _unawaited({path: ast.parse(source)})

    assert examined >= 2
    assert found == ["ingest.py:10 splitter.split_text()"]
