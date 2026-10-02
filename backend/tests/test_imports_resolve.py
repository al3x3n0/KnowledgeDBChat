"""Every `from app.x import name` names something `app.x` defines.

Most imports in this codebase are made inside functions, to avoid cycles, and
most of those functions catch `Exception`. Together that makes a wrong name
invisible: the module loads, the suite passes, and the feature answers with
its fallback for ever. Four were found that way -- synthesis output files
(three builder singletons that never existed), bulk summarisation (a task
under a name it never had), and the GitLab architecture tool twice over (a
model module that does not exist, and a service that could not be imported at
all, taking four /git routes with it).

Checked from source rather than by importing, so a missing optional
dependency cannot hide a module from the check.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

BACKEND = Path(__file__).resolve().parents[1]
APP = BACKEND / "app"


def _module_path(dotted: str):
    base = BACKEND.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _bound_names(tree: ast.Module):
    """Names a module binds at import time, wherever in the top-level flow
    (inside `try`, `if` and `with` too). None when it cannot be known."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__":
                return None
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def test_every_app_import_names_something_that_exists():
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(APP.rglob("*.py"))
    }
    bound = {path: _bound_names(tree) for path, tree in trees.items()}

    missing = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.level
                or not (node.module or "").startswith("app")
            ):
                continue
            where = f"{path.relative_to(BACKEND)}:{node.lineno}"
            target = _module_path(node.module)
            if target is None:
                missing.append(f"{where} imports {node.module}, which does not exist")
                continue
            names = bound.get(target)
            if names is None:
                continue
            for alias in node.names:
                if alias.name in names or _module_path(f"{node.module}.{alias.name}"):
                    continue
                missing.append(f"{where}: {node.module} has no {alias.name!r}")

    assert not missing, "\n".join(missing)


# --- calls on a service object -------------------------------------------
#
# The same silence one step further on: the import resolves, and the method
# called on what was imported does not exist. Agent delegation called
# `LLMService.generate_chat_response` and job finalisation called
# `DataSandboxManager.cleanup`; neither method was ever defined, both calls
# sat inside `except Exception`, so delegation always answered "failed" and no
# data sandbox was ever released at the end of a job.


def _classes(trees):
    found = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.setdefault(node.name, []).append(node)
    return {name: nodes[0] for name, nodes in found.items() if len(nodes) == 1}


def _class_attributes(node, classes, seen=()):
    """What an instance is known to have, or None when it cannot be known
    (a base defined elsewhere, or attribute access the class intercepts)."""
    names = set()
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name in ("__getattr__", "__getattribute__"):
                return None
            names.add(item.name)
        elif (
            isinstance(item, ast.Attribute)
            and isinstance(item.ctx, ast.Store)
            and isinstance(item.value, ast.Name)
            and item.value.id in ("self", "cls")
        ):
            names.add(item.attr)
        elif isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            names.add(item.id)
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else None
        if name not in classes or name in seen:
            return None
        inherited = _class_attributes(classes[name], classes, seen + (name,))
        if inherited is None:
            return None
        names |= inherited
    return names


def _instantiated(value, classes):
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in classes
    ):
        return value.func.id
    return None


def test_a_method_called_on_a_service_exists():
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(APP.rglob("*.py"))
    }
    classes = _classes(trees)

    # `thing_service = ThingService()` at module level.
    singletons = {}
    for path, tree in trees.items():
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _instantiated(node.value, classes)
            ):
                singletons[(path, node.targets[0].id)] = _instantiated(
                    node.value, classes
                )

    checked, missing = 0, []

    def check(where, spelled, class_name, attribute):
        nonlocal checked
        known = _class_attributes(classes[class_name], classes)
        if known is None or attribute.startswith("__"):
            return
        checked += 1
        if attribute not in known:
            missing.append(f"{where} {spelled}: {class_name} has no {attribute!r}")

    for path, tree in trees.items():
        relative = path.relative_to(BACKEND)

        # Imported singletons, unless the module rebinds or shadows the name.
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                source = _module_path(node.module)
                for alias in node.names:
                    if (source, alias.name) in singletons:
                        imported[alias.asname or alias.name] = singletons[
                            (source, alias.name)
                        ]
        rebound = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } | {
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            for argument in node.args.args + node.args.kwonlyargs
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in imported
                and node.value.id not in rebound
            ):
                check(
                    f"{relative}:{node.lineno}",
                    f"{node.value.id}.{node.attr}",
                    imported[node.value.id],
                    node.attr,
                )

        # `self.thing = Thing()` assigned exactly once, then `self.thing.x`.
        for owner in ast.walk(tree):
            if not isinstance(owner, ast.ClassDef):
                continue
            fields = {}
            for node in ast.walk(owner):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                ):
                    fields.setdefault(node.targets[0].attr, []).append(
                        _instantiated(node.value, classes)
                    )
            for node in ast.walk(owner):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                ):
                    held = fields.get(node.value.attr)
                    if held and len(held) == 1 and held[0]:
                        check(
                            f"{relative}:{node.lineno}",
                            f"self.{node.value.attr}.{node.attr}",
                            held[0],
                            node.attr,
                        )

    assert not missing, "\n".join(sorted(set(missing)))
    # A check that silently examines nothing passes for ever.
    assert checked > 500, checked
