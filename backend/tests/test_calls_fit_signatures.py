"""A call passes arguments its callee can take.

Python checks this when the call happens, which is too late for two kinds of
call: one inside `except Exception`, and one made with `.delay()`, where the
TypeError is raised in a worker and the caller has already answered 200.
`POST /agent/agents` passed a keyword its validator does not have, so no agent
could be created through the API; the research presentation route queued its
task without `user_id`, so the job it had just created stayed pending for ever.

Only calls whose target is certain are examined: a module-level function
called by its own or its imported name, `module.function`, `self.method`
within a class whose bases are all known, and `task.delay`. A name that is
rebound, shadowed or decorated is skipped rather than guessed at.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

BACKEND = Path(__file__).resolve().parents[1]
APP = BACKEND / "app"
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_path(dotted):
    base = BACKEND.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _top_functions(tree):
    found = {}
    for node in tree.body:
        if isinstance(node, FUNCTIONS):
            found.setdefault(node.name, []).append(node)
    return {name: nodes[0] for name, nodes in found.items() if len(nodes) == 1}


def _decorators(function):
    return [ast.unparse(d) for d in function.decorator_list]


def _is_task(function):
    return any("task" in d for d in _decorators(function))


def _misfit(call, function, *, bound=False, delayed=False):
    """Why `call` cannot be made against `function`, or None."""
    decorators = _decorators(function)
    if any(
        "task" not in d and d not in ("staticmethod", "classmethod") for d in decorators
    ):
        return None  # a decorator may change the signature
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return None  # *args / **kwargs at the call site

    spec = function.args
    everything = spec.posonlyargs + spec.args
    skip = 0
    if bound and "staticmethod" not in decorators:
        skip = 1
    if delayed and any("bind=True" in d for d in decorators):
        skip = 1
    positional = [a.arg for a in everything][skip:]
    keyword_only = [a.arg for a in spec.kwonlyargs]

    if spec.vararg is None and len(call.args) > len(positional):
        return f"{len(call.args)} positional arguments, takes {len(positional)}"
    given = set(positional[: len(call.args)])
    for keyword in call.keywords:
        if keyword.arg in given:
            return f"{keyword.arg!r} given twice"
        if (
            keyword.arg not in positional
            and keyword.arg not in keyword_only
            and spec.kwarg is None
        ):
            return f"unknown keyword {keyword.arg!r}"
        given.add(keyword.arg)

    required = [a.arg for a in everything[: len(everything) - len(spec.defaults)]]
    required = required[skip:] + [
        a.arg for a, d in zip(spec.kwonlyargs, spec.kw_defaults) if d is None
    ]
    missing = [name for name in required if name not in given]
    return f"missing {missing}" if missing else None


def _methods(name, classes, seen=()):
    """Every method an instance has, or None when a base is not ours to read."""
    node = classes[name]
    found = {}
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else None
        if base_name not in classes or base_name in seen:
            return None
        inherited = _methods(base_name, classes, seen + (base_name,))
        if inherited is None:
            return None
        found.update(inherited)
    for item in node.body:
        if isinstance(item, FUNCTIONS):
            if item.name == "__getattr__":
                return None
            found[item.name] = item
    return found


def test_calls_pass_arguments_their_callee_accepts():
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(APP.rglob("*.py"))
    }
    top = {path: _top_functions(tree) for path, tree in trees.items()}
    by_name = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                by_name.setdefault(node.name, []).append(node)
    classes = {name: nodes[0] for name, nodes in by_name.items() if len(nodes) == 1}

    checked, misfits = 0, []

    for path, tree in trees.items():
        relative = path.relative_to(BACKEND)
        functions, modules = dict(top[path]), {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and (node.module or "").startswith("app")
            ):
                source = _module_path(node.module)
                for alias in node.names:
                    submodule = _module_path(f"{node.module}.{alias.name}")
                    if submodule is not None and submodule.name != "__init__.py":
                        modules[alias.asname or alias.name] = submodule
                    elif source in top and alias.name in top[source]:
                        functions[alias.asname or alias.name] = top[source][alias.name]

        # Names this module rebinds, takes as a parameter, or defines again
        # inside something: any of those may not be the function we found.
        shadowed = {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for outer in ast.walk(tree):
            if isinstance(outer, FUNCTIONS + (ast.Lambda,)):
                shadowed |= {a.arg for a in outer.args.args + outer.args.kwonlyargs}
            if isinstance(outer, FUNCTIONS + (ast.ClassDef,)):
                shadowed |= {
                    inner.name
                    for inner in ast.walk(outer)
                    if inner is not outer
                    and isinstance(inner, FUNCTIONS + (ast.ClassDef,))
                }

        for owner in ast.walk(tree):
            if (
                not isinstance(owner, ast.ClassDef)
                or classes.get(owner.name) is not owner
            ):
                continue
            methods = _methods(owner.name, classes)
            if methods is None:
                continue
            assigned = {
                n.attr
                for n in ast.walk(owner)
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
            }
            for call in ast.walk(owner):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                    and call.func.attr in methods
                    and call.func.attr not in assigned
                ):
                    checked += 1
                    problem = _misfit(call, methods[call.func.attr], bound=True)
                    if problem:
                        misfits.append(
                            f"{relative}:{call.lineno} self.{call.func.attr}(): {problem}"
                        )

        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            target, delayed = call.func, False
            if isinstance(target, ast.Attribute) and target.attr == "delay":
                target, delayed = target.value, True
            function = spelled = None
            if isinstance(target, ast.Name) and target.id not in shadowed:
                function, spelled = functions.get(target.id), target.id
            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in modules
                and target.value.id not in shadowed
            ):
                function = top[modules[target.value.id]].get(target.attr)
                spelled = f"{target.value.id}.{target.attr}"
            if function is None or _is_task(function) != delayed:
                continue
            checked += 1
            problem = _misfit(call, function, delayed=delayed)
            if problem:
                suffix = ".delay" if delayed else ""
                misfits.append(
                    f"{relative}:{call.lineno} {spelled}{suffix}(): {problem}"
                )

    assert not misfits, "\n".join(sorted(set(misfits)))
    # A check that examines nothing passes for ever.
    assert checked > 4000, checked


def test_calls_on_held_services_pass_arguments_they_accept():
    """`self.storage.upload_file(...)` and `storage_service.delete_file(...)`.

    The check above stops at `self.method()`, so a method called on a service
    an object holds, or on an imported singleton, was never compared with its
    signature. That is how every export failed for eight months: the export
    service called `self.storage.upload_file(file_bytes, file_path, ...)`
    against `upload_file(document_id, filename, content, ...)`, inside an
    `except Exception` that recorded it as the export's own failure.

    Certain targets only: an attribute assigned once in its class, to a call
    of a class defined once in app/; a module-level `name = Class()` imported
    by name and not rebound in the caller.
    """
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(APP.rglob("*.py"))
    }
    by_name = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                by_name.setdefault(node.name, []).append(node)
    classes = {name: nodes[0] for name, nodes in by_name.items() if len(nodes) == 1}

    def instance_of(value):
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in classes
        ):
            return value.func.id
        return None

    singletons = {}  # path -> {name: class}
    for path, tree in trees.items():
        found, rebound = {}, set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    if target.id in found:
                        rebound.add(target.id)
                    cls = instance_of(node.value)
                    if cls:
                        found[target.id] = cls
        singletons[path] = {k: v for k, v in found.items() if k not in rebound}

    checked, misfits = 0, []

    def check(call, cls, spelled, relative):
        nonlocal checked
        methods = _methods(cls, classes)
        if not methods or call.func.attr not in methods:
            return
        checked += 1
        problem = _misfit(call, methods[call.func.attr], bound=True)
        if problem:
            misfits.append(f"{relative}:{call.lineno} {spelled}(): {problem}")

    for path, tree in trees.items():
        relative = path.relative_to(BACKEND)
        shadowed = {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "app"
            ):
                source = _module_path(node.module)
                for alias in node.names:
                    cls = singletons.get(source, {}).get(alias.name)
                    if cls:
                        imported[alias.asname or alias.name] = cls
        imported.update(singletons.get(path, {}))

        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in imported
                and (
                    call.func.value.id not in shadowed
                    or call.func.value.id in singletons.get(path, {})
                )
            ):
                name = call.func.value.id
                check(call, imported[name], f"{name}.{call.func.attr}", relative)

        for owner in ast.walk(tree):
            if not isinstance(owner, ast.ClassDef):
                continue
            held, counts = {}, {}
            for node in ast.walk(owner):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            counts[target.attr] = counts.get(target.attr, 0) + 1
                            cls = instance_of(node.value)
                            if (
                                cls is None
                                and isinstance(node.value, ast.Name)
                                and node.value.id in imported
                            ):
                                # `self.storage = storage_service`: a held
                                # singleton, which is how the dataset export's
                                # upload_file call went unchecked.
                                cls = imported[node.value.id]
                            if cls:
                                held[target.attr] = cls
            held = {k: v for k, v in held.items() if counts.get(k) == 1}
            for call in ast.walk(owner):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Attribute)
                    and isinstance(call.func.value.value, ast.Name)
                    and call.func.value.value.id == "self"
                    and call.func.value.attr in held
                ):
                    attr = call.func.value.attr
                    check(call, held[attr], f"self.{attr}.{call.func.attr}", relative)

    assert checked > 200, f"only {checked} calls examined; the scan found too little"
    assert not misfits, "\n".join(sorted(set(misfits)))
