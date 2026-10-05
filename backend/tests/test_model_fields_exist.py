"""A model is built and queried with fields it has.

SQLAlchemy checks a constructor keyword against the class, and every mapped
class has a `metadata` attribute -- the table registry. So a model whose JSON
column is *named* "metadata" in the database but mapped as `extra_metadata` or
`proposal_metadata` accepts `Model(metadata={...})` without complaint, sets a
plain instance attribute, and writes NULL. That is how every code patch
proposal lost what it recorded about itself (goal, scope, files touched, tests
to run) and every document chunk lost its section title.

A keyword the class does not have at all raises -- but `format_as_report`
built its `Document` inside `except Exception`, so `user_id=` there meant only
that a persisted report never was.

Read from source, like tests/test_imports_resolve.py.
"""

import ast
from pathlib import Path

import pytest

from app.models.code_patch_proposal import CodePatchProposal
from app.models.document import DocumentChunk

pytestmark = pytest.mark.unit

BACKEND = Path(__file__).resolve().parents[1]
APP = BACKEND / "app"


def _declared(node):
    names = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
        elif isinstance(item, ast.Assign):
            names |= {t.id for t in item.targets if isinstance(t, ast.Name)}
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
    return names


def _models(trees):
    """Mapped class name -> the attributes it declares, mixins and backrefs
    included. A model with a base this cannot read is left out."""
    by_name = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                by_name.setdefault(node.name, []).append((path, node))
    unique = {name: found[0] for name, found in by_name.items() if len(found) == 1}

    models = {}
    for name, (path, node) in unique.items():
        if "models" not in path.parts or "__tablename__" not in _declared(node):
            continue
        attributes, readable = _declared(node), True
        for base in node.bases:
            base_name = getattr(base, "id", None)
            if base_name == "Base":
                continue
            if base_name in unique:
                attributes |= _declared(unique[base_name][1])
            else:
                readable = False
        if readable:
            models[name] = attributes

    for tree in trees.values():
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", "") == "relationship"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in models
            ):
                continue
            for keyword in call.keywords:
                if keyword.arg != "backref":
                    continue
                value = keyword.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                if isinstance(value, ast.Constant):
                    models[call.args[0].value].add(value.value)
    return models


#: What every mapped class has without declaring it.
INHERITED = {"__table__", "__tablename__", "__mapper__", "__name__", "registry"}


def test_models_are_used_with_fields_they_have():
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(APP.rglob("*.py"))
    }
    models = _models(trees)

    checked, wrong = 0, []
    for path, tree in trees.items():
        relative = path.relative_to(BACKEND)
        # Local name -> model. An alias counts: three media tools filtered on
        # `DocModel.user_id`, a column `Document` does not have, and the check
        # walked past them because it only knew the model by its own name.
        named = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "app.models"
            ):
                for alias in node.names:
                    if alias.name in models:
                        named[alias.asname or alias.name] = alias.name
            if isinstance(node, ast.ClassDef) and node.name in models:
                named[node.name] = node.name
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in named
            ):
                checked += 1
                # `Model.metadata` is the table registry, which is real.
                known = models[named[node.value.id]] | INHERITED | {"metadata"}
                if node.attr not in known:
                    wrong.append(
                        f"{relative}:{node.lineno} {node.value.id}.{node.attr}"
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in named
            ):
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    checked += 1
                    if keyword.arg not in models[named[node.func.id]]:
                        wrong.append(
                            f"{relative}:{node.lineno} {node.func.id}({keyword.arg}=...)"
                        )

    assert not wrong, "\n".join(sorted(set(wrong)))
    assert len(models) > 80 and checked > 3000, (len(models), checked)


def test_metadata_is_stored_under_the_mapped_name():
    """The trap itself, so the reason for the check above stays demonstrable."""
    assert DocumentChunk(content="x", metadata={"a": 1}).extra_metadata is None
    assert DocumentChunk(content="x", extra_metadata={"a": 1}).extra_metadata == {
        "a": 1
    }
    assert CodePatchProposal(
        title="t", proposal_metadata={"a": 1}
    ).proposal_metadata == {"a": 1}
