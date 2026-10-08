"""No new ``except Exception: pass``.

A handler that catches everything and does nothing is how a broken call goes
unnoticed for months here: an import that named nothing, a keyword a callee
did not take, a timestamp comparison that raised on every row -- each sat
under one of these. Many of the existing ones are deliberate (a progress
message, a cache write), but none says so in a way a reader or a log can see.

This does not ask for them all to be rewritten. It stops the number growing:
a new best-effort step should catch what it expects, or log what it ignored
(``logger.debug`` is enough). Lower ``LIMIT`` when you remove some.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

#: Handlers that catch Exception (or everything) and whose whole body is
#: ``pass`` or ``continue``. May only shrink.
LIMIT = 227


def _silent_swallows(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name)
            and node.type.id in ("Exception", "BaseException")
        )
        if (
            broad
            and len(node.body) == 1
            and isinstance(node.body[0], (ast.Pass, ast.Continue))
        ):
            yield node.lineno


def _count() -> int:
    return sum(
        len(list(_silent_swallows(ast.parse(path.read_text(encoding="utf-8")))))
        for path in APP.rglob("*.py")
    )


def test_no_new_silent_swallow():
    count = _count()
    assert count <= LIMIT, (
        f"{count} handlers catch Exception and do nothing (limit {LIMIT}). "
        "Catch the error you expect, or log what you ignore."
    )


def test_the_limit_is_not_slack():
    assert _count() == LIMIT, "lower LIMIT to the current count"


def test_the_check_sees_what_it_names():
    source = """
try:
    risky()
except Exception:
    pass
for item in items:
    try:
        risky(item)
    except Exception:
        continue
try:
    risky()
except ValueError:
    pass
try:
    risky()
except Exception as exc:
    logger.debug(f"ignored: {exc}")
"""
    assert len(list(_silent_swallows(ast.parse(source)))) == 2
