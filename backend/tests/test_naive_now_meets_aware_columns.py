"""A naive "now" is not compared with a value read from an aware column.

``datetime.utcnow()`` carries no timezone; a ``DateTime(timezone=True)`` column
comes back from Postgres with one. Comparing or subtracting the two raises
TypeError -- and SQLite, which the suite runs on, returns naive values for
every column, so no ordinary test can see it. That is how an auto-sync source
stopped being scanned after its first sync. Two more were found by this scan:
the decision-trace overdue count (a 500 for any listing holding a due item)
and the validation backoff cooldown.

Read from source. The fix is ``utils.datetimes``: ``is_past``, ``age``,
``as_aware_utc``. A comparison built for SQL (``Model.col < cutoff`` inside
``where``) is not counted: the driver, not Python, compares those.
"""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"
SQL_BUILDERS = {"where", "filter", "having", "and_", "or_", "join", "outerjoin"}


def _aware_column_names() -> set:
    names = defaultdict(set)
    for path in (APP / "models").glob("*.py"):
        for cls in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(cls, ast.ClassDef):
                continue
            for statement in cls.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target, value = statement.targets[0], statement.value
                elif isinstance(statement, ast.AnnAssign) and statement.value:
                    target, value = statement.target, statement.value
                else:
                    continue
                if not isinstance(target, ast.Name):
                    continue
                source = ast.unparse(value)
                if "DateTime" in source and "timezone=True" in source:
                    names[target.id].add(cls.name)
    return set(names)


def _is_utcnow(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == "utcnow") or (
        isinstance(func, ast.Name) and func.id == "utcnow"
    )


def _find(source: str, aware: set) -> list:
    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def in_sql(node) -> bool:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in SQL_BUILDERS:
                    return True
        return False

    found = []
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in functions:
        now_names, column_names = set(), {}

        def column(node):
            if isinstance(node, ast.Attribute) and node.attr in aware:
                base = node.value
                if isinstance(base, ast.Name) and base.id[:1].isupper():
                    return None  # Model.col: a SQL expression
                return node.attr
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in aware
            ):
                return node.args[1].value
            if isinstance(node, ast.Name):
                return column_names.get(node.id)
            if isinstance(node, ast.BoolOp):
                return next(filter(None, map(column, node.values)), None)
            return None

        def naive_now(node) -> bool:
            return any(
                _is_utcnow(n) or (isinstance(n, ast.Name) and n.id in now_names)
                for n in ast.walk(node)
            )

        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            name, value = node.targets[0].id, node.value
            if _is_utcnow(value) or (
                isinstance(value, ast.BinOp)
                and (
                    _is_utcnow(value.left)
                    or (isinstance(value.left, ast.Name) and value.left.id in now_names)
                )
            ):
                now_names.add(name)
            elif column(value):
                column_names[name] = column(value)

        for node in ast.walk(function):
            if isinstance(node, ast.Compare) and len(node.comparators) == 1:
                if not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    continue
                sides = (node.left, node.comparators[0])
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                sides = (node.left, node.right)
            else:
                continue
            for one, other in (sides, sides[::-1]):
                if column(one) and naive_now(other) and not in_sql(node):
                    found.append((node.lineno, ast.unparse(node)[:100]))
                    break
    return found


def test_the_scan_sees_each_shape():
    aware = {"due_at"}
    cases = [
        "def f(item):\n    return item.due_at <= datetime.utcnow()\n",
        "def f(item):\n    now = datetime.utcnow()\n    return (now - item.due_at).days\n",
        "def f(item):\n    due = getattr(item, 'due_at', None)\n"
        "    cutoff = datetime.utcnow() - timedelta(days=1)\n    return due < cutoff\n",
        "def f(item):\n    when = item.due_at or item.other\n"
        "    return datetime.utcnow() - when\n",
    ]
    for case in cases:
        assert _find(case, aware), case


def test_the_scan_leaves_sql_and_aware_code_alone():
    aware = {"due_at"}
    cases = [
        "def f(db):\n    cutoff = datetime.utcnow()\n"
        "    return select(Item).where(Item.due_at < cutoff)\n",
        "def f(db, model):\n    cutoff = datetime.utcnow()\n"
        "    return select(model).where(model.due_at < cutoff)\n",
        "def f(item):\n    return is_past(item.due_at)\n",
        "def f(item):\n    return item.due_at <= utc_now()\n",
    ]
    for case in cases:
        assert not _find(case, aware), case


def test_no_naive_now_is_compared_with_an_aware_column():
    aware = _aware_column_names()
    assert len(aware) > 40, "the model scan found too few aware columns to trust"
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        for line, text in _find(path.read_text(encoding="utf-8"), aware):
            offenders.append(f"{path.relative_to(APP)}:{line}  {text}")
    assert not offenders, (
        "A naive utcnow() meets a timezone-aware column (TypeError on "
        "Postgres); use utils.datetimes.is_past / age / as_aware_utc:\n"
        + "\n".join(offenders)
    )
