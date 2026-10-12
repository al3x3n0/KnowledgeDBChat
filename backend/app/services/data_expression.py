"""What a dataset expression written by a model may say.

`DataFrame.query` and `DataFrame.eval` were called on strings from the
`query_data` and `transform_data` tools with a comment calling it "safe
eval". It is not: pandas resolves `@name` against the caller's Python scope,
so `a > @pd.io.common.os.system("...")` runs a command in the worker, and an
attribute chain such as `a.__class__...` or `a.to_csv("/path")` reaches
Python objects and the filesystem without any `@` at all. Checked on the real
library with harmless calls: each of those evaluated.

So the expression is parsed here first, and only what a filter or a derived
column needs is let through: columns, constants, arithmetic, comparisons,
boolean logic, a fixed set of functions, and a fixed set of methods on a
column or its `.str` / `.dt` accessor. Everything else is refused by name.
The caller also evaluates with empty scopes, so there is nothing for a name
to resolve to but a column.
"""

from __future__ import annotations

import ast
import re

#: The longest expression accepted; these are one-line filters.
MAX_EXPRESSION_CHARS = 2000

#: `sqrt(a)`: the functions pandas' own expression engine provides.
FUNCTIONS = frozenset(
    {
        "abs",
        "sqrt",
        "exp",
        "log",
        "log10",
        "log1p",
        "expm1",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "floor",
        "ceil",
    }
)

#: `col.str` and `col.dt`, and the plain properties read through them.
ACCESSORS = frozenset({"str", "dt"})
PROPERTIES = frozenset(
    {
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "dayofweek",
        "dayofyear",
        "quarter",
        "date",
    }
)

#: Methods that compute a value from a column and touch nothing else. Nothing
#: here writes a file (`to_csv`), takes a callable (`apply`, `map`, `pipe`)
#: or evaluates a string (`eval`, `query`).
METHODS = frozenset(
    {
        "isin",
        "isna",
        "notna",
        "isnull",
        "notnull",
        "between",
        "abs",
        "round",
        "clip",
        "fillna",
        "contains",
        "startswith",
        "endswith",
        "lower",
        "upper",
        "strip",
        "len",
        "sum",
        "mean",
        "min",
        "max",
        "median",
        "std",
    }
)

_PLAIN = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Load,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)

_BACKTICKED = re.compile(r"`[^`\n]+`")


class ExpressionRefused(ValueError):
    """The expression asks for something a dataset expression may not do."""


def check_expression(expression: object) -> str:
    """The expression, if it is one a dataset may be evaluated with.

    Raises ExpressionRefused naming what is not allowed, which is what the
    model needs in order to write one that is.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionRefused("An expression is required")
    text = expression.strip()
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionRefused(
            f"Expression is longer than {MAX_EXPRESSION_CHARS} characters"
        )
    if "@" in text:
        raise ExpressionRefused(
            "'@' is not allowed: an expression may refer to columns and "
            "constants only, not to variables"
        )
    if "__" in text:
        raise ExpressionRefused("Names containing '__' are not allowed")

    # A column name in backticks is pandas syntax, not Python: stand a plain
    # name in for it so the rest can be parsed. Only the check uses this.
    parseable = _BACKTICKED.sub("column", text)
    try:
        tree = ast.parse(parseable, mode="eval")
    except SyntaxError as exc:
        raise ExpressionRefused(f"Not a valid expression: {exc.msg}") from exc

    _check(tree)
    return text


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Call):
        _check_call(node)
        return
    if isinstance(node, ast.Attribute):
        _check_attribute(node, called=False)
        return
    if not isinstance(node, _PLAIN):
        raise ExpressionRefused(
            f"{type(node).__name__} is not allowed in a dataset expression"
        )
    for child in ast.iter_child_nodes(node):
        _check(child)


def _check_call(node: ast.Call) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        if func.id not in FUNCTIONS:
            raise ExpressionRefused(
                f"Function '{func.id}' is not allowed. Use one of: "
                f"{', '.join(sorted(FUNCTIONS))}"
            )
    elif isinstance(func, ast.Attribute):
        _check_attribute(func, called=True)
    else:
        raise ExpressionRefused("Only a named function or method may be called")
    for arg in node.args:
        _check(arg)
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ExpressionRefused("'**' arguments are not allowed")
        _check(keyword.value)


def _check_attribute(node: ast.Attribute, *, called: bool) -> None:
    name = node.attr
    if called:
        if name not in METHODS:
            raise ExpressionRefused(
                f"Method '.{name}()' is not allowed. Use one of: "
                f"{', '.join(sorted(METHODS))}"
            )
    elif name not in ACCESSORS and name not in PROPERTIES:
        raise ExpressionRefused(f"Attribute '.{name}' is not allowed")
    _check(node.value)


__all__ = ["ExpressionRefused", "check_expression"]
