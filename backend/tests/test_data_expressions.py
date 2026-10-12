"""A dataset expression written by a model cannot reach Python.

`query_data` and `transform_data` passed their string to `DataFrame.query`
and `DataFrame.eval`. Run on the real library with harmless calls, each of
these evaluated: `@pd.io.common.os.getpid()` (a module in the caller's
scope), `@self.max_rows` (the sandbox object), and `a.__class__.__name__`
(the object graph). A run steered by a document it read could have executed
commands in the worker, which holds the database credentials.

Every test here goes through the real sandbox and the real pandas.
"""

import os

import pytest

from app.services.data_expression import ExpressionRefused, check_expression
from app.services.data_sandbox_service import DataSandbox

pytestmark = pytest.mark.unit


@pytest.fixture
def sandbox():
    box = DataSandbox("job", "user")
    created = box.create_from_dict(
        {
            "age": [25, 40, 61],
            "city": ["Oslo", "NYC", "nyc"],
            "total sales": [10.0, 250.5, 3.0],
        },
        "people",
    )
    box.dataset_id = created["dataset_id"]
    return box


def _rows(sandbox, expression):
    return sandbox.query(sandbox.dataset_id, expression)["preview"]


# --------------------------------------------------------------------------
# what must keep working
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, ages",
    [
        ("age > 30", [40, 61]),
        ("age > 30 and city == 'NYC'", [40]),
        ("not (age < 60)", [61]),
        ("city in ['NYC', 'Oslo']", [25, 40]),
        ("city.str.lower() == 'nyc'", [40, 61]),
        ("city.str.contains('y', case=False)", [40, 61]),
        ("age.between(30, 50)", [40]),
        ("age.isin([25, 61])", [25, 61]),
        ("`total sales` > 5", [25, 40]),
        ("sqrt(age) > 6", [40, 61]),
        ("age * 2 + 1 >= 81", [40, 61]),
    ],
)
def test_an_ordinary_filter_still_works(sandbox, expression, ages):
    assert [row["age"] for row in _rows(sandbox, expression)] == ages


def test_a_derived_column_still_works(sandbox):
    result = sandbox.transform(
        sandbox.dataset_id,
        [{"op": "add_column", "name": "double", "expression": "age * 2"}],
    )

    assert [row["double"] for row in result["preview"]] == [50, 80, 122]


# --------------------------------------------------------------------------
# what must not
# --------------------------------------------------------------------------

ESCAPES = [
    # A module in the calling scope: this ran a function of `os`.
    "age > @pd.io.common.os.getpid() * 0",
    # The sandbox object itself.
    "age > @self.max_rows * 0",
    # The object graph, with no '@' at all.
    "age.__class__.__name__ == 'Series'",
    "age.__class__.__mro__[1].__subclasses__()",
    # Methods that write a file, take a callable, or evaluate a string.
    "age.to_csv('/tmp/leak.csv')",
    "age.to_pickle('/tmp/leak.pkl')",
    "age.apply(print)",
    "age.map(len)",
    "age.pipe(print)",
    # Names that are not functions of the expression language.
    "open('/etc/passwd')",
    "eval('1')",
    "getattr(age, 'values')",
    # Shapes with no place in a filter.
    "[x for x in age]",
    "(lambda: 1)()",
    "age if age else city",
    "age[0]",
    "age.values",
    "f'{age}'",
]


@pytest.mark.parametrize("expression", ESCAPES)
def test_an_expression_that_reaches_outside_the_data_is_refused(sandbox, expression):
    with pytest.raises(ValueError):
        sandbox.query(sandbox.dataset_id, expression)


@pytest.mark.parametrize("expression", ESCAPES)
def test_it_is_refused_as_a_derived_column_too(sandbox, expression):
    with pytest.raises(ValueError):
        sandbox.transform(
            sandbox.dataset_id,
            [{"op": "add_column", "name": "x", "expression": expression}],
        )


def test_a_refused_expression_ran_nothing(sandbox, tmp_path):
    """Refused before evaluation, not caught after it."""
    target = tmp_path / "written.csv"

    with pytest.raises(ValueError):
        sandbox.query(sandbox.dataset_id, f"age.to_csv({str(target)!r})")

    assert not target.exists()


def test_nothing_is_in_scope_but_the_columns(sandbox, monkeypatch):
    """Even a name the check allowed cannot resolve to a Python variable."""
    monkeypatch.setenv("LEAK", "secret")
    assert os.environ["LEAK"] == "secret"

    # `os` is a name, and names are allowed -- as columns. There is no such
    # column, and no scope behind it.
    with pytest.raises(ValueError, match="Query failed"):
        sandbox.query(sandbox.dataset_id, "age > os")


def test_the_refusal_names_what_was_wrong():
    with pytest.raises(ExpressionRefused, match="'@' is not allowed"):
        check_expression("a > @x")
    with pytest.raises(
        ExpressionRefused, match=r"Method '\.to_csv\(\)' is not allowed"
    ):
        check_expression("a.to_csv('x')")
    with pytest.raises(ExpressionRefused, match="Function 'open' is not allowed"):
        check_expression("open('x')")
    with pytest.raises(ExpressionRefused, match="required"):
        check_expression("   ")
    with pytest.raises(ExpressionRefused, match="longer than"):
        check_expression("a > 1 and " * 400 + "a > 1")


# --------------------------------------------------------------------------
# aggregation names are looked up on the column too
# --------------------------------------------------------------------------


def test_documented_aggregations_work(sandbox):
    result = sandbox.aggregate(sandbox.dataset_id, aggregations={"age": ["min", "max"]})

    assert result["preview"] == [{"age_min": 25, "age_max": 61}]


@pytest.mark.parametrize("name", ["to_csv", "__class__", "to_pickle", "memory_usage"])
def test_an_aggregation_that_is_just_a_method_name_is_refused(sandbox, name):
    with pytest.raises(ValueError, match="Unknown aggregation"):
        sandbox.aggregate(sandbox.dataset_id, aggregations={"age": [name]})
    with pytest.raises(ValueError, match="Unknown aggregation"):
        sandbox.aggregate(
            sandbox.dataset_id, group_by=["city"], aggregations={"age": [name]}
        )
