"""A custom Python tool's code, run through the real executor.

`python` is a tool type anyone may author, an agent included, and its code
runs in the server's own process under RestrictedPython. Two things were
wrong with what that code could see:

* it could leave: the "safe" modules were handed over whole, and a module
  carries what it imported -- `re.enum.sys.modules.get("os")` reached `os`
  from the defaults. A tool's config could also list any module at all;
* it could barely run: reading `input["x"]`, assigning `output["k"]`, `+=`,
  tuple unpacking and `print` each failed on a guard that was never provided.

Every case goes through `CustomToolService._execute_python`. The escapes are
written with harmless calls: what matters is whether the object was reached.
"""

import pytest

from app.services import python_tool_sandbox
from app.services.custom_tool_service import CustomToolService, ToolExecutionError

pytestmark = pytest.mark.unit

INPUT = {"a": 1, "items": [1, 2, 3], "name": "Ada"}


async def _run(code, **config):
    return await CustomToolService()._execute_python(
        config={"code": code, "timeout_seconds": 5, **config},
        inputs=dict(INPUT),
        user=None,
    )


# --------------------------------------------------------------------------
# ordinary code runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ('output = {"v": input["a"] + 1}', {"v": 2}),
        ('output["v"] = input.get("a") + 1', {"v": 2}),
        ("t = 0\nfor x in input['items']:\n    t += x\noutput = {'t': t}", {"t": 6}),
        ("a, b = 1, 2\noutput = {'s': a + b}", {"s": 3}),
        (
            "for i, x in enumerate(input['items']):\n    output[str(i)] = x",
            {"0": 1, "1": 2, "2": 3},
        ),
        ("print('working')\noutput = {'ok': True}", {"ok": True}),
        (
            "output = {'m': max(input['items']), 's': sum(input['items'])}",
            {"m": 3, "s": 6},
        ),
        ("r = []\nr.append(input['name'].upper())\noutput = {'r': r}", {"r": ["ADA"]}),
        (
            "def double(x):\n    return x * 2\noutput = {'r': [double(x) for x in input['items']]}",
            {"r": [2, 4, 6]},
        ),
        (
            'output = {"n": re.findall("[0-9]+", json.dumps(input["items"]))}',
            {"n": ["1", "2", "3"]},
        ),
        ("output = {'y': datetime.date(2026, 10, 12).year}", {"y": 2026}),
        ("output = {'c': collections.Counter('aab').most_common(1)[0][0]}", {"c": "a"}),
        ("output = {'r': math.floor(math.sqrt(17))}", {"r": 4}),
    ],
)
async def test_ordinary_tool_code_runs(code, expected):
    assert await _run(code) == expected


async def test_a_tool_may_add_a_module_from_the_list():
    result = await _run(
        "output = {'m': statistics.mean(input['items'])}",
        allowed_imports=["statistics"],
    )

    assert result == {"m": 2}


# --------------------------------------------------------------------------
# and stays inside
# --------------------------------------------------------------------------

ESCAPES = [
    # A module reached through a module: this returned `os`.
    'output = {"x": re.enum.sys.modules.get("os").getpid()}',
    'output = {"x": json.decoder.re.enum.sys.platform}',
    'output = {"x": datetime.sys.platform}',
    'output = {"x": collections.abc.Mapping}',
    # Format strings walk attributes where no guard sees them.
    'output = {"x": "{0.__class__}".format(input)}',
    'output = {"x": collections.UserString("{0.__class__}").format(input)}',
    # Underscore attributes, the object graph, and the builtins that reach them.
    'output = {"x": input.__class__}',
    'output = {"x": json.loads.__globals__}',
    'output = {"x": getattr(input, "__class__")}',
    'output = {"x": type(input)}',
    'output = {"x": vars()}',
    'output = {"x": open("/etc/hosts").read()}',
    'output = {"x": eval("1")}',
    'output = {"x": __import__("os").getpid()}',
    "import os\noutput = {'x': os.getpid()}",
    "from os import getpid\noutput = {'x': getpid()}",
    # Files, by way of an allowed module.
    'output = {"x": json.load(None)}',
]


@pytest.mark.parametrize("code", ESCAPES)
async def test_code_that_reaches_outside_gets_nothing(code):
    """Refused, or handed None: the attribute guard answers a name that is
    not there with None, so a chain through one ends in None."""
    try:
        result = await _run(code)
    except ToolExecutionError:
        return
    assert result == {"x": None}, f"reached something: {result!r}"


@pytest.mark.parametrize(
    "module", ["os", "subprocess", "sys", "socket", "operator", "functools", "builtins"]
)
async def test_a_tools_config_cannot_grant_itself_a_module(module):
    with pytest.raises(ToolExecutionError) as refused:
        await _run("output = {'ok': True}", allowed_imports=[module])

    assert module in str(refused.value)
    assert "statistics" in str(refused.value)


async def test_a_tool_cannot_assign_into_what_it_was_handed():
    """Only into the plain containers it builds its answer in."""
    with pytest.raises(ToolExecutionError):
        await _run("json.dumps = None\noutput = {'ok': True}")
    with pytest.raises(ToolExecutionError):
        await _run("math.pi = 3\noutput = {'ok': True}")


def test_no_facade_holds_a_module():
    """The property the escapes above depend on, for every allowed module."""
    from types import ModuleType

    for name in python_tool_sandbox.IMPORTABLE:
        held = vars(python_tool_sandbox.facade(name))
        assert held, name
        modules = [k for k, v in held.items() if isinstance(v, ModuleType)]
        assert modules == [], f"{name} exposes {modules}"
        assert not [k for k in held if k.startswith("_")], name


def test_the_guard_bypassing_names_are_withheld():
    assert not hasattr(python_tool_sandbox.facade("string"), "Formatter")
    assert not hasattr(python_tool_sandbox.facade("collections"), "UserString")
    assert "operator" not in python_tool_sandbox.IMPORTABLE
    assert "functools" not in python_tool_sandbox.IMPORTABLE


async def test_one_tools_changes_do_not_reach_the_next():
    """The builtins and module facades are built per run."""
    await _run("output['x'] = 1\nd = dict(a=1)\nd['b'] = 2")
    first = python_tool_sandbox.build_globals(python_tool_sandbox.modules_for(None))
    second = python_tool_sandbox.build_globals(python_tool_sandbox.modules_for(None))

    assert first["__builtins__"] is not second["__builtins__"]
    assert first["json"] is not second["json"]
