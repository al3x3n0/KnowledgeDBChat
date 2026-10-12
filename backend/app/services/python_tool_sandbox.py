"""What a custom Python tool's code can see while it runs.

The code is compiled by RestrictedPython, which rewrites attribute access,
subscripts, writes and iteration into calls to guards looked up in the
globals. What those globals hold is the sandbox, and two things were wrong
with them, each found by running code through the real executor:

* **The "safe" modules were handed over whole.** A module object carries the
  modules it imported: `re.enum.sys.modules.get("os")` reached `os` from the
  default set, with no import and no underscore. A tool's own config could
  also name any module it liked in `allowed_imports`, `os` included. Modules
  are now facades holding their public functions, classes and constants and
  none of their submodules, and the config may only add from a fixed list.
* **Most ordinary code could not run.** `_getitem_`, `_write_`,
  `_inplacevar_`, `_unpack_sequence_` and `_print_` were never provided, so
  `input["x"]`, `output["k"] = v`, `total += x`, `a, b = pair` and `print()`
  each failed with a NameError about a name the author never wrote.

This is still code running in the server's own process: the limit is what is
in scope. It cannot stop a loop that never ends -- the executor stops waiting
at the timeout and the thread runs on. Work that needs a real boundary
belongs in `agent_sandbox_runtime`.
"""

from __future__ import annotations

import builtins
import importlib
import operator
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, Optional

#: Available to every Python tool.
DEFAULT_MODULES = ("json", "re", "datetime", "math", "collections", "itertools")
#: What a tool's config may add. Pure computation only: nothing here opens a
#: file, a socket or a process.
#: Not here, on purpose: `operator` (`attrgetter("__globals__")` reads the
#: attributes the attribute guard refuses) and `functools`.
OPTIONAL_MODULES = (
    "statistics",
    "string",
    "decimal",
    "fractions",
    "textwrap",
    "hashlib",
    "base64",
    "uuid",
    "random",
)
IMPORTABLE = frozenset(DEFAULT_MODULES + OPTIONAL_MODULES)

#: Public names that are not the module's to give: file and process handles.
#: `string.Formatter` and `collections.UserString` are format strings by
#: another name: `"{0.__globals__[...]}"` walks attributes and indexes inside
#: the format call, where no guard sees it. The attribute guard refuses
#: `str.format` for that reason, and these two would hand it back.
WITHHELD = {
    "json": {"load", "dump"},
    "uuid": {"getnode", "uuid1"},
    "string": {"Formatter"},
    "collections": {"UserString"},
}

_INPLACE = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
}

#: Builtins RestrictedPython's safe set leaves out that ordinary data code
#: uses and that reach nothing: no `getattr`, `type`, `vars`, `open`, `eval`.
EXTRA_BUILTINS = (
    "min",
    "max",
    "sum",
    "all",
    "any",
    "enumerate",
    "map",
    "filter",
    "reversed",
    "dict",
    "list",
    "set",
)

#: What tool code may assign into or delete from by index or attribute.
_WRITABLE = (dict, list, set)


class ModuleNotAllowed(ValueError):
    """A tool's config asked for a module outside the allowlist."""


def facade(name: str) -> SimpleNamespace:
    """A module's public functions, classes and constants -- and no modules.

    `__all__` where the module declares one, otherwise every name without a
    leading underscore. A submodule or an imported module is left out either
    way: that is the door `re.enum.sys` went through.
    """
    module = importlib.import_module(name)
    names = getattr(module, "__all__", None) or [
        n for n in dir(module) if not n.startswith("_")
    ]
    withheld = WITHHELD.get(name, set())
    public: Dict[str, Any] = {}
    for attr in names:
        if attr.startswith("_") or attr in withheld:
            continue
        value = getattr(module, attr, None)
        if isinstance(value, ModuleType):
            continue
        public[attr] = value
    return SimpleNamespace(**public)


def modules_for(requested: Optional[Iterable[Any]]) -> Dict[str, SimpleNamespace]:
    """The modules a tool gets: the defaults, plus what its config adds."""
    extra = [str(n).strip() for n in (requested or []) if str(n).strip()]
    refused = sorted(set(extra) - IMPORTABLE)
    if refused:
        raise ModuleNotAllowed(
            f"allowed_imports names {', '.join(refused)}, which a Python tool "
            f"may not use. It may add: {', '.join(sorted(OPTIONAL_MODULES))}. "
            "This sandbox cannot run subprocesses or reach the filesystem or "
            "the network."
        )
    return {name: facade(name) for name in (*DEFAULT_MODULES, *extra)}


def _write_guard(target: Any) -> Any:
    """Allow assignment into the plain containers a tool builds its answer in."""
    if isinstance(target, _WRITABLE):
        return target
    raise TypeError(
        f"A Python tool may only assign into a dict, list or set, not "
        f"{type(target).__name__}"
    )


def _inplace(op: str, left: Any, right: Any) -> Any:
    try:
        return _INPLACE[op](left, right)
    except KeyError:
        raise SyntaxError(f"'{op}' is not supported in a Python tool") from None


def build_globals(modules: Dict[str, SimpleNamespace]) -> Dict[str, Any]:
    """The globals restricted code runs with: guards, builtins, facades."""
    from RestrictedPython import safe_globals
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
        safer_getattr,
    )
    from RestrictedPython.PrintCollector import PrintCollector

    scope = safe_globals.copy()
    # Its own copy: the builtins dict is shared by every caller otherwise.
    scope["__builtins__"] = dict(safe_globals["__builtins__"])
    for name in EXTRA_BUILTINS:
        scope["__builtins__"].setdefault(name, getattr(builtins, name))
    scope["_getattr_"] = safer_getattr
    scope["_getitem_"] = default_guarded_getitem
    scope["_getiter_"] = default_guarded_getiter
    scope["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    scope["_unpack_sequence_"] = guarded_unpack_sequence
    scope["_write_"] = _write_guard
    scope["_inplacevar_"] = _inplace
    # print() is collected and dropped; a tool answers through `output`.
    scope["_print_"] = PrintCollector
    scope.update(modules)
    return scope


__all__ = [
    "DEFAULT_MODULES",
    "IMPORTABLE",
    "ModuleNotAllowed",
    "OPTIONAL_MODULES",
    "build_globals",
    "facade",
    "modules_for",
]
