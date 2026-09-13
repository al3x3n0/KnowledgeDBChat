"""JSON columns that notice when they are changed in place.

A plain ``Column(JSON)`` holds an ordinary dict or list, and SQLAlchemy detects
a change to it only when the attribute is *reassigned*. Code that does the
natural thing --

    job.execution_log.append(entry)
    job.results["summary"] = summary

-- mutates the object the session already has, changes nothing SQLAlchemy can
see, and commits without writing anything. The append succeeds, the value is
right for the rest of the process, and it is gone on the next load. Nothing
raises, which is what makes it expensive: an agent's whole iteration record can
vanish and the only symptom is an empty log some hours later.

``sqlalchemy.ext.mutable`` solves the shallow case, but only the shallow case:
``MutableDict`` tracks assignment to its own keys and knows nothing about

    job.results["research"]["created_documents"] = [...]

which is a mutation of an ordinary dict that happens to be stored inside it.
Using it here would fix the top-level call sites, leave the nested ones broken,
and make the whole class of bug harder to find by making it rarer.

So these types wrap containers on the way in and give each child a pointer to
its parent, and a change at any depth bubbles up to the root, which is the
object the attribute actually watches. The cost is that every dict and list put
into one of these columns is copied into a wrapper; these columns hold job logs
and results, written a few times per iteration, so that is not a concern here.

Use the type matching what the column holds: ``NestedMutableList`` for a JSON
array, ``NestedMutableDict`` for an object.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.mutable import Mutable


def _wrap(value: Any, parent: Any) -> Any:
    """Wrap a child container so its own changes reach `parent`."""
    if isinstance(value, (NestedMutableDict, NestedMutableList)):
        value._parent = parent
        return value
    if isinstance(value, dict):
        return NestedMutableDict(value, _parent=parent)
    if isinstance(value, list):
        return NestedMutableList(value, _parent=parent)
    return value


class _Bubbles:
    """Report a change to the root, which is what the ORM is watching."""

    _parent: Any = None

    def changed(self) -> None:  # type: ignore[override]
        parent = getattr(self, "_parent", None)
        if parent is not None:
            parent.changed()
        else:
            super().changed()  # type: ignore[misc]


class NestedMutableDict(_Bubbles, Mutable, dict):
    """A JSON object that reports in-place changes at any depth."""

    def __init__(self, source: Any = None, _parent: Any = None):
        self._parent = _parent
        dict.__init__(self)
        for key, value in dict(source or {}).items():
            dict.__setitem__(self, key, _wrap(value, self))

    @classmethod
    def coerce(cls, key, value):  # noqa: D102 - SQLAlchemy hook
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        return Mutable.coerce(key, value)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, _wrap(value, self))
        self.changed()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.changed()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            dict.__setitem__(self, key, _wrap(value, self))
        self.changed()

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, *args):
        result = dict.pop(self, *args)
        self.changed()
        return result

    def popitem(self):
        result = dict.popitem(self)
        self.changed()
        return result

    def clear(self):
        dict.clear(self)
        self.changed()

    # Plain containers on the way out: the parent pointer must not travel into
    # a pickle or a deep copy, where it would drag the whole object graph.
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)

    def __deepcopy__(self, memo):
        from copy import deepcopy

        return NestedMutableDict({k: deepcopy(v, memo) for k, v in self.items()})

    def __reduce__(self):
        return (NestedMutableDict, (dict(self),))


class NestedMutableList(_Bubbles, Mutable, list):
    """A JSON array that reports in-place changes at any depth."""

    def __init__(self, source: Any = None, _parent: Any = None):
        self._parent = _parent
        list.__init__(self)
        for value in list(source or []):
            list.append(self, _wrap(value, self))

    @classmethod
    def coerce(cls, key, value):  # noqa: D102 - SQLAlchemy hook
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, list):
            return cls(value)
        return Mutable.coerce(key, value)

    def __setitem__(self, index, value):
        list.__setitem__(self, index, _wrap(value, self))
        self.changed()

    def __delitem__(self, index):
        list.__delitem__(self, index)
        self.changed()

    def append(self, value):
        list.append(self, _wrap(value, self))
        self.changed()

    def extend(self, values):
        for value in values:
            list.append(self, _wrap(value, self))
        self.changed()

    def __iadd__(self, values):
        self.extend(values)
        return self

    def insert(self, index, value):
        list.insert(self, index, _wrap(value, self))
        self.changed()

    def remove(self, value):
        list.remove(self, value)
        self.changed()

    def pop(self, *args):
        result = list.pop(self, *args)
        self.changed()
        return result

    def clear(self):
        list.clear(self)
        self.changed()

    def sort(self, **kwargs):
        list.sort(self, **kwargs)
        self.changed()

    def reverse(self):
        list.reverse(self)
        self.changed()

    def __getstate__(self):
        return list(self)

    def __setstate__(self, state):
        self.extend(state)

    def __deepcopy__(self, memo):
        from copy import deepcopy

        return NestedMutableList([deepcopy(v, memo) for v in self])

    def __reduce__(self):
        return (NestedMutableList, (list(self),))
