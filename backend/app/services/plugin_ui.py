"""What a plugin may contribute to the interface, and why a shape is refused.

A plugin describes its UI as data and first-party components render it. No
plugin JavaScript ever runs in the application's origin, so a plugin can never
read the viewer's token, reach their session, or see any part of the page it
was not given. The cost is a bounded vocabulary, which is a cost worth paying
and is grown deliberately.

Two rules here are security boundaries rather than conveniences.

**A view may only read.** Its source names a tool, and rendering a page calls
that tool -- so a view bound to something classified ``effects: write`` would
mean *opening a page performs a write*, repeatedly, on every render and every
revisit. Only tools the executor classifies read-only may back a view; anything
else has to be behind a button a person presses, which this vocabulary does not
yet offer.

**A view may only call its own plugin's tools.** A plugin renders its own data.
Letting a view name a built-in would turn "install this plugin" into "grant
this plugin a reader for anything in the catalogue", which is not what
installing a small bundle should mean.

Neither rule can be relaxed by the manifest, because both are checked here
against the executor type rather than against anything the author declared.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.agent_core.plugin_specs import governance_for

#: The doors a contributed nav entry may be added to.
#:
#: This mirrors `frontend/src/navigation/catalog.ts`, which is the authority on
#: what a door *is*. Validating here rather than dropping unknown doors in the
#: renderer is deliberate: a typo that installs cleanly and then does nothing
#: is the most expensive kind of failure to diagnose.
#: `frontend/src/navigation/__tests__/preferences.test.ts` fails if the two
#: lists drift.
NAV_DOORS = ("chat", "library", "rnd", "synthesis", "settings")

#: Icons a contribution may name. A plugin cannot ship a component, so icons
#: are names resolved against a first-party map.
ICONS = (
    "activity",
    "beaker",
    "bot",
    "box",
    "chart",
    "check",
    "clock",
    "database",
    "file",
    "flask",
    "gauge",
    "list",
    "package",
    "search",
    "sparkles",
    "table",
    "terminal",
    "zap",
)

#: What a view can be. Small on purpose -- each entry is a first-party
#: component that has to handle every shape a tool might return.
VIEW_KINDS = ("table", "detail", "stats", "markdown")

#: Places in existing pages a plugin may add a panel to.
#:
#: Every name here is a slot that actually renders. That is a rule rather than
#: an accident: a slot a manifest may declare but no page hosts would make a
#: panel install cleanly and appear nowhere, which is the silent-no-op failure
#: this whole validator exists to prevent. Adding a slot means adding a
#: `<PluginSlot>` to a page in the same change.
PANEL_SLOTS = (
    # The scrolling column of the agent job detail panel.
    "job.detail",
    # Beneath the header of the Runs page.
    "runs.header",
)

MAX_VIEWS = 20
MAX_NAV_ENTRIES = 10
MAX_PANELS = 10
MAX_COLUMNS = 20
MAX_TEXT = 4000


class UiContributionError(ValueError):
    """A UI contribution that cannot be installed, with the reason."""


def _text(value: Any, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def _validate_source(
    source: Any,
    *,
    where: str,
    own_tools: Mapping[str, str],
    own_required: Mapping[str, Sequence[str]],
) -> Optional[Dict[str, Any]]:
    """The tool backing a view, if it has one."""
    if source is None:
        return None
    if not isinstance(source, dict):
        raise UiContributionError(f"{where} source must be an object")

    tool = _text(source.get("tool"), 100)
    if not tool:
        raise UiContributionError(f"{where} source names no tool")

    if tool not in own_tools:
        raise UiContributionError(
            f"{where} names tool {tool!r}, which this plugin does not "
            f"contribute. A view may only read its own plugin's tools; this "
            f"manifest declares: {', '.join(sorted(own_tools)) or '(none)'}."
        )

    effects = governance_for(own_tools[tool])["effects"]
    if effects != "read":
        raise UiContributionError(
            f"{where} is backed by {tool!r}, which is a "
            f"{own_tools[tool]!r} tool and therefore classified "
            f"{effects!r}. Rendering a view calls its source, so only a "
            "read-only tool may back one -- otherwise opening the page would "
            "perform a write every time it is viewed."
        )

    params = source.get("params")
    if params is not None and not isinstance(params, dict):
        raise UiContributionError(f"{where} source params must be an object")
    params = dict(params or {})

    # A view supplies its tool's arguments from the manifest -- there is no
    # request to take them from -- so a required input the manifest does not
    # provide is not a runtime risk, it is a view that fails on every render
    # for everyone, for ever. Refusing at install is the only moment the author
    # is present to fix it.
    required = own_required.get(tool) or []
    missing = [name for name in required if name not in params]
    if missing:
        raise UiContributionError(
            f"{where} calls {tool!r} without {', '.join(missing)}, which that "
            f"tool requires. A view passes its arguments from the manifest, so "
            f"add them under source.params."
        )

    return {"tool": tool, "params": params}


def _validate_columns(value: Any, *, where: str) -> List[Dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise UiContributionError(f"{where} is a table and needs at least one column")
    columns: List[Dict[str, str]] = []
    for index, entry in enumerate(value[:MAX_COLUMNS]):
        if isinstance(entry, str):
            key, label = entry.strip(), entry.strip()
        elif isinstance(entry, dict):
            key = _text(entry.get("key"), 100)
            label = _text(entry.get("label"), 60) or key
        else:
            raise UiContributionError(
                f"{where} column {index} must be a string or an object"
            )
        if not key:
            raise UiContributionError(f"{where} column {index} has no key")
        columns.append({"key": key, "label": label})
    return columns


def _validate_view(
    view_id: str,
    raw: Any,
    *,
    own_tools: Mapping[str, str],
    own_required: Mapping[str, Sequence[str]],
):
    where = f"contributes.views[{view_id!r}]"
    if not isinstance(raw, dict):
        raise UiContributionError(f"{where} must be an object")

    kind = _text(raw.get("kind"), 32).lower()
    if kind not in VIEW_KINDS:
        raise UiContributionError(
            f"{where} has kind {kind!r}; must be one of: {', '.join(VIEW_KINDS)}"
        )

    view: Dict[str, Any] = {
        "kind": kind,
        "title": _text(raw.get("title"), 100),
        "description": _text(raw.get("description"), 300),
        "source": _validate_source(
            raw.get("source"),
            where=where,
            own_tools=own_tools,
            own_required=own_required,
        ),
        # Where in the tool's output the data lives. Dotted, applied literally;
        # a path that matches nothing renders the empty state rather than an
        # error, because a tool legitimately returning nothing is not a fault.
        "path": _text(raw.get("path"), 200),
        "empty": _text(raw.get("empty"), 200),
    }

    if kind == "table":
        view["columns"] = _validate_columns(raw.get("columns"), where=where)
        if not view["source"]:
            raise UiContributionError(f"{where} is a table and needs a source")
    elif kind == "stats":
        view["columns"] = _validate_columns(raw.get("fields"), where=where)
        if not view["source"]:
            raise UiContributionError(f"{where} is stats and needs a source")
    elif kind == "detail":
        view["columns"] = _validate_columns(raw.get("fields"), where=where)
        if not view["source"]:
            raise UiContributionError(f"{where} is a detail view and needs a source")
    elif kind == "markdown":
        view["text"] = _text(raw.get("text"), MAX_TEXT)
        if not view["text"] and not view["source"]:
            raise UiContributionError(
                f"{where} is markdown and has neither `text` nor a `source` "
                "to read it from"
            )

    return view


def validate_ui(
    contributes: Mapping[str, Any], *, tools: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Normalize `nav`, `views` and `panels`, or raise saying what is wrong.

    ``tools`` is this manifest's already-validated tool list, which is what
    lets a view's source be checked against the plugin's own declarations
    rather than against whatever exists globally.
    """
    own_tools = {
        str(t.get("name")): str(t.get("tool_type")) for t in tools if t.get("name")
    }
    own_required = {
        str(t.get("name")): list(
            (t.get("parameters_schema") or {}).get("required") or []
        )
        for t in tools
        if t.get("name")
    }

    raw_views = contributes.get("views") or {}
    if not isinstance(raw_views, dict):
        raise UiContributionError("contributes.views must be an object")
    if len(raw_views) > MAX_VIEWS:
        raise UiContributionError(
            f"contributes.views declares {len(raw_views)} views; at most "
            f"{MAX_VIEWS} are allowed"
        )
    views = {
        _text(view_id, 60): _validate_view(
            str(view_id), raw, own_tools=own_tools, own_required=own_required
        )
        for view_id, raw in raw_views.items()
    }

    raw_nav = contributes.get("nav") or []
    if not isinstance(raw_nav, list):
        raise UiContributionError("contributes.nav must be a list")
    if len(raw_nav) > MAX_NAV_ENTRIES:
        raise UiContributionError(
            f"contributes.nav declares {len(raw_nav)} entries; at most "
            f"{MAX_NAV_ENTRIES} are allowed"
        )

    nav: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw_nav):
        where = f"contributes.nav[{index}]"
        if not isinstance(entry, dict):
            raise UiContributionError(f"{where} must be an object")
        door = _text(entry.get("door"), 32).lower()
        if door not in NAV_DOORS:
            raise UiContributionError(
                f"{where} names door {door!r}; must be one of: "
                f"{', '.join(NAV_DOORS)}"
            )
        name = _text(entry.get("name"), 60)
        if not name:
            raise UiContributionError(f"{where} has no name")
        view_id = _text(entry.get("view"), 60)
        if view_id not in views:
            raise UiContributionError(
                f"{where} points at view {view_id!r}, which this manifest does "
                f"not declare. Declared views: "
                f"{', '.join(sorted(views)) or '(none)'}."
            )
        icon = _text(entry.get("icon"), 32).lower() or "package"
        if icon not in ICONS:
            raise UiContributionError(
                f"{where} names icon {icon!r}; must be one of: " f"{', '.join(ICONS)}"
            )
        nav.append({"door": door, "name": name, "view": view_id, "icon": icon})

    raw_panels = contributes.get("panels") or []
    if not isinstance(raw_panels, list):
        raise UiContributionError("contributes.panels must be a list")
    if len(raw_panels) > MAX_PANELS:
        raise UiContributionError(
            f"contributes.panels declares {len(raw_panels)} panels; at most "
            f"{MAX_PANELS} are allowed"
        )

    panels: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw_panels):
        where = f"contributes.panels[{index}]"
        if not isinstance(entry, dict):
            raise UiContributionError(f"{where} must be an object")
        slot = _text(entry.get("slot"), 60)
        if slot not in PANEL_SLOTS:
            raise UiContributionError(
                f"{where} names slot {slot!r}; must be one of: "
                f"{', '.join(PANEL_SLOTS)}"
            )
        view_id = _text(entry.get("view"), 60)
        if view_id not in views:
            raise UiContributionError(
                f"{where} points at view {view_id!r}, which this manifest does "
                f"not declare. Declared views: "
                f"{', '.join(sorted(views)) or '(none)'}."
            )
        panels.append(
            {"slot": slot, "view": view_id, "title": _text(entry.get("title"), 100)}
        )

    return {"nav": nav, "views": views, "panels": panels}
