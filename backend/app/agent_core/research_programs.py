"""Research programs: domain research code that lives outside this repository.

Core is the lab -- the run loop, contracts, the sandbox, review, the
knowledge base. What a particular line of research needs on top of that (a
simulator driver, a binary rewriter, the images they run in) is a *program*:
a Python package in its own repository, installed into the backend and worker
images, and found here through the ``kdbc.research_programs`` entry point.

Plugins and sandbox skills stay what they are -- data a user authors in the
UI. A program is code, and it is trusted the way the rest of the image is:
whoever builds the image chose to install it. That is why a program may
contribute Python handlers and a plugin may not.

Installed means available. A program found at startup is loaded, and its tools
join the built-in catalog as if they were declared here, so every registry that
reads the catalog (the model's tool list, governance, job-type policy, the
evidence map) sees them without knowing where they came from.
``RESEARCH_PROGRAMS_DISABLED`` names programs to leave out.

A program that cannot be loaded does not stop the API -- it is an optional
part -- but it is not silent either: the failure is logged and kept, and
``status()`` reports it beside the programs that loaded. A tool quietly
missing is the failure this codebase keeps relearning to avoid.

Core never imports a program. A program imports core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

#: The entry point group a program registers under. Its value is a
#: ``ResearchProgram``, or a callable returning one.
ENTRY_POINT_GROUP = "kdbc.research_programs"


@dataclass(frozen=True)
class ResearchProgram:
    """What one installed program contributes."""

    #: Short identifier, also the domain its tools are listed under.
    name: str
    version: str = ""
    description: str = ""
    #: ToolSpecs, declared exactly as a built-in tool is.
    tool_specs: Tuple[Any, ...] = ()
    #: Each called with the autonomous executor and returning an
    #: ``AgentToolProvider`` that handles some of ``tool_specs``.
    providers: Tuple[Callable[[Any], Any], ...] = ()
    #: Sandbox images the program's tools run in. Allowed beside
    #: ``SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES``: installing the program
    #: is the decision to run them.
    sandbox_images: Tuple[str, ...] = ()


@dataclass
class LoadReport:
    loaded: List[ResearchProgram] = field(default_factory=list)
    #: (name, reason) for each program found and not loaded.
    failed: List[Tuple[str, str]] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)


_REPORT: Optional[LoadReport] = None


def _disabled_names() -> set[str]:
    from app.core.config import settings

    raw = getattr(settings, "RESEARCH_PROGRAMS_DISABLED", "") or ""
    return {item.strip() for item in raw.split(",") if item.strip()}


def _entry_points() -> Sequence[Any]:
    return list(metadata.entry_points(group=ENTRY_POINT_GROUP))


def _resolve(entry_point: Any) -> ResearchProgram:
    value = entry_point.load()
    if callable(value) and not isinstance(value, ResearchProgram):
        value = value()
    if not isinstance(value, ResearchProgram):
        raise TypeError(
            f"entry point {entry_point.value!r} gave {type(value).__name__}, "
            "not a ResearchProgram"
        )
    return value


def load(reserved_tool_names: Sequence[str] = ()) -> LoadReport:
    """Find, load and check every installed program.

    ``reserved_tool_names`` are the built-in tools: a program may not shadow
    one, nor declare a tool another program already did. Either refuses the
    whole program rather than a part of it, since its tools are written to
    work together.
    """
    report = LoadReport()
    disabled = _disabled_names()
    taken = set(reserved_tool_names)
    seen_programs: set[str] = set()
    for entry_point in _entry_points():
        label = getattr(entry_point, "name", "?")
        if label in disabled:
            report.disabled.append(label)
            continue
        try:
            program = _resolve(entry_point)
        except Exception as exc:
            report.failed.append((label, f"could not be loaded: {exc}"))
            logger.error(f"Research program {label!r} could not be loaded: {exc}")
            continue
        if program.name in disabled:
            report.disabled.append(program.name)
            continue
        if program.name in seen_programs:
            report.failed.append((program.name, "installed twice"))
            logger.error(f"Research program {program.name!r} is installed twice")
            continue
        names = [getattr(spec, "name", "") for spec in program.tool_specs]
        clashes = sorted(
            {n for n in names if n in taken} | {n for n in names if names.count(n) > 1}
        )
        if clashes:
            reason = f"declares tools that already exist: {', '.join(clashes)}"
            report.failed.append((program.name, reason))
            logger.error(f"Research program {program.name!r} refused: {reason}")
            continue
        taken.update(names)
        seen_programs.add(program.name)
        report.loaded.append(program)
        logger.info(
            f"Research program {program.name!r} {program.version} loaded: "
            f"{len(names)} tools"
        )
    return report


def initialise(reserved_tool_names: Sequence[str]) -> LoadReport:
    """Load once per process; the tool catalog calls this at import."""
    global _REPORT
    if _REPORT is None:
        _REPORT = load(reserved_tool_names)
    return _REPORT


def report() -> LoadReport:
    """What was loaded, loading it first if nothing has asked yet.

    Programs are loaded when the tool catalog is built, at its import. A
    process that reaches here before importing the catalog -- a test, a script
    asking only for the allowed images -- would otherwise see no program at
    all, and an image or an evidence tool would quietly not count.
    """
    if _REPORT is None:
        from app.agent_core import tool_specs  # noqa: F401  (loads programs)
    return _REPORT if _REPORT is not None else LoadReport()


def installed() -> List[ResearchProgram]:
    return list(report().loaded)


def build_providers(executor: Any) -> List[Any]:
    """Every loaded program's tool providers, for one executor."""
    providers: List[Any] = []
    for program in installed():
        for build in program.providers:
            providers.append(build(executor))
    return providers


def sandbox_images() -> List[str]:
    return [image for program in installed() for image in program.sandbox_images]


def status() -> Dict[str, Any]:
    """What is installed, for an operator: loaded, refused, and turned off."""
    current = report()
    return {
        "loaded": [
            {
                "name": program.name,
                "version": program.version,
                "description": program.description,
                "tools": [spec.name for spec in program.tool_specs],
                "sandbox_images": list(program.sandbox_images),
            }
            for program in current.loaded
        ],
        "failed": [{"name": name, "reason": reason} for name, reason in current.failed],
        "disabled": list(current.disabled),
    }
