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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
    #: How to rebuild each image, keyed by a name its reference contains, in
    #: the shape ``agent_evidence_bundle.IMAGE_ORIGINS`` uses.
    image_origins: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: Tools whose results are evidence about the machine, archived in a run's
    #: evidence bundle (``agent_evidence_bundle.evidence_tools``).
    evidence_tools: Tuple[str, ...] = ()
    #: ((tool, ...), module path): the module whose ``describe()`` holds the
    #: methodology for those tools (``agent_evidence_map.method_notes``).
    method_sources: Tuple[Tuple[Tuple[str, ...], str], ...] = ()
    #: Per tool, fields taken as alternatives, in the shape
    #: ``agent_tool_validation.ALTERNATIVE_FIELDS`` uses.
    alternative_fields: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)


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


def _validate_catalog_contributions(program: ResearchProgram) -> List[str]:
    """Check declarations before they reach the catalog or claim tool names.

    Dataclass annotations are not runtime validation. A package can return a
    ResearchProgram whose fields would break catalog import or the status API.
    """
    from app.agent_core.tool_specs.spec import ToolSpec

    def text(value: Any, field_name: str, *, nonempty: bool = False) -> None:
        if not isinstance(value, str) or (nonempty and not value.strip()):
            raise ValueError(
                f"{field_name} must be a {'nonempty ' if nonempty else ''}string"
            )

    def sequence(value: Any, field_name: str) -> None:
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{field_name} must be a tuple or list")

    def strings(value: Any, field_name: str) -> None:
        sequence(value, field_name)
        for item in value:
            text(item, field_name, nonempty=True)

    def mapping(value: Any, field_name: str) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        for key in value:
            text(key, f"{field_name} key", nonempty=True)

    text(program.name, "name", nonempty=True)
    text(program.version, "version")
    text(program.description, "description")
    sequence(program.tool_specs, "tool_specs")
    names = []
    for spec in program.tool_specs:
        if not isinstance(spec, ToolSpec):
            raise ValueError("tool_specs must contain ToolSpec declarations")
        text(spec.name, "tool name", nonempty=True)
        text(spec.description, f"{spec.name}.description")
        if not isinstance(spec.parameters, dict):
            raise ValueError(f"{spec.name}.parameters must be a dictionary")
        for field_name in ("produces", "requires"):
            strings(getattr(spec, field_name), f"{spec.name}.{field_name}")
        if spec.job_types is not None:
            strings(spec.job_types, f"{spec.name}.job_types")
        names.append(spec.name)
    sequence(program.providers, "providers")
    if any(not callable(build) for build in program.providers):
        raise ValueError("providers must contain callable factories")
    strings(program.sandbox_images, "sandbox_images")
    strings(program.evidence_tools, "evidence_tools")

    mapping(program.image_origins, "image_origins")
    for origin in program.image_origins.values():
        mapping(origin, "image_origins entry")
        for value in origin.values():
            text(value, "image_origins value")

    sequence(program.method_sources, "method_sources")
    for entry in program.method_sources:
        sequence(entry, "method_sources entry")
        if len(entry) != 2:
            raise ValueError(
                "method_sources entries must pair tool names with a module path"
            )
        owned, module_path = entry
        strings(owned, "method_sources tool names")
        text(module_path, "method_sources module path", nonempty=True)

    mapping(program.alternative_fields, "alternative_fields")
    for rules in program.alternative_fields.values():
        sequence(rules, "alternative_fields rules")
        for rule in rules:
            sequence(rule, "alternative_fields rule")
            if len(rule) != 3:
                raise ValueError(
                    "alternative_fields rules must contain two fields and a validator"
                )
            first, alternative, validate = rule
            text(first, "alternative_fields field", nonempty=True)
            text(alternative, "alternative_fields alternative", nonempty=True)
            if not callable(validate):
                raise ValueError("alternative_fields validator must be callable")
    return names


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
            names = _validate_catalog_contributions(program)
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
    """Build providers atomically per program, for one executor.

    Factories can depend on this executor, so failures belong to this run,
    not the process-wide load report. Keep core and other programs usable,
    and answer the failed program's tools with an explicit unavailable error.
    """
    providers: List[Any] = []
    for program in installed():
        declared = {spec.name for spec in program.tool_specs}
        try:
            built = []
            handled: set[str] = set()
            for build in program.providers:
                provider = build(executor)
                supported = getattr(provider, "supported_tools", None)
                if not isinstance(supported, (set, frozenset)) or not all(
                    isinstance(name, str) for name in supported
                ):
                    raise ValueError(
                        "provider supported_tools must be a set of tool names"
                    )
                if not callable(getattr(provider, "can_handle", None)) or not callable(
                    getattr(provider, "execute", None)
                ):
                    raise ValueError("provider must implement can_handle and execute")
                if supported - declared:
                    raise ValueError(
                        f"provider handles undeclared tools: {', '.join(sorted(supported - declared))}"
                    )
                handled.update(supported)
                built.append(provider)
            if declared - handled:
                raise ValueError(
                    f"missing providers for tools: {', '.join(sorted(declared - handled))}"
                )
        except Exception as exc:
            reason = (
                f"Research program {program.name!r} is unavailable for this run: {exc}"
            )
            logger.error(reason)
            providers.append(_unavailable_provider(program, reason))
            continue
        providers.extend(built)
    return providers


def _unavailable_provider(program: ResearchProgram, reason: str) -> Any:
    from app.services.agent_tool_providers.base import FunctionToolProvider

    async def unavailable(params: Any, context: Any) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": "research_program_unavailable",
            "error": reason,
        }

    return FunctionToolProvider(
        name=program.name,
        handlers={spec.name: unavailable for spec in program.tool_specs},
    )


def sandbox_images() -> List[str]:
    return [image for program in installed() for image in program.sandbox_images]


def evidence_tools() -> set[str]:
    return {name for program in installed() for name in program.evidence_tools}


def tool_program(tool_name: str) -> Optional[Dict[str, str]]:
    """The loaded program supplying a tool, or None for an unowned name.

    Return a fresh snapshot: callers persist it as the identity at execution
    time, rather than deriving old calls from a later process's catalog.
    """
    for program in installed():
        if any(spec.name == tool_name for spec in program.tool_specs):
            return {"name": program.name, "version": program.version}
    return None


def image_origins() -> Dict[str, Mapping[str, str]]:
    return {k: v for program in installed() for k, v in program.image_origins.items()}


def method_sources() -> List[Tuple[Tuple[str, ...], str]]:
    return [entry for program in installed() for entry in program.method_sources]


def method_notes(tool_names: Iterable[str]) -> List[str]:
    """Read optional methodology, making failures visible without breaking a run.

    Resolve only the modules this plan uses. Materialize each module's notes
    before adding them, so a generator that fails midway leaves no partial
    methodology behind.
    """
    import importlib

    tools = set(tool_names)
    lines: List[str] = []
    for program in installed():
        for owned, module_path in program.method_sources:
            if not tools.intersection(owned):
                continue
            try:
                result = importlib.import_module(module_path).describe()
                if isinstance(result, str):
                    raise ValueError(
                        "describe() must return an iterable of strings, not a string"
                    )
                notes = list(result)
                if not all(isinstance(note, str) for note in notes):
                    raise ValueError(
                        "describe() returned a non-string methodology note"
                    )
            except Exception as exc:
                warning = (
                    f"Research program {program.name!r}: methodology unavailable from "
                    f"{module_path!r} ({type(exc).__name__}: {exc}). "
                    "Treat this run's methodology as incomplete."
                )
                logger.warning(warning)
                notes = [warning]
            for note in notes:
                if note not in lines:
                    lines.append(note)
    return lines


def alternative_fields() -> Dict[str, Tuple[Any, ...]]:
    return {
        k: tuple(v)
        for program in installed()
        for k, v in program.alternative_fields.items()
    }


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
