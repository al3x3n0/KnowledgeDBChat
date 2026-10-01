"""What a sandbox skill may say, checked before anything is stored or run.

A skill is a document someone composes -- a person in an editor, a model from a
sentence, a run proposing what it just did -- and all three get the same
checker, because a skill is only as trustworthy as the weakest path that can
create one.

Every refusal **names what is wrong and what would be right**. That is the
same rule `plugin_manifest` follows and for the same reason: the message is
what the next attempt is built from, whether a person or a model makes it.

Three requirements here are decisions rather than hygiene.

**A skill must declare the result it leaves behind.** The evidence a skill
yields is a `result.json` checked against declared fields. A skill declaring
none would yield evidence nothing can check, and a contract requiring it would
be satisfied by any file at all.

**A skill must carry a control.** The control is the smallest invocation that
should work -- it is what a dry run executes, and a skill with none can never
be shown to work, so it could never honestly be activated.

**A skill says whether its result goes stale.** `perishable: true` means the
result describes the working directory as it stood -- a test run, a size, a
timing of a binary -- and is invalidated by whatever changes it. Such a result
is never inherited by a later stage and only its latest reading is bounded,
the same rule `test_result` follows. It is a declaration on the skill because
the evidence map is fixed at import and cannot know one user's skills; the
finding carries the flag to the two places that act on it.

**Unknown keys are refused, not ignored.** `judge_cmd` for `judge_command`
would otherwise be dropped in silence, and the skill would run with no judge
while its author believed it had one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping

#: Evidence a skill yields is named `skill_<slug>`. The prefix is a namespace
#: no built-in evidence type occupies, so a skill can never be mistaken for --
#: or satisfy a contract written against -- a first-party measurement.
EVIDENCE_PREFIX = "skill_"

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
#: Relative, no parent traversal, and a charset that needs no shell quoting.
PATH_PATTERN = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")

#: Where a skill's result is read from, relative to the run directory.
RESULT_FILE = "result.json"
#: Where a skill's own files are placed, so they cannot collide with -- or be
#: overwritten by -- what a run writes beside them.
SKILL_DIR = "skill"

FIELD_TYPES = ("number", "string", "boolean", "array", "object")

MAX_FILES = 20
MAX_FILE_CHARS = 100_000
MAX_TOTAL_FILE_CHARS = 400_000
MAX_PROCEDURE_CHARS = 20_000
MAX_COMMAND_CHARS = 4_000
MIN_DESCRIPTION_CHARS = 10
MAX_DESCRIPTION_CHARS = 600
MAX_RESULT_FIELDS = 40

DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 5
MAX_TIMEOUT_SECONDS = 1800

KNOWN_KEYS = (
    "id",
    "name",
    "description",
    "image",
    "procedure",
    "files",
    "result",
    "judge_command",
    "control",
    "timeout_seconds",
    "perishable",
)


class SkillError(ValueError):
    """A skill that cannot be stored, with the reason."""


def evidence_type(slug: str) -> str:
    """The finding type a skill's result is recorded as."""
    return f"{EVIDENCE_PREFIX}{str(slug or '').strip()}"


def slug_of_evidence(finding_type: str) -> str:
    """The skill a finding type names, or "" if it is not skill evidence."""
    name = str(finding_type or "").strip()
    return name[len(EVIDENCE_PREFIX) :] if name.startswith(EVIDENCE_PREFIX) else ""


def content_hash(manifest: Mapping[str, Any]) -> str:
    """Identity of a manifest's content, stable across key order."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip() if isinstance(value, (str, int, float)) else ""


def _reject_path(path: str) -> str:
    """Why this path cannot be a skill file, or "" if it can."""
    if not path:
        return "a file has an empty path"
    if not PATH_PATTERN.match(path) or ".." in path.split("/"):
        return (
            f"file path {path!r} is not allowed: use a relative path of "
            "letters, digits, dot, underscore and hyphen, with no '..'"
        )
    return ""


def _files(raw: Any, where: str) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SkillError(
            f"{where} must be an object mapping a relative path to the file's "
            f"text, got {type(raw).__name__}"
        )
    if len(raw) > MAX_FILES:
        raise SkillError(f"{where} has {len(raw)} files; at most {MAX_FILES}")
    out: Dict[str, str] = {}
    total = 0
    for key, value in raw.items():
        path = str(key or "").strip()
        problem = _reject_path(path)
        if problem:
            raise SkillError(f"{where}: {problem}")
        if not isinstance(value, str):
            raise SkillError(
                f"{where}: {path!r} must be text, got {type(value).__name__}"
            )
        if len(value) > MAX_FILE_CHARS:
            raise SkillError(
                f"{where}: {path!r} is {len(value)} characters; at most "
                f"{MAX_FILE_CHARS}"
            )
        total += len(value)
        out[path] = value
    if total > MAX_TOTAL_FILE_CHARS:
        raise SkillError(
            f"{where} totals {total} characters; at most {MAX_TOTAL_FILE_CHARS}"
        )
    return out


def _command(raw: Any, where: str, *, required: bool) -> str:
    command = raw.strip() if isinstance(raw, str) else ""
    if not command:
        if required:
            raise SkillError(f"{where} is required: the shell command to run")
        return ""
    if len(command) > MAX_COMMAND_CHARS:
        raise SkillError(
            f"{where} is {len(command)} characters; at most {MAX_COMMAND_CHARS}. "
            "Put a long script in `files` and call it."
        )
    return command


def _result_fields(raw: Any) -> Dict[str, str]:
    fields = raw.get("fields") if isinstance(raw, Mapping) else None
    if not isinstance(fields, Mapping) or not fields:
        raise SkillError(
            "result.fields is required: an object naming each field "
            f"{RESULT_FILE} must contain and its type "
            f"({', '.join(FIELD_TYPES)}). A skill that declares no result "
            "yields evidence nothing can check."
        )
    if len(fields) > MAX_RESULT_FIELDS:
        raise SkillError(
            f"result.fields has {len(fields)} fields; at most {MAX_RESULT_FIELDS}"
        )
    out: Dict[str, str] = {}
    for key, value in fields.items():
        name = str(key or "").strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", name):
            raise SkillError(
                f"result field {name!r} is not a usable name: letters, digits "
                "and underscore, starting with a letter"
            )
        kind = str(value or "").strip().lower()
        if kind not in FIELD_TYPES:
            raise SkillError(
                f"result field {name!r} has type {value!r}; it must be one of "
                f"{', '.join(FIELD_TYPES)}"
            )
        out[name] = kind
    return out


def validate_skill(raw: Any, *, known_images: Iterable[str]) -> Dict[str, Any]:
    """The skill, normalised, or a :class:`SkillError` saying why not.

    ``known_images`` is every image a skill may currently name: the server's
    allowlist plus authored images that have been built. It is a parameter
    because the second half lives in the database and this module does not.
    """
    if not isinstance(raw, Mapping):
        raise SkillError(f"a skill must be an object, got {type(raw).__name__}")

    unknown = sorted(str(k) for k in raw if str(k) not in KNOWN_KEYS)
    if unknown:
        raise SkillError(
            f"unknown key(s) {', '.join(unknown)}. A skill may carry only: "
            f"{', '.join(KNOWN_KEYS)}"
        )

    slug = _text(raw.get("id"))
    if not SLUG_PATTERN.match(slug):
        raise SkillError(
            f"id {slug!r} is not usable: 2 to 32 characters, lowercase letters, "
            "digits and underscore, starting with a letter. It names the "
            f"evidence the skill yields ({EVIDENCE_PREFIX}<id>)."
        )

    name = _text(raw.get("name"))
    if not name or len(name) > 200:
        raise SkillError("name is required, at most 200 characters")

    description = _text(raw.get("description"))
    if len(description) < MIN_DESCRIPTION_CHARS:
        raise SkillError(
            "description is required: one or two sentences saying WHEN to use "
            "this skill. It is all an agent sees before deciding to load it."
        )
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise SkillError(
            f"description is {len(description)} characters; at most "
            f"{MAX_DESCRIPTION_CHARS}. The detail belongs in `procedure`."
        )

    procedure = raw.get("procedure")
    procedure = procedure.strip() if isinstance(procedure, str) else ""
    if not procedure:
        raise SkillError(
            "procedure is required: the steps an agent follows, as text. "
            "Without it a skill is an image and a result shape, and the agent "
            "has nothing to carry out."
        )
    if len(procedure) > MAX_PROCEDURE_CHARS:
        raise SkillError(
            f"procedure is {len(procedure)} characters; at most "
            f"{MAX_PROCEDURE_CHARS}"
        )

    images = sorted({str(i).strip() for i in known_images if str(i).strip()})
    image = _text(raw.get("image"))
    if image not in images:
        raise SkillError(
            f"image {image!r} may not be used. A skill runs in an image the "
            "server already allows"
            + (f": {', '.join(images)}" if images else ", and none is allowed")
            + ". For a toolchain none of them has, propose a skill image and "
            "have an administrator build it."
        )

    files = _files(raw.get("files"), "files")
    if RESULT_FILE in files or f"{SKILL_DIR}/{RESULT_FILE}" in files:
        raise SkillError(
            f"files may not include {RESULT_FILE}: it is what a run produces, "
            "and shipping one would make every run look like it had a result"
        )

    fields = _result_fields(raw.get("result"))
    judge_command = _command(raw.get("judge_command"), "judge_command", required=False)

    control = raw.get("control")
    if not isinstance(control, Mapping):
        raise SkillError(
            "control is required: {command, files?} -- the smallest invocation "
            f"that should work and leave a valid {RESULT_FILE}. It is what a "
            "dry run executes; a skill without one can never be shown to work."
        )
    unknown_control = sorted(str(k) for k in control if k not in ("command", "files"))
    if unknown_control:
        raise SkillError(
            f"control has unknown key(s) {', '.join(unknown_control)}; it may "
            "carry only command and files"
        )
    control_out: Dict[str, Any] = {
        "command": _command(control.get("command"), "control.command", required=True)
    }
    control_files = _files(control.get("files"), "control.files")
    if control_files:
        control_out["files"] = control_files

    timeout_raw = raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout_raw, bool) or not isinstance(timeout_raw, (int, float)):
        raise SkillError("timeout_seconds must be a number of seconds")
    timeout = int(timeout_raw)
    if not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
        raise SkillError(
            f"timeout_seconds is {timeout}; it must be between "
            f"{MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
        )

    perishable = raw.get("perishable", False)
    if not isinstance(perishable, bool):
        raise SkillError(
            "perishable must be true or false: true when the result describes "
            "files that a later change invalidates"
        )

    manifest: Dict[str, Any] = {
        "id": slug,
        "name": name,
        "description": description,
        "image": image,
        "procedure": procedure,
        "files": files,
        "result": {"fields": fields},
        "control": control_out,
        "timeout_seconds": timeout,
    }
    if judge_command:
        manifest["judge_command"] = judge_command
    if perishable:
        manifest["perishable"] = True
    return manifest


def _matches(kind: str, value: Any) -> bool:
    if kind == "number":
        # bool is an int in Python, and NaN/inf are numbers no bound can hold.
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, Mapping)
    return False


def result_problems(manifest: Mapping[str, Any], payload: Any) -> List[str]:
    """Everything wrong with a result, against the fields the skill declared.

    The whole list rather than the first, for the same reason the pipeline
    checker is not fail-fast: fixing one field only to be told about the next
    is the slow way to find out a result is unusable.
    """
    if not isinstance(payload, Mapping):
        return [f"{RESULT_FILE} must be a JSON object, got {type(payload).__name__}"]
    fields = (manifest.get("result") or {}).get("fields") or {}
    problems: List[str] = []
    for name, kind in fields.items():
        if name not in payload:
            problems.append(f"{RESULT_FILE} has no {name!r} ({kind})")
        elif not _matches(kind, payload[name]):
            problems.append(
                f"{RESULT_FILE} field {name!r} should be a {kind}, got "
                f"{type(payload[name]).__name__}"
                + (
                    " that is not finite"
                    if kind == "number"
                    and isinstance(payload[name], float)
                    and not math.isfinite(payload[name])
                    else ""
                )
            )
    return problems
