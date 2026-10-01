"""Drafting a sandbox skill from a description, and running it before anyone sees it.

A skill has one closed vocabulary -- the images it may run in -- and one thing
no model can know in advance: whether the commands it wrote actually work in
that image. A toolchain that is not installed, a helper script with a typo, a
judge that writes `speed_up` where the skill declared `speedup`: each validates
perfectly and fails the first time a run uses it, in a way that reads as the
run's mistake.

So, as with plugins, this does two things rather than one.

**It repairs against the real validator.** `validate_skill` decides here
exactly as it decides on create, and its refusal goes back to the model
verbatim. Nothing in this module restates a rule.

**It runs the control it wrote.** A draft that validates is dry-run in the
sandbox, and when the control fails, what the sandbox said goes back to the
model. That is safe for the reason a transform is safe to dry-run in the
plugin drafter and a webhook is not: the sandbox has no network and no
capabilities, so running an unreviewed draft reaches nothing.

When the sandbox cannot run at all -- execution disabled, no daemon -- the
draft is returned *unverified and saying so*. Looping on a failure no edit can
fix would spend three model calls to learn nothing.

Passing ``current`` makes it a **revision**: the instruction is applied to the
skill the author has in the editor, not to the model's memory of its own last
answer, so a hand edit between passes is kept. A revision keeps its id, since
the evidence a skill yields is named after it.

Drafting never stores anything. The skill comes back for review with ``notes``
saying what had to be repaired.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from loguru import logger

from app.services import (
    sandbox_skill_manifest,
    sandbox_skill_runtime,
    sandbox_skill_service,
)
from app.services.plugin_author_service import _payload
from app.services.sandbox_skill_manifest import SkillError

#: The same bound the plugin drafter uses, for the same observed reason: the
#: mistakes that happen are fixed by the third attempt or not at all.
MAX_ATTEMPTS = 3

DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "image": {"type": "string"},
        "procedure": {"type": "string"},
        "files": {"type": "object"},
        "result": {
            "type": "object",
            "properties": {"fields": {"type": "object"}},
        },
        "judge_command": {"type": "string"},
        "control": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "files": {"type": "object"},
            },
        },
        "timeout_seconds": {"type": "integer"},
    },
    "required": [
        "id",
        "name",
        "description",
        "image",
        "procedure",
        "result",
        "control",
    ],
}


def _system_prompt(images: List[str]) -> str:
    m = sandbox_skill_manifest
    return f"""You write sandbox skills for an autonomous research agent.

A skill packages ONE kind of sandboxed work: a procedure the agent follows,
helper files, the image it runs in, and the result it must leave behind. The
agent reads the procedure and decides which commands to run; the platform
checks the result. Write for an agent that has never seen this work before.

Return ONE JSON object with these keys and no others:
{', '.join(m.KNOWN_KEYS)}

id          lowercase letters, digits, underscore; starts with a letter; 2-32
            characters. The evidence is named {m.EVIDENCE_PREFIX}<id>.
name        a short human name.
description WHEN to use the skill, in one or two sentences. This is all the
            agent sees before deciding to load it.
image       exactly one of: {', '.join(images) if images else '(none available)'}
procedure   the steps, as text. Say what to write, what to run, what the
            output means, and what commonly goes wrong.
files       {{relative_path: text}} helper files. They appear in the sandbox
            under ./{m.SKILL_DIR}/ -- so a file "judge.py" is run as
            "python3 {m.SKILL_DIR}/judge.py".
result      {{"fields": {{name: type}}}} where type is one of
            {', '.join(m.FIELD_TYPES)}. These are the fields ./{m.RESULT_FILE}
            must contain for a run to count.
judge_command   optional but preferred: a command that computes
            ./{m.RESULT_FILE} from what the agent's run left in the working
            directory. With a judge, the agent is not the author of its own
            result.
control     {{"command": "...", "files": {{...}}}} -- the SMALLEST invocation
            that should succeed and leave a valid ./{m.RESULT_FILE}. It is
            executed to prove the skill works before anyone can use it, so it
            must be self-contained and finish in well under
            {sandbox_skill_runtime.DRY_RUN_TIMEOUT_SECONDS} seconds.
timeout_seconds  optional, {m.MIN_TIMEOUT_SECONDS}-{m.MAX_TIMEOUT_SECONDS}.

Facts about the sandbox that decide whether a skill works:
- There is NO network. Nothing can be downloaded or installed at run time, so
  use only what the image already contains.
- Commands run with /bin/sh in a working directory that is the only writable
  path, as an unprivileged user.
- The working directory persists between the agent's calls for one skill.
- ./{m.SKILL_DIR}/ is restored before every call; do not write results there.
- ./{m.RESULT_FILE} is deleted before every call, and before the judge runs.

Keep it small: a procedure, at most a few short helper files, a handful of
result fields. Output JSON only, no prose and no code fences."""


def _report_to(on_progress: Optional[Callable[[str, int, List[str]], None]]):
    def _report(stage: str, attempt: int, notes: List[str]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(stage, attempt, list(notes))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Skill draft progress callback failed: {exc}")

    return _report


def _control_complaint(outcome: Mapping[str, Any]) -> str:
    """What the sandbox said, in the form a repair can act on."""
    parts = [f"The control did not pass: {outcome.get('detail')}"]
    stdout = str(outcome.get("stdout") or "").strip()
    stderr = str(outcome.get("stderr") or "").strip()
    if stderr:
        parts.append(f"stderr:\n{stderr[-1200:]}")
    if stdout:
        parts.append(f"stdout:\n{stdout[-800:]}")
    return "\n".join(parts)


async def draft_skill(
    description: str,
    *,
    db: Any,
    user_id: Any = None,
    current: Optional[Mapping[str, Any]] = None,
    on_progress: Optional[Callable[[str, int, List[str]], None]] = None,
) -> Dict[str, Any]:
    """Draft a skill, repair it against the validator, and run its control.

    Returns ``{manifest, notes, attempts, dry_run}``. ``manifest`` is None when
    no attempt validated. ``dry_run`` is the last control outcome, or None when
    none was attempted -- a draft with ``dry_run.ok`` false is still returned,
    because a flaw a person can see is worth more than nothing.
    """
    from app.services.llm_service import LLMService

    report = _report_to(on_progress)
    text = str(description or "").strip()
    if not text:
        return {
            "manifest": None,
            "notes": ["Describe what the skill should do."],
            "attempts": 0,
            "dry_run": None,
        }

    images = await sandbox_skill_service.known_images(db)
    system = _system_prompt(images)
    revising = isinstance(current, Mapping) and bool(current)
    if revising:
        message = (
            "Revise this skill as asked, and return the whole skill. Keep its "
            f"id ({current.get('id')!r}) unchanged.\n\nThe skill now:\n"
            f"{json.dumps(dict(current), indent=2)}\n\nThe change wanted:\n{text}"
        )
    else:
        message = f"Write a sandbox skill for this request:\n\n{text}"

    llm = LLMService()
    notes: List[str] = []
    manifest: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None
    attempt = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        report("drafting", attempt, notes)
        try:
            completion = await llm.generate_structured(
                system_prompt=system,
                user_message=message,
                response_schema=DRAFT_SCHEMA,
                task_type="balanced",
                user_id=user_id,
                db=db,
                snapshot_context={"phase": "sandbox_skill_draft"},
            )
        except Exception as exc:
            logger.warning(f"Skill draft call failed on attempt {attempt}: {exc}")
            notes.append(f"The model could not be reached: {exc}")
            break

        payload = _payload(completion)
        if not payload:
            notes.append(f"Attempt {attempt}: the reply was not JSON.")
            message = (
                f"{message}\n\nYour last reply was not a JSON object. Reply "
                "with one JSON object and nothing else."
            )
            continue

        try:
            candidate = sandbox_skill_manifest.validate_skill(
                payload, known_images=images
            )
            if revising and candidate["id"] != str(current.get("id")):
                raise SkillError(
                    f"id changed from {current.get('id')!r} to "
                    f"{candidate['id']!r}; a revision keeps its id"
                )
        except SkillError as exc:
            notes.append(f"Attempt {attempt}: {exc}")
            message = (
                f"{message}\n\nYour last skill was rejected:\n{exc}\n\n"
                "Fix exactly that and return the whole skill again."
            )
            continue

        manifest = candidate
        report("checking", attempt, notes)
        outcome = await sandbox_skill_runtime.dry_run(manifest, image_allowed=True)
        if outcome["ok"]:
            break
        if not outcome.get("ran"):
            # Nothing could be tested, so there is nothing to repair. Say that
            # the draft is unverified and why, and stop spending model calls.
            notes.append(
                "The control was not run, so this draft is unverified: "
                f"{outcome.get('detail')}"
            )
            break

        complaint = _control_complaint(outcome)
        notes.append(f"Attempt {attempt}: {outcome.get('detail')}")
        message = (
            f"{message}\n\nThe skill validates, but running its control in the "
            f"sandbox showed:\n{complaint}\n\nFix the skill so its control "
            "passes and return the whole skill again. Remember there is no "
            "network and only what the image already contains is available."
        )
        # The manifest is kept: a control that fails is a flaw a person can
        # read and fix, and handing back nothing would be worse.

    report("done", attempt, notes)
    return {
        "manifest": manifest,
        "notes": notes,
        "attempts": attempt,
        "dry_run": outcome,
    }
