"""What an autonomous run can do with sandbox skills.

Four handlers behind four specs (`agent_core/tool_specs/skills.py`): find a
skill, read it, run a command in its sandbox, propose another. They are plain
functions taking the dispatcher's `(params, ctx)` so the provider in
`agent_tool_dispatch` stays a table.

How each outcome is reported is deliberate, because the failure-diagnosis
machinery reads it:

* **Could not run** -- sandbox disabled, image missing, daemon unreachable --
  is an ``error`` with the cause, worded to say that no edit to the command
  can help. An unreachable daemon carries Docker's own words, which
  `agent_failure_diagnosis.could_not_run` recognises and escalates on the
  first attempt; that predicate is narrow on purpose and is not widened here,
  so a disabled sandbox relies on its message saying it is a server setting.
* **Ran and exited non-zero** is an ``error`` carrying the output: the command
  is what the run should change, and the stderr tail is the remedy.
* **Ran, but the result was not accepted** is an ``error`` naming each field
  that was missing or mistyped. It is not a finding. A result that fails its
  own skill's declared shape is not weak evidence; it is none.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.services import (
    sandbox_skill_manifest,
    sandbox_skill_runtime,
    sandbox_skill_service,
)
from app.services.sandbox_skill_manifest import SkillError

#: Finding keys a result's own fields may not overwrite. A skill declaring a
#: field called `type` would otherwise rename its own evidence.
RESERVED_FINDING_KEYS = frozenset(
    {
        "type",
        "skill",
        "subject",
        "title",
        "image",
        "judged_by",
        "command",
        "perishable",
        "inherited",
        "inherited_from_job_id",
    }
)


def _user_id(ctx: Any) -> Any:
    return getattr(getattr(ctx, "job", None), "user_id", None) or getattr(
        ctx, "user_id", None
    )


def _workdir(ctx: Any):
    """Where this caller works: a job's directory, or a conversation's.

    A job's is seeded from its parent stage if it has one. Chat has no job, so
    the conversation is what persists between calls -- "build it, now measure
    it" two messages apart must find the build. A call with neither gets a
    directory per user rather than one shared by everybody.
    """
    job = getattr(ctx, "job", None)
    if getattr(job, "id", None):
        return sandbox_skill_runtime.run_dir(
            str(job.id), getattr(job, "parent_job_id", None)
        )
    extra = getattr(ctx, "extra", None) or {}
    conversation = extra.get("conversation_id") if isinstance(extra, dict) else None
    if conversation:
        return sandbox_skill_runtime.run_dir(f"chat-{conversation}")
    return sandbox_skill_runtime.run_dir(f"user-{_user_id(ctx)}")


def _directory_view(workdir: Any) -> Dict[str, Any]:
    view = sandbox_skill_runtime.list_files(workdir)
    note = sandbox_skill_runtime.inheritance_note(workdir)
    if note:
        view["inherited"] = note
    return view


async def _unknown_skill(ctx: Any, wanted: str) -> Dict[str, Any]:
    active = await sandbox_skill_service.active_skills(ctx.db, _user_id(ctx))
    names = ", ".join(skill.slug for skill in active)
    return {
        "error": (
            f"No active sandbox skill is called {wanted!r}. "
            + (
                f"Active skills: {names}."
                if names
                else "There are no active skills at all, so a contract "
                "requiring skill evidence cannot be met by this run; say so "
                "rather than working around it."
            )
        )
    }


async def list_sandbox_skills(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    active = await sandbox_skill_service.active_skills(ctx.db, _user_id(ctx))
    return {
        "success": True,
        "data": {
            "skills": [sandbox_skill_service.describe_for_run(s) for s in active],
            "note": (
                "Call load_sandbox_skill for the procedure before running one."
                if active
                else "No skills are active. One can be proposed with "
                "propose_sandbox_skill, but a person has to activate it."
            ),
        },
    }


async def load_sandbox_skill(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    wanted = str(params.get("skill") or "").strip()
    skill = await sandbox_skill_service.get_active(ctx.db, _user_id(ctx), wanted)
    if skill is None:
        return await _unknown_skill(ctx, wanted)
    manifest = skill.manifest or {}
    judge = str(manifest.get("judge_command") or "")
    return {
        "success": True,
        "data": {
            **sandbox_skill_service.describe_for_run(skill),
            "procedure": manifest.get("procedure"),
            "files": {
                f"{sandbox_skill_manifest.SKILL_DIR}/{path}": content
                for path, content in (manifest.get("files") or {}).items()
            },
            "result_fields": (manifest.get("result") or {}).get("fields") or {},
            "result_file": sandbox_skill_manifest.RESULT_FILE,
            "judged_by": "judge_command" if judge else "command",
            "perishable": bool(manifest.get("perishable")),
            "how_the_result_is_decided": (
                "When you pass collect_result=true the skill's judge runs "
                f"after your command ({judge}) and writes "
                f"{sandbox_skill_manifest.RESULT_FILE}; anything your own "
                "command wrote there is discarded."
                if judge
                else "This skill has no judge, so your command must write "
                f"{sandbox_skill_manifest.RESULT_FILE} itself, with every "
                "field listed in result_fields."
            ),
            "control_command": (manifest.get("control") or {}).get("command"),
            "timeout_seconds": manifest.get("timeout_seconds"),
            # What is already there. A later pipeline stage starts with a copy
            # of the previous stage's files, and is only spared rebuilding
            # them if it is told they exist.
            "working_directory": _directory_view(_workdir(ctx)),
        },
    }


async def run_sandbox_skill(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    wanted = str(params.get("skill") or "").strip()
    skill = await sandbox_skill_service.get_active(ctx.db, _user_id(ctx), wanted)
    if skill is None:
        return await _unknown_skill(ctx, wanted)

    files = params.get("files")
    problem = sandbox_skill_runtime.reject_caller_files(files)
    if problem:
        return {"error": problem}

    manifest = skill.manifest or {}
    image = str(manifest.get("image") or "")
    collect = bool(params.get("collect_result"))
    command = str(params.get("command") or "")

    sandbox_skill_runtime.prune_stale()
    workdir = _workdir(ctx)
    run = await sandbox_skill_runtime.execute(
        manifest,
        workdir=workdir,
        command=command,
        files=files or None,
        collect_result=collect,
        timeout_seconds=params.get("timeout_seconds"),
        image_allowed=image in await sandbox_skill_service.known_images(ctx.db),
    )

    data: Dict[str, Any] = {
        "skill": skill.slug,
        "image": image,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "working_directory": _directory_view(workdir),
    }

    if not run.ran or run.timed_out:
        return {"error": run.error, "data": data}

    if run.returncode != 0:
        tail = (run.stderr or run.stdout)[-600:].strip()
        return {
            "error": f"The command exited {run.returncode}"
            + (f": {tail}" if tail else "."),
            "data": data,
        }

    if not collect:
        data["note"] = (
            "Ran. No result was collected; pass collect_result=true on the "
            "call that should count."
        )
        return {"success": True, "data": data}

    if run.result is None:
        return {
            "error": "The command ran, but no result was recorded: "
            + "; ".join(run.result_problems),
            "data": {
                **data,
                "result_fields": (manifest.get("result") or {}).get("fields") or {},
            },
        }

    label = str(params.get("label") or "").strip()
    finding_type = sandbox_skill_manifest.evidence_type(skill.slug)
    finding: Dict[str, Any] = {
        **{k: v for k, v in run.result.items() if k not in RESERVED_FINDING_KEYS},
        "type": finding_type,
        "skill": skill.slug,
        "subject": label or skill.name,
        "title": f"{skill.name}" + (f": {label}" if label else ""),
        "image": image,
        # Who wrote the accepted result. A judge shipped with the skill and a
        # run reporting on itself are not equally strong, and a reader of the
        # finding should not have to open the skill to learn which this was.
        "judged_by": run.judged_by,
        "command": command[:500],
    }
    if manifest.get("perishable"):
        # Said on the finding, because that is what the inheritance and
        # bounds checks read: the evidence map is fixed at import and cannot
        # know that this user's skill declared its result perishable.
        finding["perishable"] = True
    data["result"] = run.result
    data["judged_by"] = run.judged_by
    data["recorded_as"] = finding_type
    return {"success": True, "data": data, "findings": [finding]}


async def propose_sandbox_skill(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    from app.core.config import settings

    if not bool(getattr(settings, "SANDBOX_SKILLS_AUTHORING_ENABLED", True)):
        return {
            "error": "Authoring sandbox skills is disabled on this deployment "
            "(SANDBOX_SKILLS_AUTHORING_ENABLED=false). This is a server "
            "setting and retrying will fail the same way."
        }

    job = getattr(ctx, "job", None)
    job_id = getattr(job, "id", None)
    already = await sandbox_skill_service.agent_drafts_for_job(ctx.db, job_id)
    if already >= sandbox_skill_service.MAX_AGENT_DRAFTS_PER_JOB:
        return {
            "error": (
                f"This run has already proposed {already} skills, which is the "
                "limit. Each one is a draft somebody has to read; propose the "
                "ones worth their time."
            )
        }
    # Chat has no job to count against, so the bound there is the review
    # queue itself: proposals nobody has yet looked at.
    waiting = await sandbox_skill_service.unreviewed_agent_drafts(ctx.db, _user_id(ctx))
    if job_id is None and waiting >= sandbox_skill_service.MAX_UNREVIEWED_AGENT_DRAFTS:
        return {
            "error": (
                f"{waiting} proposed skills are already waiting to be reviewed, "
                "which is the limit. Review or delete some in the Sandbox "
                "Skills panel before proposing more."
            )
        }

    control: Dict[str, Any] = {"command": params.get("control_command")}
    if params.get("control_files"):
        control["files"] = params.get("control_files")
    raw: Dict[str, Any] = {
        "id": params.get("id"),
        "name": params.get("name"),
        "description": params.get("description"),
        "image": params.get("image"),
        "procedure": params.get("procedure"),
        "files": params.get("files") or {},
        "result": {"fields": params.get("result_fields")},
        "control": control,
    }
    if params.get("judge_command"):
        raw["judge_command"] = params.get("judge_command")
    if "perishable" in params:
        raw["perishable"] = params.get("perishable")

    notes: List[str] = [
        (
            f"Proposed by run {job_id}"
            + (f" ({getattr(job, 'name', '')})" if getattr(job, "name", "") else "")
            + "."
        )
        if job_id
        else "Proposed by the assistant in a chat."
    ]
    why = str(params.get("why") or "").strip()
    if why:
        notes.append(f"The run's reason: {why[:600]}")

    try:
        skill = await sandbox_skill_service.create_skill(
            ctx.db,
            user_id=_user_id(ctx),
            raw=raw,
            origin="agent",
            origin_job_id=job_id,
            notes=notes,
        )
    except SkillError as exc:
        # Verbatim: the refusal names what is wrong, which is what the next
        # attempt needs.
        return {"error": f"The skill was refused: {exc}"}

    return {
        "success": True,
        "data": {
            "skill": skill.slug,
            "status": skill.status,
            "note": (
                "Stored as a draft. A person has to run its control and "
                "activate it before any run is offered it, so it is not "
                "available to you and is not evidence of anything."
            ),
        },
    }
