"""An evolutionary search over rewrites of a kernel, judged like every rewrite.

`propose_restructurings` asks for independent ideas, each from the original,
and repairs a failed one once. That finds an idea; it does not build on one.
ucb-bar/chia's gem5 alignment loop and its AlphaEvolve-style discovery flows
do the other thing: keep an archive of candidates with a fitness, sample the
next parent from the best few, ask a model to improve that parent, score the
child, and keep going -- a failed build scores 0 and the search continues.
This is that loop, over `evaluate_restructuring`, and three things in it are
not chia's because this machine and this codebase measured otherwise:

  CHILDREN ARE TIMED ONE AT A TIME. chia evaluates N candidates at once on N
  machines. Here every candidate is timed on one host, and a run timed beside
  another measures the other (the sandbox slot cap exists for that reason).
  The search is sequential, and its cost is said up front.

  THE WINNER MUST PASS INPUTS THE SEARCH NEVER SAW. The proposer's own rule is
  "one repair, not a loop": iterating until something passes selects for
  whatever slips past these particular inputs. A search is that loop on
  purpose, so the inputs it optimises against are not the inputs that decide.
  `holdout_inputs` are never shown to the model and never used for fitness;
  the best candidates are re-judged on them, and one that diverges there is
  reported as fitting the search inputs, not as a result.

  NOISE IS NOT FITNESS. An `unresolved` candidate's median ratio is whatever
  the host's load did that minute; selecting on it would breed luck. An
  equivalent candidate whose gain did not resolve scores 1.0, the original's
  own fitness, and only a resolved verdict carries its measured median.

The archive is returned and, through the tool, kept in the run's state, so a
second call continues the search where the first stopped -- chia's loop
resumes from its database the same way.
"""

from __future__ import annotations

import random
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.services.agent_compiler_sandbox import _clean_flags
from app.services.agent_restructure import (
    DEFAULT_FLAGS,
    DEFAULT_IMAGE,
    check_inputs,
    evaluate_restructuring,
    sandbox_blocked,
)
from app.services.agent_restructure_proposer import (
    SOURCE_SCHEMA,
    _call,
    _first_proposal,
    _source_system_prompt,
    compiler_listing,
)

MAX_GENERATIONS = 5
DEFAULT_GENERATIONS = 3
MAX_CHILDREN = 3
DEFAULT_CHILDREN = 2
#: Parents are sampled from this many of the best (chia samples from 2).
TOP_K = 2
#: Generations without a better candidate before the search stops early.
PATIENCE = 2
#: Candidates re-judged on the held-out inputs, best first, until one holds.
HOLDOUT_TRIES = 3
#: Archive entries a call returns in full; the rest are summarised.
_SHOWN = 12
#: Entries kept to continue from: the original and the fittest. Each holds a
#: whole kernel and the archive lives in the run's checkpointed state.
_KEPT = 16

#: Verdicts whose measured median is a real gain or loss. Anything else that
#: is equivalent scores the original's 1.0; anything not equivalent scores 0.
_RESOLVED = ("faster", "faster_than_original", "compiler_already_can", "slower")
_FAILED = ("did_not_compile", "crashed", "diverged", "baseline_broken", "error")
#: Best first, as the proposer ranks.
_RANK = {
    "faster": 0,
    "faster_than_original": 1,
    "compiler_already_can": 2,
    "unresolved": 3,
    "slower": 4,
}

#: A `static` that is not const and not a function: state a kernel keeps
#: between calls.
_MUTABLE_STATIC = re.compile(r"\bstatic\s+(?!const\b)(?!inline\b)[^;(){}]*[;=]")

Evaluate = Callable[..., Awaitable[Dict[str, Any]]]
Propose = Callable[[str], Awaitable[Optional[Dict[str, Any]]]]


def fitness(verdict: Optional[str], median_speedup: Any) -> float:
    """What a candidate is worth as a parent; see the module docstring."""
    if verdict in _FAILED or verdict is None:
        return 0.0
    if verdict in _RESOLVED:
        try:
            return max(0.0, float(median_speedup))
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def _entry(
    name: str,
    idea: str,
    kernel: str,
    *,
    generation: int,
    parent: Optional[str],
    result: Optional[Dict[str, Any]] = None,
    proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = (result or {}).get("data") or {}
    timing = data.get("timing") or {}
    verdict = data.get("verdict") or ("error" if (result or {}).get("error") else None)
    if result is None:
        verdict = "original"
    entry = {
        "name": name,
        "idea": idea,
        "kernel": kernel,
        "generation": generation,
        "parent": parent,
        "verdict": verdict,
        "median_speedup": timing.get("median_speedup"),
        "speedup_over_ceiling": timing.get("speedup_over_ceiling"),
        "fitness": 1.0
        if result is None
        else fitness(verdict, timing.get("median_speedup")),
        "problem": data.get("compile_errors")
        or (data.get("equivalence") or {}).get("first_problem")
        or data.get("detail")
        or (result or {}).get("error"),
    }
    if proposal:
        entry["invariant"] = proposal.get("invariant")
        entry["value_preserving"] = proposal.get("value_preserving", True)
    return entry


def kept(archive: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What a later call continues from, in the order the entries were made."""
    original = [e for e in archive if e.get("verdict") == "original"][:1]
    rest = [e for e in archive if e.get("verdict") != "original"]
    best = {id(e) for e in _top(rest, _KEPT - len(original))}
    return original + [e for e in rest if id(e) in best]


def keeps_state(candidate: str, original: str) -> bool:
    """Whether a candidate keeps mutable state between calls the original did not.

    The live search's best candidate cached the whole result in a static,
    keyed on the array pointer, because the driver calls the kernel twenty
    times with the same arguments in order to time it. "Work repeated across
    calls" is a fair invariant when the APPLICATION repeats it, and the
    measurement cannot tell that from the harness repeating it. A lazily
    built lookup table looks the same to this check, so it flags and does
    not refuse.
    """
    count = len(_MUTABLE_STATIC.findall(candidate or ""))
    return count > len(_MUTABLE_STATIC.findall(original or ""))


def _top(archive: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return sorted(archive, key=lambda e: -float(e.get("fitness") or 0.0))[:k]


def _ranked(archive: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (e for e in archive if e["verdict"] in _RANK),
        key=lambda e: (_RANK[e["verdict"]], -float(e.get("fitness") or 0.0)),
    )


def _history(archive: List[Dict[str, Any]]) -> str:
    lines = []
    for e in archive:
        if e["verdict"] == "original":
            continue
        speed = f", median {e['median_speedup']}x" if e.get("median_speedup") else ""
        lines.append(
            f"- {e['name']} (from {e['parent']}): {e['verdict']}{speed}. {e['idea'][:160]}"
        )
    return "\n".join(lines[-20:])


def _ask(base: str, parent: Dict[str, Any], archive: List[Dict[str, Any]]) -> str:
    measured = (
        "the original"
        if parent["verdict"] == "original"
        else f"{parent['verdict']}, median {parent.get('median_speedup')}x over the original"
    )
    tried = _history(archive)
    return (
        base
        + f"\n\n=== the PARENT to improve: {parent['name']} ({measured}) ===\n"
        + parent["kernel"]
        + (
            f"\n\n=== already tried, and what each measured ===\n{tried}"
            if tried
            else ""
        )
        + "\n\nPropose ONE change to the parent that makes it faster still. Return the "
        "complete kernel. Do not repeat an idea above that failed; building on one "
        "that worked is the point. The driver repeats its calls only so it can "
        "be timed: a change that pays because those calls repeat (keeping a "
        "result between calls) measures the harness, not the application."
    )


async def search(
    *,
    kernel: str,
    base_message: str,
    propose: Propose,
    evaluate: Evaluate,
    holdout: Evaluate,
    generations: int = DEFAULT_GENERATIONS,
    children: int = DEFAULT_CHILDREN,
    archive: Optional[List[Dict[str, Any]]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """The loop itself, with the model and the measurements passed in."""
    generations = max(1, min(int(generations or DEFAULT_GENERATIONS), MAX_GENERATIONS))
    children = max(1, min(int(children or DEFAULT_CHILDREN), MAX_CHILDREN))
    archive = [
        dict(e) for e in (archive or []) if isinstance(e, dict) and e.get("kernel")
    ]
    if not archive:
        archive = [
            _entry("original", "the kernel as given", kernel, generation=0, parent=None)
        ]
    start = max(int(e.get("generation") or 0) for e in archive)
    rng = random.Random(seed + start)
    best = max(float(e.get("fitness") or 0.0) for e in archive)
    stale = 0
    failures: List[str] = []
    ran = 0
    names = {e["name"] for e in archive}

    for generation in range(start + 1, start + 1 + generations):
        improved = False
        for _ in range(children):
            parent = rng.choice(_top(archive, TOP_K))
            try:
                proposal = await propose(_ask(base_message, parent, archive))
            except Exception as exc:
                # Every later call would meet the same outage. Measured: with
                # the provider unreachable, retrying each child spent 47
                # minutes and reported "0 judged" as if nothing had won.
                failures.append(
                    f"generation {generation}: the model could not be reached "
                    f"({str(exc).strip() or exc.__class__.__name__})"
                )
                return _finish(archive, failures, ran, [], stopped="model")
            if not proposal or proposal.get("_failure"):
                failures.append(
                    f"generation {generation}: "
                    + str((proposal or {}).get("_failure") or "no usable proposal")
                )
                continue
            name = str(proposal.get("name") or "candidate")[:60]
            while name in names:
                name = f"{name}'"
            names.add(name)
            result = await evaluate(proposal)
            entry = _entry(
                name,
                str(proposal.get("idea") or ""),
                proposal["kernel"],
                generation=generation,
                parent=parent["name"],
                result=result,
                proposal=proposal,
            )
            if keeps_state(entry["kernel"], archive[0]["kernel"]):
                entry["keeps_state_across_calls"] = True
            archive.append(entry)
            if entry["verdict"] == "baseline_broken":
                return {
                    "error": (
                        "the ORIGINAL kernel does not build or run with this driver "
                        f"and these inputs: {entry['problem'] or 'no detail'}"
                    ),
                    "archive": archive,
                }
            if entry["verdict"] == "error" and not (result.get("data")):
                # The comparison could not run at all: no further child
                # would be judged either.
                failures.append(f"generation {generation}: {entry['problem']}")
                return _finish(archive, failures, ran + 1, [], stopped="sandbox")
            ran += 1
            if entry["fitness"] > best:
                best, improved = entry["fitness"], True
        stale = 0 if improved else stale + 1
        if stale >= PATIENCE:
            break

    checked: List[Dict[str, Any]] = []
    winner = None
    for candidate in [
        e
        for e in _ranked(archive)
        if e["verdict"] in ("faster", "faster_than_original")
    ][:HOLDOUT_TRIES]:
        result = await holdout(candidate)
        data = result.get("data") or {}
        record = {
            "name": candidate["name"],
            "verdict": data.get("verdict")
            or ("error" if result.get("error") else None),
            "median_speedup": (data.get("timing") or {}).get("median_speedup"),
            "findings": result.get("findings") or [],
        }
        checked.append(record)
        if record["verdict"] in ("faster", "faster_than_original"):
            winner = candidate
            break
    return _finish(archive, failures, ran, checked, winner=winner)


def _finish(
    archive: List[Dict[str, Any]],
    failures: List[str],
    ran: int,
    checked: List[Dict[str, Any]],
    *,
    winner: Optional[Dict[str, Any]] = None,
    stopped: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "archive": archive,
        "candidates_judged": ran,
        "holdout": [{k: v for k, v in c.items() if k != "findings"} for c in checked],
        "winner": winner,
        "findings": [f for c in checked for f in c["findings"]],
    }
    if failures:
        out["proposal_failures"] = failures
    if stopped:
        out["stopped"] = stopped
    return out


async def search_restructurings(
    *,
    kernel: str,
    driver: str,
    inputs: List[str],
    holdout_inputs: List[str],
    focus: str = "",
    generations: int = DEFAULT_GENERATIONS,
    children: int = DEFAULT_CHILDREN,
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    label: str = "",
    archive: Optional[List[Dict[str, Any]]] = None,
    seed: int = 0,
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Search for a faster kernel, and believe only what held-out inputs confirm."""
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    held_problem, held = check_inputs(holdout_inputs, 0)
    if held_problem or not held:
        return {
            "error": (
                "holdout_inputs are required: inputs the search never sees, on "
                "which its best candidates are judged again. A search that is "
                "judged on the inputs it optimised against selects for whatever "
                "passes those inputs. " + (held_problem or "")
            ).strip()
        }
    if set(held) <= set(cleaned):
        return {
            "error": (
                "every holdout input is also a search input, so the holdout "
                "would confirm only what the search already optimised against. "
                "Give at least one input the search does not use."
            )
        }
    if not (kernel or "").strip() or not (driver or "").strip():
        return {"error": "kernel and driver are both required"}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    blocked = sandbox_blocked(DEFAULT_IMAGE)
    if blocked:
        return {"error": blocked}

    subject = (label or "").strip() or "kernel"
    listing = await compiler_listing(kernel, safe)
    base = (
        "Make this application's kernel faster with an application-specific "
        "change, building on what has already been measured.\n\n"
        + (f"Where to look: {focus}\n\n" if focus else "")
        + f"=== the original kernel.c ===\n{kernel}\n\n=== driver.c (fixed) ===\n"
        f"{driver}\n\n=== an input (bench) ===\n{cleaned[bench_input][:2000]}\n\n"
        + (
            f"=== what clang {safe} made of the original ===\n{listing}\n"
            if listing
            else ""
        )
    )

    async def propose(message: str) -> Optional[Dict[str, Any]]:
        # An unreachable model raises out of here and ends the search.
        payload = await _call(
            _source_system_prompt(), message, SOURCE_SCHEMA, user_id=user_id, db=db
        )
        found = _first_proposal(payload, "kernel")
        if found:
            return found
        return {
            "_failure": payload.get("_unparsed")
            or f"no proposal with a kernel; the reply's keys were {sorted(payload)[:12]}"
        }

    async def evaluate(proposal: Dict[str, Any]) -> Dict[str, Any]:
        # Always against the ORIGINAL, so every fitness is on one scale.
        return await evaluate_restructuring(
            kernel=kernel,
            candidate=proposal["kernel"],
            driver=driver,
            inputs=cleaned,
            value_preserving=proposal.get("value_preserving", True),
            invariant=str(proposal.get("invariant") or ""),
            flags=safe,
            bench_input=bench_input,
            label=f"{subject}/search/{proposal.get('name') or 'candidate'}",
        )

    async def holdout(candidate: Dict[str, Any]) -> Dict[str, Any]:
        # Equivalence on the held-out inputs, timing on the search's own
        # bench input. Timed on the first held-out input instead, the live
        # run's holdout timed a 5-element array -- the edge case the holdout
        # was given to check -- and read a 1.86x win as 0.66x.
        return await evaluate_restructuring(
            kernel=kernel,
            candidate=candidate["kernel"],
            driver=driver,
            inputs=held + [cleaned[bench_input]],
            value_preserving=candidate.get("value_preserving", True),
            invariant=str(candidate.get("invariant") or ""),
            flags=safe,
            bench_input=len(held),
            label=f"{subject}/holdout/{candidate['name']}",
        )

    out = await search(
        kernel=kernel,
        base_message=base,
        propose=propose,
        evaluate=evaluate,
        holdout=holdout,
        generations=generations,
        children=children,
        archive=archive,
        seed=seed,
    )
    return as_result(out, subject)


def as_result(out: Dict[str, Any], subject: str) -> Dict[str, Any]:
    """The tool's answer: what won on held-out inputs, and the archive behind it."""
    archive = out.get("archive") or []
    shown = [{k: v for k, v in e.items() if k != "kernel"} for e in archive[-_SHOWN:]]
    if out.get("error"):
        return {
            "success": False,
            "error": out["error"],
            "data": {"archive": shown},
            "_archive": kept(archive),
        }
    if not out.get("candidates_judged") and out.get("proposal_failures"):
        # Nothing was judged, so nothing was found or ruled out. As a
        # success this read "no candidate beat the original".
        return {
            "success": False,
            "error": "no candidate was judged: "
            + "; ".join(out["proposal_failures"][:3]),
            "data": {"archive": shown, "stopped": out.get("stopped")},
            "_archive": kept(archive),
        }
    winner = out.get("winner")
    held = out.get("holdout") or []
    if winner:
        confirmed = held[-1]
        summary = (
            f"{winner['name']} is faster than the original {subject} on inputs the "
            f"search never saw (median {confirmed.get('median_speedup')}x held out, "
            f"{winner.get('median_speedup')}x during the search), after "
            f"{out['candidates_judged']} candidate(s)."
            + (
                " It keeps state between calls: check that what it keeps is "
                "not keyed on the driver repeating the same call to time it."
                if winner.get("keeps_state_across_calls")
                else ""
            )
        )
    elif held:
        summary = (
            "No candidate held on the held-out inputs: "
            + "; ".join(f"{h['name']} was {h['verdict']}" for h in held)
            + ". A win on the search inputs alone is a fit to those inputs."
        )
    else:
        summary = (
            f"No candidate beat the original {subject} during the search "
            f"({out['candidates_judged']} judged)."
        )
    data = {
        "summary": summary,
        "winner": dict(winner) if winner else None,
        "holdout": held,
        "candidates_judged": out.get("candidates_judged", 0),
        "archive": shown,
        "how_to_read": (
            "Fitness is the median speedup over the original for a resolved "
            "verdict, 1.0 for an equivalent candidate whose gain did not "
            "resolve, 0 for one that failed. Only a candidate confirmed on the "
            "held-out inputs is a result; the archive is the search's history. "
            "Call again to continue from it."
        ),
    }
    for key in ("proposal_failures", "stopped"):
        if out.get(key):
            data[key] = out[key]
    # `_archive` is the whole archive, kernels included, for the caller to
    # keep and pass back; the tool strips it before the model sees the reply.
    return {
        "success": True,
        "data": data,
        "findings": out.get("findings") or [],
        "_archive": kept(archive),
    }
