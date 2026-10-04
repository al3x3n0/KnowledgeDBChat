"""A key read from a job's results is one something writes.

`job.results` is written only by the system -- nobody types it -- so a key
read from it and written nowhere is a reader waiting for a writer that does
not exist. The run's library record read `results["methods"]` and listed
methods verbatim; `record_method` stored a memory and a finding and never that
key, so the section never rendered, and its own test set the key by hand.
The pipeline spawn threshold was the same shape in `chain_config`.

Read from source with the AST, like the other scans in this directory, so a
missing optional dependency cannot hide a module.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"
RESULTS = re.compile(r"job\.results|\bresults\b|parent_results|stored_results")

#: Read from something named `results` that is not a job's, or written by a
#: helper that takes the key as an argument. Each says which.
NOT_A_JOB_RESULT_GAP = {
    # A vector store's own query result (Chroma's `distances`).
    "distances": "services/vector_store.py",
    # Written by `_upsert(parent_results, "verification_reconciliations", ...)`.
    "verification_reconciliations": (
        "services/autonomous_rnd_verification_reconciliation_service.py"
    ),
}


def _scan():
    reads, writes = {}, set()
    for path in APP.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        rel = str(path.relative_to(APP))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                key = node.args[0].value
                if node.func.attr == "setdefault":
                    writes.add(key)
                elif RESULTS.search(ast.unparse(node.func.value)):
                    reads.setdefault(key, []).append(f"{rel}:{node.lineno}")
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                key = node.slice.value
                if isinstance(node.ctx, ast.Store):
                    writes.add(key)
                elif RESULTS.search(ast.unparse(node.value)):
                    reads.setdefault(key, []).append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        writes.add(key.value)
            elif isinstance(node, ast.keyword) and node.arg:
                writes.add(node.arg)
    return reads, writes


def test_the_scan_sees_result_reads():
    reads, _ = _scan()
    assert len(reads) > 50, "too few reads found; this guard would pass vacuously"


def test_every_result_key_read_is_written_somewhere():
    reads, writes = _scan()
    orphans = [
        f"{key} (read at {sites[0]})"
        for key, sites in sorted(reads.items())
        if key not in writes and key not in NOT_A_JOB_RESULT_GAP
    ]
    assert not orphans, "Read from job results, written nowhere:\n" + "\n".join(orphans)


def test_the_exceptions_are_still_where_they_say():
    for key, module in NOT_A_JOB_RESULT_GAP.items():
        assert key in (APP / module).read_text(), f"{key} no longer in {module}"
