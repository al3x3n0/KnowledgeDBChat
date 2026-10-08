"""The stats response keeps every document count the stats task produces.

`GET /admin/stats` answers through `SystemStatsResponse`, and a response model
drops any key it does not declare. The task counted documents without a
summary; the schema did not name the field, so the dashboard line reading it
never rendered and nothing failed. Found by the frontend's type-drift check.
"""

import ast
from pathlib import Path

import pytest

from app.schemas.admin import DocumentStatsResponse, SystemStatsResponse

pytestmark = pytest.mark.unit

TASK = Path(__file__).resolve().parents[1] / "app/tasks/monitoring_tasks.py"


def _document_keys_the_task_writes() -> set:
    keys = set()
    for node in ast.walk(ast.parse(TASK.read_text(encoding="utf-8"))):
        # stats["documents"]["<key>"] = ...
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value == "documents"
            and isinstance(node.slice, ast.Constant)
        ):
            keys.add(node.slice.value)
        # stats["documents"] = {"<key>": ...}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "documents"
            ):
                keys.update(
                    k.value for k in node.value.keys if isinstance(k, ast.Constant)
                )
    return keys


def test_every_document_count_the_task_writes_is_declared():
    written = _document_keys_the_task_writes()
    assert "without_summary" in written, "the scan no longer sees the task's keys"
    assert written <= set(DocumentStatsResponse.model_fields), written - set(
        DocumentStatsResponse.model_fields
    )


def test_the_count_survives_the_response_model():
    documents = {
        "total": 5,
        "processed": 3,
        "failed": 0,
        "pending": 2,
        "success_rate": 60.0,
        "without_summary": 2,
    }
    body = SystemStatsResponse(
        timestamp="2026-10-08T00:00:00", documents=documents
    ).model_dump()
    assert body["documents"]["without_summary"] == 2
