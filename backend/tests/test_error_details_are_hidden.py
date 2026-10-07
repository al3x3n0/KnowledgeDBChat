"""A 500 response does not carry the exception's text unless asked to.

Sixty-seven routes answered a failure with ``detail=str(e)``, and so did the
handler for anything unhandled: a SQL statement, a file path or an upstream's
reply went to whoever made the request. They now go through
``internal_error_detail``, which gives a reference and logs the exception
under it; ``EXPOSE_ERROR_DETAILS`` restores the text for development.
"""

import ast
import re
from pathlib import Path

import pytest

from app.core import exceptions
from app.core.config import Settings, settings

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"
SECRET = 'relation "users" does not exist at /srv/app/secret.py'


def test_the_default_is_to_hide():
    assert Settings.model_fields["EXPOSE_ERROR_DETAILS"].default is False


def test_hidden_the_client_gets_a_reference_and_the_log_gets_the_error(monkeypatch):
    logged = []
    monkeypatch.setattr(settings, "EXPOSE_ERROR_DETAILS", False)
    monkeypatch.setattr(exceptions.logger, "error", logged.append)

    detail = exceptions.internal_error_detail(RuntimeError(SECRET), "Failed to export")

    assert SECRET not in detail and "users" not in detail
    reference = re.fullmatch(r"Failed to export \(reference ([0-9a-f]{8})\)", detail)
    assert reference, detail
    # The same reference is in the log line that does carry the error.
    assert len(logged) == 1
    assert reference.group(1) in logged[0] and SECRET in logged[0]


def test_exposed_it_reads_as_it_used_to(monkeypatch):
    monkeypatch.setattr(settings, "EXPOSE_ERROR_DETAILS", True)
    error = RuntimeError(SECRET)
    assert exceptions.internal_error_detail(error) == SECRET
    assert exceptions.internal_error_detail(error, "Failed to export") == (
        f"Failed to export: {SECRET}"
    )


async def test_an_unhandled_exception_is_hidden_too(monkeypatch):
    monkeypatch.setattr(settings, "EXPOSE_ERROR_DETAILS", False)
    response = await exceptions.generic_exception_handler(None, RuntimeError(SECRET))
    body = response.body.decode()
    assert response.status_code == 500
    assert SECRET not in body
    # Not the exception's class name either: that names the library in use.
    assert "RuntimeError" not in body and "InternalServerError" in body


def _leaking_500s():
    names = {"e", "exc", "err", "error", "ex"}
    found = []
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if callee != "HTTPException":
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            status, detail = keywords.get("status_code"), keywords.get("detail")
            if status is None or detail is None:
                continue
            code = (
                status.value
                if isinstance(status, ast.Constant)
                else ast.unparse(status)
            )
            if code not in (500, "status.HTTP_500_INTERNAL_SERVER_ERROR"):
                continue
            if isinstance(detail, ast.Call) and "internal_error_detail" in ast.unparse(
                detail.func
            ):
                continue
            used = {n.id for n in ast.walk(detail) if isinstance(n, ast.Name)}
            if used & names:
                found.append(f"{path.relative_to(APP)}:{node.lineno}")
    return found


def test_no_500_builds_its_detail_from_the_exception():
    leaking = _leaking_500s()
    assert not leaking, (
        "These 500 responses put the exception's text in `detail`; use "
        "internal_error_detail(exc, 'what failed'): " + ", ".join(leaking)
    )
