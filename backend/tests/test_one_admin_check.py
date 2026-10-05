"""Whether a user is an administrator is decided in one place.

It was decided in about a dozen: `user.is_admin()`, a raw `role == "admin"` in
seven places, three private `_is_admin` helpers, and a second `require_admin`
with a different message. One variant was `not current_user.is_admin` -- the
method, never called, so always truthy -- and it guarded three pipeline routes
that therefore let any signed-in user read or restart anyone's run.

A check that exists in one form cannot be written wrongly in another, so the
other forms are refused here.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.services.auth_service import ensure_admin, is_admin

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

#: Where the role is allowed to be read directly: the model that defines it.
ROLE_IS_READ_IN = {"models/user.py"}


def _lines():
    for path in sorted(APP.rglob("*.py")):
        relative = str(path.relative_to(APP))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            yield relative, number, line


def test_the_role_is_compared_in_one_place():
    raw = re.compile(r"""role\s*(==|!=)\s*["']admin["']|["']role["'],\s*None\)\s*==""")
    offenders = [
        f"{rel}:{n}: {line.strip()[:80]}"
        for rel, n, line in _lines()
        if rel not in ROLE_IS_READ_IN and raw.search(line)
    ]
    assert (
        not offenders
    ), "Use auth_service.is_admin(user) instead of comparing the role:\n" + "\n".join(
        offenders
    )


def test_nobody_keeps_a_private_admin_helper():
    private = re.compile(
        r"^\s*(async\s+)?def\s+(_is_admin|is_admin_user|_user_is_admin)\b"
    )
    offenders = [f"{rel}:{n}" for rel, n, line in _lines() if private.search(line)]
    assert not offenders, f"private admin helpers: {offenders}"


def test_there_is_one_require_admin():
    defined = [
        rel
        for rel, _n, line in _lines()
        if re.match(r"^(async\s+)?def require_admin\b", line)
    ]
    assert defined == ["services/auth_service.py"], defined


def test_the_method_is_never_used_without_being_called():
    """`not current_user.is_admin` is a bound method and always truthy: the
    check type-checks, reads correctly, and never refuses anyone."""
    uncalled = re.compile(r"\b(current_user|user|admin|owner)\.is_admin\b(?!\s*\()")
    offenders = [
        f"{rel}:{n}: {line.strip()[:80]}"
        for rel, n, line in _lines()
        if uncalled.search(line)
    ]
    assert not offenders, "is_admin used without calling it:\n" + "\n".join(offenders)


def test_the_login_dependency_has_one_home():
    stray = [
        f"{rel}:{n}"
        for rel, n, line in _lines()
        if "from app.api.endpoints.users import get_current_user" in line
    ]
    assert not stray, f"import get_current_user from auth_service instead: {stray}"


class TestThePredicate:
    def test_nobody_is_not_an_administrator(self):
        assert is_admin(None) is False

    def test_it_asks_the_user(self):
        assert is_admin(SimpleNamespace(is_admin=lambda: True)) is True
        assert is_admin(SimpleNamespace(is_admin=lambda: False)) is False

    def test_ensure_admin_refuses_with_the_callers_words(self):
        admin = SimpleNamespace(is_admin=lambda: True)
        assert ensure_admin(admin) is admin
        with pytest.raises(HTTPException) as refused:
            ensure_admin(SimpleNamespace(is_admin=lambda: False), "Only an admin may.")
        assert refused.value.status_code == 403
        assert refused.value.detail == "Only an admin may."
        with pytest.raises(HTTPException):
            ensure_admin(None)
