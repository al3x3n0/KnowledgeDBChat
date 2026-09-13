"""CORS_ALLOWED_ORIGINS has to be settable from an ordinary env var.

It was first written as `List[str]`, which looks right and works fine from the
default -- but pydantic-settings JSON-decodes complex types straight out of the
environment, before any `mode="before"` validator can normalise them. Setting
it the way the documentation says to (`a,b`) raised SettingsError and took the
process down at startup. These tests pin the env path, not just the default.
"""

import pytest

from app.core.config import Settings


def _settings(monkeypatch, raw=None):
    # _env_file=None so a developer's own backend/.env cannot change the result.
    if raw is None:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", raw)
    return Settings(_env_file=None)


def test_default_covers_the_ui_origin(monkeypatch):
    origins = _settings(monkeypatch).cors_allowed_origins
    assert "http://localhost:23000" in origins
    assert "http://127.0.0.1:23000" in origins


def test_comma_separated_env_var_parses(monkeypatch):
    s = _settings(monkeypatch, "http://a.test:1,http://b.test:2")
    assert s.cors_allowed_origins == ["http://a.test:1", "http://b.test:2"]


def test_surrounding_whitespace_is_tolerated(monkeypatch):
    s = _settings(monkeypatch, " http://a.test:1 , http://b.test:2 ")
    assert s.cors_allowed_origins == ["http://a.test:1", "http://b.test:2"]


def test_single_origin_is_not_split_into_characters(monkeypatch):
    s = _settings(monkeypatch, "http://only.test:1")
    assert s.cors_allowed_origins == ["http://only.test:1"]


def test_empty_entries_are_dropped(monkeypatch):
    s = _settings(monkeypatch, "http://a.test:1,,  ,http://b.test:2,")
    assert s.cors_allowed_origins == ["http://a.test:1", "http://b.test:2"]


def test_empty_value_yields_no_origins_rather_than_one_blank(monkeypatch):
    assert _settings(monkeypatch, "").cors_allowed_origins == []


@pytest.mark.parametrize("raw", ["http://a.test:1,http://b.test:2", ""])
def test_env_value_never_raises_on_construction(monkeypatch, raw):
    # The original bug: constructing Settings() at import time blew up.
    _settings(monkeypatch, raw)
