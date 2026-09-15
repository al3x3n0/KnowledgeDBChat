"""Installing a plugin, and what it then offers.

Two properties matter here and neither is about the happy path: a manifest that
is wrong is refused *with the reason*, and a plugin that is not enabled
contributes nothing at all.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.plugin import Plugin, PluginInstallation
from app.services import plugin_registry as registry
from app.services.plugin_manifest import ManifestError, validate_manifest


def _manifest(**overrides):
    base = {
        "id": "bench",
        "name": "Benchmarks",
        "version": "0.1.0",
        "contributes": {
            "tools": [
                {
                    "name": "note",
                    "description": "Record a note",
                    "tool_type": "transform",
                    "parameters_schema": {"type": "object", "properties": {}},
                    "config": {"template": "noted: {{ text }}"},
                    "job_types": ["research"],
                }
            ]
        },
    }
    base.update(overrides)
    return base


async def _installed_plugin(db, user, *, enabled=True, manifest=None):
    validated = validate_manifest(manifest or _manifest())
    plugin = Plugin(
        slug=validated["id"],
        name=validated["name"],
        version=validated["version"],
        source="user",
        owner_id=user.id,
        manifest=validated,
    )
    db.add(plugin)
    await db.flush()
    db.add(PluginInstallation(plugin_id=plugin.id, user_id=user.id, is_enabled=enabled))
    await db.flush()
    return plugin


# --------------------------------------------------------------------------
# The manifest is refused, and says why
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken,fragment",
    [
        ({"id": "Bench!"}, "lowercase letters"),
        ({"version": "v1"}, "look like 1.2.3"),
        ({"name": ""}, "name is required"),
        ({"contributes": {"tols": []}}, "unknown key"),
        ({"contributes": {"tools": []}}, "contributes nothing"),
    ],
)
def test_a_manifest_is_refused_with_the_reason(broken, fragment):
    with pytest.raises(ManifestError) as excinfo:
        validate_manifest(_manifest(**broken))

    assert fragment in str(excinfo.value)


def test_an_unknown_job_type_names_the_ones_that_exist():
    """A contributor told only "invalid" will guess, and guess the same way."""
    bad = _manifest(
        contributes={
            "tools": [{"name": "n", "tool_type": "transform", "job_types": ["reserch"]}]
        }
    )

    with pytest.raises(ManifestError) as excinfo:
        validate_manifest(bad)

    message = str(excinfo.value)
    assert "reserch" in message
    assert "research" in message


def test_a_tool_declared_twice_is_refused():
    duplicated = _manifest(
        contributes={
            "tools": [
                {"name": "n", "tool_type": "transform"},
                {"name": "n", "tool_type": "transform"},
            ]
        }
    )

    with pytest.raises(ManifestError, match="declared twice"):
        validate_manifest(duplicated)


def test_a_manifest_carries_undelivered_contributions_through_unharmed():
    """Dropping `nav` because this release cannot render it would silently
    destroy half of a manifest written against the documented format."""
    with_ui = _manifest()
    with_ui["contributes"]["nav"] = [{"door": "R&D", "name": "Benchmarks"}]

    out = validate_manifest(with_ui)

    assert out["contributes"]["nav"] == [{"door": "R&D", "name": "Benchmarks"}]


# --------------------------------------------------------------------------
# What an enabled plugin contributes
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_enabled_plugin_contributes_its_tools(db_session, test_user):
    await _installed_plugin(db_session, test_user)

    resolved = await registry.contributions_for_user(db_session, test_user.id)

    assert resolved.spec_names() == ["p_bench_note"]
    assert resolved.tools["p_bench_note"].tool_type == "transform"
    assert resolved.skipped == []


@pytest.mark.asyncio
async def test_a_disabled_plugin_contributes_nothing(db_session, test_user):
    """Disabling is the reversible switch, and it has to actually switch."""
    await _installed_plugin(db_session, test_user, enabled=False)

    resolved = await registry.contributions_for_user(db_session, test_user.id)

    assert resolved.specs == []


@pytest.mark.asyncio
async def test_a_plugin_nobody_installed_contributes_nothing(db_session, test_user):
    validated = validate_manifest(_manifest())
    db_session.add(
        Plugin(
            slug="bench",
            name="Benchmarks",
            version="0.1.0",
            source="user",
            owner_id=test_user.id,
            manifest=validated,
        )
    )
    await db_session.flush()

    resolved = await registry.contributions_for_user(db_session, test_user.id)

    assert resolved.specs == []


@pytest.mark.asyncio
async def test_one_users_plugin_is_not_another_users_tool(db_session, test_user):
    await _installed_plugin(db_session, test_user)

    other = await registry.contributions_for_user(db_session, uuid4())

    assert other.specs == []


@pytest.mark.asyncio
async def test_a_tool_is_offered_only_to_the_job_types_it_declared(
    db_session, test_user
):
    await _installed_plugin(db_session, test_user)

    research = await registry.contributions_for_user(
        db_session, test_user.id, job_type="research"
    )
    coding = await registry.contributions_for_user(
        db_session, test_user.id, job_type="coding"
    )

    assert research.spec_names() == ["p_bench_note"]
    assert coding.spec_names() == []


@pytest.mark.asyncio
async def test_resolving_a_tool_finds_what_the_model_would_call(db_session, test_user):
    await _installed_plugin(db_session, test_user)

    found = await registry.resolve_tool(db_session, test_user.id, "p_bench_note")
    missing = await registry.resolve_tool(db_session, test_user.id, "p_bench_absent")

    assert found is not None
    assert found.declared_name == "note"
    # The policy engine names a tool by this id, so it must be stable and
    # readable rather than a uuid.
    assert found.id == "bench:note"
    assert missing is None


# --------------------------------------------------------------------------
# Bundles shipped on disk
# --------------------------------------------------------------------------


def _write_bundle(root, name, payload):
    import json

    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.json").write_text(json.dumps(payload))
    return directory


def test_a_shipped_bundle_is_loaded_from_disk(tmp_path, monkeypatch):
    from app.core.config import settings

    _write_bundle(tmp_path, "hygiene", _manifest(id="hygiene", name="Hygiene"))
    monkeypatch.setattr(settings, "PLUGIN_BUILTIN_DIR", str(tmp_path))

    loaded = registry.load_builtin_manifests()

    assert [m["id"] for m in loaded] == ["hygiene"]


def test_one_broken_bundle_does_not_stop_the_others_loading(tmp_path, monkeypatch):
    """A bad file shipped in one bundle must not stop the application starting.

    Raising here would take down every deployment that shipped a typo, and the
    bundles that are fine would be unavailable for a reason none of them caused.
    """
    from app.core.config import settings

    _write_bundle(tmp_path, "good", _manifest(id="good", name="Good"))
    _write_bundle(tmp_path, "broken", {"id": "Nope!", "name": "Broken"})
    (tmp_path / "unparseable").mkdir()
    (tmp_path / "unparseable" / "plugin.json").write_text("{ not json")
    monkeypatch.setattr(settings, "PLUGIN_BUILTIN_DIR", str(tmp_path))

    loaded = registry.load_builtin_manifests()

    assert [m["id"] for m in loaded] == ["good"]


def test_a_missing_plugin_directory_is_not_an_error(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PLUGIN_BUILTIN_DIR", str(tmp_path / "absent"))

    assert registry.load_builtin_manifests() == []


@pytest.mark.asyncio
async def test_syncing_a_bundle_twice_updates_rather_than_duplicates(
    db_session, tmp_path, monkeypatch
):
    """Startup runs on every boot, so the sync has to be idempotent."""
    from sqlalchemy import select

    from app.core.config import settings

    _write_bundle(tmp_path, "hygiene", _manifest(id="hygiene", name="Hygiene"))
    monkeypatch.setattr(settings, "PLUGIN_BUILTIN_DIR", str(tmp_path))

    await registry.sync_builtin_plugins(db_session)
    _write_bundle(
        tmp_path, "hygiene", _manifest(id="hygiene", name="Hygiene", version="0.2.0")
    )
    await registry.sync_builtin_plugins(db_session)

    rows = list(
        (
            await db_session.execute(select(Plugin).where(Plugin.slug == "hygiene"))
        ).scalars()
    )
    assert len(rows) == 1
    assert rows[0].version == "0.2.0"
    assert rows[0].owner_id is None
    assert rows[0].source == "builtin"
