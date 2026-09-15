"""Loading plugins, and resolving what they contribute for one user.

Two sources, one shape. Bundles shipped in this repository live under
``PLUGIN_BUILTIN_DIR`` as a directory each holding a ``plugin.json``; they have
no owner and are synced into the database at startup so that installing one is
the same operation as installing a user's own. Plugins authored in the
application are rows already.

A contributed tool is **not materialized** as a ``UserTool``. It could be --
the manifest carries exactly the fields that table holds -- but then the same
declaration would exist twice, and the copy would drift the first time a
manifest was updated without reinstalling. Instead the manifest stays the only
source of truth and execution builds a ``ContributedTool`` at call time.
``CustomToolService.execute_tool`` reads only ``id``, ``name``, ``config``,
``parameters_schema`` and ``tool_type`` off its argument, and uses ``id`` to
name a policy rather than to join anything -- so the ephemeral object is
governed by the policy engine, the approval gate and the audit log exactly as a
stored tool is, with none of them needing to learn what a plugin is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core import tool_specs
from app.agent_core.plugin_specs import contributed_tool_name, spec_from_user_tool
from app.agent_core.tool_specs.spec import ToolSpec
from app.models.plugin import Plugin, PluginInstallation
from app.services.plugin_manifest import ManifestError, validate_manifest


@dataclass(frozen=True)
class ContributedTool:
    """A manifest-declared tool, in the shape the executor already accepts."""

    id: str
    name: str
    tool_type: str
    config: Dict[str, Any]
    parameters_schema: Dict[str, Any]
    plugin_slug: str
    #: The tool's own name inside its manifest, before namespacing. Kept for
    #: messages, which should say what the author called it.
    declared_name: str
    is_enabled: bool = True


@dataclass
class ResolvedContributions:
    """Everything one user's enabled plugins offer, and what was skipped."""

    specs: List[ToolSpec] = field(default_factory=list)
    tools: Dict[str, ContributedTool] = field(default_factory=dict)
    #: (tool name, why) for anything that could not be offered. Resolution
    #: never raises: one unusable tool must not cost a job the others.
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def spec_names(self) -> List[str]:
        return [spec.name for spec in self.specs]


def builtin_dir() -> Path:
    from app.core.config import settings

    configured = str(getattr(settings, "PLUGIN_BUILTIN_DIR", "") or "plugins")
    path = Path(configured)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def load_builtin_manifests() -> List[Dict[str, Any]]:
    """Every valid ``plugin.json`` under the builtin directory.

    A malformed bundle is logged and skipped rather than raised: a bad file
    shipped in one bundle must not stop the application from starting with the
    others.
    """
    root = builtin_dir()
    if not root.is_dir():
        return []

    manifests: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/plugin.json")):
        try:
            manifests.append(validate_manifest(json.loads(path.read_text())))
        except (ManifestError, json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping builtin plugin at {path}: {exc}")
    return manifests


async def sync_builtin_plugins(db: AsyncSession) -> List[Plugin]:
    """Upsert the shipped bundles. Safe to call on every startup."""
    synced: List[Plugin] = []
    for manifest in load_builtin_manifests():
        slug = manifest["id"]
        existing = (
            await db.execute(
                select(Plugin).where(Plugin.slug == slug, Plugin.owner_id.is_(None))
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = Plugin(slug=slug, source="builtin", owner_id=None)
            db.add(existing)
        existing.name = manifest["name"]
        existing.description = manifest["description"] or None
        existing.version = manifest["version"]
        existing.manifest = manifest
        synced.append(existing)
    await db.flush()
    return synced


async def available_plugins(db: AsyncSession, user_id: Any) -> List[Plugin]:
    """Plugins this user may install: the builtins, plus their own."""
    rows = await db.execute(
        select(Plugin)
        .where(or_(Plugin.owner_id.is_(None), Plugin.owner_id == user_id))
        .order_by(Plugin.name)
    )
    return list(rows.scalars())


async def installations_for(
    db: AsyncSession, user_id: Any, *, enabled_only: bool = False
) -> List[Tuple[Plugin, PluginInstallation]]:
    query = (
        select(Plugin, PluginInstallation)
        .join(PluginInstallation, PluginInstallation.plugin_id == Plugin.id)
        .where(PluginInstallation.user_id == user_id)
    )
    if enabled_only:
        query = query.where(PluginInstallation.is_enabled.is_(True))
    return [tuple(row) for row in (await db.execute(query.order_by(Plugin.name))).all()]


def contributions_of(
    plugin: Plugin, *, job_type: Optional[str] = None
) -> ResolvedContributions:
    """What one plugin offers, as specs plus the tools behind them."""
    resolved = ResolvedContributions()
    manifest = plugin.manifest if isinstance(plugin.manifest, dict) else {}
    contributes = manifest.get("contributes") or {}
    reserved = tool_specs.spec_names()

    for entry in contributes.get("tools") or []:
        if not isinstance(entry, dict):
            continue
        declared = str(entry.get("name") or "").strip()
        declared_job_types = tuple(entry.get("job_types") or ())

        # A tool that names job types is offered only to those. A tool that
        # names none is offered to none -- `job_types=None` would mean "every
        # job type", which is not something an author should be able to claim
        # by leaving a field out.
        if job_type is not None and job_type not in declared_job_types:
            continue

        holder = ContributedTool(
            id=f"{plugin.slug}:{declared}",
            name=contributed_tool_name(plugin.slug, declared),
            tool_type=str(entry.get("tool_type") or ""),
            config=dict(entry.get("config") or {}),
            parameters_schema=dict(entry.get("parameters_schema") or {}),
            plugin_slug=plugin.slug,
            declared_name=declared,
        )
        try:
            spec = spec_from_user_tool(
                holder,
                slug=plugin.slug,
                declared_name=declared,
                job_types=declared_job_types,
                reserved=reserved,
            )
        except ValueError as exc:
            # Refused at install, so reaching here means the built-in tool set
            # changed under an installed plugin. Skipping is right: the job
            # keeps every other tool, and the reason travels to the caller.
            resolved.skipped.append((declared, str(exc)))
            continue

        resolved.specs.append(spec)
        resolved.tools[spec.name] = holder

    return resolved


async def contributions_for_user(
    db: AsyncSession,
    user_id: Any,
    *,
    job_type: Optional[str] = None,
) -> ResolvedContributions:
    """Everything this user's enabled plugins contribute right now."""
    merged = ResolvedContributions()
    if user_id is None:
        return merged

    seen: set = set()
    for plugin, _installation in await installations_for(
        db, user_id, enabled_only=True
    ):
        part = contributions_of(plugin, job_type=job_type)
        for spec in part.specs:
            if spec.name in seen:
                # Two plugins cannot produce the same name -- the slug is part
                # of it and is unique per owner -- but a builtin and a user
                # plugin could share a slug. First wins, and the loss is said
                # out loud rather than resolved silently.
                merged.skipped.append(
                    (spec.name, "another installed plugin already offers this tool")
                )
                continue
            seen.add(spec.name)
            merged.specs.append(spec)
            merged.tools[spec.name] = part.tools[spec.name]
        merged.skipped.extend(part.skipped)

    if merged.skipped:
        logger.warning(
            "Plugin tools not offered to user {}: {}",
            user_id,
            "; ".join(f"{n}: {why}" for n, why in merged.skipped),
        )
    return merged


async def resolve_tool(
    db: AsyncSession, user_id: Any, tool_name: str
) -> Optional[ContributedTool]:
    """The contributed tool behind this name, if the user has it enabled."""
    name = str(tool_name or "").strip()
    if not name:
        return None
    return (await contributions_for_user(db, user_id)).tools.get(name)
