"""Plugins over HTTP.

Listing shows every plugin this user *may* run -- the bundles shipped with the
deployment plus their own -- each annotated with whether they have installed it
and whether it is enabled. That is deliberately not two endpoints: "what can I
run" and "what am I running" are the same question asked from either end, and
splitting them makes the catalogue page do two round trips to render one list.

Each response carries the tools the plugin contributes **with their governance
resolved**, because that is the thing a person needs before installing: not
what the author said the tool does, but what it will be allowed to do.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core.plugin_specs import contributed_tool_name, governance_for
from app.api.endpoints.auth import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.plugin import Plugin, PluginInstallation
from app.models.user import User
from app.schemas.plugin import (
    ContributedToolView,
    PluginCreate,
    PluginDraftRequest,
    PluginDraftResponse,
    PluginInstallationUpdate,
    PluginInstallRequest,
    PluginListResponse,
    PluginResponse,
    PluginUpdate,
)
from app.services import plugin_registry
from app.services.custom_tool_service import CustomToolService, ToolExecutionError
from app.services.plugin_author_service import draft_manifest
from app.services.plugin_manifest import ManifestError, validate_manifest

router = APIRouter()


def _tool_views(plugin: Plugin) -> List[ContributedToolView]:
    manifest = plugin.manifest if isinstance(plugin.manifest, dict) else {}
    views: List[ContributedToolView] = []
    for entry in (manifest.get("contributes") or {}).get("tools") or []:
        if not isinstance(entry, dict):
            continue
        tool_type = str(entry.get("tool_type") or "")
        gov = governance_for(tool_type)
        views.append(
            ContributedToolView(
                name=contributed_tool_name(plugin.slug, str(entry.get("name") or "")),
                declared_name=str(entry.get("name") or ""),
                description=str(entry.get("description") or ""),
                tool_type=tool_type,
                effects=gov["effects"],
                network=gov["network"],
                cost_tier=gov["cost_tier"],
                job_types=list(entry.get("job_types") or []),
            )
        )
    return views


def _respond(
    plugin: Plugin, installation: Optional[PluginInstallation]
) -> PluginResponse:
    resolved = plugin_registry.contributions_of(plugin)
    return PluginResponse(
        id=plugin.id,
        slug=plugin.slug,
        name=plugin.name,
        description=plugin.description,
        version=plugin.version,
        source=plugin.source,
        owner_id=plugin.owner_id,
        created_at=plugin.created_at,
        updated_at=plugin.updated_at,
        installed=installation is not None,
        enabled=bool(installation and installation.is_enabled),
        tools=_tool_views(plugin),
        unavailable=[{"tool": n, "reason": why} for n, why in resolved.skipped],
    )


@router.get("", response_model=PluginListResponse)
async def list_plugins(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Every plugin this user may run, with their own install state on each."""
    plugins = await plugin_registry.available_plugins(db, current_user.id)
    installed = {
        inst.plugin_id: inst
        for _p, inst in await plugin_registry.installations_for(db, current_user.id)
    }
    items = [_respond(p, installed.get(p.id)) for p in plugins]
    return PluginListResponse(items=items, total=len(items))


@router.post("", response_model=PluginResponse, status_code=status.HTTP_201_CREATED)
async def create_plugin(
    payload: PluginCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Author a plugin. The manifest is validated before anything is stored."""
    if not bool(getattr(settings, "PLUGINS_USER_AUTHORING_ENABLED", True)):
        raise HTTPException(
            status_code=403,
            detail=(
                "Authoring plugins is disabled on this deployment "
                "(PLUGINS_USER_AUTHORING_ENABLED=false)."
            ),
        )
    try:
        manifest = validate_manifest(payload.manifest)
    except ManifestError as exc:
        # 400 rather than 422: the document is well-formed JSON and wrong about
        # itself, which is a different thing from a malformed request body.
        raise HTTPException(status_code=400, detail=str(exc))

    existing = (
        await db.execute(
            select(Plugin).where(
                Plugin.slug == manifest["id"], Plugin.owner_id == current_user.id
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"You already have a plugin with id {manifest['id']!r}. "
                "Update it instead, or choose another id."
            ),
        )

    plugin = Plugin(
        slug=manifest["id"],
        name=manifest["name"],
        description=manifest["description"] or None,
        version=manifest["version"],
        source="user",
        owner_id=current_user.id,
        manifest=manifest,
    )
    db.add(plugin)
    await db.commit()
    await db.refresh(plugin)
    return _respond(plugin, None)


async def _owned(db: AsyncSession, plugin_id: UUID, user: User) -> Plugin:
    plugin = await db.get(Plugin, plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    if plugin.owner_id != user.id:
        # A builtin has no owner and belongs to the deployment; a user's plugin
        # belongs to them. Neither is editable by anybody else.
        raise HTTPException(
            status_code=403, detail="Only the author of a plugin can change it"
        )
    return plugin


@router.get("/{plugin_id}", response_model=PluginResponse)
async def get_plugin(
    plugin_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plugin = await db.get(Plugin, plugin_id)
    if plugin is None or (
        plugin.owner_id is not None and plugin.owner_id != current_user.id
    ):
        raise HTTPException(status_code=404, detail="Plugin not found")
    installation = (
        await db.execute(
            select(PluginInstallation).where(
                PluginInstallation.plugin_id == plugin.id,
                PluginInstallation.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    return _respond(plugin, installation)


@router.put("/{plugin_id}", response_model=PluginResponse)
async def update_plugin(
    plugin_id: UUID,
    payload: PluginUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plugin = await _owned(db, plugin_id, current_user)
    try:
        manifest = validate_manifest(payload.manifest)
    except ManifestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if manifest["id"] != plugin.slug:
        raise HTTPException(
            status_code=400,
            detail=(
                f"A plugin's id cannot change: this one is {plugin.slug!r} and "
                f"the manifest says {manifest['id']!r}. Its id is part of every "
                "tool name it contributes, so changing it would rename tools "
                "that running jobs may be planning to call."
            ),
        )
    plugin.name = manifest["name"]
    plugin.description = manifest["description"] or None
    plugin.version = manifest["version"]
    plugin.manifest = manifest
    await db.commit()
    await db.refresh(plugin)
    return _respond(plugin, None)


@router.delete("/{plugin_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_plugin(
    plugin_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plugin = await _owned(db, plugin_id, current_user)
    await db.delete(plugin)
    await db.commit()


@router.post("/{plugin_id}/install", response_model=PluginResponse)
async def install_plugin(
    plugin_id: UUID,
    payload: PluginInstallRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Install a plugin for the calling user. Installing twice is not an error."""
    plugin = await db.get(Plugin, plugin_id)
    if plugin is None or (
        plugin.owner_id is not None and plugin.owner_id != current_user.id
    ):
        raise HTTPException(status_code=404, detail="Plugin not found")

    installation = (
        await db.execute(
            select(PluginInstallation).where(
                PluginInstallation.plugin_id == plugin.id,
                PluginInstallation.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if installation is None:
        installation = PluginInstallation(plugin_id=plugin.id, user_id=current_user.id)
        db.add(installation)
    installation.is_enabled = bool(payload.enabled)
    if payload.settings:
        installation.settings = dict(payload.settings)
    await db.commit()
    await db.refresh(installation)
    return _respond(plugin, installation)


@router.patch("/{plugin_id}/install", response_model=PluginResponse)
async def update_installation(
    plugin_id: UUID,
    payload: PluginInstallationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Enable, disable or reconfigure an installed plugin."""
    plugin = await db.get(Plugin, plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    installation = (
        await db.execute(
            select(PluginInstallation).where(
                PluginInstallation.plugin_id == plugin.id,
                PluginInstallation.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if installation is None:
        raise HTTPException(status_code=404, detail="Plugin is not installed")
    if payload.enabled is not None:
        installation.is_enabled = bool(payload.enabled)
    if payload.settings is not None:
        installation.settings = dict(payload.settings)
    await db.commit()
    await db.refresh(installation)
    return _respond(plugin, installation)


@router.delete("/{plugin_id}/install", status_code=status.HTTP_204_NO_CONTENT)
async def uninstall_plugin(
    plugin_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Uninstall. Disabling is the reversible option; this discards settings."""
    installation = (
        await db.execute(
            select(PluginInstallation).where(
                PluginInstallation.plugin_id == plugin_id,
                PluginInstallation.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if installation is None:
        return
    await db.delete(installation)
    await db.commit()


@router.post("/draft", response_model=PluginDraftResponse)
async def draft_plugin(
    payload: PluginDraftRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Draft a manifest from a description. Drafting never installs anything.

    The result is handed back for review rather than created, because a
    manifest that validates is not the same as a manifest that does what
    somebody meant -- and the only person who can tell is the one who asked.
    """
    if not bool(getattr(settings, "PLUGINS_USER_AUTHORING_ENABLED", True)):
        raise HTTPException(
            status_code=403,
            detail=(
                "Authoring plugins is disabled on this deployment "
                "(PLUGINS_USER_AUTHORING_ENABLED=false)."
            ),
        )
    result = await draft_manifest(
        payload.description, user=current_user, db=db, user_id=current_user.id
    )
    return PluginDraftResponse(**result)


@router.get("/me/ui", response_model=Dict[str, Any])
async def my_contributed_ui(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The interface this user's enabled plugins contribute.

    Nav entries, views and panels as declared -- already validated at install,
    so the renderer can trust the shape and does not re-check it. What it must
    not trust is the *content*: every string here was written by whoever
    authored the plugin and is rendered as text, never as markup.
    """
    return {
        "plugins": await plugin_registry.ui_contributions_for_user(db, current_user.id)
    }


@router.post("/me/views/{slug}/{view_id}/data", response_model=Dict[str, Any])
async def read_view_data(
    slug: str,
    view_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Run the tool that backs one view, and return what it produced.

    Deliberately *not* a general "run a plugin tool from the browser"
    endpoint. The only thing callable here is a tool a view already declares
    as its source -- which install-time validation guarantees is read-only and
    belongs to the plugin that declared the view. A general runner would let a
    page call any contributed tool with any arguments, which is a much larger
    surface for the same feature.

    Arguments come from the manifest, not the request, for the same reason.
    """
    for contribution in await plugin_registry.ui_contributions_for_user(
        db, current_user.id
    ):
        if contribution["slug"] != slug:
            continue
        view = (contribution["views"] or {}).get(view_id)
        if not view:
            break

        source = view.get("source")
        if not source:
            # A static view -- markdown with its own text. Nothing to run.
            return {"data": None, "static": True}

        tool_name = contributed_tool_name(slug, str(source.get("tool") or ""))
        contributed = await plugin_registry.resolve_tool(db, current_user.id, tool_name)
        if contributed is None:
            raise HTTPException(
                status_code=404,
                detail=f"This view is backed by {tool_name!r}, which is no longer available",
            )

        try:
            result = await CustomToolService().execute_tool(
                tool=contributed,
                inputs=dict(source.get("params") or {}),
                user=current_user,
                db=db,
            )
        except ToolExecutionError as exc:
            # A policy denial or approval gate reaches here. Surfaced as 403
            # rather than 500: the request was well-formed and was refused.
            raise HTTPException(status_code=403, detail=str(exc))

        return {"data": result.get("output"), "static": False}

    raise HTTPException(status_code=404, detail="No such view")


@router.get("/me/tools", response_model=Dict[str, Any])
async def my_contributed_tools(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Exactly what this user's enabled plugins currently offer an agent.

    The same resolution a job performs at start, so this answers "why can my
    job not see my tool" without reading logs.
    """
    resolved = await plugin_registry.contributions_for_user(db, current_user.id)
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "effects": spec.effects,
                "network": spec.network,
                "cost_tier": spec.cost_tier,
                "job_types": list(spec.job_types or []),
            }
            for spec in resolved.specs
        ],
        "unavailable": [{"tool": n, "reason": why} for n, why in resolved.skipped],
    }
