"""
Tool policy engine.

Semantics:
- allow-by-default
- explicit denies win
- approvals can be required by policy
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_policy import ToolPolicy
from app.models.user import User
from app.services.tool_registry import get_tool_metadata
from urllib.parse import urlparse


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    require_approval: bool
    denied_reason: Optional[str] = None
    matched_policies: Optional[list[dict]] = None


def _match_policy(policy: ToolPolicy, tool_name: str) -> bool:
    sel = str(policy.tool_name or "").strip()
    if not sel:
        return False
    if sel == "*":
        return True
    if sel.endswith("*"):
        return tool_name.startswith(sel[:-1])
    return sel == tool_name


def _tier_leq(actual: str, maximum: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2}
    return order.get(str(actual).strip().lower(), 999) <= order.get(str(maximum).strip().lower(), 999)


def _extract_url_from_args(args: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(args, dict):
        return None
    for k in ("url", "repo_url"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_host_from_url_like(url_like: str) -> str:
    """
    Best-effort host extraction for constraints enforcement.

    Supports:
    - http(s)://host/path
    - host/path (assumes https)
    - host:port/path (assumes http)
    - git@host:org/repo(.git) scp-style
    - ssh://git@host:port/org/repo
    """
    raw = str(url_like or "").strip()
    if not raw:
        return ""

    # scp-style git remote: git@github.com:org/repo.git
    # urlparse doesn't treat this as a URL, so handle it first.
    if "://" not in raw and "@" in raw and ":" in raw.split("@", 1)[1]:
        after_at = raw.split("@", 1)[1]
        host = after_at.split(":", 1)[0].strip()
        return host.strip("[]").strip().lower()

    parsed = urlparse(raw)
    if parsed.hostname:
        return str(parsed.hostname).strip().lower()

    # Missing scheme forms like "github.com/org/repo" or "localhost:8000/path".
    # If it starts with "//", urlparse treats it as netloc-only.
    if raw.startswith("//"):
        parsed2 = urlparse("https:" + raw)
        if parsed2.hostname:
            return str(parsed2.hostname).strip().lower()

    # Heuristic: if it looks like host/path, assume https; if it looks like host:port/path, assume http.
    prefix = "http://" if (":" in raw.split("/", 1)[0]) else "https://"
    parsed3 = urlparse(prefix + raw)
    if parsed3.hostname:
        return str(parsed3.hostname).strip().lower()

    return ""


def _is_private_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return False
    if h in {"localhost"}:
        return True
    if h.endswith(".local") or h.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except Exception:
        return False


def _host_allowed(host: str, allowed_domains: list[str]) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return False
    for d in allowed_domains:
        dom = str(d or "").strip().lower().rstrip(".")
        if not dom:
            continue
        if h == dom:
            return True
        if h.endswith("." + dom):
            return True
    return False


def _constraints_ok(*, constraints: Optional[dict], tool_name: str, tool_args: Optional[dict]) -> tuple[bool, Optional[str]]:
    if not isinstance(constraints, dict) or not constraints:
        return True, None

    meta = get_tool_metadata(tool_name)
    if meta is None:
        return False, f"Unknown tool '{tool_name}'"

    max_cost = constraints.get("max_cost_tier")
    if isinstance(max_cost, str) and max_cost.strip():
        if not _tier_leq(meta.cost_tier, max_cost):
            return False, f"Tool '{tool_name}' exceeds max_cost_tier={max_cost}"

    # Network URL constraints (best-effort)
    url = _extract_url_from_args(tool_args)
    if url:
        host = _extract_host_from_url_like(url)

        if constraints.get("deny_private_networks") is True:
            if not host:
                return False, f"Tool '{tool_name}' could not parse URL host and deny_private_networks=true"
            if _is_private_host(host):
                return False, f"Tool '{tool_name}' URL host is private and deny_private_networks=true"

        allowed_domains = constraints.get("allowed_domains")
        if isinstance(allowed_domains, list) and allowed_domains:
            if not host:
                return False, f"Tool '{tool_name}' could not parse URL host for allowed_domains"
            if not _host_allowed(host, allowed_domains):
                return False, f"Tool '{tool_name}' URL host not in allowed_domains"

    return True, None


async def evaluate_tool_policy(
    *,
    db: AsyncSession,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    user: Optional[User] = None,
    agent_definition_id: Optional[UUID] = None,
    api_key_id: Optional[UUID] = None,
) -> ToolDecision:
    """
    Evaluate policies for this tool call.

    Policy precedence isn't order-based; any deny blocks.
    """
    tn = str(tool_name or "").strip()
    if not tn:
        return ToolDecision(allowed=False, require_approval=False, denied_reason="Missing tool name")

    # Validate tool identifier (fail closed) while supporting dynamic user tools.
    if tn.startswith("user_tool:"):
        # Dynamic tools are allowed to exist without registry entries.
        pass
    else:
        if get_tool_metadata(tn) is None:
            return ToolDecision(allowed=False, require_approval=False, denied_reason=f"Unknown tool '{tn}'")

    subject_clauses = [ToolPolicy.subject_type == "global"]
    if user is not None:
        subject_clauses.append((ToolPolicy.subject_type == "user") & (ToolPolicy.subject_id == user.id))
        subject_clauses.append((ToolPolicy.subject_type == "role") & (ToolPolicy.subject_key == user.role))
    if agent_definition_id:
        subject_clauses.append((ToolPolicy.subject_type == "agent_definition") & (ToolPolicy.subject_id == agent_definition_id))
    if api_key_id:
        subject_clauses.append((ToolPolicy.subject_type == "api_key") & (ToolPolicy.subject_id == api_key_id))

    stmt = select(ToolPolicy).where(or_(*subject_clauses))
    res = await db.execute(stmt)
    policies = list(res.scalars().all())

    matched: list[ToolPolicy] = [p for p in policies if _match_policy(p, tn)]

    # Determine denies
    denies = [p for p in matched if str(p.effect).strip().lower() == "deny"]
    if denies:
        reason = f"Tool '{tn}' denied by policy"
        logger.info(reason)
        return ToolDecision(
            allowed=False,
            require_approval=False,
            denied_reason=reason,
            matched_policies=[{"id": str(p.id), "subject_type": p.subject_type, "tool_name": p.tool_name, "effect": p.effect} for p in denies[:10]],
        )

    # Enforce constraints (if any). Any violated constraint denies the tool.
    for p in matched:
        ok, reason = _constraints_ok(constraints=p.constraints, tool_name=tn, tool_args=tool_args)
        if not ok:
            return ToolDecision(
                allowed=False,
                require_approval=False,
                denied_reason=reason or f"Tool '{tn}' denied by policy constraints",
                matched_policies=[{"id": str(p.id), "subject_type": p.subject_type, "tool_name": p.tool_name, "effect": p.effect}],
            )

    require_approval = any(bool(p.require_approval) for p in matched)
    return ToolDecision(
        allowed=True,
        require_approval=require_approval,
        denied_reason=None,
        matched_policies=[{"id": str(p.id), "subject_type": p.subject_type, "tool_name": p.tool_name, "effect": p.effect, "require_approval": p.require_approval} for p in matched[:20]],
    )
