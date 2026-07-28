"""Hardened outbound gateway for registered external agents."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional
from urllib.parse import quote, urlencode, urlparse
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.secret import UserSecret
from app.models.user import User
from app.models.workflow import UserTool
from app.services.secret_service import SecretService


class ExternalAgentGatewayError(RuntimeError):
    """Raised when an external-agent request is unsafe or unsuccessful."""


AddressResolver = Callable[[str, int], Awaitable[Iterable[str]]]


class ExternalAgentGatewayService:
    MAX_REQUEST_BYTES = 1_000_000
    MAX_RESPONSE_BYTES = 1_000_000
    CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,119}$")
    REMOTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
    COMPOPS_CAPABILITIES = {
        "compops.health",
        "compops.operators.list",
        "compops.runs.get",
        "compops.runs.submit",
        "compops.artifacts.get",
        "compops.artifacts.lineage",
        "compops.studies.get",
        "compops.studies.report",
        "compops.studies.gates.evaluate",
        "compops.batches.create",
        "compops.actions.get",
        "compops.actions.approve",
        "compops.actions.reject",
    }

    def __init__(
        self,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        resolver: Optional[AddressResolver] = None,
    ) -> None:
        self._transport = transport
        self._resolver = resolver or self._resolve_addresses

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        provider_type = (
            str(config.get("provider_type") or "generic_agent").strip().lower()
        )
        if provider_type not in {"generic_agent", "compops"}:
            raise ExternalAgentGatewayError(
                "provider_type must be generic_agent or compops"
            )
        endpoint_url = str(config.get("endpoint_url") or "").strip()
        parsed = urlparse(endpoint_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ExternalAgentGatewayError(
                "External agent endpoint_url must be an absolute HTTPS URL"
            )
        if parsed.username or parsed.password or parsed.fragment:
            raise ExternalAgentGatewayError(
                "External agent endpoint_url cannot contain credentials or fragments"
            )
        if provider_type == "compops" and (parsed.query or parsed.params):
            raise ExternalAgentGatewayError(
                "CompOps endpoint_url must be a base URL without query parameters"
            )

        capabilities = []
        for raw in config.get("capabilities") or []:
            capability = str(raw or "").strip().lower()
            if not self.CAPABILITY_PATTERN.fullmatch(capability):
                raise ExternalAgentGatewayError(
                    f"Invalid external agent capability: {capability!r}"
                )
            if capability not in capabilities:
                capabilities.append(capability)
        if not capabilities:
            raise ExternalAgentGatewayError(
                "External agent requires at least one capability"
            )
        if provider_type == "compops":
            unsupported = sorted(set(capabilities) - self.COMPOPS_CAPABILITIES)
            if unsupported:
                raise ExternalAgentGatewayError(
                    f"Unsupported CompOps capabilities: {unsupported}"
                )

        auth_type = str(config.get("auth_type") or "none").strip().lower()
        if auth_type not in {"none", "bearer", "api_key"}:
            raise ExternalAgentGatewayError(
                "auth_type must be none, bearer, or api_key"
            )
        secret_id = str(config.get("secret_id") or "").strip()
        if auth_type != "none" and not secret_id:
            raise ExternalAgentGatewayError(
                "Authenticated external agents require secret_id"
            )
        if secret_id:
            try:
                UUID(secret_id)
            except (TypeError, ValueError) as exc:
                raise ExternalAgentGatewayError("secret_id must be a UUID") from exc

        header_name = str(config.get("auth_header_name") or "X-API-Key").strip()
        if auth_type == "api_key" and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", header_name
        ):
            raise ExternalAgentGatewayError("Invalid API key header name")

        try:
            timeout_seconds = int(config.get("timeout_seconds") or 30)
        except (TypeError, ValueError):
            timeout_seconds = 30
        timeout_seconds = max(2, min(timeout_seconds, 120))
        return {
            "protocol": (
                "compops_rest_v1" if provider_type == "compops" else "http_json"
            ),
            "provider_type": provider_type,
            "endpoint_url": (
                endpoint_url.rstrip("/") if provider_type == "compops" else endpoint_url
            ),
            "capabilities": capabilities,
            "auth_type": auth_type,
            "secret_id": secret_id or None,
            "auth_header_name": header_name if auth_type == "api_key" else None,
            "timeout_seconds": timeout_seconds,
        }

    async def invoke(
        self,
        *,
        tool: UserTool,
        user: User,
        db: AsyncSession,
        capability: str,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not bool(tool.is_enabled):
            raise ExternalAgentGatewayError("External agent is disabled")
        config = self.validate_config(
            tool.config if isinstance(tool.config, dict) else {}
        )
        normalized_capability = str(capability or "").strip().lower()
        if normalized_capability not in set(config["capabilities"]):
            raise ExternalAgentGatewayError(
                f"Capability is not allowed for this external agent: "
                f"{normalized_capability!r}"
            )

        parsed = urlparse(config["endpoint_url"])
        port = parsed.port or 443
        addresses = list(await self._resolver(str(parsed.hostname), port))
        if not addresses:
            raise ExternalAgentGatewayError("External agent host did not resolve")
        for address in addresses:
            self._assert_public_address(address, hostname=str(parsed.hostname))

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "KnowledgeOps-Lab-External-Agent-Gateway/1",
        }
        await self._apply_authentication(
            headers=headers,
            config=config,
            user=user,
            db=db,
        )

        invocation_id = str(request_id or "").strip() or str(uuid4())
        method, target_url, request_body = self._request_for_capability(
            config=config,
            capability=normalized_capability,
            payload=payload,
            request_id=invocation_id,
        )
        try:
            encoded_request = (
                json.dumps(
                    request_body, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                if request_body is not None
                else b""
            )
        except (TypeError, ValueError) as exc:
            raise ExternalAgentGatewayError(
                "External agent request must be JSON serializable"
            ) from exc
        if len(encoded_request) > self.MAX_REQUEST_BYTES:
            raise ExternalAgentGatewayError(
                "External agent request exceeded size limit"
            )
        started = time.perf_counter()
        content = bytearray()
        try:
            async with httpx.AsyncClient(
                timeout=float(config["timeout_seconds"]),
                follow_redirects=False,
                transport=self._transport,
                trust_env=False,
            ) as client:
                async with client.stream(
                    method,
                    target_url,
                    headers=headers,
                    content=encoded_request or None,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ExternalAgentGatewayError(
                            "External agent redirects are not allowed"
                        )
                    if response.status_code < 200 or response.status_code >= 300:
                        raise ExternalAgentGatewayError(
                            f"External agent returned HTTP {response.status_code}"
                        )
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > self.MAX_RESPONSE_BYTES:
                            raise ExternalAgentGatewayError(
                                "External agent response exceeded size limit"
                            )
        except ExternalAgentGatewayError:
            raise
        except httpx.TimeoutException as exc:
            raise ExternalAgentGatewayError(
                f"External agent timed out after {config['timeout_seconds']}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExternalAgentGatewayError(
                f"External agent request failed: {type(exc).__name__}"
            ) from exc

        try:
            decoded = json.loads(bytes(content))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalAgentGatewayError(
                "External agent must return valid JSON"
            ) from exc
        output = decoded if isinstance(decoded, dict) else {"result": decoded}
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "output": output,
            "provenance": {
                "external_agent_id": str(tool.id),
                "external_agent_name": str(tool.name),
                "endpoint_origin": f"{parsed.scheme}://{parsed.netloc}",
                "provider_type": config["provider_type"],
                "capability": normalized_capability,
                "request_id": invocation_id,
                "remote_references": self._remote_references(payload),
                "received_at": datetime.now(timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(bytes(content)).hexdigest(),
                "response_bytes": len(content),
                "execution_time_ms": elapsed_ms,
            },
        }

    def _request_for_capability(
        self,
        *,
        config: Dict[str, Any],
        capability: str,
        payload: Dict[str, Any],
        request_id: str,
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        if config["provider_type"] != "compops":
            return (
                "POST",
                config["endpoint_url"],
                {
                    "request_id": request_id,
                    "capability": capability,
                    "input": payload,
                },
            )

        base = config["endpoint_url"]
        read_routes = {
            "compops.health": ("/healthz", None, ()),
            "compops.operators.list": ("/v1/operators", None, ()),
            "compops.runs.get": ("/v1/runs/{run_id}", "run_id", ()),
            "compops.artifacts.get": (
                "/v1/artifacts/{artifact_id}",
                "artifact_id",
                (),
            ),
            "compops.artifacts.lineage": (
                "/v1/artifacts/{artifact_id}/lineage",
                "artifact_id",
                ("direction", "depth"),
            ),
            "compops.studies.get": ("/v1/studies/{study_id}", "study_id", ()),
            "compops.studies.report": (
                "/v1/studies/{study_id}/report",
                "study_id",
                ("metric", "order"),
            ),
            "compops.studies.gates.evaluate": (
                "/v1/studies/{study_id}/gates/evaluate",
                "study_id",
                (),
            ),
            "compops.actions.get": ("/v1/actions/{action_id}", "action_id", ()),
        }
        if capability in read_routes:
            route, id_field, query_fields = read_routes[capability]
            if id_field:
                route = route.replace(
                    "{" + id_field + "}",
                    quote(self._remote_id(payload, id_field), safe=""),
                )
            query = {
                field: str(payload[field])
                for field in query_fields
                if payload.get(field) is not None
            }
            return (
                "GET",
                f"{base}{route}" + (f"?{urlencode(query)}" if query else ""),
                None,
            )

        write_routes = {
            "compops.runs.submit": ("/v1/runs", None),
            "compops.batches.create": ("/v1/batches", None),
            "compops.actions.approve": (
                "/v1/actions/{action_id}/approve",
                "action_id",
            ),
            "compops.actions.reject": (
                "/v1/actions/{action_id}/reject",
                "action_id",
            ),
        }
        route, id_field = write_routes[capability]
        if id_field:
            route = route.replace(
                "{" + id_field + "}",
                quote(self._remote_id(payload, id_field), safe=""),
            )
        body = payload.get("request")
        if not isinstance(body, dict):
            raise ExternalAgentGatewayError(
                f"{capability} requires an object payload.request"
            )
        body = {**body, "knowledgeops_request_id": request_id}
        return "POST", f"{base}{route}", body

    def _remote_id(self, payload: Dict[str, Any], field: str) -> str:
        value = str(payload.get(field) or "").strip()
        if not self.REMOTE_ID_PATTERN.fullmatch(value):
            raise ExternalAgentGatewayError(
                f"{field} must be a valid CompOps identifier"
            )
        return value

    @staticmethod
    def _remote_references(payload: Dict[str, Any]) -> Dict[str, str]:
        fields = (
            "project_id",
            "workflow_id",
            "run_id",
            "batch_id",
            "study_id",
            "artifact_id",
            "action_id",
        )
        return {
            field: str(payload[field])[:200]
            for field in fields
            if payload.get(field) is not None
        }

    async def _apply_authentication(
        self,
        *,
        headers: Dict[str, str],
        config: Dict[str, Any],
        user: User,
        db: AsyncSession,
    ) -> None:
        auth_type = config["auth_type"]
        if auth_type == "none":
            return
        secret_uuid = UUID(str(config["secret_id"]))
        secret = (
            await db.execute(
                select(UserSecret).where(
                    UserSecret.id == secret_uuid,
                    UserSecret.user_id == user.id,
                )
            )
        ).scalar_one_or_none()
        if secret is None:
            raise ExternalAgentGatewayError("External agent secret was not found")
        value = SecretService().decrypt(secret.encrypted_value)
        if not value:
            raise ExternalAgentGatewayError("External agent secret could not be read")
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {value}"
        else:
            headers[str(config["auth_header_name"])] = value

    @staticmethod
    async def _resolve_addresses(host: str, port: int) -> Iterable[str]:
        try:
            rows = await asyncio.get_running_loop().getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ExternalAgentGatewayError(
                "External agent host could not be resolved"
            ) from exc
        return sorted({str(row[4][0]) for row in rows if row[4]})

    @staticmethod
    def _assert_public_address(address: str, *, hostname: str) -> None:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ExternalAgentGatewayError(
                "External agent resolved to an invalid address"
            ) from exc
        allowed_private_hosts = {
            item.strip().lower()
            for item in settings.EXTERNAL_GATEWAY_PRIVATE_HOST_ALLOWLIST.split(",")
            if item.strip()
        }
        if (
            not parsed.is_global
            and hostname.strip().lower() not in allowed_private_hosts
        ):
            raise ExternalAgentGatewayError(
                "External agent endpoint resolved to a non-public network"
            )


external_agent_gateway_service = ExternalAgentGatewayService()
