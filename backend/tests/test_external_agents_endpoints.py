import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, Mock
from uuid import UUID

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.external_agent_gateway_service import external_agent_gateway_service


def _create_payload(**overrides):
    return {
        "name": "External Research Agent",
        "description": "A capability-scoped research collaborator.",
        "endpoint_url": "https://agent.example.test/invoke",
        "capabilities": ["research.summarize", "research.critique"],
        "auth_type": "none",
        "timeout_seconds": 15,
        **overrides,
    }


def test_registers_and_lists_external_agent(client, auth_headers):
    response = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(),
    )

    assert response.status_code == 201
    created = response.json()
    assert created["capabilities"] == [
        "research.summarize",
        "research.critique",
    ]
    assert created["auth_type"] == "none"
    assert created["provider_type"] == "generic_agent"
    assert created["secret_id"] is None

    listed = client.get("/api/v1/external-agents", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["agents"][0]["id"] == created["id"]

    registry = client.get("/api/v1/tools/registry", headers=auth_headers)
    assert registry.status_code == 200
    registered_tool = next(
        tool
        for tool in registry.json()["tools"]
        if tool["name"] == f"user_tool:{created['id']}"
    )
    assert registered_tool["tool_type"] == "external_agent"
    assert registered_tool["effects"] == "write"
    assert registered_tool["network"] == "egress"


def test_registers_compops_as_a_typed_external_system(client, auth_headers):
    response = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(
            name="CompOps Compiler Research",
            provider_type="compops",
            endpoint_url="https://compops.example.test",
            capabilities=[
                "compops.operators.list",
                "compops.runs.submit",
                "compops.studies.report",
                "compops.studies.gates.evaluate",
            ],
        ),
    )

    assert response.status_code == 201
    created = response.json()
    assert created["provider_type"] == "compops"
    assert created["endpoint_url"] == "https://compops.example.test"
    assert created["capabilities"][0] == "compops.operators.list"


def test_invocation_is_audited(
    client,
    auth_headers,
    db_session,
    monkeypatch,
):
    created = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(name="Audited Agent"),
    ).json()
    invoke = AsyncMock(
        return_value={
            "output": {"status": "completed", "answer": "Evidence-backed result"},
            "provenance": {
                "request_id": "request-1",
                "response_sha256": "a" * 64,
            },
        }
    )
    monkeypatch.setattr(external_agent_gateway_service, "invoke", invoke)

    response = client.post(
        f"/api/v1/external-agents/{created['id']}/invoke",
        headers=auth_headers,
        json={
            "capability": "research.summarize",
            "payload": {"topic": "compiler vectorization"},
            "request_id": "request-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["output"]["provenance"]["response_sha256"] == "a" * 64

    from app.models.tool_audit import ToolExecutionAudit

    async def _load_audit():
        return await db_session.get(ToolExecutionAudit, UUID(payload["audit_id"]))

    audit = asyncio.get_event_loop().run_until_complete(_load_audit())
    assert audit.status == "completed"
    assert audit.tool_name == f"user_tool:{created['id']}"
    assert audit.tool_input["capability"] == "research.summarize"
    assert "secret" not in str(audit.tool_input).lower()


def test_compops_invocation_links_sanitized_evidence_to_owned_rnd_job(
    client,
    auth_headers,
    db_session,
    test_user,
    monkeypatch,
):
    parent = AgentJob(
        name="Compiler research",
        goal="Investigate a compiler optimization",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        results={"evaluation_outcome": {"claims": [], "evidence": [], "actions": []}},
        output_artifacts=[],
    )

    async def _seed():
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_seed())
    created = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(
            name="CompOps evidence source",
            provider_type="compops",
            endpoint_url="https://compops.example.test",
            capabilities=["compops.studies.report"],
        ),
    ).json()
    invoke = AsyncMock(
        return_value={
            "output": {
                "study_id": "study-1",
                "raw_report": "sensitive remote compiler output",
            },
            "provenance": {
                "external_agent_id": created["id"],
                "external_agent_name": "CompOps evidence source",
                "endpoint_origin": "https://compops.example.test",
                "provider_type": "compops",
                "capability": "compops.studies.report",
                "request_id": "compops-import-1",
                "received_at": "2026-07-28T12:00:00+00:00",
                "response_sha256": "a" * 64,
                "response_bytes": 128,
                "execution_time_ms": 12,
                "remote_references": {"study_id": "study-1"},
            },
        }
    )
    monkeypatch.setattr(external_agent_gateway_service, "invoke", invoke)

    response = client.post(
        f"/api/v1/external-agents/{created['id']}/invoke",
        headers=auth_headers,
        json={
            "capability": "compops.studies.report",
            "payload": {"study_id": "study-1", "metric": "cycles"},
            "request_id": "compops-import-1",
            "agent_job_id": str(parent.id),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["evidence_linked"] is True
    assert payload["output"] is None

    from app.models.tool_audit import ToolExecutionAudit

    async def _load():
        await db_session.refresh(parent)
        audit = await db_session.get(ToolExecutionAudit, UUID(payload["audit_id"]))
        return audit

    audit = asyncio.get_event_loop().run_until_complete(_load())
    external = next(
        item
        for item in parent.results["evaluation_outcome"]["evidence"]
        if item["kind"] == "external_system_response"
    )
    assert external["id"] == "external-system:compops-import-1"
    assert external["external_system_type"] == "compops"
    assert external["remote_references"] == {"study_id": "study-1"}
    assert external["audit_id"] == payload["audit_id"]
    assert external["verification_status"] == "unverified"
    assert parent.results["evaluation_outcome"]["verification_plan"]["task_count"] == 1
    assert "sensitive remote compiler output" not in repr(parent.results)
    assert "sensitive remote compiler output" in repr(audit.tool_output)


def test_rejects_insecure_external_agent_endpoint(client, auth_headers):
    response = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(endpoint_url="http://agent.example.test/invoke"),
    )

    assert response.status_code == 422
    assert "HTTPS" in response.json()["detail"]


def test_policy_can_require_approval_before_external_invocation(
    client,
    auth_headers,
    db_session,
    test_user,
    monkeypatch,
):
    created = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(name="Approval-Gated Agent"),
    ).json()

    from app.models.tool_policy import ToolPolicy

    async def _seed_policy():
        db_session.add(
            ToolPolicy(
                subject_type="user",
                subject_id=test_user.id,
                tool_name=f"user_tool:{created['id']}",
                effect="allow",
                require_approval=True,
            )
        )
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed_policy())
    invoke = AsyncMock()
    monkeypatch.setattr(external_agent_gateway_service, "invoke", invoke)

    response = client.post(
        f"/api/v1/external-agents/{created['id']}/invoke",
        headers=auth_headers,
        json={
            "capability": "research.critique",
            "payload": {"claim": "Candidate claim"},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "requires_approval"
    assert response.json()["audit_id"]
    invoke.assert_not_awaited()


def test_approved_compops_call_links_evidence_using_captured_job_scope(
    client,
    auth_headers,
    admin_headers,
    db_session,
    test_user,
    monkeypatch,
):
    parent = AgentJob(
        name="Approval-gated compiler research",
        goal="Import an approved compiler study",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        results={"evaluation_outcome": {"claims": [], "evidence": [], "actions": []}},
        output_artifacts=[],
    )

    async def _seed():
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_seed())
    created = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(
            name="Approval-gated CompOps",
            provider_type="compops",
            endpoint_url="https://compops.example.test",
            capabilities=["compops.studies.gates.evaluate"],
        ),
    ).json()

    from app.models.tool_policy import ToolPolicy

    async def _seed_policy():
        db_session.add(
            ToolPolicy(
                subject_type="user",
                subject_id=test_user.id,
                tool_name=f"user_tool:{created['id']}",
                effect="allow",
                require_approval=True,
            )
        )
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed_policy())
    invoke = AsyncMock(
        return_value={
            "output": {"decision": "pass", "private_details": "audit-only"},
            "provenance": {
                "external_agent_id": created["id"],
                "external_agent_name": "Approval-gated CompOps",
                "endpoint_origin": "https://compops.example.test",
                "provider_type": "compops",
                "capability": "compops.studies.gates.evaluate",
                "request_id": "approved-import-1",
                "received_at": "2026-07-28T12:10:00+00:00",
                "response_sha256": "b" * 64,
                "response_bytes": 64,
                "execution_time_ms": 8,
                "remote_references": {"study_id": "study-approved"},
            },
        }
    )
    monkeypatch.setattr(external_agent_gateway_service, "invoke", invoke)

    requested = client.post(
        f"/api/v1/external-agents/{created['id']}/invoke",
        headers=auth_headers,
        json={
            "capability": "compops.studies.gates.evaluate",
            "payload": {"study_id": "study-approved"},
            "request_id": "approved-import-1",
            "agent_job_id": str(parent.id),
        },
    )
    assert requested.status_code == 200
    audit_id = requested.json()["audit_id"]
    assert requested.json()["status"] == "requires_approval"
    invoke.assert_not_awaited()

    owner_approval = client.post(
        f"/api/v1/audit/tools/{audit_id}/approve",
        headers=auth_headers,
        json={"note": "Owner approved bounded evidence import"},
    )
    assert owner_approval.status_code == 200
    assert owner_approval.json()["approval_status"] == "pending_admin"
    admin_approval = client.post(
        f"/api/v1/audit/tools/{audit_id}/approve",
        headers=admin_headers,
        json={"note": "Admin approved bounded evidence import"},
    )
    assert admin_approval.status_code == 200
    assert admin_approval.json()["approval_status"] == "approved"

    executed = client.post(
        f"/api/v1/audit/tools/{audit_id}/run",
        headers=auth_headers,
    )
    assert executed.status_code == 200
    assert executed.json()["status"] == "completed"
    invoke.assert_awaited_once()

    async def _refresh_parent():
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_refresh_parent())
    external = next(
        item
        for item in parent.results["evaluation_outcome"]["evidence"]
        if item["kind"] == "external_system_response"
    )
    assert external["remote_references"] == {"study_id": "study-approved"}
    assert external["audit_id"] == audit_id
    assert "audit-only" not in repr(parent.results)


def test_compops_subscription_reconciles_changed_digest_without_duplicate_evidence(
    client,
    auth_headers,
    db_session,
    test_user,
    monkeypatch,
):
    parent = AgentJob(
        name="Synchronized compiler research",
        goal="Track a long-running compiler run",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        results={"evaluation_outcome": {"claims": [], "evidence": [], "actions": []}},
        output_artifacts=[],
    )

    async def _seed():
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_seed())
    created = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(
            name="Synchronized CompOps",
            provider_type="compops",
            endpoint_url="https://compops.example.test",
            capabilities=["compops.runs.get"],
        ),
    ).json()

    def _result(digest, status):
        return {
            "output": {
                "run_id": "run-42",
                "status": status,
                "private_log": f"audit-only-{status}",
            },
            "provenance": {
                "external_agent_id": created["id"],
                "external_agent_name": "Synchronized CompOps",
                "endpoint_origin": "https://compops.example.test",
                "provider_type": "compops",
                "capability": "compops.runs.get",
                "request_id": f"request-{status}",
                "received_at": "2026-07-28T12:20:00+00:00",
                "response_sha256": digest,
                "response_bytes": 96,
                "execution_time_ms": 9,
                "remote_references": {"run_id": "run-42"},
            },
        }

    invoke = AsyncMock(
        side_effect=[
            _result("a" * 64, "running"),
            _result("a" * 64, "running"),
            _result("c" * 64, "completed"),
        ]
    )
    monkeypatch.setattr(external_agent_gateway_service, "invoke", invoke)

    created_subscription = client.post(
        f"/api/v1/external-agents/jobs/{parent.id}/compops-sync-subscriptions",
        headers=auth_headers,
        json={
            "tool_id": created["id"],
            "capability": "compops.runs.get",
            "payload": {"run_id": "run-42"},
            "interval_minutes": 5,
            "sync_immediately": True,
        },
    )
    assert created_subscription.status_code == 201
    subscription_id = created_subscription.json()["subscription"]["id"]
    assert created_subscription.json()["evidence_changed"] is True

    unchanged = client.post(
        (
            f"/api/v1/external-agents/jobs/{parent.id}/"
            f"compops-sync-subscriptions/{subscription_id}/sync"
        ),
        headers=auth_headers,
    )
    assert unchanged.status_code == 200
    assert unchanged.json()["evidence_changed"] is False

    changed = client.post(
        (
            f"/api/v1/external-agents/jobs/{parent.id}/"
            f"compops-sync-subscriptions/{subscription_id}/sync"
        ),
        headers=auth_headers,
    )
    assert changed.status_code == 200
    assert changed.json()["evidence_changed"] is True
    assert changed.json()["subscription"]["last_response_sha256"] == "c" * 64

    listed = client.get(
        f"/api/v1/external-agents/jobs/{parent.id}/compops-sync-subscriptions",
        headers=auth_headers,
    )
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    async def _refresh_parent():
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_refresh_parent())
    evidence = [
        item
        for item in parent.results["evaluation_outcome"]["evidence"]
        if item["kind"] == "external_system_response"
    ]
    assert len(evidence) == 1
    assert evidence[0]["id"] == f"external-system:{subscription_id}"
    assert evidence[0]["response_sha256"] == "c" * 64
    assert evidence[0]["remote_references"] == {"run_id": "run-42"}
    assert "audit-only-running" not in repr(parent.results)
    assert "audit-only-completed" not in repr(parent.results)


def test_signed_compops_webhook_is_replay_safe_and_queues_authoritative_refresh(
    client,
    auth_headers,
    db_session,
    test_user,
    monkeypatch,
):
    parent = AgentJob(
        name="Push-triggered compiler research",
        goal="Refresh a run after a signed CompOps event",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        results={"evaluation_outcome": {"claims": [], "evidence": [], "actions": []}},
        output_artifacts=[],
    )

    async def _seed():
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)

    asyncio.get_event_loop().run_until_complete(_seed())
    connection = client.post(
        "/api/v1/external-agents",
        headers=auth_headers,
        json=_create_payload(
            name="Webhook CompOps",
            provider_type="compops",
            endpoint_url="https://compops.example.test",
            capabilities=["compops.runs.get"],
        ),
    ).json()
    subscription_response = client.post(
        f"/api/v1/external-agents/jobs/{parent.id}/compops-sync-subscriptions",
        headers=auth_headers,
        json={
            "tool_id": connection["id"],
            "capability": "compops.runs.get",
            "payload": {"run_id": "run-webhook"},
            "interval_minutes": 15,
            "sync_immediately": False,
        },
    )
    assert subscription_response.status_code == 201
    subscription_id = subscription_response.json()["subscription"]["id"]
    setup = client.post(
        (
            f"/api/v1/external-agents/jobs/{parent.id}/"
            f"compops-sync-subscriptions/{subscription_id}/webhook"
        ),
        headers=auth_headers,
    )
    assert setup.status_code == 200
    secret = setup.json()["signing_secret"]
    assert setup.json()["subscription"]["webhook_enabled"] is True
    assert setup.json()["callback_path"].endswith(subscription_id)
    assert "webhook_secret_id" not in setup.json()["subscription"]

    from app.tasks.compops_sync_tasks import sync_compops_webhook_event

    monkeypatch.setattr(
        sync_compops_webhook_event,
        "delay",
        Mock(),
    )
    body = json.dumps(
        {"type": "run.completed", "untrusted_result": "must not be evidence"},
        separators=(",", ":"),
    ).encode("utf-8")
    timestamp = str(int(time.time()))
    event_id = "compops-event-1"
    signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + event_id.encode("utf-8") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    webhook_headers = {
        "Content-Type": "application/json",
        "X-CompOps-Timestamp": timestamp,
        "X-CompOps-Event-ID": event_id,
        "X-CompOps-Event-Type": "run.completed",
        "X-CompOps-Signature": f"v1={signature}",
    }
    received = client.post(
        f"/api/v1/external-agents/compops-webhooks/{subscription_id}",
        headers=webhook_headers,
        content=body,
    )
    assert received.status_code == 202
    assert received.json() == {
        "accepted": True,
        "duplicate": False,
        "event_id": event_id,
    }
    sync_compops_webhook_event.delay.assert_called_once()

    replay = client.post(
        f"/api/v1/external-agents/compops-webhooks/{subscription_id}",
        headers=webhook_headers,
        content=body,
    )
    assert replay.status_code == 202
    assert replay.json()["duplicate"] is True
    sync_compops_webhook_event.delay.assert_called_once()

    changed_body = b'{"type":"run.failed"}'
    changed_signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii")
        + b"."
        + event_id.encode("utf-8")
        + b"."
        + changed_body,
        hashlib.sha256,
    ).hexdigest()
    conflicting_replay = client.post(
        f"/api/v1/external-agents/compops-webhooks/{subscription_id}",
        headers={
            **webhook_headers,
            "X-CompOps-Signature": f"v1={changed_signature}",
        },
        content=changed_body,
    )
    assert conflicting_replay.status_code == 409

    stale_timestamp = str(int(time.time()) - 600)
    stale_event_id = "compops-event-stale"
    stale_signature = hmac.new(
        secret.encode("utf-8"),
        stale_timestamp.encode("ascii")
        + b"."
        + stale_event_id.encode("utf-8")
        + b"."
        + body,
        hashlib.sha256,
    ).hexdigest()
    stale = client.post(
        f"/api/v1/external-agents/compops-webhooks/{subscription_id}",
        headers={
            **webhook_headers,
            "X-CompOps-Timestamp": stale_timestamp,
            "X-CompOps-Event-ID": stale_event_id,
            "X-CompOps-Signature": f"v1={stale_signature}",
        },
        content=body,
    )
    assert stale.status_code == 401

    invalid = client.post(
        f"/api/v1/external-agents/compops-webhooks/{subscription_id}",
        headers={**webhook_headers, "X-CompOps-Signature": f"v1={'0' * 64}"},
        content=body,
    )
    assert invalid.status_code == 401

    from app.models.compops_evidence_subscription import CompOpsWebhookEvent

    async def _load_events():
        from sqlalchemy import select

        return list(
            (await db_session.execute(select(CompOpsWebhookEvent))).scalars().all()
        )

    events = asyncio.get_event_loop().run_until_complete(_load_events())
    assert len(events) == 1
    assert events[0].payload_sha256 == hashlib.sha256(body).hexdigest()
    assert "must not be evidence" not in repr(events[0].__dict__)
    assert "must not be evidence" not in repr(parent.results)
