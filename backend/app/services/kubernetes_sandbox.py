"""Bounded sandbox steps as Kubernetes Jobs, reattached by journal identity.

The worker submits and waits; it does not own the lifetime of the Job. A lost
worker leaves the experiment running, and replaying its journaled tool intent
reattaches to the same Job and persistent workspace. This is submission
deduplication, not an exactly-once guarantee for code inside a Kubernetes Pod.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

import httpx

from app.core.config import settings
from app.services import kubernetes_sandbox_artifacts as artifacts
from app.services.sandbox_invocation import (
    SandboxStepPending,
    current,
    synchronous_wait_required,
)

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


@dataclass
class Configuration:
    namespace: str
    claim: str
    root: Path
    subpath: str = "kubernetes-sandbox"
    owner: str = "kdbc"
    service_account: str = "default"
    profiles: dict = field(default_factory=dict)
    image_profiles: dict = field(default_factory=dict)
    poll_seconds: float = 5
    ttl_seconds: int = 86400
    input_bytes: int = 512 * 1024 * 1024
    output_bytes: int = 1024 * 1024 * 1024
    log_bytes: int = 2 * 1024 * 1024
    max_seconds: int = 21600
    max_memory_mb: int = 8192
    max_cpus: float = 8

    def profile(self, image: str) -> dict:
        name = self.image_profiles.get(image, "default")
        if name != "default" and name not in self.profiles:
            raise ValueError(f"Unknown Kubernetes sandbox resource profile: {name}")
        return {**self.profiles.get("default", {}), **self.profiles.get(name, {})}

    @classmethod
    def configured(cls):
        namespace = settings.KUBERNETES_SANDBOX_NAMESPACE
        if not namespace:
            namespace = (SERVICE_ACCOUNT / "namespace").read_text().strip()
        return cls(
            namespace=namespace,
            claim=settings.KUBERNETES_SANDBOX_PVC,
            root=Path(settings.KUBERNETES_SANDBOX_ROOT),
            subpath=settings.KUBERNETES_SANDBOX_PVC_SUBPATH,
            owner=settings.KUBERNETES_SANDBOX_OWNER,
            service_account=settings.KUBERNETES_SANDBOX_SERVICE_ACCOUNT,
            profiles=json.loads(settings.KUBERNETES_SANDBOX_PROFILES),
            image_profiles=json.loads(settings.KUBERNETES_SANDBOX_IMAGE_PROFILES),
            poll_seconds=settings.KUBERNETES_SANDBOX_POLL_SECONDS,
            ttl_seconds=settings.KUBERNETES_SANDBOX_TTL_SECONDS,
            input_bytes=settings.KUBERNETES_SANDBOX_MAX_INPUT_MB * 1024 * 1024,
            output_bytes=settings.KUBERNETES_SANDBOX_MAX_OUTPUT_MB * 1024 * 1024,
            log_bytes=settings.KUBERNETES_SANDBOX_LOG_BYTES,
            max_seconds=settings.KUBERNETES_SANDBOX_MAX_TIMEOUT_SECONDS,
            max_memory_mb=settings.SCIENTIFIC_VALIDATION_MAX_MEMORY_MB,
            max_cpus=settings.SCIENTIFIC_VALIDATION_MAX_CPUS,
        )

    def validate(self):
        if not self.namespace or not self.claim or not self.root.is_absolute():
            raise ValueError(
                "Kubernetes sandbox requires namespace, PVC and an absolute artifact root"
            )
        path = PurePosixPath(self.subpath)
        if (
            not self.subpath
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != self.subpath
        ):
            raise ValueError(
                "Kubernetes sandbox PVC subpath must be a normalized relative directory"
            )
        if (
            min(
                self.poll_seconds,
                self.ttl_seconds,
                self.input_bytes,
                self.output_bytes,
                self.log_bytes,
                self.max_seconds,
                self.max_memory_mb,
                self.max_cpus,
            )
            <= 0
        ):
            raise ValueError("Kubernetes sandbox limits must be positive")


class KubernetesAPI:
    """In-cluster REST client; reload projected tokens on every request."""

    def __init__(self):
        if (
            not (SERVICE_ACCOUNT / "ca.crt").is_file()
            or not (SERVICE_ACCOUNT / "token").is_file()
        ):
            raise RuntimeError(
                "Kubernetes sandbox requires mounted in-cluster ServiceAccount credentials"
            )
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        self.client = httpx.AsyncClient(
            base_url=f"https://{host}:{port}",
            verify=str(SERVICE_ACCOUNT / "ca.crt"),
            timeout=15,
            trust_env=False,
        )

    async def request(self, method, path, **kwargs):
        try:
            token = (SERVICE_ACCOUNT / "token").read_text().strip()
        except OSError as exc:
            raise RuntimeError(
                "Kubernetes sandbox ServiceAccount token is unavailable"
            ) from exc
        response = await self.client.request(
            method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )
        if response.status_code not in (404, 409):
            response.raise_for_status()
        return response

    async def close(self):
        await self.client.aclose()


def resources(memory: str, cpus: str, config: Configuration) -> dict:
    match = re.fullmatch(r"([0-9]+)([kmg]?)i?b?", str(memory).lower())
    if not match:
        raise ValueError(
            "Sandbox memory must be bytes or a whole number with k/m/g units"
        )
    size = int(match[1]) * {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[match[2]]
    try:
        cpu = Decimal(str(cpus))
    except InvalidOperation as exc:
        raise ValueError("Invalid sandbox CPU request") from exc
    if (
        not cpu.is_finite()
        or cpu < Decimal("0.001")
        or cpu > Decimal(str(config.max_cpus))
    ):
        raise ValueError("Sandbox CPU request exceeds the configured limit")
    if cpu % Decimal("0.001"):
        raise ValueError("Sandbox CPU request must use whole millicores")
    if size <= 0 or size > config.max_memory_mb * 1024 * 1024:
        raise ValueError("Sandbox memory request exceeds the configured limit")
    return {"cpu": str(cpu), "memory": str(size), "ephemeral-storage": "4Gi"}


def manifest(name: str, request: dict, fingerprint: str, config: Configuration) -> dict:
    profile = config.profile(request["image"])
    limits = dict(request["resources"])
    for key, value in profile.get("extraResources", {}).items():
        if "/" not in key or not str(value).isdigit() or int(value) <= 0:
            raise ValueError(
                "Sandbox profile extraResources must be positive extended resources"
            )
        limits[key] = str(value)
    prefix = f"{config.subpath}/{name}"
    pod = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "serviceAccountName": config.service_account,
        "enableServiceLinks": False,
        "terminationGracePeriodSeconds": 15,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65534,
            "runAsGroup": 65534,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "sandbox",
                "image": request["image"],
                "imagePullPolicy": "IfNotPresent",
                "command": ["python3", "/runner/launcher.py"],
                "workingDir": "/work",
                "env": [
                    {"name": "HOME", "value": "/tmp"},
                    {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                ],
                "resources": {"requests": limits, "limits": dict(limits)},
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    {
                        "name": "artifacts",
                        "mountPath": "/work",
                        "subPath": f"{prefix}/work",
                    },
                    {
                        "name": "artifacts",
                        "mountPath": "/runner",
                        "subPath": f"{prefix}/control",
                        "readOnly": True,
                    },
                    {
                        "name": "artifacts",
                        "mountPath": "/runner-output",
                        "subPath": f"{prefix}/io",
                    },
                    {"name": "tmp", "mountPath": "/tmp"},
                ],
            }
        ],
        "volumes": [
            {"name": "artifacts", "persistentVolumeClaim": {"claimName": config.claim}},
            {"name": "tmp", "emptyDir": {"sizeLimit": "4Gi"}},
        ],
    }
    for key in ("nodeSelector", "tolerations", "runtimeClassName", "imagePullSecrets"):
        if profile.get(key):
            pod[key] = profile[key]
    labels = {
        "kdbc.io/runner": config.owner,
        "app.kubernetes.io/component": "sandbox-job",
    }
    deadline_seconds = request["timeout_seconds"]
    if request.get("agent_deadline_at") is not None:
        deadline_seconds = min(
            deadline_seconds, max(1, int(request["agent_deadline_at"] - time.time()))
        )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": config.namespace,
            "labels": labels,
            "annotations": {"kdbc.io/request-sha256": fingerprint},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": deadline_seconds,
            "ttlSecondsAfterFinished": config.ttl_seconds,
            "parallelism": 1,
            "completions": 1,
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }


class SandboxJobCancelled(RuntimeError):
    pass


async def cancel(
    api, config: Configuration, name: str, *, fingerprint: str, uid: str = ""
) -> bool:
    """Delete only the observed execution, and report confirmed removal."""
    path = f"/apis/batch/v1/namespaces/{config.namespace}/jobs/{name}"
    response = await api.request("GET", path)
    if response.status_code == 404:
        return True
    response.raise_for_status()
    metadata = response.json()["metadata"]
    observed_uid = metadata["uid"]
    if (uid and uid != observed_uid) or (
        metadata.get("annotations", {}).get("kdbc.io/request-sha256") != fingerprint
    ):
        raise ValueError("Sandbox Job identity changed; refusing to cancel it")
    response = await api.request(
        "DELETE",
        path,
        json={
            "propagationPolicy": "Foreground",
            "gracePeriodSeconds": 15,
            "preconditions": {"uid": observed_uid},
        },
    )
    if response.status_code == 409:
        raise ValueError("Sandbox Job identity changed during cancellation")
    if response.status_code not in (200, 202, 404):
        response.raise_for_status()
    response = await api.request("GET", path)
    if response.status_code == 404:
        return True
    response.raise_for_status()
    if response.json()["metadata"]["uid"] != observed_uid:
        raise ValueError("Sandbox Job was replaced during cancellation")
    return False


def _job_name(config: Configuration, identity: str) -> str:
    key = hashlib.sha256(
        f"{config.namespace}:{config.owner}:{identity}".encode()
    ).hexdigest()[:40]
    return f"kdbc-step-{key}"


def _replay_states(
    config: Configuration, context, identity: str, workspace_key
) -> tuple:
    """Authorize earlier-step replay using the saved later step's snapshots."""
    reference = context.resume_reference if context else None
    if not reference or "step" not in reference or "workspace" not in reference:
        return ()
    step = reference["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Invalid sandbox continuation reference step")
    workspace = reference["workspace"]
    if isinstance(workspace, bool) or not isinstance(workspace, int) or workspace <= 0:
        raise ValueError("Invalid sandbox continuation reference workspace")
    prefix = f"{context.job_id}:{context.key}:"
    name = _job_name(config, f"{prefix}{step}")
    if reference.get("namespace") != config.namespace or reference.get("name") != name:
        raise ValueError("Sandbox continuation reference belongs to another action")
    if identity != f"{prefix}{context.calls}" or context.calls >= step:
        return ()
    # Derive the path from identity, not a stored/user-supplied filesystem path.
    record = json.loads((config.root / name / "record.json").read_text())
    if (
        record["fingerprint"] != reference.get("fingerprint")
        or (reference.get("uid") and record.get("uid") != reference["uid"])
        or record.get("workspace") != reference["workspace"]
    ):
        raise ValueError("Sandbox continuation reference no longer matches its record")
    if workspace_key != reference["workspace"]:
        return ()
    return tuple(
        record[key]
        for key in ("inputs", "outputs")
        if isinstance(record.get(key), dict)
    )


async def run(
    script: str,
    workdir: str,
    *,
    image: str,
    timeout_seconds: int,
    memory: str,
    cpus: str,
    step_identity: str = "",
    config=None,
    api=None,
) -> tuple:
    # The public runtime dispatch gates execution and the image allowlist.
    config = config or Configuration.configured()
    config.validate()
    if not 1 <= timeout_seconds <= config.max_seconds:
        raise ValueError("Sandbox timeout exceeds the configured Kubernetes limit")
    context = current.get()
    defer = bool(context and context.defer and not synchronous_wait_required())
    identity = step_identity or (
        context.next_key() if context and context.key else uuid.uuid4().hex
    )
    workspace_key = context.workspace_key(workdir) if context else None
    name = _job_name(config, identity)
    request = {
        "script": script,
        "image": image,
        "timeout_seconds": timeout_seconds,
        "resources": resources(memory, cpus, config),
        "pids": 256,
        "max_file_bytes": config.output_bytes,
        "claim": config.claim,
        "subpath": config.subpath,
        "profile": config.profile(image),
    }
    if context and context.deadline_at is not None:
        request["agent_deadline_at"] = context.deadline_at
    replay_states = await asyncio.to_thread(
        _replay_states, config, context, identity, workspace_key
    )
    directory, record = await asyncio.to_thread(
        artifacts.prepare,
        config.root,
        name,
        Path(workdir),
        request,
        config.input_bytes,
        replay_states=replay_states,
        workspace_key=workspace_key,
    )
    reference = {"name": name, "namespace": config.namespace, "path": str(directory)}
    reference["fingerprint"] = record["fingerprint"]
    if record.get("uid"):
        reference["uid"] = record["uid"]
    if context:
        if identity == f"{context.job_id}:{context.key}:{context.calls}":
            reference["step"] = context.calls
        if record.get("workspace") is not None:
            reference["workspace"] = record["workspace"]
        context.runs.append(reference)
    owned_api = api is None
    api = api or KubernetesAPI()
    path = f"/apis/batch/v1/namespaces/{config.namespace}/jobs/{name}"
    try:
        if (
            record.get("stop_reason") == "timed_out"
            and record["status"] != "timed_out"
            and defer
        ):
            # An early/manual delivery must leave cleanup to the poller too.
            # In particular, it cannot turn a late result into a success.
            raise SandboxStepPending(reference)
        if record["status"] not in ("completed", "failed", "timed_out", "cancelled"):
            response = await api.request("GET", path)
            if response.status_code == 404:
                if context and context.cancelled and await context.cancelled():
                    raise SandboxJobCancelled(
                        "Agent was cancelled before sandbox submission"
                    )
                if record["status"] != "prepared":
                    raise RuntimeError(
                        "Sandbox Job disappeared before its result was recorded; inspect its persistent artifacts"
                    )
                if (
                    context
                    and context.deadline_at is not None
                    and time.time() >= context.deadline_at
                ):
                    raise asyncio.TimeoutError(
                        "Agent runtime budget expired before sandbox submission"
                    )
                response = await api.request(
                    "POST",
                    f"/apis/batch/v1/namespaces/{config.namespace}/jobs",
                    json=manifest(name, request, record["fingerprint"], config),
                )
                if response.status_code == 409:
                    response = await api.request("GET", path)
            response.raise_for_status()
            job = response.json()
            if (
                job["metadata"].get("annotations", {}).get("kdbc.io/request-sha256")
                != record["fingerprint"]
            ):
                raise RuntimeError(
                    "Existing Kubernetes Job does not match this sandbox invocation"
                )
            if record.get("uid") and record["uid"] != job["metadata"]["uid"]:
                raise RuntimeError(
                    "Sandbox Job was replaced; refusing to consume another execution's result"
                )
            record.update(status="running", uid=job["metadata"]["uid"])
            reference.update(uid=record["uid"], fingerprint=record["fingerprint"])
            await asyncio.to_thread(artifacts.save, directory / "record.json", record)
            # The Job deadline is authoritative. The local bound also protects
            # against an API outage; a retry reattaches instead of resubmitting.
            deadline = time.monotonic() + timeout_seconds + 60
            while True:
                if job["metadata"]["uid"] != record["uid"]:
                    raise RuntimeError(
                        "Sandbox Job was replaced while waiting; refusing its result"
                    )
                if context and context.cancelled and await context.cancelled():
                    await cancel(
                        api,
                        config,
                        name,
                        fingerprint=record["fingerprint"],
                        uid=record["uid"],
                    )
                    record["status"] = "cancelled"
                    await asyncio.to_thread(
                        artifacts.save, directory / "record.json", record
                    )
                    raise SandboxJobCancelled("Sandbox Job cancelled by its owner")
                conditions = job.get("status", {}).get("conditions", [])
                terminal = next(
                    (
                        c
                        for c in conditions
                        if c.get("status") == "True"
                        and c["type"] in ("Complete", "Failed")
                    ),
                    None,
                )
                if terminal:
                    if terminal.get("reason") == "DeadlineExceeded":
                        record["status"] = "timed_out"
                    else:
                        pods = await api.request(
                            "GET",
                            f"/api/v1/namespaces/{config.namespace}/pods",
                            params={
                                "labelSelector": f"batch.kubernetes.io/controller-uid={record['uid']}"
                            },
                        )
                        pods.raise_for_status()
                        exits = [
                            {
                                "exit": s["state"]["terminated"],
                                "pod_uid": p.get("metadata", {}).get("uid", ""),
                                "node": p.get("spec", {}).get("nodeName", ""),
                                "image_id": s.get("imageID", ""),
                            }
                            for p in pods.json()["items"]
                            if p.get("metadata", {})
                            .get("labels", {})
                            .get("batch.kubernetes.io/controller-uid")
                            == record["uid"]
                            for s in p.get("status", {}).get("containerStatuses", [])
                            if s.get("name") == "sandbox"
                            and "terminated" in s.get("state", {})
                        ]
                        if len(exits) != 1:
                            raise RuntimeError(
                                "Sandbox Job has no unique container result; inspect the Job and artifacts"
                            )
                        record.update(
                            status="completed"
                            if terminal["type"] == "Complete"
                            else "failed",
                            returncode=exits[0]["exit"]["exitCode"],
                            reason=exits[0]["exit"].get("reason", ""),
                            execution=exits[0],
                        )
                        record["outputs"] = await asyncio.to_thread(
                            artifacts.files, directory / "work", config.output_bytes
                        )
                    await asyncio.to_thread(
                        artifacts.save, directory / "record.json", record
                    )
                    break
                if defer:
                    raise SandboxStepPending(reference)
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Stopped waiting for Kubernetes; the bounded Job can be reattached on retry"
                    )
                await asyncio.sleep(config.poll_seconds)
                try:
                    response = await api.request("GET", path)
                    response.raise_for_status()
                    job = response.json()
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 500
                    ):
                        raise
                    continue
        if record["status"] == "timed_out" or record.get("returncode") == 124:
            raise asyncio.TimeoutError("Kubernetes sandbox exceeded its deadline")
        if record["status"] == "cancelled":
            raise SandboxJobCancelled("Sandbox Job was cancelled")
        reference.update(record.get("execution", {}))
        if (
            not (directory / "io" / "stdout").exists()
            or not (directory / "io" / "stderr").exists()
        ):
            raise RuntimeError(
                "Kubernetes sandbox launcher did not start; the tool image must provide python3 and /bin/sh"
            )
        await asyncio.to_thread(
            artifacts.restore,
            directory,
            Path(workdir),
            config.output_bytes,
            expected=record["outputs"],
        )
        stdout = await asyncio.to_thread(
            artifacts.log, directory, "stdout", config.log_bytes
        )
        stderr = await asyncio.to_thread(
            artifacts.log, directory, "stderr", config.log_bytes
        )
        if record.get("reason") == "OOMKilled":
            stderr += (
                "\nKubernetes killed the sandbox after it exceeded its memory limit."
            )
        return record["returncode"], stdout, stderr
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        transient = (
            isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
        )
        if defer and transient:
            raise SandboxStepPending(reference) from exc
        raise
    finally:
        # Cancellation of the waiter (lease loss, worker shutdown) intentionally
        # leaves the independent Job alive for another owner to reattach.
        if owned_api:
            await api.close()
