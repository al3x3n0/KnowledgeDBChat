"""Render real Helm manifests to check worker/scaler integration (no cluster)."""

import subprocess
import unittest
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[1] / "helm/knowledgedbchat"


def render(*settings, overlay=False, production=False, kube_version="1.31.0"):
    command = ["helm", "template", "test", str(CHART)]
    command.extend(["--kube-version", kube_version])
    if production:
        command.extend(["-f", str(CHART / "values-prod.example.yaml")])
    if overlay:
        command.extend(["-f", str(CHART / "values-keda.yaml")])
    for setting in settings:
        command.extend(["--set", setting])
    return subprocess.run(command, capture_output=True, text=True, check=False)


class CeleryAutoscalingTests(unittest.TestCase):
    def documents(self, *settings, overlay=False, production=False):
        result = render(*settings, overlay=overlay, production=production)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [doc for doc in yaml.safe_load_all(result.stdout) if doc]

    def test_defaults_and_cpu_scaling_remain_available(self):
        for settings, expected_hpas in [
            ([], 0),
            (["celery.autoscaling.enabled=true"], 1),
        ]:
            with self.subTest(settings=settings):
                docs = self.documents(*settings)
                self.assertFalse(any(d["kind"] == "ScaledObject" for d in docs))
                self.assertEqual(
                    sum(d["kind"] == "HorizontalPodAutoscaler" for d in docs),
                    expected_hpas,
                )

    def test_every_scaler_targets_a_worker_and_its_actual_queue(self):
        docs = self.documents(
            "celeryLatex.enabled=true",
            "celeryTranscription.enabled=true",
            "config.extra.LATEX_COMPILER_CELERY_QUEUE=custom-latex",
            "config.extra.TRANSCRIPTION_CELERY_QUEUE=custom-audio",
            "secrets.redisPassword=test-password",
            overlay=True,
        )
        deployments = {
            d["metadata"]["name"]: d for d in docs if d["kind"] == "Deployment"
        }
        scalers = [d for d in docs if d["kind"] == "ScaledObject"]
        self.assertEqual(len(scalers), 4)
        self.assertFalse(any(d["kind"] == "HorizontalPodAutoscaler" for d in docs))
        for scaler in scalers:
            spec = scaler["spec"]
            deployment = deployments[spec["scaleTargetRef"]["name"]]
            self.assertNotIn("replicas", deployment["spec"])
            pod = deployment["spec"]["template"]["spec"]
            worker = pod["containers"][0]
            command = worker["command"]
            queue = command[command.index("-Q") + 1]
            if spec["triggers"][0]["type"] != "redis":
                continue  # Agent jobs use the database demand query.
            self.assertEqual(
                spec["scaleTargetRef"]["envSourceContainerName"], worker["name"]
            )
            self.assertEqual(spec["triggers"][0]["metadata"]["listName"], queue)
            self.assertGreaterEqual(pod["terminationGracePeriodSeconds"], 1860)
            env = {e["name"]: e for e in worker["env"]}
            self.assertIn("secretKeyRef", env["REDIS_PASSWORD"]["valueFrom"])
            for trigger in spec["triggers"]:
                self.assertEqual(
                    trigger["metadata"]["passwordFromEnv"], "REDIS_PASSWORD"
                )
                self.assertNotIn("test-password", str(trigger))

    def test_agent_fallback_and_multiple_queues_are_all_measured(self):
        docs = self.documents(
            "celeryAgents.enabled=false",
            r"celery.queues=celery\,imports",
            "config.extra.AGENT_JOB_CELERY_QUEUE=custom-agents",
            overlay=True,
        )
        scalers = [d for d in docs if d["kind"] == "ScaledObject"]
        self.assertEqual(len(scalers), 1)
        self.assertEqual(
            [t["metadata"]["listName"] for t in scalers[0]["spec"]["triggers"]],
            ["celery", "imports", "custom-agents"],
        )

    def test_overlay_preserves_production_replica_bounds(self):
        docs = self.documents(production=True, overlay=True)
        scalers = {
            d["metadata"]["labels"]["app.kubernetes.io/component"]: d["spec"]
            for d in docs
            if d["kind"] == "ScaledObject"
        }
        self.assertEqual(len(scalers), 4)
        for component, minimum, maximum in [
            ("celery", 3, 10),
            ("celery-agents", 2, 5),
            ("celery-latex", 2, 5),
            ("celery-transcription", 1, 5),
        ]:
            with self.subTest(component=component):
                self.assertEqual(scalers[component]["minReplicaCount"], minimum)
                self.assertEqual(scalers[component]["maxReplicaCount"], maximum)

    def test_queue_target_defaults_to_concurrency_and_can_be_tuned(self):
        for settings, expected in [
            (["celery.concurrency=7"], "7"),
            (["celery.concurrency=7", "celery.autoscaling.queueLength=12"], "12"),
        ]:
            with self.subTest(settings=settings):
                docs = self.documents(*settings, overlay=True)
                scaler = next(
                    d
                    for d in docs
                    if d["kind"] == "ScaledObject"
                    and d["metadata"]["labels"]["app.kubernetes.io/component"]
                    == "celery"
                )
                self.assertEqual(
                    scaler["spec"]["triggers"][0]["metadata"]["listLength"], expected
                )

    def test_external_redis_and_existing_secret(self):
        docs = self.documents(
            "redis.enabled=false",
            "redis.external.host=broker.example",
            "redis.external.port=6380",
            "secrets.create=false",
            "secrets.existingSecret=managed",
            "secrets.redisPassword=external",
            "networkPolicy.enabled=true",
            overlay=True,
        )
        scaler = next(
            d
            for d in docs
            if d["kind"] == "ScaledObject"
            and d["spec"]["triggers"][0]["type"] == "redis"
        )
        self.assertEqual(
            scaler["spec"]["triggers"][0]["metadata"]["address"], "broker.example:6380"
        )
        self.assertFalse(
            any(d["metadata"]["name"].endswith("redis-from-keda") for d in docs)
        )

    def test_unauthenticated_redis_does_not_request_password(self):
        docs = self.documents(overlay=True)
        scaler = next(
            d
            for d in docs
            if d["kind"] == "ScaledObject"
            and d["spec"]["triggers"][0]["type"] == "redis"
        )
        self.assertNotIn("passwordFromEnv", scaler["spec"]["triggers"][0]["metadata"])

    def test_network_policy_allows_operator_namespace_to_redis_only(self):
        docs = self.documents(
            "networkPolicy.enabled=true",
            "celeryQueueAutoscaling.kedaNamespace=scaling",
            overlay=True,
        )
        policy = next(
            d for d in docs if d["metadata"]["name"].endswith("redis-from-keda")
        )
        spec = policy["spec"]
        self.assertEqual(
            spec["podSelector"]["matchLabels"]["app.kubernetes.io/component"], "redis"
        )
        self.assertEqual(
            spec["ingress"][0]["ports"], [{"protocol": "TCP", "port": 6379}]
        )
        self.assertEqual(
            spec["ingress"][0]["from"][0]["namespaceSelector"]["matchLabels"][
                "kubernetes.io/metadata.name"
            ],
            "scaling",
        )

    def test_disabled_workers_have_no_scalers(self):
        docs = self.documents(
            "celery.enabled=false", "celeryAgents.enabled=false", overlay=True
        )
        self.assertFalse(any(d["kind"] == "ScaledObject" for d in docs))

    def test_worker_daemon_outlives_draining_tasks_with_keda(self):
        docs = self.documents("sandbox.enabled=true", overlay=True)
        for component in ("celery", "celery-agents"):
            deployment = next(
                d
                for d in docs
                if d["kind"] == "Deployment"
                and d["metadata"]["labels"]["app.kubernetes.io/component"] == component
            )
            pod = deployment["spec"]["template"]["spec"]
            self.assertEqual([c["name"] for c in pod["containers"]], [component])
            daemon = pod["initContainers"][0]
            self.assertEqual(daemon["name"], "sandbox-docker")
            self.assertEqual(daemon["restartPolicy"], "Always")
            self.assertTrue(daemon["securityContext"]["privileged"])
            mounts = {m["name"]: m["mountPath"] for m in daemon["volumeMounts"]}
            self.assertEqual(mounts["sandbox-work"], "/sandbox-work")
            self.assertEqual(mounts["sandbox-run"], "/run/sandbox")

    def test_disabled_or_excluded_sandbox_has_no_native_sidecar(self):
        for settings in [[], ["sandbox.enabled=true", "sandbox.components={backend}"]]:
            docs = self.documents(*settings, overlay=True)
            for deployment in (d for d in docs if d["kind"] == "Deployment"):
                component = deployment["metadata"]["labels"][
                    "app.kubernetes.io/component"
                ]
                if component in ("celery", "celery-agents"):
                    self.assertNotIn(
                        "initContainers", deployment["spec"]["template"]["spec"]
                    )

    def test_legacy_sidecar_remains_available_without_keda_overlay(self):
        docs = self.documents("sandbox.enabled=true")
        for deployment in (d for d in docs if d["kind"] == "Deployment"):
            component = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
            if component in ("backend", "celery", "celery-agents"):
                pod = deployment["spec"]["template"]["spec"]
                self.assertNotIn("initContainers", pod)
                daemon = pod["containers"][1]
                self.assertEqual(daemon["name"], "sandbox-docker")
                self.assertNotIn("restartPolicy", daemon)

    def test_native_sidecars_require_supported_kubernetes_version(self):
        result = render("sandbox.enabled=true", overlay=True, kube_version="1.28.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Kubernetes >=1.29", result.stderr)
        # The overlay does not require sidecar support when sandbox is off.
        result = render(overlay=True, kube_version="1.28.0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_independent_jobs_wire_only_worker_permissions_and_shared_storage(self):
        docs = self.documents(
            "kubernetesSandbox.enabled=true", "sandbox.enabled=true", overlay=True
        )
        role = next(
            d
            for d in docs
            if d["kind"] == "Role" and d["metadata"]["name"].endswith("sandbox-runner")
        )
        self.assertEqual(role["rules"][0]["resources"], ["jobs"])
        self.assertEqual(role["rules"][0]["verbs"], ["create", "get", "delete"])
        self.assertEqual(role["rules"][1]["resources"], ["pods"])
        self.assertEqual(role["rules"][1]["verbs"], ["list"])
        experiment_account = next(
            d
            for d in docs
            if d["kind"] == "ServiceAccount"
            and d["metadata"]["name"].endswith("sandbox-experiment")
        )
        self.assertFalse(experiment_account["automountServiceAccountToken"])
        for component in ("celery", "celery-agents"):
            deployment = next(
                d
                for d in docs
                if d["kind"] == "Deployment"
                and d["metadata"]["labels"]["app.kubernetes.io/component"] == component
            )
            pod = deployment["spec"]["template"]["spec"]
            self.assertTrue(pod["serviceAccountName"].endswith("sandbox-runner"))
            self.assertNotIn("initContainers", pod)
            self.assertEqual(len(pod["containers"]), 1)
            env = {e["name"]: e.get("value") for e in pod["containers"][0]["env"]}
            self.assertEqual(env["SANDBOX_EXECUTION_BACKEND"], "kubernetes")
            self.assertEqual(env["KUBERNETES_SANDBOX_DEFER_ENABLED"], "true")
            self.assertEqual(env["KUBERNETES_SANDBOX_POLL_BATCH_SIZE"], "20")
            self.assertTrue(env["KUBERNETES_SANDBOX_PVC"].endswith("-data"))
        policy = next(
            d
            for d in docs
            if d["kind"] == "NetworkPolicy"
            and d["metadata"]["name"].endswith("sandbox-experiments")
        )
        self.assertEqual(policy["spec"]["policyTypes"], ["Ingress", "Egress"])
        self.assertEqual(policy["spec"]["ingress"], [])
        self.assertEqual(policy["spec"]["egress"], [])
        self.assertEqual(
            policy["metadata"]["annotations"]["helm.sh/resource-policy"], "keep"
        )

    def test_independent_jobs_require_rwx_storage(self):
        for setting in (
            "backend.persistence.enabled=false",
            "backend.persistence.accessMode=ReadWriteOnce",
        ):
            result = render("kubernetesSandbox.enabled=true", setting)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ReadWriteMany", result.stderr)

    def test_continuations_can_be_disabled_without_disabling_independent_jobs(self):
        docs = self.documents(
            "kubernetesSandbox.enabled=true", "kubernetesSandbox.defer.enabled=false"
        )
        worker = next(
            d
            for d in docs
            if d["kind"] == "Deployment"
            and d["metadata"]["labels"]["app.kubernetes.io/component"] == "celery"
        )
        env = {
            e["name"]: e.get("value")
            for e in worker["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        self.assertEqual(env["SANDBOX_EXECUTION_BACKEND"], "kubernetes")
        self.assertEqual(env["KUBERNETES_SANDBOX_DEFER_ENABLED"], "false")

    def test_independent_jobs_can_retain_daemons_for_mixed_docker_workloads(self):
        docs = self.documents(
            "kubernetesSandbox.enabled=true",
            "kubernetesSandbox.keepDockerSidecar=true",
            "sandbox.enabled=true",
            overlay=True,
        )
        deployment = next(
            d
            for d in docs
            if d["kind"] == "Deployment"
            and d["metadata"]["labels"]["app.kubernetes.io/component"]
            == "celery-agents"
        )
        pod = deployment["spec"]["template"]["spec"]
        self.assertEqual(pod["initContainers"][0]["name"], "sandbox-docker")
        self.assertEqual(pod["initContainers"][0]["restartPolicy"], "Always")

    def test_invalid_scaling_settings_fail_before_deployment(self):
        for setting, message in [
            ("celery.autoscaling.mode=unknown", "must be cpu or redis"),
            ("celery.autoscaling.minReplicas=0", "minReplicas"),
            ("celery.autoscaling.maxReplicas=0", "maxReplicas"),
            ("celery.autoscaling.queueLength=-1", "queueLength"),
            ("celery.autoscaling.scaleDownStabilizationSeconds=3601", "timing"),
            ("celery.autoscaling.scaleDownPeriodSeconds=1801", "timing"),
        ]:
            with self.subTest(setting=setting):
                result = render(setting, overlay=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
