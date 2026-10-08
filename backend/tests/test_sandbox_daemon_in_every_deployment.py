"""Every deployment can be given the daemon sandboxed tools run in.

The development stack had `sandbox-docker`; the production compose file and
the Helm chart had nothing, so every compiler, gem5, BOLT and skill tool
answered "could not run" there. Both now have an opt-in (a privileged
container is a decision): `docker-compose.sandbox.yml` and `sandbox.enabled`.

What these check is the part that fails silently. `-v <dir>:/work` is resolved
on the daemon's filesystem, so a client whose TMPDIR the daemon does not mount
at the same path gives every sandbox an empty /work -- and the dind entrypoint
covers /tmp with a tmpfs, which hides a volume mounted there.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CLIENTS = ("backend", "celery", "celery_agents")
HELPERS = ROOT / "deploy/helm/knowledgedbchat/templates/_helpers.tpl"


def _env(service: dict) -> dict:
    env = service.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env if "=" in item)
    return env


def _mounts(service: dict) -> dict:
    """volume name -> path inside the container."""
    mounts = {}
    for volume in service.get("volumes") or []:
        source, target = str(volume).split(":")[:2]
        mounts[source] = target
    return mounts


@pytest.mark.parametrize(
    "compose", ["docker-compose.yml", "docker-compose.sandbox.yml"]
)
def test_each_client_shares_its_work_directory_with_the_daemon(compose):
    path = ROOT / compose
    if not path.exists():  # pragma: no cover - a checkout without deploy files
        pytest.skip(f"{compose} not present")
    services = yaml.safe_load(path.read_text())["services"]
    daemon = services["sandbox-docker"]
    assert daemon.get("privileged") is True
    daemon_mounts = _mounts(daemon)

    for name in CLIENTS:
        service = services[name]
        if "extends" in service:  # copies a client checked here
            assert service["extends"]["service"] in CLIENTS
            continue
        env = _env(service)
        assert env.get("DOCKER_HOST") == "tcp://sandbox-docker:2376", name
        assert env.get("DOCKER_TLS_VERIFY") == "1", name
        work = env.get("TMPDIR")
        assert work and not work.startswith("/tmp"), (name, work)
        volume = next((v for v, t in _mounts(service).items() if t == work), None)
        assert volume, f"{name}: TMPDIR {work} is not a mounted volume"
        assert (
            daemon_mounts.get(volume) == work
        ), f"{name}: the daemon must mount {volume} at {work} too"


def test_the_production_stack_alone_has_no_privileged_container():
    services = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text())[
        "services"
    ]
    assert not [n for n, s in services.items() if s.get("privileged")]


def test_the_chart_sidecar_shares_socket_and_work_directory():
    if not HELPERS.exists():  # pragma: no cover
        pytest.skip("chart not present")
    text = HELPERS.read_text()

    def block(name: str) -> str:
        start = text.index(f'define "kdbc.sandbox.{name}"')
        return text[start : text.index("{{- end }}\n\n", start)]

    env, mounts, container = block("env"), block("volumeMounts"), block("container")
    assert "unix:///run/sandbox/docker.sock" in env
    assert "--host=unix:///run/sandbox/docker.sock" in container
    for where in (mounts, container):
        assert "mountPath: /run/sandbox" in where
        assert "mountPath: /sandbox-work" in where
    assert "value: /sandbox-work" in env
    assert "privileged: true" in container
    # A dead daemon must not take the API pod out of service.
    assert "readinessProbe" not in container


def test_the_chart_leaves_the_daemon_off_by_default():
    values = yaml.safe_load(
        (ROOT / "deploy/helm/knowledgedbchat/values.yaml").read_text()
    )
    assert values["sandbox"]["enabled"] is False
    assert set(values["sandbox"]["components"]) == {"backend", "celery", "celeryAgents"}
