"""Where a custom webhook tool may send its request.

A webhook tool's URL is written by whoever authored the tool -- an agent can
author one -- and may be templated from the call's inputs. It went straight
to the HTTP client. The scraper has refused private addresses for a long
time; this path had no check at all, so a tool could read the cloud metadata
address or any service on the deployment's own network.

DNS is replaced (a name resolves to what the test says); the request itself
goes to an `httpx.MockTransport`, so a URL that is allowed is seen to be sent
and one that is refused is seen not to be.
"""

import socket

import httpx
import pytest

from app.core.config import settings
from app.services import web_scraper_service
from app.services.custom_tool_service import CustomToolService, ToolExecutionError
from app.services.web_scraper_service import OutboundGuard, PinnedNetworkBackend

pytestmark = pytest.mark.unit

DNS = {
    "api.example.com": "93.184.216.34",
    "jira.lab.internal": "10.20.0.7",
    "ci.build.lab": "10.20.0.8",
    "rebind.example.com": "169.254.169.254",
    "vector-store": "172.18.0.5",
}


@pytest.fixture
def network(monkeypatch):
    """Names resolve as DNS says; requests are answered here and recorded."""
    sent = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in DNS:
            raise socket.gaierror(f"unknown host {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS[host], port or 0))]

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.method, str(request.url)))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(
        OutboundGuard,
        "client",
        lambda self, *, timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=timeout
        ),
    )
    monkeypatch.setattr(settings, "CUSTOM_TOOL_WEBHOOK_PRIVATE_HOSTS", "")
    return sent


async def _call(url, inputs=None, method="GET"):
    return await CustomToolService()._execute_webhook(
        {"url": url, "method": method}, inputs or {}
    )


async def test_a_public_address_is_called(network):
    result = await _call("https://api.example.com/v1/items")

    assert result == {"ok": True}
    assert network == [("GET", "https://api.example.com/v1/items")]


@pytest.mark.parametrize(
    "url",
    [
        # The cloud metadata address, by number and by a name that resolves to it.
        "http://169.254.169.254/latest/meta-data/",
        "http://rebind.example.com/latest/meta-data/",
        # This machine.
        "http://127.0.0.1:6333/collections",
        "http://localhost:9000/",
        "http://[::1]/",
        "http://0.0.0.0/",
        # The deployment's own network, by address and by service name.
        "http://10.20.0.7/rest/api",
        "http://vector-store:6333/collections",
        "http://jira.lab.internal/rest/api",
        # Not HTTP, and credentials in the URL.
        "file:///etc/passwd",
        "gopher://api.example.com/",
        "http://user:secret@api.example.com/",
        "",
    ],
)
async def test_a_url_that_is_not_a_public_address_is_refused_unsent(network, url):
    with pytest.raises(ToolExecutionError, match="not allowed"):
        await _call(url)

    assert network == []


async def test_a_url_built_from_the_calls_inputs_is_checked_after_rendering(network):
    with pytest.raises(ToolExecutionError, match="not allowed"):
        await _call("{{ base }}/latest/", inputs={"base": "http://169.254.169.254"})

    assert network == []


async def test_an_admin_can_name_private_hosts_a_webhook_may_call(network, monkeypatch):
    monkeypatch.setattr(
        settings, "CUSTOM_TOOL_WEBHOOK_PRIVATE_HOSTS", " jira.lab.internal , build.lab"
    )

    await _call("http://jira.lab.internal/rest/api", method="POST")
    # A listed name covers its subdomains.
    await _call("http://ci.build.lab/job")

    assert [url for _m, url in network] == [
        "http://jira.lab.internal/rest/api",
        "http://ci.build.lab/job",
    ]


async def test_listing_a_host_does_not_open_the_others(network, monkeypatch):
    monkeypatch.setattr(
        settings, "CUSTOM_TOOL_WEBHOOK_PRIVATE_HOSTS", "jira.lab.internal"
    )

    with pytest.raises(ToolExecutionError):
        await _call("http://vector-store:6333/collections")
    # And no listing reaches loopback or link-local.
    monkeypatch.setattr(
        settings, "CUSTOM_TOOL_WEBHOOK_PRIVATE_HOSTS", "rebind.example.com,localhost"
    )
    with pytest.raises(ToolExecutionError):
        await _call("http://rebind.example.com/")
    with pytest.raises(ToolExecutionError):
        await _call("http://localhost/")

    assert network == []


def test_the_real_client_connects_only_to_an_address_it_checked(monkeypatch):
    """Without this a name could resolve public for the check and private for
    the connection."""
    monkeypatch.setattr(web_scraper_service.urllib.request, "getproxies", lambda: {})

    client = OutboundGuard().client(timeout=5)

    backend = client._transport._pool._network_backend
    assert isinstance(backend, PinnedNetworkBackend)
    assert client.follow_redirects is False


async def test_the_pinned_backend_refuses_a_private_answer(monkeypatch):
    import asyncio
    import ipaddress

    import httpcore

    guard = OutboundGuard()

    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    backend = PinnedNetworkBackend(guard._rule._address_allowed)

    with pytest.raises(httpcore.ConnectError, match="Disallowed IP"):
        await backend.connect_tcp("api.example.com", 443)
    assert guard._rule._address_allowed(
        "api.example.com", ipaddress.ip_address("93.184.216.34")
    )


@pytest.mark.parametrize(
    "url, fetched",
    [
        ("https://arxiv.org/abs/2401.00001", "https://arxiv.org/abs/2401.00001"),
        # What arXiv's own API returns: fetched over https all the same.
        ("http://arxiv.org/abs/2401.00001v2", "https://arxiv.org/abs/2401.00001v2"),
        ("https://export.arxiv.org/abs/1", "https://export.arxiv.org/abs/1"),
        ("https://arxiv.org.evil.example/abs/1", None),
        ("https://evilarxiv.org/abs/1", None),
        ("https://user:pw@arxiv.org/abs/1", None),
        ("https://arxiv.org:8443/abs/1", None),
        ("http://169.254.169.254/", None),
        ("file:///etc/passwd", None),
        ("", None),
    ],
)
def test_the_repo_scout_fetches_arxiv_pages_only(url, fetched):
    """An inbox item's stored URL was fetched wherever it pointed."""
    from app.api.endpoints.research_inbox import _arxiv_page_url

    assert _arxiv_page_url(url) == fetched
