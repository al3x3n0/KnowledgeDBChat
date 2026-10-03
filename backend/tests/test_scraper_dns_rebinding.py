"""The scraper connects only to an address it has checked.

The URL check resolved a hostname, and the HTTP client then resolved it again
to connect. A name that answers with a public address first and a private one
second (DNS rebinding) passed the check and reached the private address.
"""

import inspect
import ipaddress
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpcore
import pytest

from app.services.web_scraper_service import PinnedNetworkBackend, WebScraperService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Behind a proxy the proxy resolves names and pinning does not apply;
    these tests are about the direct path. (macOS reports system proxies to
    urllib, so a developer machine may have one without any variable set.)"""
    import urllib.request

    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})


PUBLIC = "93.184.216.34"


@pytest.fixture
def local_server():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b"<html><title>internal</title><body>secret</body></html>"
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], hits
    server.shutdown()


@pytest.fixture
def rebinding_dns(monkeypatch):
    """`rebind.test` answers public to the URL check and loopback to every
    other lookup -- the one a connection makes.

    That is the attack: with a short TTL the attacker decides which answer
    each lookup gets, so however many checks come first, the answer that
    matters is the connection's.
    """
    real = socket.getaddrinfo
    answers = []

    def getaddrinfo(host, port, *args, **kwargs):
        # anyio resolves the IDNA-encoded name, so the lookup a plain client
        # makes arrives as bytes.
        name = host.decode() if isinstance(host, bytes) else host
        if name != "rebind.test":
            return real(host, port, *args, **kwargs)
        checking = any(
            frame.function == "_validate_safe_url" for frame in inspect.stack()
        )
        ip = PUBLIC if checking else "127.0.0.1"
        answers.append(ip)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return answers


async def test_a_name_that_rebinds_to_loopback_is_never_requested(
    local_server, rebinding_dns
):
    port, hits = local_server
    service = WebScraperService(enforce_network_safety=True)
    try:
        result = await service.scrape(f"http://rebind.test:{port}/", max_pages=1)
    finally:
        await service.aclose()

    # The check saw the public answer and passed; the connection asked again,
    # got loopback, and refused before sending anything.
    assert rebinding_dns[0] == PUBLIC
    assert "127.0.0.1" in rebinding_dns
    assert result["pages"] == []
    assert "Disallowed IP address" in result["errors"][0]["error"]
    assert hits == []


class _Recorder(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.connected = []

    async def connect_tcp(self, host, port, timeout=None, **kwargs):
        self.connected.append((host, port))
        return object()

    async def connect_unix_socket(self, *args, **kwargs):
        raise AssertionError("not used")

    async def sleep(self, seconds):
        pass


def _resolves_to(monkeypatch, *ips):
    def getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in ip else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
            for ip in ips
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


async def test_it_connects_to_the_address_it_checked(monkeypatch):
    _resolves_to(monkeypatch, PUBLIC)
    inner = _Recorder()
    backend = PinnedNetworkBackend(lambda host, ip: ip.is_global, inner)

    await backend.connect_tcp("example.com", 443)

    assert inner.connected == [(PUBLIC, 443)]


async def test_one_disallowed_address_refuses_the_name(monkeypatch):
    # Any answer could be the one a later connection uses.
    _resolves_to(monkeypatch, PUBLIC, "10.0.0.7")
    inner = _Recorder()
    backend = PinnedNetworkBackend(lambda host, ip: ip.is_global, inner)

    with pytest.raises(httpcore.ConnectError, match="10.0.0.7"):
        await backend.connect_tcp("example.com", 80)
    assert inner.connected == []


async def test_the_rule_is_the_services_own(monkeypatch):
    """A named private host is allowed its private address, and only it."""
    service = WebScraperService(enforce_network_safety=True)
    service._private_hosts = ["wiki.corp"]
    private = ipaddress.ip_address("10.0.0.7")

    assert service._address_allowed("wiki.corp", private) is True
    assert service._address_allowed("docs.wiki.corp", private) is True
    assert service._address_allowed("elsewhere.corp", private) is False
    assert (
        service._address_allowed("wiki.corp", ipaddress.ip_address("127.0.0.1"))
        is False
    )


async def test_the_services_own_client_uses_it():
    service = WebScraperService(enforce_network_safety=True)
    try:
        client = await service._get_client(timeout_s=5, headers={})
        assert isinstance(
            client._transport._pool._network_backend, PinnedNetworkBackend
        )
    finally:
        await service.aclose()


async def test_behind_a_proxy_the_client_keeps_the_proxy(monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "getproxies", lambda: {"https": "http://proxy.corp:3128"}
    )
    service = WebScraperService(enforce_network_safety=True)
    try:
        client = await service._get_client(timeout_s=5, headers={})
        # No pinned transport, so httpx honours the proxy it was given.
        assert not isinstance(
            getattr(
                getattr(client._transport, "_pool", None), "_network_backend", None
            ),
            PinnedNetworkBackend,
        )
    finally:
        await service.aclose()
