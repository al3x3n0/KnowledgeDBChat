"""The web research tools: search_web, fetch_url_content and summarize_url.

These call the real handlers, which in turn run the real `WebScraperService`
(and the real BeautifulSoup -- it is installed, so conftest does not stub it).
The file used to restate each handler inline and assert on the restatement --
`params = {}; url = str(params.get("url", "")).strip(); assert not url` -- so
nineteen tests passed whatever the tools did.

Three edges are replaced, and nothing else:

* the wire: every `httpx.AsyncClient` the code builds is given a
  `MockTransport`, so requests are recorded and answered from a table and no
  packet leaves the machine;
* the resolver: `socket.getaddrinfo` answers from a table, so the scraper's
  private-address check runs for real against addresses the test chose;
* the model: `generate_response` records what it was given.
"""

import socket
from types import SimpleNamespace
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_web_research_provider,
)

pytestmark = pytest.mark.unit

PUBLIC_IP = "93.184.216.34"
DDG_HOST = "html.duckduckgo.com"


class _Web:
    """The network as the tools see it: a routing table and a resolver."""

    def __init__(self):
        self.requests = []
        self.routes = {}
        self.dns = {DDG_HOST: PUBLIC_IP}

    def serve(self, url, response, ip=PUBLIC_IP):
        parsed = httpx.URL(url)
        self.routes[(parsed.host, parsed.path or "/")] = response
        if ip is not None:
            self.dns.setdefault(parsed.host, ip)

    def page(self, url, html, **kwargs):
        self.serve(
            url,
            httpx.Response(
                200, content=html.encode(), headers={"content-type": "text/html"}
            ),
            **kwargs,
        )

    def handle(self, request):
        self.requests.append(request)
        answer = self.routes.get((request.url.host, request.url.path))
        if answer is None:
            return httpx.Response(404, text="not found")
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(request)
        return httpx.Response(
            answer.status_code, headers=answer.headers, content=answer.content
        )

    def getaddrinfo(self, host, *args, **kwargs):
        literal = host.strip("[]")
        try:
            socket.inet_pton(
                socket.AF_INET6 if ":" in literal else socket.AF_INET, literal
            )
            address = literal
        except OSError:
            address = self.dns.get(host)
        if address is None:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 0))]

    @property
    def hosts(self):
        return [request.url.host for request in self.requests]


@pytest.fixture
def web(monkeypatch):
    net = _Web()
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(net.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(socket, "getaddrinfo", net.getaddrinfo)
    return net


class _LLM:
    def __init__(self, reply="A short summary.", fail=None):
        self.calls = []
        self._reply = reply
        self._fail = fail

    async def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise self._fail
        return self._reply


async def _run(tool, params, llm=None, job=None):
    llm = llm or _LLM()
    provider = build_autonomous_web_research_provider(SimpleNamespace(llm_service=llm))
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id=str(uuid4()),
        job=job or SimpleNamespace(id=uuid4(), iteration=3, config={}),
        state={},
    )
    return await provider._handlers[tool](dict(params), ctx)


def _refused(result):
    assert isinstance(result, dict)
    assert result.get("error")
    assert "success" not in result
    assert "data" not in result
    return result["error"]


# --------------------------------------------------------------------------
# search_web
# --------------------------------------------------------------------------


def _ddg_result(title_html, target, snippet_html):
    href = f"//duckduckgo.com/l/?uddg={quote(target, safe='')}&amp;rut=4f6e1c"
    snippet = (
        f'<a class="result__snippet" href="{href}">{snippet_html}</a>'
        if snippet_html is not None
        else ""
    )
    return f"""
    <div class="result results_links results_links_deep web-result ">
      <div class="links_main links_deep result__body">
        <h2 class="result__title">
          <a rel="nofollow" class="result__a" href="{href}">{title_html}</a>
        </h2>
        <div class="result__extras">
          <div class="result__extras__url">
            <span class="result__icon">
              <a rel="nofollow" href="{href}"><img class="result__icon__img"
                 width="16" height="16" alt="" src="//external.example/i.ico"/></a>
            </span>
            <a class="result__url" href="{href}">{target.split('//')[1]}</a>
          </div>
        </div>
        {snippet}
        <div class="clear"></div>
      </div>
    </div>"""


def _ddg_page(results):
    body = "\n".join(_ddg_result(*result) for result in results)
    return f"""<!DOCTYPE html>
<html><head><title>python asyncio at DuckDuckGo</title></head>
<body class="body--html">
  <div class="header"><form action="/html/" method="post">
    <input name="q" value="python asyncio"/></form></div>
  <div class="serp__results"><div id="links" class="results">
    {body}
    <div class="nav-link"><form action="/html/" method="post">
      <input type="submit" class="btn btn--alt" value="Next"/></form></div>
  </div></div>
</body></html>"""


THREE = [
    (
        "asyncio — <b>Asynchronous</b> I/O — <b>Python</b> 3 documentation",
        "https://docs.python.org/3/library/asyncio.html",
        "<b>asyncio</b> is a library to write concurrent code using async/await.",
    ),
    (
        "Async IO in <b>Python</b>: A Complete Walkthrough",
        "https://realpython.com/async-io-python/?utm=a&ref=b",
        "This tutorial gives a full picture of <b>asyncio</b>.",
    ),
    (
        "PEP 3156",
        "https://peps.python.org/pep-3156/",
        "Asynchronous IO Support Rebooted: the asyncio module.",
    ),
]


def _numbered(count):
    return [
        (f"Result {i}", f"https://site{i}.example/page", f"Snippet {i}")
        for i in range(1, count + 1)
    ]


def _serve_search(web, results):
    web.page(f"https://{DDG_HOST}/html/", _ddg_page(results))


@pytest.mark.parametrize("params", [{}, {"query": ""}, {"query": "   "}])
async def test_search_refuses_without_a_query(web, params):
    error = _refused(await _run("search_web", params))

    assert "query" in error
    assert web.requests == []


async def test_search_sends_the_query_and_parses_the_results_page(web):
    _serve_search(web, THREE)

    result = await _run("search_web", {"query": "  python asyncio  "})

    assert result["success"] is True
    assert len(web.requests) == 1
    request = web.requests[0]
    assert request.url.host == DDG_HOST
    assert request.url.params["q"] == "python asyncio"
    data = result["data"]
    assert data["query"] == "python asyncio"
    assert data["count"] == 3
    assert data["results"] == [
        {
            "title": "asyncio — Asynchronous I/O — Python 3 documentation",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "snippet": (
                "asyncio is a library to write concurrent code using async/await."
            ),
        },
        {
            "title": "Async IO in Python: A Complete Walkthrough",
            "url": "https://realpython.com/async-io-python/?utm=a&ref=b",
            "snippet": "This tutorial gives a full picture of asyncio.",
        },
        {
            "title": "PEP 3156",
            "url": "https://peps.python.org/pep-3156/",
            "snippet": "Asynchronous IO Support Rebooted: the asyncio module.",
        },
    ]


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, 5),
        ({"max_results": None}, 5),
        ({"max_results": 2}, 2),
        ({"max_results": "3"}, 3),
        ({"max_results": 10}, 10),
        ({"max_results": 50}, 10),
    ],
)
async def test_search_returns_at_most_max_results_in_page_order(web, params, expected):
    _serve_search(web, _numbered(12))

    result = await _run("search_web", {"query": "q", **params})

    assert result["data"]["count"] == expected
    assert [r["title"] for r in result["data"]["results"]] == [
        f"Result {i}" for i in range(1, expected + 1)
    ]


async def test_a_negative_max_results_never_exceeds_the_cap(web):
    _serve_search(web, _numbered(12))

    result = await _run("search_web", {"query": "q", "max_results": -1})

    assert "error" in result or result["data"]["count"] <= 10


async def test_a_page_with_no_results_is_an_empty_success(web):
    _serve_search(web, [])

    result = await _run("search_web", {"query": "xyzzynotfound"})

    assert result["success"] is True
    assert result["data"] == {"query": "xyzzynotfound", "results": [], "count": 0}


async def test_long_fields_are_capped(web):
    _serve_search(web, [("T" * 400, "https://long.example/" + "p" * 900, "S" * 900)])

    result = await _run("search_web", {"query": "q"})

    only = result["data"]["results"][0]
    assert len(only["title"]) == 200
    assert len(only["url"]) == 500
    assert len(only["snippet"]) == 500


async def test_html_entities_in_titles_and_snippets_are_decoded(web):
    _serve_search(
        web,
        [
            (
                "Compilers Q&amp;A: what&#x27;s a pass?",
                "https://qa.example/passes",
                "It&#x27;s &quot;a transformation&quot; &amp; an analysis.",
            )
        ],
    )

    result = await _run("search_web", {"query": "q"})

    only = result["data"]["results"][0]
    assert only["title"] == "Compilers Q&A: what's a pass?"
    assert only["snippet"] == 'It\'s "a transformation" & an analysis.'


async def test_a_result_without_a_snippet_does_not_swallow_the_next_one(web):
    _serve_search(
        web,
        [
            ("First", "https://one.example/", "Snippet one"),
            ("Second", "https://two.example/", None),
            ("Third", "https://three.example/", "Snippet three"),
        ],
    )

    result = await _run("search_web", {"query": "q"})

    results = result["data"]["results"]
    assert [r["title"] for r in results] == ["First", "Second", "Third"]
    assert results[1]["snippet"] != "Snippet three"
    assert results[2]["url"] == "https://three.example/"


@pytest.mark.parametrize("status", [403, 429, 500, 503])
async def test_a_search_answered_with_an_error_status_is_an_error(web, status):
    web.serve(f"https://{DDG_HOST}/html/", httpx.Response(status, text="blocked"))

    error = _refused(await _run("search_web", {"query": "q"}))

    assert str(status) in error


async def test_a_search_that_times_out_is_an_error(web):
    web.serve(
        f"https://{DDG_HOST}/html/", httpx.ReadTimeout("The read operation timed out")
    )

    error = _refused(await _run("search_web", {"query": "q"}))

    assert "timed out" in error


async def test_a_non_numeric_max_results_is_refused_before_any_request(web):
    _serve_search(web, THREE)

    _refused(await _run("search_web", {"query": "q", "max_results": "many"}))

    assert web.requests == []


# --------------------------------------------------------------------------
# fetch_url_content
# --------------------------------------------------------------------------

ARTICLE_URL = "https://blog.example.org/posts/prefetchers"

ARTICLE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>How Stride Prefetchers Work</title>
  <style>body { font-family: serif; } .hidden { display: none; }</style>
  <script>window.dataLayer = []; trackPageview("SECRET-TRACKER");</script>
</head>
<body>
  <header><a href="/">Example Blog</a></header>
  <nav><ul><li><a href="/archive">Archive link text</a></li></ul></nav>
  <article>
    <h1>How Stride Prefetchers Work</h1>
    <p>A stride prefetcher watches the addresses one instruction touches.</p>
    <p>When two consecutive deltas match, it fetches the next address early.</p>
    <aside>Subscribe to our newsletter</aside>
    <h2>Limits</h2>
    <p>Pointer chasing defeats it because the deltas never repeat.</p>
  </article>
  <footer>Copyright 2026 Example Blog</footer>
</body>
</html>"""


def _long_article(chars):
    paragraph = "<p>" + "cache line " * 40 + "</p>\n"
    count = chars // (len(paragraph) - 8) + 1
    return (
        "<html><head><title>Long Read</title></head><body><article>"
        + paragraph * count
        + "</article></body></html>"
    )


@pytest.mark.parametrize("tool", ["fetch_url_content", "summarize_url"])
@pytest.mark.parametrize("params", [{}, {"url": ""}, {"url": "   "}])
async def test_a_url_is_required(web, tool, params):
    llm = _LLM()

    error = _refused(await _run(tool, params, llm=llm))

    assert "url" in error
    assert web.requests == []
    assert llm.calls == []


async def test_fetch_returns_the_readable_text_of_the_page(web):
    web.page(ARTICLE_URL, ARTICLE)

    result = await _run("fetch_url_content", {"url": f"  {ARTICLE_URL}  "})

    assert result["success"] is True
    assert [str(r.url) for r in web.requests] == [ARTICLE_URL]
    data = result["data"]
    assert data["url"] == ARTICLE_URL
    assert data["title"] == "How Stride Prefetchers Work"
    content = data["content"]
    assert data["content_length"] == len(content)
    first = content.index("A stride prefetcher watches the addresses")
    second = content.index("When two consecutive deltas match")
    third = content.index("Pointer chasing defeats it")
    assert first < second < third
    assert "Limits" in content
    for noise in (
        "SECRET-TRACKER",
        "font-family",
        "Archive link text",
        "Subscribe to our newsletter",
        "Copyright 2026",
        "<p>",
    ):
        assert noise not in content


async def test_plain_text_is_returned_as_it_is(web):
    web.serve(
        "https://files.example.org/notes.txt",
        httpx.Response(
            200,
            content=b"line one\nline two\n",
            headers={"content-type": "text/plain; charset=utf-8"},
        ),
    )

    result = await _run(
        "fetch_url_content", {"url": "https://files.example.org/notes.txt"}
    )

    assert result["success"] is True
    assert result["data"]["content"] == "line one\nline two\n"


@pytest.mark.parametrize(
    "params, low, high",
    [
        ({}, 40_000, 50_000),
        ({"max_chars": None}, 40_000, 50_000),
        ({"max_chars": 1000}, 500, 1000),
        ({"max_chars": "2000"}, 1000, 2000),
        ({"max_chars": 80_000}, 70_000, 80_000),
        ({"max_chars": 900_000}, 90_000, 100_000),
    ],
)
async def test_content_is_capped_at_max_chars(web, params, low, high):
    web.page("https://long.example.org/read", _long_article(150_000))

    result = await _run(
        "fetch_url_content", {"url": "https://long.example.org/read", **params}
    )

    assert result["success"] is True
    content = result["data"]["content"]
    assert low < len(content) <= high
    assert result["data"]["content_length"] == len(content)
    assert content.rstrip().endswith("[truncated]")


async def test_a_short_page_is_not_marked_truncated(web):
    web.page(ARTICLE_URL, ARTICLE)

    result = await _run("fetch_url_content", {"url": ARTICLE_URL, "max_chars": 5000})

    assert "[truncated]" not in result["data"]["content"]
    assert "Pointer chasing defeats it" in result["data"]["content"]


@pytest.mark.parametrize("tool", ["fetch_url_content", "summarize_url"])
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://ftp.example.org/pub/readme.txt",
        "gopher://example.org/",
        "javascript:alert(1)",
        "data:text/html,<p>hi</p>",
        "blog.example.org/posts/prefetchers",
        "https://",
        "https://user:hunter2@blog.example.org/posts/prefetchers",
    ],
)
async def test_only_plain_http_urls_are_fetched(web, tool, url):
    web.page(ARTICLE_URL, ARTICLE)
    llm = _LLM()

    _refused(await _run(tool, {"url": url}, llm=llm))

    assert web.requests == []
    assert llm.calls == []


@pytest.mark.parametrize("tool", ["fetch_url_content", "summarize_url"])
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/api/v1/admin",
        "http://localhost:6379/",
        "http://backend.localhost/",
        "http://printer.local/status",
        "http://10.0.0.5/",
        "http://172.16.4.4/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0:8000/",
        "http://[::1]:8000/",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
async def test_private_and_loopback_addresses_are_never_fetched(web, tool, url):
    # A page is on offer at every one of them: only the check stands between
    # the tool and the content.
    web.page(url, ARTICLE, ip=None)
    llm = _LLM()

    _refused(await _run(tool, {"url": url}, llm=llm))

    assert web.requests == []
    assert llm.calls == []


async def test_a_public_name_resolving_to_a_private_address_is_not_fetched(web):
    web.page("https://intranet.example.org/wiki", ARTICLE, ip="10.20.30.40")

    error = _refused(
        await _run("fetch_url_content", {"url": "https://intranet.example.org/wiki"})
    )

    assert web.requests == []
    assert "Disallowed" in error


async def test_a_name_that_does_not_resolve_is_an_error(web):
    error = _refused(
        await _run("fetch_url_content", {"url": "https://nowhere.invalid/page"})
    )

    assert web.requests == []
    assert "resolve" in error


async def test_content_behind_a_redirect_to_a_private_address_is_withheld(web):
    web.serve(
        "https://short.example.org/x",
        httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        ),
    )
    web.serve(
        "http://169.254.169.254/latest/meta-data/",
        httpx.Response(
            200,
            content=b"<html><body><p>iam-role-credentials</p></body></html>",
            headers={"content-type": "text/html"},
        ),
        ip=None,
    )
    llm = _LLM()

    for tool in ("fetch_url_content", "summarize_url"):
        result = await _run(tool, {"url": "https://short.example.org/x"}, llm=llm)

        _refused(result)
        assert "iam-role-credentials" not in str(result)
    assert llm.calls == []


async def test_a_redirect_to_a_private_address_is_not_followed(web):
    web.serve(
        "https://short.example.org/x",
        httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        ),
    )

    await _run("fetch_url_content", {"url": "https://short.example.org/x"})

    assert "169.254.169.254" not in web.hosts


async def test_a_redirect_between_public_pages_is_followed(web):
    web.serve(
        "https://short.example.org/x",
        httpx.Response(301, headers={"location": ARTICLE_URL}),
    )
    web.page(ARTICLE_URL, ARTICLE)

    result = await _run("fetch_url_content", {"url": "https://short.example.org/x"})

    assert result["success"] is True
    assert "A stride prefetcher watches" in result["data"]["content"]


def _failures():
    return {
        "404": httpx.Response(404, text="<html><body>Not Found</body></html>"),
        "500": httpx.Response(500, text="<html><body>Server Error</body></html>"),
        "503": httpx.Response(503, text=""),
        "timeout": httpx.ReadTimeout("The read operation timed out"),
        "refused": httpx.ConnectError("All connection attempts failed"),
        "pdf": httpx.Response(
            200, content=b"%PDF-1.7 ...", headers={"content-type": "application/pdf"}
        ),
    }


@pytest.mark.parametrize("tool", ["fetch_url_content", "summarize_url"])
@pytest.mark.parametrize("case", ["404", "500", "503", "timeout", "refused", "pdf"])
async def test_a_failed_fetch_is_an_error_and_never_an_empty_success(web, tool, case):
    web.serve("https://down.example.org/page", _failures()[case])
    llm = _LLM()

    _refused(await _run(tool, {"url": "https://down.example.org/page"}, llm=llm))

    assert len(web.requests) == 1
    assert llm.calls == []


@pytest.mark.parametrize(
    "tool, case, expected",
    [
        ("fetch_url_content", "404", "404"),
        ("fetch_url_content", "503", "503"),
        ("fetch_url_content", "timeout", "timed out"),
        ("fetch_url_content", "refused", "connection attempts failed"),
        ("fetch_url_content", "pdf", "application/pdf"),
        ("summarize_url", "404", "404"),
        ("summarize_url", "timeout", "timed out"),
    ],
)
async def test_a_failed_fetch_says_why_it_failed(web, tool, case, expected):
    web.serve("https://down.example.org/page", _failures()[case])

    error = _refused(await _run(tool, {"url": "https://down.example.org/page"}))

    assert expected in error
    assert "list index out of range" not in error


@pytest.mark.parametrize("tool", ["fetch_url_content", "summarize_url"])
async def test_a_page_with_no_text_is_an_error(web, tool):
    web.page(
        "https://empty.example.org/",
        "<html><head><script>render()</script></head><body><div id='app'>"
        "</div></body></html>",
    )
    llm = _LLM()

    error = _refused(await _run(tool, {"url": "https://empty.example.org/"}, llm=llm))

    assert "No content" in error
    assert llm.calls == []


@pytest.mark.parametrize("max_chars", [0, -5, "lots"])
async def test_a_nonsense_max_chars_never_yields_more_than_the_default(web, max_chars):
    web.page("https://long.example.org/read", _long_article(150_000))

    result = await _run(
        "fetch_url_content",
        {"url": "https://long.example.org/read", "max_chars": max_chars},
    )

    if "error" not in result:
        assert len(result["data"]["content"]) <= 50_000


# --------------------------------------------------------------------------
# summarize_url
# --------------------------------------------------------------------------


async def test_the_summariser_is_given_the_fetched_page(web):
    web.page(ARTICLE_URL, ARTICLE)
    llm = _LLM(reply="Stride prefetchers predict addresses from repeating deltas.")
    job = SimpleNamespace(id=uuid4(), iteration=7, config={})

    result = await _run("summarize_url", {"url": ARTICLE_URL}, llm=llm, job=job)

    assert result["success"] is True
    assert [str(r.url) for r in web.requests] == [ARTICLE_URL]
    assert len(llm.calls) == 1
    call = llm.calls[0]
    message = call["user_message"]
    assert "A stride prefetcher watches the addresses one instruction touches." in (
        message
    )
    assert "Pointer chasing defeats it because the deltas never repeat." in message
    assert "SECRET-TRACKER" not in message
    assert "ummar" in call["system_prompt"]
    assert "focus on" not in call["system_prompt"]
    assert call["snapshot_context"] == {
        "job_id": str(job.id),
        "iteration": 7,
        "phase": "tool:summarize_url",
    }
    data = result["data"]
    assert data["url"] == ARTICLE_URL
    assert data["summary"] == (
        "Stride prefetchers predict addresses from repeating deltas."
    )
    assert data["focus"] is None
    assert data["content_length"] == len(message)


async def test_the_focus_reaches_the_prompt_and_the_result(web):
    web.page(ARTICLE_URL, ARTICLE)
    llm = _LLM()

    result = await _run(
        "summarize_url",
        {"url": ARTICLE_URL, "focus": "  pointer chasing workloads  "},
        llm=llm,
    )

    assert "pointer chasing workloads" in llm.calls[0]["system_prompt"]
    assert result["data"]["focus"] == "pointer chasing workloads"


async def test_a_long_page_is_cut_before_it_reaches_the_model(web):
    web.page("https://long.example.org/read", _long_article(150_000))
    llm = _LLM()

    result = await _run(
        "summarize_url", {"url": "https://long.example.org/read"}, llm=llm
    )

    assert result["success"] is True
    message = llm.calls[0]["user_message"]
    assert 20_000 < len(message) <= 30_000
    assert message.startswith("Long Read")


async def test_the_reported_length_is_what_the_model_was_shown(web):
    web.page("https://long.example.org/read", _long_article(150_000))
    llm = _LLM()

    result = await _run(
        "summarize_url", {"url": "https://long.example.org/read"}, llm=llm
    )

    shown = len(llm.calls[0]["user_message"])
    data = result["data"]
    assert data["content_length"] == shown or data.get("truncated") is True


async def test_a_model_failure_is_an_error(web):
    web.page(ARTICLE_URL, ARTICLE)
    llm = _LLM(fail=RuntimeError("provider returned 529 overloaded"))

    error = _refused(await _run("summarize_url", {"url": ARTICLE_URL}, llm=llm))

    assert "529 overloaded" in error


async def test_an_empty_summary_is_not_a_success(web):
    web.page(ARTICLE_URL, ARTICLE)

    result = await _run("summarize_url", {"url": ARTICLE_URL}, llm=_LLM(reply="  "))

    _refused(result)


# --------------------------------------------------------------------------
# Which call sites turn the scraper's network safety on
# --------------------------------------------------------------------------


def test_the_scraper_enforces_network_safety_unless_told_otherwise():
    from app.services.web_scraper_service import WebScraperService

    assert WebScraperService()._enforce_network_safety is True


def test_no_call_site_switches_network_safety_off():
    import re
    from pathlib import Path

    import app

    root = Path(app.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"enforce_network_safety\s*=\s*False", line)
        or re.search(r"allow_private_networks\s*=\s*True", line)
    ]

    assert offenders == []


# --------------------------------------------------------------------------
# Schemas and registry classification
# --------------------------------------------------------------------------


class TestResearchToolSchemas:
    """Tests for research tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "search_web" in names
        assert "summarize_url" in names
        assert "fetch_url_content" in names

    def test_search_web_requires_query(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("search_web")
        assert tool is not None
        assert "query" in tool["parameters"].get("required", [])

    def test_fetch_url_requires_url(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("fetch_url_content")
        assert tool is not None
        assert "url" in tool["parameters"].get("required", [])

    def test_summarize_url_requires_url(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("summarize_url")
        assert tool is not None
        assert "url" in tool["parameters"].get("required", [])


class TestResearchToolRegistry:
    """Tests for research tool registry classification."""

    def test_search_web_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("search_web")
        assert meta is not None
        assert meta.network == "egress"

    def test_search_web_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("search_web")
        assert meta is not None
        assert meta.cost_tier == "medium"

    def test_fetch_url_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("fetch_url_content")
        assert meta is not None
        assert meta.network == "egress"

    def test_summarize_url_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("summarize_url")
        assert meta is not None
        assert meta.network == "egress"
        assert meta.cost_tier == "medium"
