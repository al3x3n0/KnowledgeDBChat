import httpx
import pytest

from app.services.web_scraper_service import WebScraperService


@pytest.mark.asyncio
async def test_web_scraper_extracts_readable_text_and_links():
    html = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <nav>Navigation</nav>
        <main>
          <div id="toc">Table of contents</div>
          <h1>Heading</h1>
          <p>Hello world <sup class="reference">[1]</sup></p>
          <a href="/wiki/Next">Next</a>
        </main>
        <footer>Footer</footer>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/wiki/Test"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        service = WebScraperService(client=client, enforce_network_safety=False)
        result = await service.scrape("https://example.com/wiki/Test", include_links=True)

    assert result["total_pages"] == 1
    page = result["pages"][0]
    assert page["title"] == "Test Page"
    assert page["url"] == "https://example.com/wiki/Test"
    assert page["status_code"] == 200
    assert "Navigation" not in page["content"]
    assert "Footer" not in page["content"]
    assert "Table of contents" not in page["content"]
    assert "[1]" not in page["content"]
    assert page["content"].startswith("Test Page")
    assert "https://example.com/wiki/Next" in page["links"]


@pytest.mark.asyncio
async def test_web_scraper_follow_links_with_depth_and_page_limit():
    root_html = """
    <html><head><title>Root</title></head>
      <body><main><a href="/wiki/Next">Next</a></main></body>
    </html>
    """
    next_html = """
    <html><head><title>Next</title></head>
      <body><main><p>Second page</p></main></body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/wiki/Root":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=root_html,
                request=request,
            )
        if str(request.url) == "https://example.com/wiki/Next":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=next_html,
                request=request,
            )
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        service = WebScraperService(client=client, enforce_network_safety=False)
        result = await service.scrape(
            "https://example.com/wiki/Root",
            follow_links=True,
            max_pages=2,
            max_depth=1,
            same_domain_only=True,
            include_links=True,
        )

    assert result["total_pages"] == 2
    urls = {p["url"] for p in result["pages"]}
    assert urls == {"https://example.com/wiki/Root", "https://example.com/wiki/Next"}


@pytest.mark.asyncio
async def test_web_scraper_blocks_localhost_and_private_ips():
    service = WebScraperService(enforce_network_safety=True)

    with pytest.raises(ValueError):
        await service.scrape("http://localhost/")

    with pytest.raises(ValueError):
        await service.scrape("http://127.0.0.1/")

