"""
Web scraping utilities for fetching and extracting readable text from web pages.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger


class WebScraperService:
    DEFAULT_TIMEOUT_S = 20.0
    DEFAULT_MAX_BYTES = 2_000_000
    DEFAULT_MAX_CONTENT_CHARS = 50_000
    DEFAULT_USER_AGENT = "KnowledgeDBChat/1.0"

    def __init__(
        self,
        *,
        client: Optional[httpx.AsyncClient] = None,
        enforce_network_safety: bool = True,
    ):
        self._client = client
        self._owns_client = client is None
        self._enforce_network_safety = enforce_network_safety

    async def _get_client(self, *, timeout_s: float, headers: Dict[str, str]) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=timeout_s,
                follow_redirects=True,
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def scrape(
        self,
        url: str,
        *,
        follow_links: bool = False,
        max_pages: int = 1,
        max_depth: int = 0,
        same_domain_only: bool = True,
        include_links: bool = True,
        allow_private_networks: bool = False,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_bytes: int = DEFAULT_MAX_BYTES,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if max_pages < 1 or max_pages > 25:
            raise ValueError("max_pages must be between 1 and 25")
        if max_depth < 0 or max_depth > 5:
            raise ValueError("max_depth must be between 0 and 5")
        if max_content_chars < 1 or max_content_chars > 500_000:
            raise ValueError("max_content_chars must be between 1 and 500000")
        if max_bytes < 1024 or max_bytes > 25_000_000:
            raise ValueError("max_bytes must be between 1024 and 25000000")

        normalized_url = self._normalize_url(url)
        if self._enforce_network_safety:
            self._validate_safe_url(normalized_url, allow_private_networks=allow_private_networks)

        root_netloc = urlparse(normalized_url).netloc.lower()
        request_headers = {
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        }
        if headers:
            request_headers.update(headers)

        pages: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(normalized_url, 0)])

        while queue and len(pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                page = await self._scrape_single(
                    current_url,
                    include_links=include_links,
                    allow_private_networks=allow_private_networks,
                    max_content_chars=max_content_chars,
                    timeout_s=timeout_s,
                    max_bytes=max_bytes,
                    headers=request_headers,
                )
                pages.append(page)
            except Exception as e:
                logger.warning(f"Web scrape failed for {current_url}: {e}")
                errors.append({"url": current_url, "error": str(e)})
                continue

            if not follow_links or depth >= max_depth or not include_links:
                continue

            discovered = 0
            for link in page.get("links", []):
                if discovered >= 200:
                    break
                if link in visited:
                    continue
                if same_domain_only and urlparse(link).netloc.lower() != root_netloc:
                    continue
                try:
                    normalized_link = self._normalize_url(link)
                    if self._enforce_network_safety:
                        self._validate_safe_url(normalized_link, allow_private_networks=allow_private_networks)
                except Exception:
                    continue
                queue.append((normalized_link, depth + 1))
                discovered += 1

        return {
            "root_url": normalized_url,
            "total_pages": len(pages),
            "pages": pages,
            "errors": errors,
        }

    async def _scrape_single(
        self,
        url: str,
        *,
        include_links: bool,
        allow_private_networks: bool,
        max_content_chars: int,
        timeout_s: float,
        max_bytes: int,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        final_url, response = await self._fetch(
            url,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            headers=headers,
            allow_private_networks=allow_private_networks,
        )

        content_type = (response.headers.get("content-type") or "").lower()
        if "text/html" in content_type or "application/xhtml+xml" in content_type or content_type == "":
            soup = BeautifulSoup(response.content, "html.parser")
            title = self._extract_title(soup)
            main = self._select_main_content(soup)
            text = self._extract_text(main, title=title)
            links = self._extract_links(main, base_url=final_url) if include_links else []
        elif "text/plain" in content_type:
            title = ""
            text = response.text
            links = []
        else:
            raise ValueError(f"Unsupported content-type: {content_type}")

        text = self._truncate_text(text, max_content_chars)

        return {
            "url": final_url,
            "title": title,
            "content": text,
            "content_type": response.headers.get("content-type", ""),
            "status_code": response.status_code,
            "links": links,
        }

    async def _fetch(
        self,
        url: str,
        *,
        timeout_s: float,
        max_bytes: int,
        headers: Dict[str, str],
        allow_private_networks: bool,
    ) -> Tuple[str, httpx.Response]:
        client = await self._get_client(timeout_s=timeout_s, headers=headers)
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = 0
            chunks: List[bytes] = []
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Response too large (>{max_bytes} bytes)")
                chunks.append(chunk)
            content = b"".join(chunks)

        final_url = str(response.url)
        if self._enforce_network_safety:
            self._validate_safe_url(final_url, allow_private_networks=allow_private_networks)
        hydrated = httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=content,
            request=response.request,
            extensions=response.extensions,
        )
        return final_url, hydrated

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported")
        if not parsed.netloc:
            raise ValueError("URL must include a hostname")
        if parsed.username or parsed.password:
            raise ValueError("Userinfo in URL is not supported")
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.params, parsed.query, ""))
        return normalized

    def _is_allowed_ip(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private_networks: bool) -> bool:
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False
        if ip.is_global:
            return True
        if allow_private_networks and ip.is_private:
            return True
        return False

    def _validate_safe_url(self, url: str, *, allow_private_networks: bool) -> None:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("URL must include a hostname")

        host_lower = hostname.lower()
        if host_lower in {"localhost"} or host_lower.endswith(".localhost") or host_lower.endswith(".local"):
            raise ValueError("Localhost domains are not allowed")

        # If it's a literal IP, check it directly.
        try:
            ip = ipaddress.ip_address(host_lower)
            if not self._is_allowed_ip(ip, allow_private_networks=allow_private_networks):
                raise ValueError("Disallowed IP address")
            return
        except ValueError:
            pass

        try:
            infos = socket.getaddrinfo(hostname, None)
        except OSError as e:
            raise ValueError(f"Failed to resolve hostname: {e}") from e

        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if not self._is_allowed_ip(ip, allow_private_networks=allow_private_networks):
                raise ValueError("Disallowed IP address")

    def _select_main_content(self, soup: BeautifulSoup):
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        for selector in [
            "#mw-content-text",
            "#bodyContent",
            "main",
            "article",
            '[role="main"]',
            ".wiki-content",
            ".documentation",
            ".article-content",
            ".entry-content",
            ".post-content",
            "#content",
            "#main-content",
            ".content",
        ]:
            node = soup.select_one(selector)
            if node:
                return node

        return soup.find("body") or soup

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find("title")
        if not title_tag:
            return ""
        return title_tag.get_text(strip=True)

    def _extract_text(self, root, *, title: str) -> str:
        # Remove common non-content containers within the selected root.
        for tag in root.find_all(["nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        for selector in [
            "#toc",
            ".toc",
            ".mw-editsection",
            ".navbox",
            ".catlinks",
            "#catlinks",
            ".mw-footer",
            ".vector-header",
            ".vector-sidebar",
            ".vector-toc",
        ]:
            for node in root.select(selector):
                node.decompose()

        for node in root.select("sup.reference, span.mw-cite-backlink, span.mw-cite-backlink *"):
            node.decompose()

        text = root.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if title and title.lower() not in text[:200].lower():
            return f"{title}\n\n{text}"
        return text

    def _extract_links(self, root, *, base_url: str) -> List[str]:
        links: List[str] = []
        seen: Set[str] = set()
        for a in root.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("#"):
                continue
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            absolute = urljoin(base_url, href)
            try:
                normalized = self._normalize_url(absolute)
            except Exception:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            links.append(normalized)
            if len(links) >= 500:
                break
        return links

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 200].rstrip() + "\n\n[truncated]"
