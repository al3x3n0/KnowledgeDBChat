"""
MCP tool for fetching and extracting readable content from web pages.
"""

from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.mcp.auth import MCPAuthContext
from app.services.web_scraper_service import WebScraperService


class WebScrapeTool:
    name = "web_scrape"
    description = "Fetch a web page (or small set of pages) and extract readable text and links"

    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch (http/https)"},
            "follow_links": {"type": "boolean", "description": "Crawl links from the page", "default": False},
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages to fetch when crawling",
                "default": 1,
                "minimum": 1,
                "maximum": 25,
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum crawl depth when follow_links is true",
                "default": 0,
                "minimum": 0,
                "maximum": 5,
            },
            "same_domain_only": {
                "type": "boolean",
                "description": "Only follow links on the same domain as the start URL",
                "default": True,
            },
            "include_links": {"type": "boolean", "description": "Include extracted links in the response", "default": True},
            "allow_private_networks": {
                "type": "boolean",
                "description": "Allow private-network hosts (admin-scope only)",
                "default": False,
            },
            "max_content_chars": {
                "type": "integer",
                "description": "Maximum characters to return per page",
                "default": 50000,
                "minimum": 1000,
                "maximum": 500000,
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.scraper = WebScraperService(enforce_network_safety=True)

    async def execute(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,  # unused, kept for MCP signature consistency
        url: str,
        follow_links: bool = False,
        max_pages: int = 1,
        max_depth: int = 0,
        same_domain_only: bool = True,
        include_links: bool = True,
        allow_private_networks: bool = False,
        max_content_chars: int = 50_000,
    ) -> Dict[str, Any]:
        auth.require_scope("read")
        if allow_private_networks:
            auth.require_scope("admin")

        logger.info(f"MCP web_scrape: url={url}, user={auth.user.username}")

        try:
            result = await self.scraper.scrape(
                url,
                follow_links=follow_links,
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
                include_links=include_links,
                allow_private_networks=allow_private_networks,
                max_content_chars=max_content_chars,
            )
            return result
        except Exception as e:
            logger.error(f"MCP web_scrape error: {e}")
            return {"error": str(e)}
