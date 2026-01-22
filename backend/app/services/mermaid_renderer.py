"""
Mermaid diagram rendering service.

Renders Mermaid diagram code to PNG images using local or external Kroki service.
"""

import base64
import zlib
from typing import Optional
import httpx
from loguru import logger

from app.core.config import settings


class MermaidRenderError(Exception):
    """Raised when Mermaid rendering fails."""
    pass


class MermaidRenderer:
    """
    Renders Mermaid diagrams to PNG images.

    Uses local Kroki Docker container by default, with optional fallback
    to external kroki.io if local service is unavailable.
    """

    # Request timeout in seconds
    TIMEOUT = 30

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        # Primary: local Kroki Docker container
        self._kroki_url = settings.KROKI_URL.rstrip('/')
        # Fallback: external kroki.io
        self._fallback_url = settings.KROKI_FALLBACK_URL.rstrip('/')
        self._use_fallback = settings.KROKI_USE_FALLBACK

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.TIMEOUT)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _encode_diagram(self, code: str) -> str:
        """
        Encode Mermaid code for Kroki API.

        Uses deflate compression + base64 URL-safe encoding.
        """
        # Compress with zlib (deflate)
        compressed = zlib.compress(code.encode('utf-8'), level=9)
        # Base64 URL-safe encode
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        return encoded

    def _validate_mermaid_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Basic validation of Mermaid code.

        Returns (is_valid, error_message).
        """
        code = code.strip()

        if not code:
            return False, "Empty diagram code"

        # Check for valid diagram type declarations
        valid_starts = [
            'flowchart', 'graph', 'sequenceDiagram', 'classDiagram',
            'stateDiagram', 'erDiagram', 'gantt', 'pie', 'mindmap',
            'journey', 'gitGraph', 'C4Context', 'sankey', 'timeline',
            'quadrantChart', 'requirementDiagram', 'architecture'
        ]

        first_line = code.split('\n')[0].strip().lower()
        has_valid_start = any(first_line.startswith(v.lower()) for v in valid_starts)

        if not has_valid_start:
            return False, f"Invalid diagram type. Code starts with: {first_line[:50]}"

        return True, None

    def _clean_mermaid_code(self, code: str) -> str:
        """
        Clean and normalize Mermaid code.

        Removes markdown code blocks and normalizes whitespace.
        """
        code = code.strip()

        # Remove markdown code blocks if present
        if code.startswith('```mermaid'):
            code = code[len('```mermaid'):].strip()
        elif code.startswith('```'):
            code = code[3:].strip()

        if code.endswith('```'):
            code = code[:-3].strip()

        return code

    async def render_to_png(self, code: str) -> bytes:
        """
        Render Mermaid diagram to PNG image.

        Args:
            code: Mermaid diagram code

        Returns:
            PNG image as bytes

        Raises:
            MermaidRenderError: If rendering fails
        """
        code = self._clean_mermaid_code(code)

        is_valid, error = self._validate_mermaid_code(code)
        if not is_valid:
            raise MermaidRenderError(f"Invalid Mermaid code: {error}")

        try:
            # Try local Kroki first
            return await self._render_via_kroki(code, format="png", base_url=self._kroki_url)
        except Exception as e:
            logger.warning(f"Local Kroki rendering failed: {e}")

            # Try fallback if enabled
            if self._use_fallback and self._fallback_url != self._kroki_url:
                try:
                    logger.info("Trying fallback Kroki service...")
                    return await self._render_via_kroki(code, format="png", base_url=self._fallback_url)
                except Exception as fallback_error:
                    logger.error(f"Fallback Kroki also failed: {fallback_error}")
                    raise MermaidRenderError(f"Failed to render diagram: {e} (fallback: {fallback_error})")

            raise MermaidRenderError(f"Failed to render diagram: {e}")

    async def render_to_svg(self, code: str) -> str:
        """
        Render Mermaid diagram to SVG.

        Args:
            code: Mermaid diagram code

        Returns:
            SVG content as string

        Raises:
            MermaidRenderError: If rendering fails
        """
        code = self._clean_mermaid_code(code)

        is_valid, error = self._validate_mermaid_code(code)
        if not is_valid:
            raise MermaidRenderError(f"Invalid Mermaid code: {error}")

        try:
            # Try local Kroki first
            result = await self._render_via_kroki(code, format="svg", base_url=self._kroki_url)
            return result.decode('utf-8')
        except Exception as e:
            logger.warning(f"Local Kroki SVG rendering failed: {e}")

            # Try fallback if enabled
            if self._use_fallback and self._fallback_url != self._kroki_url:
                try:
                    logger.info("Trying fallback Kroki service for SVG...")
                    result = await self._render_via_kroki(code, format="svg", base_url=self._fallback_url)
                    return result.decode('utf-8')
                except Exception as fallback_error:
                    logger.error(f"Fallback Kroki SVG also failed: {fallback_error}")
                    raise MermaidRenderError(f"Failed to render diagram: {e} (fallback: {fallback_error})")

            raise MermaidRenderError(f"Failed to render diagram: {e}")

    async def _render_via_kroki(self, code: str, format: str = "png", base_url: Optional[str] = None) -> bytes:
        """
        Render diagram using Kroki API.

        Args:
            code: Mermaid diagram code
            format: Output format (png, svg)
            base_url: Kroki base URL (defaults to local)

        Returns:
            Rendered image/SVG as bytes
        """
        client = await self._get_client()
        kroki_base = base_url or self._kroki_url

        # Method 1: GET with encoded URL (preferred for caching)
        encoded = self._encode_diagram(code)
        url = f"{kroki_base}/mermaid/{format}/{encoded}"

        try:
            response = await client.get(url)

            if response.status_code == 200:
                return response.content

            # If GET fails, try POST
            logger.warning(f"Kroki GET failed with status {response.status_code}, trying POST")

        except httpx.TimeoutException:
            logger.warning("Kroki GET timed out, trying POST")

        # Method 2: POST with raw code (fallback)
        endpoint = f"{kroki_base}/mermaid/{format}"
        headers = {"Content-Type": "text/plain"}

        response = await client.post(endpoint, content=code, headers=headers)

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "Unknown error"
            raise MermaidRenderError(
                f"Kroki API error (status {response.status_code}): {error_text}"
            )

        return response.content

    async def render_multiple(
        self,
        diagrams: dict[int, str]
    ) -> dict[int, bytes]:
        """
        Render multiple diagrams concurrently.

        Args:
            diagrams: Dict mapping slide_number to Mermaid code

        Returns:
            Dict mapping slide_number to PNG bytes
        """
        import asyncio

        results = {}
        errors = []

        async def render_one(slide_num: int, code: str):
            try:
                png_bytes = await self.render_to_png(code)
                results[slide_num] = png_bytes
            except MermaidRenderError as e:
                errors.append((slide_num, str(e)))
                logger.warning(f"Failed to render diagram for slide {slide_num}: {e}")

        # Render all diagrams concurrently
        tasks = [render_one(num, code) for num, code in diagrams.items()]
        await asyncio.gather(*tasks)

        if errors:
            logger.warning(f"Some diagrams failed to render: {errors}")

        return results


# Singleton instance
_renderer: Optional[MermaidRenderer] = None


def get_mermaid_renderer() -> MermaidRenderer:
    """Get the singleton MermaidRenderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = MermaidRenderer()
    return _renderer
