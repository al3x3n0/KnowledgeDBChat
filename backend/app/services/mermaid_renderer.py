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
from app.utils.per_loop import PerLoop


class MermaidRenderError(Exception):
    """Raised when Mermaid rendering fails."""


class MermaidRenderer:
    """
    Renders Mermaid diagrams to PNG images.

    Uses local Kroki Docker container by default, with optional fallback
    to external kroki.io if local service is unavailable.
    """

    # Request timeout in seconds
    TIMEOUT = 30

    def __init__(self):
        self._clients: PerLoop[httpx.AsyncClient] = PerLoop(
            lambda: httpx.AsyncClient(timeout=self.TIMEOUT)
        )

    # Read when used, not when the singleton is built: the singleton outlives
    # whatever configuration the process had when it was first asked for. An
    # instance may still be told a value explicitly, which then wins.
    def _setting(self, name: str, value: object) -> object:
        return self.__dict__.get(f"_override{name}", value)

    @property
    def _kroki_url(self) -> str:
        """Primary: the stack's own renderer."""
        return str(self._setting("_kroki_url", settings.KROKI_URL)).rstrip("/")

    @_kroki_url.setter
    def _kroki_url(self, value: str) -> None:
        self.__dict__["_override_kroki_url"] = value

    @property
    def _fallback_url(self) -> str:
        """Fallback: external kroki.io."""
        return str(self._setting("_fallback_url", settings.KROKI_FALLBACK_URL)).rstrip(
            "/"
        )

    @_fallback_url.setter
    def _fallback_url(self, value: str) -> None:
        self.__dict__["_override_fallback_url"] = value

    @property
    def _use_fallback(self) -> bool:
        return bool(self._setting("_use_fallback", settings.KROKI_USE_FALLBACK))

    @_use_fallback.setter
    def _use_fallback(self, value: bool) -> None:
        self.__dict__["_override_use_fallback"] = value

    async def _get_client(self) -> httpx.AsyncClient:
        """An HTTP client bound to the running event loop.

        The renderer is a process-wide singleton, and a Celery task runs each
        job under its own `asyncio.run`. A client kept from an earlier task is
        bound to a loop that has closed, so in a worker every render after the
        first failed and the diagram was quietly left out.
        """
        return self._clients.get()

    async def close(self):
        """Close the running loop's HTTP client."""
        client = self._clients.pop()
        if client:
            await client.aclose()

    def _is_companion(self, base_url: str) -> bool:
        """Whether this endpoint is a Kroki companion rather than a gateway.

        The two speak different shapes: a gateway takes
        /mermaid/svg/<encoded>, a companion takes the raw source POSTed to
        /svg. Configured rather than probed, because probing costs a failed
        request on every render and the answer never changes at run time.
        """
        return bool(settings.KROKI_LOCAL_IS_COMPANION) and base_url == self._kroki_url

    def _encode_diagram(self, code: str) -> str:
        """
        Encode Mermaid code for Kroki API.

        Uses deflate compression + base64 URL-safe encoding.
        """
        # Compress with zlib (deflate)
        compressed = zlib.compress(code.encode("utf-8"), level=9)
        # Base64 URL-safe encode
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
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
            "flowchart",
            "graph",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "gantt",
            "pie",
            "mindmap",
            "journey",
            "gitGraph",
            "C4Context",
            "sankey",
            "timeline",
            "quadrantChart",
            "requirementDiagram",
            "architecture",
        ]

        first_line = code.split("\n")[0].strip().lower()
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
        if code.startswith("```mermaid"):
            code = code[len("```mermaid") :].strip()
        elif code.startswith("```"):
            code = code[3:].strip()

        if code.endswith("```"):
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

        self.last_render_used_fallback = False
        try:
            # Try local Kroki first
            return await self._render_via_kroki(
                code, format="png", base_url=self._kroki_url
            )
        except Exception as e:
            logger.warning(f"Local Kroki rendering failed: {e}")

            # Try fallback if enabled
            if self._use_fallback and self._fallback_url != self._kroki_url:
                try:
                    # The fallback is a public service by default, so the
                    # diagram source leaves the deployment. Say so at warning
                    # level: a local Kroki missing its Mermaid companion sent
                    # every diagram to kroki.io and nothing recorded it.
                    logger.warning(
                        "Rendering diagram via the external fallback "
                        f"{self._fallback_url}; diagram source leaves this "
                        "deployment. Local Kroki failed: %s" % e
                    )
                    self.last_render_used_fallback = True
                    return await self._render_via_kroki(
                        code, format="png", base_url=self._fallback_url
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback Kroki also failed: {fallback_error}")
                    raise MermaidRenderError(
                        f"Failed to render diagram: {e} (fallback: {fallback_error})"
                    )

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
            result = await self._render_via_kroki(
                code, format="svg", base_url=self._kroki_url
            )
            return result.decode("utf-8")
        except Exception as e:
            logger.warning(f"Local Kroki SVG rendering failed: {e}")

            # Try fallback if enabled
            if self._use_fallback and self._fallback_url != self._kroki_url:
                try:
                    logger.info("Trying fallback Kroki service for SVG...")
                    result = await self._render_via_kroki(
                        code, format="svg", base_url=self._fallback_url
                    )
                    return result.decode("utf-8")
                except Exception as fallback_error:
                    logger.error(f"Fallback Kroki SVG also failed: {fallback_error}")
                    raise MermaidRenderError(
                        f"Failed to render diagram: {e} (fallback: {fallback_error})"
                    )

            raise MermaidRenderError(f"Failed to render diagram: {e}")

    async def _render_via_kroki(
        self, code: str, format: str = "png", base_url: Optional[str] = None
    ) -> bytes:
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
        companion = self._is_companion(kroki_base)

        if not companion:
            # A full Kroki gateway. Method 1: GET with the encoded diagram in
            # the path, which is cacheable.
            encoded = self._encode_diagram(code)
            url = f"{kroki_base}/mermaid/{format}/{encoded}"

            try:
                response = await client.get(url)

                if response.status_code == 200:
                    return response.content

                # If GET fails, try POST
                logger.warning(
                    f"Kroki GET failed with status {response.status_code}, "
                    "trying POST"
                )

            except httpx.TimeoutException:
                logger.warning("Kroki GET timed out, trying POST")

        # Method 2: POST the raw diagram.
        #
        # A Kroki companion renders one format and serves it at /svg and /png,
        # with no `mermaid` segment and no encoded-URL form -- the gateway that
        # provides those is 3.76 GB of diagram backends this project never
        # asks for, since the only caller renders Mermaid. Talking to the
        # companion directly drops it.
        endpoint = (
            f"{kroki_base}/{format}" if companion else f"{kroki_base}/mermaid/{format}"
        )
        headers = {"Content-Type": "text/plain"}

        response = await client.post(endpoint, content=code, headers=headers)

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "Unknown error"
            raise MermaidRenderError(
                f"Kroki API error (status {response.status_code}): {error_text}"
            )

        return response.content

    async def render_multiple(self, diagrams: dict[int, str]) -> dict[int, bytes]:
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
