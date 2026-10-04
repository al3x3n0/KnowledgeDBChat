"""What the DOCX and PDF builders share about styles.

Both describe the same three built-in styles to the UI and both accept a
custom theme as a flat mapping laid over the professional defaults; each
carried its own copy. (The PPTX builder is deliberately not here: its theme is
nested under colors/fonts/sizes and it has a fourth style.)
"""

from typing import Any, Dict


class FlatThemeMixin:
    """Expects a ``STYLES`` mapping with a ``professional`` entry."""

    STYLES: Dict[str, Dict[str, Any]]

    def _parse_custom_theme(self, theme: Dict[str, Any]) -> Dict[str, Any]:
        """Professional defaults, overridden by the keys of ``theme`` that the
        style config knows; unknown keys are dropped."""
        config = dict(self.STYLES["professional"])
        for key, value in theme.items():
            # A field the person left unset arrives as None (the endpoint
            # sends the whole model); laid over a default, it broke the
            # build -- `None * int` in DOCX, `None.lstrip` in PDF.
            if value is None:
                continue
            if key in config:
                config[key] = value
        # The UI's one heading size names no builder key, so it was dropped.
        # It sets the top level; the second keeps the defaults' proportion.
        heading = theme.get("heading_size")
        if isinstance(heading, (int, float)) and heading > 0:
            config["heading1_size"] = heading
            config["heading2_size"] = max(1, round(heading * 14 / 18))
        return config

    @classmethod
    def get_available_styles(cls) -> Dict[str, Dict[str, str]]:
        """Built-in style names with the description the UI shows."""
        return {
            "professional": {
                "name": "Professional",
                "description": "Clean, corporate look with dark blue accents",
                "primary_color": "#1a365d",
            },
            "casual": {
                "name": "Casual",
                "description": "Friendly and approachable with warm colors",
                "primary_color": "#4a90d9",
            },
            "technical": {
                "name": "Technical",
                "description": "Developer-focused with monospace code blocks",
                "primary_color": "#007acc",
            },
        }
