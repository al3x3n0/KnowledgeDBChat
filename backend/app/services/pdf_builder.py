"""
PDF document builder service.

Creates PDF files from structured content using reportlab.
"""

from io import BytesIO
from typing import Dict, Optional, Any, List
from loguru import logger

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted, ListFlowable, ListItem, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def hex_to_color(hex_color: str) -> colors.Color:
    """Convert hex color string to reportlab Color."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return colors.Color(r, g, b)


class PDFBuilder:
    """
    Builds PDF documents from structured content.

    Supports multiple built-in styles, custom themes, and various content types.
    """

    # Built-in style definitions
    STYLES = {
        "professional": {
            "title_color": "#1a365d",
            "heading_color": "#2e86ab",
            "text_color": "#333333",
            "accent_color": "#2e86ab",
            "title_font": "Helvetica-Bold",
            "body_font": "Helvetica",
            "code_font": "Courier",
            "title_size": 24,
            "heading1_size": 16,
            "heading2_size": 13,
            "body_size": 10,
            "code_size": 9,
        },
        "casual": {
            "title_color": "#4a90d9",
            "heading_color": "#ff6b6b",
            "text_color": "#2d3a4a",
            "accent_color": "#ff6b6b",
            "title_font": "Helvetica-Bold",
            "body_font": "Helvetica",
            "code_font": "Courier",
            "title_size": 26,
            "heading1_size": 18,
            "heading2_size": 14,
            "body_size": 11,
            "code_size": 9,
        },
        "technical": {
            "title_color": "#007acc",
            "heading_color": "#28a745",
            "text_color": "#24292e",
            "accent_color": "#28a745",
            "title_font": "Helvetica-Bold",
            "body_font": "Helvetica",
            "code_font": "Courier",
            "title_size": 22,
            "heading1_size": 14,
            "heading2_size": 12,
            "body_size": 10,
            "code_size": 8,
        },
    }

    def __init__(
        self,
        style: str = "professional",
        custom_theme: Optional[Dict[str, Any]] = None,
        page_size: str = "letter"
    ):
        """
        Initialize the builder with a style or custom theme.

        Args:
            style: Built-in style name (used if no custom_theme)
            custom_theme: Custom theme configuration dict
            page_size: Page size - "letter" or "a4"
        """
        self.style = style
        self.page_size = letter if page_size.lower() == "letter" else A4

        if custom_theme:
            self.style_config = self._parse_custom_theme(custom_theme)
        else:
            self.style_config = self.STYLES.get(style, self.STYLES["professional"])

        self._styles = None

    def _parse_custom_theme(self, theme: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse custom theme configuration into style config.

        Args:
            theme: Custom theme dict

        Returns:
            Style config dict compatible with built-in styles
        """
        # Start with professional defaults
        config = dict(self.STYLES["professional"])

        # Override with custom values
        for key in theme:
            if key in config:
                config[key] = theme[key]

        return config

    @classmethod
    def get_available_styles(cls) -> Dict[str, Dict[str, str]]:
        """
        Get list of available built-in styles with descriptions.

        Returns:
            Dict mapping style name to description
        """
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

    def _get_styles(self) -> Dict[str, ParagraphStyle]:
        """Get configured paragraph styles."""
        if self._styles:
            return self._styles

        styles = getSampleStyleSheet()

        # Title style
        styles.add(ParagraphStyle(
            name='DocTitle',
            parent=styles['Title'],
            fontName=self.style_config["title_font"],
            fontSize=self.style_config["title_size"],
            textColor=hex_to_color(self.style_config["title_color"]),
            alignment=TA_CENTER,
            spaceAfter=20,
        ))

        # Heading 1 style
        styles.add(ParagraphStyle(
            name='DocHeading1',
            parent=styles['Heading1'],
            fontName=self.style_config["title_font"],
            fontSize=self.style_config["heading1_size"],
            textColor=hex_to_color(self.style_config["heading_color"]),
            spaceBefore=16,
            spaceAfter=8,
        ))

        # Heading 2 style
        styles.add(ParagraphStyle(
            name='DocHeading2',
            parent=styles['Heading2'],
            fontName=self.style_config["title_font"],
            fontSize=self.style_config["heading2_size"],
            textColor=hex_to_color(self.style_config["heading_color"]),
            spaceBefore=12,
            spaceAfter=6,
        ))

        # Heading 3 style
        styles.add(ParagraphStyle(
            name='DocHeading3',
            fontName=self.style_config["body_font"],
            fontSize=self.style_config["body_size"] + 1,
            textColor=hex_to_color(self.style_config["heading_color"]),
            spaceBefore=10,
            spaceAfter=4,
            fontWeight='bold',
        ))

        # Body style
        styles.add(ParagraphStyle(
            name='DocBody',
            parent=styles['Normal'],
            fontName=self.style_config["body_font"],
            fontSize=self.style_config["body_size"],
            textColor=hex_to_color(self.style_config["text_color"]),
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14,
        ))

        # Code style
        styles.add(ParagraphStyle(
            name='DocCode',
            fontName=self.style_config["code_font"],
            fontSize=self.style_config["code_size"],
            textColor=hex_to_color(self.style_config["text_color"]),
            backColor=colors.Color(0.95, 0.95, 0.95),
            leftIndent=10,
            rightIndent=10,
            spaceBefore=8,
            spaceAfter=8,
        ))

        # Quote style
        styles.add(ParagraphStyle(
            name='DocQuote',
            parent=styles['Normal'],
            fontName=self.style_config["body_font"],
            fontSize=self.style_config["body_size"],
            textColor=hex_to_color(self.style_config["text_color"]),
            leftIndent=20,
            borderLeftWidth=3,
            borderLeftColor=hex_to_color(self.style_config["accent_color"]),
            spaceBefore=10,
            spaceAfter=10,
        ))

        # List item style
        styles.add(ParagraphStyle(
            name='DocListItem',
            parent=styles['Normal'],
            fontName=self.style_config["body_font"],
            fontSize=self.style_config["body_size"],
            textColor=hex_to_color(self.style_config["text_color"]),
            leftIndent=20,
            spaceAfter=4,
        ))

        self._styles = styles
        return styles

    def build(
        self,
        title: str,
        content_items: List[Dict[str, Any]],
        author: Optional[str] = None,
        subject: Optional[str] = None
    ) -> bytes:
        """
        Build a PDF document from content items.

        Args:
            title: Document title
            content_items: List of content items with type and data
            author: Optional author name for metadata
            subject: Optional subject for metadata

        Returns:
            PDF file as bytes

        Content item types:
            - heading: {"type": "heading", "level": 1|2|3, "text": "..."}
            - paragraph: {"type": "paragraph", "text": "..."}
            - bullet_list: {"type": "bullet_list", "items": ["...", "..."]}
            - numbered_list: {"type": "numbered_list", "items": ["...", "..."]}
            - code_block: {"type": "code_block", "code": "...", "language": "python"}
            - table: {"type": "table", "headers": [...], "rows": [[...], ...]}
            - quote: {"type": "quote", "text": "..."}
            - horizontal_rule: {"type": "horizontal_rule"}
            - page_break: {"type": "page_break"}
        """
        buffer = BytesIO()
        styles = self._get_styles()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=title,
            author=author or "",
            subject=subject or "",
        )

        story = []

        # Add title
        story.append(Paragraph(self._escape_html(title), styles['DocTitle']))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=hex_to_color(self.style_config["accent_color"]),
            spaceBefore=0,
            spaceAfter=20,
        ))

        # Process content items
        for item in content_items:
            item_type = item.get("type", "paragraph")

            if item_type == "heading":
                level = item.get("level", 1)
                text = self._escape_html(item.get("text", ""))
                style_name = f"DocHeading{min(level, 3)}"
                story.append(Paragraph(text, styles[style_name]))

            elif item_type == "paragraph":
                text = self._escape_html(item.get("text", ""))
                story.append(Paragraph(text, styles['DocBody']))

            elif item_type == "bullet_list":
                items = item.get("items", [])
                list_items = []
                for list_item in items:
                    list_items.append(
                        ListItem(Paragraph(self._escape_html(list_item), styles['DocListItem']))
                    )
                story.append(ListFlowable(list_items, bulletType='bullet', start='bulletchar'))
                story.append(Spacer(1, 10))

            elif item_type == "numbered_list":
                items = item.get("items", [])
                list_items = []
                for list_item in items:
                    list_items.append(
                        ListItem(Paragraph(self._escape_html(list_item), styles['DocListItem']))
                    )
                story.append(ListFlowable(list_items, bulletType='1'))
                story.append(Spacer(1, 10))

            elif item_type == "code_block":
                code = item.get("code", "")
                language = item.get("language", "")

                # Add language label if provided
                if language:
                    story.append(Paragraph(
                        f"<b>{self._escape_html(language)}</b>",
                        styles['DocCode']
                    ))

                # Add code as preformatted text
                story.append(Preformatted(code, styles['DocCode']))
                story.append(Spacer(1, 10))

            elif item_type == "table":
                headers = item.get("headers", [])
                rows = item.get("rows", [])
                table_data = []

                if headers:
                    table_data.append([self._escape_html(h) for h in headers])

                for row in rows:
                    table_data.append([self._escape_html(str(cell)) for cell in row])

                if table_data:
                    table = Table(table_data, repeatRows=1 if headers else 0)
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), hex_to_color(self.style_config["heading_color"])),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), self.style_config["title_font"]),
                        ('FONTSIZE', (0, 0), (-1, 0), self.style_config["body_size"]),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),
                        ('FONTNAME', (0, 1), (-1, -1), self.style_config["body_font"]),
                        ('FONTSIZE', (0, 1), (-1, -1), self.style_config["body_size"] - 1),
                        ('TEXTCOLOR', (0, 1), (-1, -1), hex_to_color(self.style_config["text_color"])),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
                    ])
                    table.setStyle(table_style)
                    story.append(table)
                    story.append(Spacer(1, 15))

            elif item_type == "quote":
                text = self._escape_html(item.get("text", ""))
                # Create a quote with left border effect using table
                quote_table = Table([[Paragraph(f"<i>{text}</i>", styles['DocBody'])]], colWidths=[self.page_size[0] - 200])
                quote_table.setStyle(TableStyle([
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('LINEBEFOERE', (0, 0), (0, -1), 3, hex_to_color(self.style_config["accent_color"])),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.98, 0.98, 0.98)),
                ]))
                story.append(quote_table)
                story.append(Spacer(1, 10))

            elif item_type == "horizontal_rule":
                story.append(HRFlowable(
                    width="100%",
                    thickness=1,
                    color=hex_to_color(self.style_config["accent_color"]),
                    spaceBefore=10,
                    spaceAfter=10,
                ))

            elif item_type == "page_break":
                story.append(PageBreak())

            else:
                # Default to paragraph
                text = self._escape_html(str(item.get("text", item)))
                story.append(Paragraph(text, styles['DocBody']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for reportlab paragraphs."""
        if not text:
            return ""
        return (
            text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )
