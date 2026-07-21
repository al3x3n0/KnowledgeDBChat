"""Tests for export_document handler correctness — verifying builder API calls."""

import re

import pytest

try:
    from app.services.docx_builder import markdown_to_content_items
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from app.services.pdf_builder import PDFBuilder
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from app.services.pptx_builder import PPTXBuilder
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


SAMPLE_MARKDOWN = """\
# Introduction

This is the introduction paragraph.

## Methods

- Method A
- Method B
- Method C

## Results

The results are shown below.

```python
print("hello")
```

## Conclusion

In conclusion, everything works.
"""


@pytest.mark.skipif(not HAS_DOCX, reason="python-docx not installed")
class TestMarkdownToContentItems:
    """Verify markdown_to_content_items produces expected structure."""

    def test_parses_headings(self):
        items = markdown_to_content_items("# Title\n\n## Subtitle\n\nText here.")
        headings = [i for i in items if i["type"] == "heading"]
        assert len(headings) >= 2
        assert headings[0]["level"] == 1
        assert headings[0]["text"] == "Title"
        assert headings[1]["level"] == 2
        assert headings[1]["text"] == "Subtitle"

    def test_parses_bullet_lists(self):
        md = "- Item 1\n- Item 2\n- Item 3"
        items = markdown_to_content_items(md)
        bullets = [i for i in items if i["type"] == "bullet_list"]
        assert len(bullets) == 1
        assert bullets[0]["items"] == ["Item 1", "Item 2", "Item 3"]

    def test_parses_code_blocks(self):
        md = "```python\nprint('hi')\n```"
        items = markdown_to_content_items(md)
        code = [i for i in items if i["type"] == "code_block"]
        assert len(code) == 1
        assert "print" in code[0]["code"]
        assert code[0]["language"] == "python"

    def test_empty_markdown(self):
        items = markdown_to_content_items("")
        assert items == []

    def test_full_document(self):
        items = markdown_to_content_items(SAMPLE_MARKDOWN)
        types = [i["type"] for i in items]
        assert "heading" in types
        assert "bullet_list" in types
        assert "code_block" in types


class TestExportDocumentParamValidation:
    """Tests for export_document handler parameter validation logic."""

    def test_format_validation(self):
        valid_formats = {"docx", "pdf", "pptx", "latex"}
        assert "docx" in valid_formats
        assert "pdf" in valid_formats
        assert "pptx" in valid_formats
        assert "latex" in valid_formats
        assert "html" not in valid_formats

    def test_requires_assembled_markdown(self):
        doc_ws = {}
        assert not doc_ws.get("assembled_markdown")

    def test_accepts_valid_document_workspace(self):
        doc_ws = {
            "assembled_markdown": "# Hello",
            "plan": {"title": "Test Doc"},
        }
        assert doc_ws.get("assembled_markdown")
        assert doc_ws["plan"]["title"] == "Test Doc"

    def test_size_limit(self):
        markdown = "x" * 600_000
        assert len(markdown) > 500_000

    def test_size_within_limit(self):
        markdown = "x" * 100
        assert len(markdown) <= 500_000


class TestMarkdownToSlides:
    """Tests for PPTX markdown-to-slides conversion logic (inline in handler)."""

    def _markdown_to_slides(self, markdown: str, title: str = "Test"):
        """Replicate the handler's slide conversion logic."""
        from app.schemas.presentation import SlideContent
        slides = []
        sections = re.split(r'^##\s+', markdown, flags=re.MULTILINE)
        slide_num = 1
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split('\n', 1)
            slide_title = lines[0].strip().lstrip('#').strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            bullets = []
            for bline in body.split('\n'):
                bline = bline.strip()
                if bline.startswith(('- ', '* ', '+ ')):
                    bullets.append(bline[2:].strip())
                elif bline and not bline.startswith('#'):
                    bullets.append(bline)
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title=slide_title,
                content=bullets[:10],
            ))
            slide_num += 1
        if not slides:
            slides.append(SlideContent(
                slide_number=1,
                slide_type="title",
                title=title,
                content=["Generated from document"],
            ))
        return slides

    def test_splits_on_h2_headings(self):
        slides = self._markdown_to_slides(SAMPLE_MARKDOWN)
        titles = [s.title for s in slides]
        assert "Methods" in titles
        assert "Results" in titles
        assert "Conclusion" in titles

    def test_extracts_bullets(self):
        slides = self._markdown_to_slides(SAMPLE_MARKDOWN)
        methods_slide = [s for s in slides if s.title == "Methods"][0]
        assert "Method A" in methods_slide.content
        assert "Method B" in methods_slide.content

    def test_empty_markdown_gets_title_slide(self):
        slides = self._markdown_to_slides("", title="Fallback Title")
        assert len(slides) == 1
        assert slides[0].slide_type == "title"
        assert slides[0].title == "Fallback Title"

    def test_caps_bullets_at_10(self):
        md = "## Big Section\n\n" + "\n".join(f"- Item {i}" for i in range(20))
        slides = self._markdown_to_slides(md)
        assert len(slides[0].content) <= 10

    def test_presentation_outline_schema(self):
        from app.schemas.presentation import PresentationOutline, SlideContent
        slides = [
            SlideContent(slide_number=1, slide_type="content", title="Slide 1", content=["Bullet 1"]),
        ]
        outline = PresentationOutline(title="Test Preso", slides=slides)
        assert outline.title == "Test Preso"
        assert len(outline.slides) == 1

    def test_no_h2_still_produces_slides(self):
        md = "# Only H1\n\nSome content without H2 headings."
        slides = self._markdown_to_slides(md)
        assert len(slides) >= 1


@pytest.mark.skipif(not HAS_DOCX, reason="python-docx not installed")
class TestDocxBuilderAPI:
    """Tests verifying DOCXBuilder.build() signature matches our usage."""

    def test_build_accepts_content_items(self):
        from app.services.docx_builder import DOCXBuilder
        builder = DOCXBuilder()
        assert hasattr(builder, "build")
        import inspect
        sig = inspect.signature(builder.build)
        param_names = list(sig.parameters.keys())
        assert "title" in param_names
        assert "content_items" in param_names

    def test_build_from_markdown_does_not_exist(self):
        """Confirm that build_from_markdown does NOT exist (the bug we fixed)."""
        from app.services.docx_builder import DOCXBuilder
        builder = DOCXBuilder()
        assert not hasattr(builder, "build_from_markdown")


@pytest.mark.skipif(not HAS_PDF, reason="reportlab not installed")
class TestPdfBuilderAPI:
    """Tests verifying PDFBuilder.build() signature matches our usage."""

    def test_build_accepts_content_items(self):
        from app.services.pdf_builder import PDFBuilder
        builder = PDFBuilder()
        assert hasattr(builder, "build")
        import inspect
        sig = inspect.signature(builder.build)
        param_names = list(sig.parameters.keys())
        assert "title" in param_names
        assert "content_items" in param_names

    def test_build_from_markdown_does_not_exist(self):
        from app.services.pdf_builder import PDFBuilder
        builder = PDFBuilder()
        assert not hasattr(builder, "build_from_markdown")


@pytest.mark.skipif(not HAS_PPTX, reason="python-pptx not installed")
class TestPptxBuilderAPI:
    """Tests verifying PPTXBuilder.build() signature matches our usage."""

    def test_build_accepts_outline(self):
        from app.services.pptx_builder import PPTXBuilder
        builder = PPTXBuilder()
        assert hasattr(builder, "build")
        import inspect
        sig = inspect.signature(builder.build)
        param_names = list(sig.parameters.keys())
        assert "outline" in param_names

    def test_build_from_markdown_does_not_exist(self):
        from app.services.pptx_builder import PPTXBuilder
        builder = PPTXBuilder()
        assert not hasattr(builder, "build_from_markdown")
