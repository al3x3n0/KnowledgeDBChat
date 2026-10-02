"""A synthesis job asked for a file gets one.

`_generate_output_file` imported three builder singletons that never existed.
The ImportError was caught and logged, the job completed with no file, and the
download link led nowhere -- for every format, since the first two imports ran
before the format was looked at.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services import storage_service as storage_module
from app.services.synthesis_service import synthesis_service

pytestmark = pytest.mark.unit

CONTENT = """# Findings

## Caches

- The L2 prefetcher helps
- The L1 does not

## Next

Measure it again on a quiet host.
"""


@pytest.fixture
def uploads(monkeypatch):
    stored = {}

    async def upload_to_path(path, data, mime):
        stored[path] = (data, mime)

    monkeypatch.setattr(
        storage_module.storage_service, "upload_to_path", upload_to_path
    )
    return stored


def _job(output_format):
    return SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        title="Cache study",
        output_format=output_format,
        output_style="professional",
    )


@pytest.mark.parametrize(
    "output_format, magic",
    [("docx", b"PK"), ("pdf", b"%PDF")],
)
async def test_a_file_is_built_and_stored(uploads, output_format, magic):
    job = _job(output_format)

    result = await synthesis_service._generate_output_file(job, CONTENT, [])

    assert result["file_path"].endswith(f"synthesis_{job.id}.{output_format}")
    data, _mime = uploads[result["file_path"]]
    assert data.startswith(magic)
    assert result["file_size"] == len(data)


def test_slides_fit_the_outline_the_builder_takes():
    from app.schemas.presentation import SlideContent

    slides = synthesis_service._content_to_slides(CONTENT, "Cache study")

    assert [s["type"] for s in slides] == ["title", "content", "content"]
    for number, slide in enumerate(slides, start=1):
        SlideContent(
            slide_number=number,
            slide_type=slide["type"],
            title=slide["title"],
            subtitle=slide.get("subtitle"),
            content=slide.get("bullets") or [],
        )
