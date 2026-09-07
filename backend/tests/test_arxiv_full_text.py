"""A paper's abstract is not a paper.

The reproduce-a-paper pipeline asks a stage to "read the paper into an
implementable specification". Ingestion stored arXiv metadata and the abstract
-- 1520 characters for Lemire's random-integer paper -- so the stage was being
asked to implement an algorithm from a summary. Measured live: the agent read
the document, correctly judged it insufficient, and tried to fetch the PDF
itself.

A specification written from an abstract is the model's recall wearing the
paper's name, and a contract counting `algorithm_spec` cannot tell the two
apart. Hence the full text, and hence the document saying which one it holds.
"""

import pytest

from app.services.connectors.arxiv_connector import (
    ABSTRACT_ONLY_HEADING,
    FULL_TEXT_HEADING,
    ArxivConnector,
)

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _Stream:
    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        if self._error:
            raise self._error
        return _Response(self._chunks)

    async def __aexit__(self, *args):
        return False


class _Session:
    def __init__(self, chunks=(b"%PDF-1.4 fake",), error=None):
        self.stream = _Stream(list(chunks), error)


def _connector(session=None, entry=None):
    connector = ArxivConnector()
    connector.session = session or _Session()
    connector.is_initialized = True
    connector.entry_cache = {
        "paper-1": entry
        or {
            "title": "Fast Random Integer Generation in an Interval",
            "summary": "We show that ...",
            "pdf_url": "https://arxiv.org/pdf/1805.10941v4",
        }
    }
    return connector


@pytest.mark.asyncio
class TestTheDocumentSaysWhichItIs:
    async def test_full_text_is_appended_under_its_own_heading(self, monkeypatch):
        monkeypatch.setattr(
            ArxivConnector, "_pdf_to_text", staticmethod(lambda data: "SECTION 1 ...")
        )

        content = await _connector().get_document_content("paper-1")

        assert FULL_TEXT_HEADING in content
        assert "SECTION 1 ..." in content
        assert ABSTRACT_ONLY_HEADING not in content
        # The metadata is still there; full text is added, not substituted.
        assert "Fast Random Integer Generation" in content

    async def test_a_failed_fetch_says_so_in_the_document(self, monkeypatch):
        """The whole point. A document that silently contains only the abstract
        is indistinguishable from a paper that is simply short, and the stage
        reading it cannot know it is specifying from a summary."""
        connector = _connector(_Session(error=RuntimeError("connection refused")))

        content = await connector.get_document_content("paper-1")

        assert ABSTRACT_ONLY_HEADING in content
        assert "connection refused" in content
        assert FULL_TEXT_HEADING not in content

    async def test_an_oversized_pdf_is_not_downloaded_and_says_why(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.connectors.arxiv_connector.settings.ARXIV_FULL_TEXT_MAX_BYTES",
            10,
            raising=False,
        )
        connector = _connector(_Session(chunks=(b"x" * 50,)))

        content = await connector.get_document_content("paper-1")

        assert ABSTRACT_ONLY_HEADING in content
        assert "exceeded" in content

    async def test_a_scanned_paper_reports_no_text_rather_than_a_blank(
        self, monkeypatch
    ):
        """An image-only PDF extracts to nothing. Storing that as the full text
        would be a blank document that looks like a successful extraction."""
        monkeypatch.setattr(
            ArxivConnector, "_pdf_to_text", staticmethod(lambda data: "")
        )

        content = await _connector().get_document_content("paper-1")

        assert ABSTRACT_ONLY_HEADING in content
        assert "no text could be extracted" in content

    async def test_it_can_be_turned_off(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.connectors.arxiv_connector.settings.ARXIV_FULL_TEXT_ENABLED",
            False,
            raising=False,
        )

        content = await _connector().get_document_content("paper-1")

        assert ABSTRACT_ONLY_HEADING in content
        assert "disabled" in content

    async def test_an_entry_with_no_pdf_is_not_an_error(self):
        connector = _connector(
            entry={"title": "A paper", "summary": "abstract", "pdf_url": None}
        )

        content = await connector.get_document_content("paper-1")

        assert ABSTRACT_ONLY_HEADING in content
        assert "no PDF" in content


class TestUnreadableBytesAreNotAnException:
    def test_garbage_extracts_to_empty_rather_than_raising(self):
        """The caller turns "" into a stated reason; a raise would fail the
        whole ingestion over one bad paper."""
        assert ArxivConnector._pdf_to_text(b"not a pdf at all") == ""
