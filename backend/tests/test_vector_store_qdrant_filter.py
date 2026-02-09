import pytest


def test_qdrant_filter_from_metadata_builds_must_conditions():
    from app.services.vector_store import VectorStoreService

    svc = VectorStoreService()
    svc.provider = "qdrant"

    filt = svc._qdrant_filter_from_metadata(
        {
            "$and": [
                {"source_type": "file"},
                {"document_id": {"$in": ["a", "b"]}},
            ]
        }
    )

    assert filt is not None
    # The Qdrant models expose `must` as a list of conditions.
    must = getattr(filt, "must", None)
    assert isinstance(must, list)
    assert len(must) == 2
