"""The ONNX embedding backend that replaced torch.

The interesting property -- that its vectors agree with the torch pipeline's --
cannot be asserted here without downloading two models and installing the very
stack this replaced. It was measured before the switch (per-vector cosine
1.000000, identical top-5 rankings on 30 passages and 8 queries) and is
recorded in `services/onnx_embeddings.py`. What is asserted here is everything
that can go wrong without a network: the repo-id resolution, the backend
selection, and the shape contract the call sites depend on.
"""

import sys
import types

import numpy as np
import pytest

from app.services.onnx_embeddings import (
    OnnxTextEmbedder,
    load_cross_encoder,
    load_text_embedder,
    resolve_repo_id,
)

pytestmark = pytest.mark.unit


class TestRepoResolution:
    def test_bare_name_gets_the_sentence_transformers_org(self):
        # SentenceTransformer("all-MiniLM-L6-v2") resolved this way, and an
        # index built under the old pipeline must keep pointing at one model.
        assert resolve_repo_id("all-MiniLM-L6-v2") == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_qualified_name_is_left_alone(self):
        assert (
            resolve_repo_id("cross-encoder/ms-marco-MiniLM-L-6-v2")
            == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


class TestBackendSelection:
    """`EMBEDDING_BACKEND=sentence-transformers` has to still mean something."""

    @pytest.fixture
    def fake_sentence_transformers(self, monkeypatch):
        module = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            def __init__(self, model_id):
                self.model_id = model_id

        class CrossEncoder:
            def __init__(self, model_id):
                self.model_id = model_id

        module.SentenceTransformer = SentenceTransformer
        module.CrossEncoder = CrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return module

    def test_opting_out_loads_sentence_transformers(
        self, monkeypatch, fake_sentence_transformers
    ):
        from app.core.config import settings

        monkeypatch.setattr(
            settings, "EMBEDDING_BACKEND", "sentence-transformers", raising=False
        )
        embedder = load_text_embedder("all-MiniLM-L6-v2")
        reranker = load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert isinstance(embedder, fake_sentence_transformers.SentenceTransformer)
        assert isinstance(reranker, fake_sentence_transformers.CrossEncoder)

    def test_opting_out_without_the_package_says_how_to_get_it(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(
            settings, "EMBEDDING_BACKEND", "sentence-transformers", raising=False
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(RuntimeError) as excinfo:
            load_text_embedder("all-MiniLM-L6-v2")
        # A missing optional dependency should name its own install command.
        assert "pip install sentence-transformers" in str(excinfo.value)


class TestEncodeShapeContract:
    """Call sites do `.encode(query).tolist()` and index `[i]` into batches."""

    class _Embedder(OnnxTextEmbedder):
        """The pooling and shape logic, without the ONNX session under it."""

        def __init__(self, dim=4):
            self._dimension = None
            self._dim = dim

        def _run(self, encodings):
            n = len(encodings)
            hidden = np.tile(np.arange(1, self._dim + 1, dtype=np.float32), (n, 3, 1))
            mask = np.ones((n, 3), dtype=np.int64)
            return hidden, mask

        @property
        def _tokenizer(self):
            class _Tok:
                @staticmethod
                def encode_batch(texts):
                    return [object() for _ in texts]

            return _Tok()

    def test_a_single_string_returns_one_flat_vector(self):
        vector = self._Embedder().encode("one passage")
        assert vector.ndim == 1
        assert len(vector.tolist()) == 4

    def test_a_list_returns_a_matrix(self):
        matrix = self._Embedder().encode(["a", "b", "c"])
        assert matrix.shape == (3, 4)

    def test_vectors_are_unit_length(self):
        # Qdrant collections here use cosine distance; a stored vector that is
        # not normalised is not wrong, but it is not what the old pipeline
        # wrote either.
        matrix = self._Embedder().encode(["a", "b"])
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)

    def test_batching_does_not_change_the_answer(self):
        one = self._Embedder().encode(["a", "b", "c", "d"], batch_size=1)
        many = self._Embedder().encode(["a", "b", "c", "d"], batch_size=32)
        assert np.allclose(one, many)

    def test_empty_input_is_an_empty_matrix_not_an_error(self):
        assert self._Embedder().encode([]).shape == (0, 4)
