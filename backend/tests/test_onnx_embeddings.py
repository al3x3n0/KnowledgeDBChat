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
    ApiTextEmbedder,
    OnnxTextEmbedder,
    load_cross_encoder,
    load_text_embedder,
    resolve_embedder,
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


class _FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _FakeHttpxModule(types.ModuleType):
    """A stand-in for `httpx` whose `post` returns a canned response.

    `ApiTextEmbedder._post` imports `httpx` at call time, so this is swapped
    into `sys.modules["httpx"]` for the duration of a test.
    """

    def __init__(self, responses, error: Exception | None = None):
        super().__init__("httpx")
        self._responses = list(responses)
        self._error = error
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.posts.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self._error is not None:
            raise self._error
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _install_fake_httpx(monkeypatch, module: _FakeHttpxModule):
    monkeypatch.setitem(sys.modules, "httpx", module)
    return module


class TestApiTextEmbedder:
    """The external OpenAI-compatible `/embeddings` backend."""

    BASE = "http://embedding-service.invalid:10999/api/v1"

    def _embedder(self, monkeypatch, payload, *, key=None, error=None):
        fake = _FakeHttpxModule([_FakeResponse(payload)], error=error)
        _install_fake_httpx(monkeypatch, fake)
        return ApiTextEmbedder("model-x", base_url=self.BASE, api_key=key), fake

    def test_posts_to_the_embeddings_path_and_returns_a_single_vector(
        self, monkeypatch
    ):
        payload = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}
        embedder, fake = self._embedder(monkeypatch, payload)
        vector = embedder.encode("a passage")
        # The contract the call sites rely on: one string -> one flat vector.
        assert vector.ndim == 1
        # float32 cannot hold 0.1 exactly, so compare within tolerance.
        assert np.allclose(vector, [0.1, 0.2, 0.3, 0.4])
        # It must hit {base}/embeddings and name the model from EMBEDDING_MODEL.
        call = fake.posts[0]
        assert call["url"] == f"{self.BASE}/embeddings"
        assert call["json"]["model"] == "model-x"
        assert call["json"]["input"] == ["a passage"]

    def test_a_sequence_returns_a_matrix_in_request_order(self, monkeypatch):
        payload = {
            "data": [
                {"embedding": [1.0, 0.0], "index": 0},
                {"embedding": [0.0, 1.0], "index": 1},
                {"embedding": [0.5, 0.5], "index": 2},
            ]
        }
        embedder, _ = self._embedder(monkeypatch, payload)
        matrix = embedder.encode(["a", "b", "c"])
        assert matrix.shape == (3, 2)
        # 1.0/0.0 are exact in float32; 0.5 is not, so use tolerance.
        assert np.allclose(matrix[0], [1.0, 0.0])
        assert np.allclose(matrix[2], [0.5, 0.5])

    def test_sends_the_api_key_as_a_bearer_token(self, monkeypatch):
        payload = {"data": [{"embedding": [0.0]}]}
        embedder, fake = self._embedder(monkeypatch, payload, key="sekret")
        embedder.encode("x")
        assert fake.posts[0]["headers"]["Authorization"] == "Bearer sekret"

    def test_no_key_sends_no_authorization_header(self, monkeypatch):
        payload = {"data": [{"embedding": [0.0]}]}
        embedder, fake = self._embedder(monkeypatch, payload, key=None)
        embedder.encode("x")
        assert "Authorization" not in fake.posts[0]["headers"]

    def test_get_sentence_embedding_dimension_probes_the_api(self, monkeypatch):
        payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        embedder, _ = self._embedder(monkeypatch, payload)
        assert embedder.get_sentence_embedding_dimension() == 3

    def test_http_error_is_surfaced_not_swallowed(self, monkeypatch):
        fake = _FakeHttpxModule([_FakeResponse({}, status_code=503, text="boom")])
        _install_fake_httpx(monkeypatch, fake)
        embedder = ApiTextEmbedder("m", base_url=self.BASE)
        with pytest.raises(RuntimeError) as excinfo:
            embedder.encode("x")
        assert "503" in str(excinfo.value)

    def test_missing_embedding_field_is_an_error(self, monkeypatch):
        fake = _FakeHttpxModule([_FakeResponse({"data": [{"index": 0}]})])
        _install_fake_httpx(monkeypatch, fake)
        embedder = ApiTextEmbedder("m", base_url=self.BASE)
        with pytest.raises(RuntimeError):
            embedder.encode("x")

    def test_vector_count_mismatch_is_an_error(self, monkeypatch):
        # The API answered for one input when three were requested.
        fake = _FakeHttpxModule([_FakeResponse({"data": [{"embedding": [0.0]}]})])
        _install_fake_httpx(monkeypatch, fake)
        embedder = ApiTextEmbedder("m", base_url=self.BASE)
        with pytest.raises(RuntimeError):
            embedder.encode(["a", "b", "c"])

    def test_empty_input_returns_an_empty_matrix_without_a_call(self, monkeypatch):
        # An empty batch must not hit the network: dimension stays whatever it
        # already is (0 until the first real encode), not a fresh probe.
        embedder, fake = self._embedder(monkeypatch, {"data": []})
        assert embedder.encode([]).shape == (0, 0)
        assert fake.posts == []
        embedder._dimension = 4
        assert embedder.encode([]).shape == (0, 4)


class TestResolveEmbedder:
    """`EMBEDDING_API_BASE` set -> remote API; unset -> the local model."""

    BASE = "http://embedding-service.invalid:10999/api/v1"

    def test_api_base_set_returns_the_api_embedder(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "EMBEDDING_API_BASE", self.BASE, raising=False)
        monkeypatch.setattr(settings, "EMBEDDING_API_KEY", None, raising=False)
        embedder = resolve_embedder("model-x")
        # A real API embedder would need the network only at encode() time, so
        # construction is safe to assert on here.
        assert isinstance(embedder, ApiTextEmbedder)
        assert embedder._url == f"{self.BASE}/embeddings"

    def test_api_base_unset_falls_back_to_local_onnx(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "EMBEDDING_API_BASE", None, raising=False)
        monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "onnx", raising=False)

        from app.services import onnx_embeddings as mod

        class _FakeOnnx(OnnxTextEmbedder):
            def __init__(self, model_id):
                self.model_id = model_id  # skip the real session

        monkeypatch.setattr(mod, "OnnxTextEmbedder", _FakeOnnx)
        embedder = resolve_embedder("all-MiniLM-L6-v2")
        assert isinstance(embedder, _FakeOnnx)

    def test_api_base_beats_the_local_backend_flag(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "EMBEDDING_API_BASE", self.BASE, raising=False)
        monkeypatch.setattr(
            settings, "EMBEDDING_BACKEND", "sentence-transformers", raising=False
        )
        embedder = resolve_embedder("model-x")
        assert isinstance(embedder, ApiTextEmbedder)
