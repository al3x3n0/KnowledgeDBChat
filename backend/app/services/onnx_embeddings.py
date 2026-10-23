"""The embedding and reranking models, run by ONNX Runtime instead of torch.

Four call sites in this codebase used sentence-transformers, and behind them
sat 578 MB: torch, transformers, sympy, networkx, scipy and scikit-learn, none
of which anything else here needs. ONNX Runtime is 53 MB and runs the same
weights: every model this project configures publishes an `onnx/model.onnx` in
its own Hugging Face repo, so nothing is re-exported, re-trained or
substituted -- the file is downloaded from the same repository
sentence-transformers would have downloaded, into the same cache.

Measured before the switch, on a 30-passage corpus and 8 queries: per-vector
cosine agreement with the torch pipeline of **1.000000** (minimum, not mean),
and identical top-1 and top-5 rankings for every query. An index built by the
old pipeline stays valid.

It also runs a model torch could not. On aarch64 this image's torch fails the
cross-encoder's forward pass with "could not create a primitive descriptor for
a matmul primitive"; `vector_store._load_models` already had a branch matching
that message by hand and switching reranking off. Under ONNX Runtime the same
cross-encoder loads and scores.

The classes below deliberately mirror the sentence-transformers API the rest of
the codebase already calls -- `encode`, `get_sentence_embedding_dimension`,
`predict` -- so the call sites did not have to learn a new one.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

#: Bare names mean the sentence-transformers org, the same resolution
#: `SentenceTransformer("all-MiniLM-L6-v2")` performs.
_DEFAULT_ORG = "sentence-transformers"

#: Where an ONNX export lives inside a repo. `onnx/model.onnx` is the
#: convention optimum exports to and every model configured here uses; the bare
#: name is what older Qdrant-published exports use.
_MODEL_FILE_CANDIDATES = ("onnx/model.onnx", "model.onnx")

#: A transformer's own limit is in its config, but nothing here benefits from
#: feeding a 512-token model a longer sequence than it was trained on.
_MAX_SEQUENCE_LENGTH = 512

#: Bounded on purpose. ONNX Runtime otherwise sizes its pool from the host's
#: core count, and these run inside forked Celery workers that already cap
#: OMP_NUM_THREADS -- which ORT does not read.
_INTRA_OP_THREADS = min(4, os.cpu_count() or 1)


def resolve_repo_id(model_id: str) -> str:
    """`all-MiniLM-L6-v2` -> `sentence-transformers/all-MiniLM-L6-v2`."""
    return model_id if "/" in model_id else f"{_DEFAULT_ORG}/{model_id}"


def _download(repo_id: str, filename: str) -> Optional[str]:
    """The file's local path, or None if the repo genuinely does not have it.

    Only absence returns None. A refused connection, a TLS reset or a proxy in
    the way propagates: swallowing those turns a network problem into "this
    model publishes no ONNX export", which sends whoever reads it looking for
    a different model instead of at their network.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.debug(f"{repo_id} does not publish {filename}")
        return None


class _OnnxModel:
    """A tokenizer and an ONNX session for one Hugging Face model id."""

    def __init__(self, model_id: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_id = model_id
        repo_id = resolve_repo_id(model_id)

        model_path = None
        for candidate in _MODEL_FILE_CANDIDATES:
            model_path = _download(repo_id, candidate)
            if model_path:
                break
        if not model_path:
            raise RuntimeError(
                f"{repo_id} publishes no ONNX export "
                f"({' or '.join(_MODEL_FILE_CANDIDATES)}). Either pick a model "
                f"that does, or set EMBEDDING_BACKEND=sentence-transformers and "
                f"install sentence-transformers."
            )

        tokenizer_path = _download(repo_id, "tokenizer.json")
        if not tokenizer_path:
            raise RuntimeError(f"{repo_id} has no tokenizer.json")

        options = ort.SessionOptions()
        options.intra_op_num_threads = _INTRA_OP_THREADS
        self._session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=_MAX_SEQUENCE_LENGTH)
        self._tokenizer.enable_padding()

        # One session is not thread-safe to configure, and encode() is called
        # from asyncio.to_thread all over this codebase.
        self._lock = threading.Lock()

    def _run(self, encodings: Sequence[Any]) -> np.ndarray:
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # XLM-R based models take no token_type_ids; feeding an input the graph
        # does not declare is an error rather than something ORT ignores.
        feed = {k: v for k, v in feed.items() if k in self._input_names}
        with self._lock:
            outputs = self._session.run(None, feed)
        return outputs[0], feed["attention_mask"]


class OnnxTextEmbedder(_OnnxModel):
    """Mean-pooled, L2-normalised sentence embeddings.

    Mean pooling is what every model in EMBEDDING_MODEL_OPTIONS declares in its
    sentence-transformers config. Normalisation is unconditional: Qdrant
    collections here are created with `Distance.COSINE`, under which the norm
    cannot change a ranking, and a unit vector is what the old pipeline stored
    for the default model anyway.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._dimension: Optional[int] = None

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # noqa: ARG002 - API compatibility
        convert_to_numpy: bool = True,  # noqa: ARG002 - API compatibility
        **_: Any,
    ) -> np.ndarray:
        """A single string returns one vector; a sequence returns a matrix.

        The shape contract is sentence-transformers': callers here do
        `.encode(query).tolist()` and expect a flat list of floats.
        """
        single = isinstance(sentences, str)
        texts = [sentences] if single else [str(s) for s in sentences]
        if not texts:
            return np.zeros(
                (0, self.get_sentence_embedding_dimension()), dtype=np.float32
            )

        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            hidden, mask = self._run(self._tokenizer.encode_batch(chunk))
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(
                weights.sum(axis=1), 1e-9, None
            )
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            vectors.append(pooled / np.clip(norms, 1e-12, None))

        stacked = np.concatenate(vectors, axis=0).astype(np.float32)
        self._dimension = int(stacked.shape[1])
        return stacked[0] if single else stacked

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            # Cheapest way to learn the width is to use the thing once.
            self.encode("dimension probe")
        return int(self._dimension or 0)


class OnnxCrossEncoder(_OnnxModel):
    """Relevance scores for (query, passage) pairs, the reranker's contract."""

    def predict(
        self,
        pairs: Sequence[Sequence[str]],
        batch_size: int = 32,
        **_: Any,
    ) -> np.ndarray:
        if not pairs:
            return np.zeros((0,), dtype=np.float32)

        scores: List[np.ndarray] = []
        for start in range(0, len(pairs), max(1, batch_size)):
            chunk: List[Tuple[str, str]] = [
                (str(pair[0]), str(pair[1]))
                for pair in pairs[start : start + max(1, batch_size)]
            ]
            logits, _ = self._run(self._tokenizer.encode_batch(chunk))
            # A single-logit head scores directly; a two-logit head scores by
            # its positive class.
            logits = np.asarray(logits, dtype=np.float32)
            scores.append(logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1])
        return np.concatenate(scores, axis=0)


def available() -> bool:
    """Whether this backend can be used at all in this image."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        return True
    except ImportError:
        return False


def _sentence_transformers_missing(kind: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"EMBEDDING_BACKEND=sentence-transformers but sentence-transformers is "
        f"not installed, so the {kind} cannot be loaded ({exc}). It was removed "
        f"from requirements.txt with torch behind it; `pip install "
        f"sentence-transformers` restores it, or use the default onnx backend."
    )


class ApiTextEmbedder:
    """Sentence embeddings from an external OpenAI-compatible ``/embeddings`` API.

    Mirrors the local embedder contract the rest of the codebase calls --
    ``encode`` and ``get_sentence_embedding_dimension`` -- so a call site does
    not have to know whether the vectors come from a local ONNX model or a
    remote service. ``encode("query")`` returns a single vector; a sequence
    returns a matrix, the same as :class:`OnnxTextEmbedder`.

    The endpoint is the base URL ``EMBEDDING_API_BASE`` with ``/embeddings``
    appended. ``EMBEDDING_MODEL`` names the model the API must serve and is
    sent as the ``model`` field. When ``EMBEDDING_API_KEY`` is set it is sent
    as a Bearer token.
    """

    def __init__(
        self,
        model_id: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        from app.core.config import settings

        self.model_id = model_id
        self._base_url = (base_url or settings.EMBEDDING_API_BASE or "").rstrip("/")
        self._api_key = api_key if api_key is not None else settings.EMBEDDING_API_KEY
        self._timeout_seconds = float(
            timeout_seconds or settings.EMBEDDING_API_TIMEOUT_SECONDS
        )
        self._url = f"{self._base_url}/embeddings"
        self._dimension: Optional[int] = None
        if not self._base_url:
            raise RuntimeError(
                "EMBEDDING_API_BASE is empty; the external embedding API cannot "
                "be used. Set it to the base URL up to the version segment, e.g. "
                "http://<embedding-service>:10999/api/v1."
            )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, payload: Dict[str, Any]) -> Any:
        import httpx

        logger.debug(f"Requesting embeddings from {self._url} (model={self.model_id})")
        try:
            response = httpx.post(
                self._url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"External embedding API {self._url} is unreachable: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise RuntimeError(
                f"External embedding API {self._url} returned "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"External embedding API {self._url} returned a non-JSON body"
            ) from exc

    def _vectors(self, data: Any, expected: int) -> List[List[float]]:
        vectors: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError(
                    f"External embedding API {self._url} returned an item without "
                    f"an 'embedding' list: {item!r}"
                )
            vectors.append([float(v) for v in embedding])
        if len(vectors) != expected:
            raise RuntimeError(
                f"External embedding API {self._url} returned {len(vectors)} "
                f"vectors for {expected} inputs"
            )
        return vectors

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # noqa: ARG002 - API compatibility
        convert_to_numpy: bool = True,  # noqa: ARG002 - API compatibility
        **_: Any,
    ) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else [str(s) for s in sentences]
        if not texts:
            # No round-trip for an empty batch: callers handle the empty case,
            # and probing the API just to learn the dimension would spend a
            # request on nothing.
            return np.zeros((0, self._dimension or 0), dtype=np.float32)

        vectors = self._post({"model": self.model_id, "input": texts})
        data = vectors.get("data") if isinstance(vectors, dict) else None
        if not isinstance(data, list):
            raise RuntimeError(
                f"External embedding API {self._url} returned no 'data' list: "
                f"{vectors!r}"
            )
        rows = self._vectors(data, len(texts))
        stacked = np.array(rows, dtype=np.float32)
        self._dimension = int(stacked.shape[1])
        return stacked[0] if single else stacked

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            self.encode("dimension probe")
        return int(self._dimension or 0)


def resolve_embedder(model_id: str) -> Any:
    """The embedding model for this deployment.

    Returns an external :class:`ApiTextEmbedder` when ``EMBEDDING_API_BASE``
    is configured (embeddings come from the remote API), otherwise the local
    model on whichever backend ``EMBEDDING_BACKEND`` names.
    """
    from app.core.config import settings

    if (getattr(settings, "EMBEDDING_API_BASE", None) or "").strip():
        logger.info(
            f"Using external embedding API {settings.EMBEDDING_API_BASE} "
            f"(model={model_id})"
        )
        return ApiTextEmbedder(model_id)

    if getattr(settings, "EMBEDDING_BACKEND", "onnx") == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise _sentence_transformers_missing("embedding model", exc) from exc
        return SentenceTransformer(model_id)

    logger.info(f"Loading embedding model {model_id} under ONNX Runtime")
    return OnnxTextEmbedder(model_id)


def load_text_embedder(model_id: str) -> Any:
    """The embedding model, on whichever backend this deployment asked for."""
    return resolve_embedder(model_id)


def load_cross_encoder(model_id: str) -> Any:
    """The reranking model, on whichever backend this deployment asked for."""
    from app.core.config import settings

    if getattr(settings, "EMBEDDING_BACKEND", "onnx") == "sentence-transformers":
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise _sentence_transformers_missing("reranking model", exc) from exc
        return CrossEncoder(model_id)

    logger.info(f"Loading reranking model {model_id} under ONNX Runtime")
    return OnnxCrossEncoder(model_id)
