from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

from config import MY_FILES_EMBEDDING_MODEL


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, *, status: str = "ERROR") -> None:
        super().__init__(message)
        self.status = status


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalFastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str = MY_FILES_EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._dimension = 0
        self._status = "NOT_LOADED"
        self.last_metrics: dict[str, int | float | str] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def status(self) -> str:
        return self._status

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            for line in open("/proc/self/status", encoding="utf-8"):
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
        except (OSError, ValueError):
            return None
        return None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        started_at = time.perf_counter()
        rss_before = self._rss_mb()
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        except (ImportError, MemoryError, OSError) as error:
            self._status = "RESOURCE_LIMIT"
            raise EmbeddingError(
                "The local embedding runtime is unavailable.",
                status="RESOURCE_LIMIT",
            ) from error
        except Exception as error:
            self._status = "ERROR"
            raise EmbeddingError(
                "The configured local embedding model could not be loaded.",
                status="ERROR",
            ) from error
        self._status = "READY"
        self.last_metrics = {
            "model_load_ms": int((time.perf_counter() - started_at) * 1000),
            "rss_before_mb": rss_before or 0,
            "rss_after_mb": self._rss_mb() or 0,
        }
        return self._model

    @staticmethod
    def _validate_vectors(vectors: list[list[float]]) -> int:
        if not vectors:
            return 0
        dimension = len(vectors[0])
        if dimension == 0:
            raise EmbeddingError("The local embedding model returned empty vectors.")
        for vector in vectors:
            if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("The local embedding model returned invalid vectors.")
        return dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = time.perf_counter()
        model = self._ensure_model()
        vectors = [list(map(float, vector)) for vector in model.embed(texts)]
        self._dimension = self._validate_vectors(vectors)
        self.last_metrics["document_embed_ms"] = int((time.perf_counter() - started_at) * 1000)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        model = self._ensure_model()
        vector = list(map(float, next(model.query_embed(text))))
        self._dimension = self._validate_vectors([vector])
        self.last_metrics["query_embed_ms"] = int((time.perf_counter() - started_at) * 1000)
        return vector


_provider: LocalFastEmbedProvider | None = None


def get_embedding_provider() -> LocalFastEmbedProvider:
    global _provider
    if _provider is None:
        _provider = LocalFastEmbedProvider()
    return _provider


def embedding_health() -> dict[str, Any]:
    provider = get_embedding_provider()
    return {
        "status": provider.status,
        "model": provider.model_name,
        "dimension": provider.dimension or None,
        "last_metrics": dict(provider.last_metrics),
        "safe_error_code": None,
    }


def vector_literal(vector: list[float]) -> str:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("The vector is invalid.")
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"