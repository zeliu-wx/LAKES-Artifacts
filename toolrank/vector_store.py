"""Vector store using OpenAI-compatible embedding API."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener, getproxies, urlopen

from toolrank.openai_compat import DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL


WHATAI_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
WHATAI_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_WHATAI_COMPATIBLE_API_KEY = ""
DEFAULT_SILICONFLOW_COMPATIBLE_API_KEY = DEFAULT_WHATAI_COMPATIBLE_API_KEY
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_INDEX_SCHEMA_VERSION = "toolrank_vector_index_v1"


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    base_url: str
    api_key: str
    model: str


def _looks_like_whatai(base_url: str | None, model: str | None) -> bool:
    normalized_base_url = (base_url or "").lower()
    normalized_model = (model or "").lower()
    return bool(
        (base_url and ("whatai" in normalized_base_url or "siliconflow" in normalized_base_url))
        or (model and (normalized_model.startswith("qwen/") or normalized_model.startswith("qwen")))
    )


def _whatai_compatible_api_key() -> str:
    return (
        os.getenv("WHATAI_API_KEY")
        or os.getenv("WHATAI_COMPATIBLE_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or DEFAULT_WHATAI_COMPATIBLE_API_KEY
        or DEFAULT_SILICONFLOW_COMPATIBLE_API_KEY
        or ""
    )


def resolve_embedding_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> EmbeddingConfig:
    """Resolve embedding provider settings from explicit args and environment."""
    whatai_key = _whatai_compatible_api_key()
    explicit_openai_compat = (
        (base_url is not None or model is not None)
        and not _looks_like_whatai(base_url, model)
    )
    use_whatai = not explicit_openai_compat

    if use_whatai:
        return EmbeddingConfig(
            provider="whatai",
            base_url=(
                base_url
                or os.getenv("WHATAI_BASE_URL")
                or os.getenv("SILICONFLOW_BASE_URL")
                or WHATAI_DEFAULT_BASE_URL
            ).rstrip("/"),
            api_key=api_key or whatai_key,
            model=(
                model
                or os.getenv("WHATAI_EMBEDDING_MODEL")
                or os.getenv("SILICONFLOW_EMBEDDING_MODEL")
                or WHATAI_DEFAULT_EMBEDDING_MODEL
            ),
        )

    return EmbeddingConfig(
        provider="openai_compat",
        base_url=(base_url or os.getenv("TOOLRANK_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/"),
        api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY,
        model=model or os.getenv("TOOLRANK_OPENAI_EMBEDDING_MODEL", OPENAI_DEFAULT_EMBEDDING_MODEL),
    )


def _extract_response_embeddings(payload: dict[str, Any]) -> list[list[float]]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise RuntimeError("Embedding response did not contain a data list")

    indexed_embeddings: list[tuple[int, list[float]]] = []
    for fallback_index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise RuntimeError("Embedding response data item was not an object")
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Embedding response data item did not contain an embedding list")
        index = item.get("index", fallback_index)
        indexed_embeddings.append((int(index), [float(value) for value in embedding]))

    indexed_embeddings.sort(key=lambda item: item[0])
    return [embedding for _, embedding in indexed_embeddings]


def _run_curl(cmd: list[str]) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "curl embedding request failed")
    return result.stdout


def _system_https_proxy() -> str:
    proxy = getproxies().get("https") or getproxies().get("http") or ""
    return proxy if isinstance(proxy, str) else ""


def _should_bypass_proxy(config: EmbeddingConfig, *, no_proxy: bool) -> bool:
    return no_proxy or config.provider == "whatai"


def get_embeddings(
    texts: list[str],
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 30,
    no_proxy: bool = False,
) -> list[list[float]]:
    """Get embeddings from an OpenAI-compatible endpoint."""
    if not texts:
        return []
    config = resolve_embedding_config(base_url=base_url, api_key=api_key, model=model)
    if not config.api_key:
        if config.provider == "whatai":
            raise RuntimeError(
                "Missing Whatai embedding key. "
                "Set WHATAI_API_KEY, WHATAI_COMPATIBLE_API_KEY, SILICONFLOW_API_KEY, "
                "QWEN_API_KEY, or DASHSCOPE_API_KEY."
            )
        raise RuntimeError(f"Missing API key for {config.provider} embeddings")

    url = f"{config.base_url}/embeddings"
    payload_dict: dict[str, Any] = {"input": texts, "model": config.model}
    if config.provider == "whatai":
        payload_dict["encoding_format"] = "float"
        if config.model.lower().startswith("qwen/qwen3-embedding"):
            payload_dict["dimensions"] = 1024
    payload = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    bypass_proxy = _should_bypass_proxy(config, no_proxy=no_proxy)

    def _curl_fallback() -> dict[str, Any]:
        cmd = ["curl", "-sS",
               "--connect-timeout", str(int(timeout)),
               "--max-time", str(int(timeout)),
               "-X", "POST", url,
               "-H", "Content-Type: application/json",
               "-H", f"Authorization: Bearer {config.api_key}",
               "--data-binary", payload.decode("utf-8")]
        proxy = _system_https_proxy()
        if bypass_proxy:
            cmd.extend(["--noproxy", "*"])
        elif proxy:
            cmd.extend(["--proxy", proxy])
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return json.loads(_run_curl(cmd))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"curl embedding failed after 3 attempts: {last_error}")

    req = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
    )
    try:
        if bypass_proxy:
            opener = build_opener(ProxyHandler({}))
            response = opener.open(req, timeout=timeout)
        else:
            response = urlopen(req, timeout=timeout)
        with response as resp:
            data = json.loads(resp.read())
    except Exception:
        data = _curl_fallback()
    return _extract_response_embeddings(data)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _validate_embedding_dimensions(query: list[float], stored: list[float]) -> None:
    if len(query) != len(stored):
        raise ValueError(
            f"Embedding dimension mismatch: query_dim={len(query)} index_dim={len(stored)}"
        )


class VectorIndex:
    """In-memory vector index with cosine similarity search."""

    def __init__(self, embeddings: list[list[float]] | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._embeddings = embeddings or []
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self._embeddings)

    def build(self, texts: list[str], batch_size: int | None = None, **kwargs: Any) -> None:
        if not texts:
            self._embeddings = []
            return
        if batch_size is None or batch_size <= 0:
            self._embeddings = get_embeddings(texts, **kwargs)
            return
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(get_embeddings(texts[start : start + batch_size], **kwargs))
        self._embeddings = embeddings

    def query(self, query_text: str, top_k: int = 5, **kwargs: Any) -> list[tuple[int, float]]:
        return self.query_candidates(query_text, range(len(self._embeddings)), top_k=top_k, **kwargs)

    def query_candidates(
        self,
        query_text: str,
        candidate_indexes: set[int] | list[int] | range,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        if not self._embeddings:
            return []
        candidates = sorted({index for index in candidate_indexes if 0 <= index < len(self._embeddings)})
        if not candidates:
            return []
        query_emb = get_embeddings([query_text], **kwargs)[0]
        scores = []
        for index in candidates:
            stored = self._embeddings[index]
            _validate_embedding_dimensions(query_emb, stored)
            scores.append((index, cosine_similarity(query_emb, stored)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def batch_query_candidates(
        self,
        queries: list[tuple[str, set[int] | list[int] | range, int]],
        **kwargs: Any,
    ) -> list[list[tuple[int, float]]]:
        if not self._embeddings or not queries:
            return [[] for _ in queries]
        texts = [q[0] for q in queries]
        embedding_chunk = 16
        all_embs: list[list[float]] = []
        for start in range(0, len(texts), embedding_chunk):
            chunk = texts[start : start + embedding_chunk]
            all_embs.extend(get_embeddings(chunk, **kwargs))
        results: list[list[tuple[int, float]]] = []
        for (_, candidate_indexes, top_k), query_emb in zip(queries, all_embs):
            candidates = sorted({i for i in candidate_indexes if 0 <= i < len(self._embeddings)})
            if not candidates:
                results.append([])
                continue
            scores = []
            for index in candidates:
                stored = self._embeddings[index]
                _validate_embedding_dimensions(query_emb, stored)
                scores.append((index, cosine_similarity(query_emb, stored)))
            scores.sort(key=lambda item: item[1], reverse=True)
            results.append(scores[:top_k])
        return results

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        payload_metadata = dict(metadata or self.metadata)
        if self._embeddings and "embedding_dim" not in payload_metadata:
            payload_metadata["embedding_dim"] = len(self._embeddings[0])
        payload = {
            "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
            "metadata": payload_metadata,
            "embeddings": self._embeddings,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != VECTOR_INDEX_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported vector index schema: {payload.get('schema_version')}")
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Vector index did not contain embeddings")
        metadata = payload.get("metadata")
        return cls(embeddings=embeddings, metadata=metadata if isinstance(metadata, dict) else {})
