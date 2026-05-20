from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen


DEFAULT_OPENAI_BASE_URL = os.getenv("TOOLRANK_OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")
DEFAULT_OPENAI_MODEL = os.getenv("TOOLRANK_OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_CONNECT_TIMEOUT_SEC = float(os.getenv("TOOLRANK_OPENAI_CONNECT_TIMEOUT_SEC", "0.25"))
DEFAULT_REQUEST_TIMEOUT_SEC = float(os.getenv("TOOLRANK_OPENAI_REQUEST_TIMEOUT_SEC", "360"))
DEFAULT_RETRY_COUNT = int(os.getenv("TOOLRANK_OPENAI_RETRY_COUNT", "2"))
DEFAULT_RETRY_BACKOFF_SEC = float(os.getenv("TOOLRANK_OPENAI_RETRY_BACKOFF_SEC", "1.0"))

_NO_PROXY_OPENER = build_opener(ProxyHandler({}))


@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    timeout_sec: float


class OpenAICompatError(RuntimeError):
    """Raised when the OpenAI-compatible endpoint is unavailable or returns unusable data."""


def _is_retryable_status(code: int) -> bool:
    return code in {429, 500, 502, 503, 504}


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return None
    return None


def _coerce_content(raw_content: Any) -> Optional[str]:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None


def _open_request(request: Request, timeout_sec: float) -> str:
    parsed = urlparse(request.full_url)
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        response = _NO_PROXY_OPENER.open(request, timeout=timeout_sec)
    else:
        response = urlopen(request, timeout=timeout_sec)
    with response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_stream_content(raw: str) -> Optional[str]:
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        try:
            delta = chunk["choices"][0]["delta"]
        except Exception:
            continue
        text = _coerce_content(delta.get("content"))
        if isinstance(text, str):
            parts.append(text)
            continue
        reasoning_text = _coerce_content(delta.get("reasoning_content"))
        if isinstance(reasoning_text, str):
            reasoning_parts.append(reasoning_text)
    joined = "".join(parts).strip()
    if joined:
        return joined
    joined_reasoning = "".join(reasoning_parts).strip()
    return joined_reasoning or None


def _build_request(
    *,
    client: OpenAICompatClient,
    payload: Dict[str, Any],
) -> Request:
    return Request(
        url=f"{client.base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client.api_key}",
        },
        method="POST",
    )


def _is_local_server_reachable(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return True
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=DEFAULT_CONNECT_TIMEOUT_SEC):
            return True
    except OSError:
        return False


def load_openai_client() -> Optional[OpenAICompatClient]:
    base_url = DEFAULT_OPENAI_BASE_URL.rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY
    if not api_key:
        return None
    if not _is_local_server_reachable(base_url) and not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAICompatClient(base_url=base_url, api_key=api_key, timeout_sec=DEFAULT_REQUEST_TIMEOUT_SEC)


def create_json_chat_completion(
    *,
    client: OpenAICompatClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False,
    timeout_sec: float | None = None,
) -> Optional[dict]:
    schema_suffix = ""
    if schema is not None:
        schema_suffix = (
            "\n\nReturn JSON only. It must satisfy the following schema exactly:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + schema_suffix},
        ],
        "temperature": 0,
    }

    def request_payload(
        request_payload: Dict[str, Any],
        *,
        raise_errors: bool,
        retry_count: int,
    ) -> Optional[str]:
        raw = None
        last_error: Optional[Exception] = None
        for attempt in range(retry_count + 1):
            try:
                effective_timeout = timeout_sec or client.timeout_sec
                raw = _open_request(_build_request(client=client, payload=request_payload), effective_timeout)
                break
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                last_error = OpenAICompatError(
                    f"LLM request failed with HTTP {exc.code}: {(body or str(exc.reason))[:300]}"
                )
                if attempt < retry_count and _is_retryable_status(exc.code):
                    time.sleep(DEFAULT_RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                if raise_errors:
                    raise last_error from exc
                return None
            except URLError as exc:
                last_error = OpenAICompatError(f"LLM request failed: {exc}")
                if attempt < retry_count:
                    time.sleep(DEFAULT_RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                if raise_errors:
                    raise last_error from exc
                return None
            except (TimeoutError, socket.timeout, OSError) as exc:
                last_error = OpenAICompatError(f"LLM request timed out or failed at transport layer: {exc}")
                if attempt < retry_count:
                    time.sleep(DEFAULT_RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                if raise_errors:
                    raise last_error from exc
                return None
        if raw is None and raise_errors:
            raise last_error or OpenAICompatError("LLM request failed without a response.")
        return raw

    stream_payload = {**payload, "stream": True}
    raw = request_payload(stream_payload, raise_errors=raise_on_error, retry_count=DEFAULT_RETRY_COUNT)
    if raw is None:
        return None
    content = _extract_stream_content(raw)
    if not isinstance(content, str):
        if raise_on_error:
            raise OpenAICompatError(f"LLM endpoint returned no parseable stream content: {raw[:300]}")
        return None
    parsed = _parse_json(content)
    if parsed is None and raise_on_error:
        raise OpenAICompatError(f"LLM returned non-parseable content: {content[:300]}")
    return parsed
