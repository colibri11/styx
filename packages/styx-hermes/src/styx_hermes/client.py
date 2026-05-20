"""HTTP клиент для styx-core daemon.

Тонкий synchronous wrapper над ``requests.Session``. Все вызовы — sync,
потому что Hermes invoke'ит plugin methods синхронно.

Конфигурация:
- ``base_url`` — обычно ``STYX_DAEMON_URL`` (default ``http://127.0.0.1:8788``)
- ``token`` — ``STYX_HTTP_TOKEN`` (если задан, daemon на non-loopback'е).

Контракт endpoint'ов — ``.design/host-agnostic-split-v1.md`` § 6.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5.0
"""Sync HTTP timeout для всех вызовов кроме длинных (recall, context).

Hermes call-path синхронный — больший timeout не критичен. recall
может занимать до 1-3 сек на p99 (embed + search), сделаем отдельный
больший timeout для recall/context.
"""

LONG_TIMEOUT_S = 30.0


class StyxCoreClient:
    """Sync HTTP клиент к styx-core daemon."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        long_timeout_s: float = LONG_TIMEOUT_S,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("STYX_DAEMON_URL", "http://127.0.0.1:8788")
        ).rstrip("/")
        self._token = token if token is not None else os.environ.get("STYX_HTTP_TOKEN")
        self._timeout = timeout_s
        self._long_timeout = long_timeout_s
        self._session = requests.Session()
        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass

    # ── healthcheck ────────────────────────────────────────────────────

    def healthz(self) -> dict[str, Any]:
        return self._get("/healthz", auth=False)

    def readyz(self) -> dict[str, Any]:
        return self._get("/readyz", auth=False)

    # ── agent lifecycle ────────────────────────────────────────────────

    def initialize_agent(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        agent_identity: str | None = None,
        platform: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/agent/initialize",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "agent_identity": agent_identity or agent_id,
                "platform": platform,
                "model": model,
            },
        )

    def shutdown_agent(self, agent_id: str) -> None:
        self._post("/agent/shutdown", {"agent_id": agent_id})

    # ── per-turn ───────────────────────────────────────────────────────

    def sync_turn(
        self,
        agent_id: str,
        *,
        user_content: str = "",
        assistant_content: str = "",
        session_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/sync_turn",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "tool_calls": tool_calls,
            },
        )

    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/recall",
            {
                "agent_id": agent_id,
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "session_id": session_id,
            },
            timeout=self._long_timeout,
            wrap_for_llm=True,
        )

    def build_context(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        *,
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/context/build",
            {
                "agent_id": agent_id,
                "messages": messages,
                "current_tokens": current_tokens,
                "focus_topic": focus_topic,
            },
            timeout=self._long_timeout,
        )

    def assemble_context(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        """`POST /context/assemble` — runtime channel split (волна 26.7).

        Возвращает eviction-normalized messages БЕЗ inject'а salient
        в array, плюс ``system_prompt_addition`` — pre-formatted
        ``<styx-salient>...</styx-salient>`` строка (или None).

        Используется ``StyxMemoryProvider.prefetch()`` (волна 29 Phase B)
        как основной recall-канал в Hermes path: prefetch строит
        minimal messages = [{user: query}], вызывает assemble,
        возвращает ``system_prompt_addition`` как text для inject в
        Hermes input.
        """
        return self._post(
            "/context/assemble",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "messages": messages,
                "token_budget": token_budget,
            },
            timeout=self._long_timeout,
        )

    def pre_llm_inject(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        user_message: str | None = None,
        is_first_turn: bool = False,
        model: str | None = None,
        platform: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/pre_llm_inject",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "user_message": user_message,
                "is_first_turn": is_first_turn,
                "model": model,
                "platform": platform,
                "extra": extra or {},
            },
        )

    def get_agent_state(self, agent_id: str) -> dict[str, Any]:
        return self._get(f"/agent_state?agent_id={agent_id}")

    def push_cache_stats(
        self,
        agent_id: str,
        *,
        cache_read_tokens: int,
        cache_creation_tokens: int,
    ) -> None:
        """`POST /agent/cache_stats` — observability push (волна 29 Phase E).

        Fire-and-forget: вызывается из ``StyxAnthropicTransport.extract_cache_stats``
        после каждого LLM call'а с Anthropic backend. Tokens — кумулятив
        per-agent в core daemon, доступен через ``GET /analytics``.

        Возвращает None (204 No Content). Любой fail прокидывает
        exception — caller (transport) сам fail-open'ит.
        """
        self._post(
            "/agent/cache_stats",
            {
                "agent_id": agent_id,
                "cache_read_tokens": int(cache_read_tokens),
                "cache_creation_tokens": int(cache_creation_tokens),
            },
        )

    # ── search archive (волна 20) ─────────────────────────────────────

    def search_archive(
        self,
        agent_id: str,
        query: str,
        *,
        scope: str = "all",
        limit: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        snapshot_cycle_start: str | None = None,
    ) -> dict[str, Any]:
        """POST /search_archive — pull-канал к архиву.

        ``scope`` ∈ {'documents', 'chunks', 'dialogue', 'all'}.
        ``snapshot_cycle_start`` принимаем как ISO-8601 string —
        FastAPI route декодирует в datetime.
        """
        return self._post(
            "/search_archive",
            {
                "agent_id": agent_id,
                "query": query,
                "scope": scope,
                "limit": limit,
                "date_from": date_from,
                "date_to": date_to,
                "snapshot_cycle_start": snapshot_cycle_start,
            },
            timeout=self._long_timeout,
            wrap_for_llm=True,
        )

    # ── dialogue tools (волна 24) ──────────────────────────────────────

    def dialogue_search(
        self,
        agent_id: str,
        query: str,
        *,
        session_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        semantic_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """POST /dialogue/search — hybrid либо pure-vector search.

        ``semantic_only=True`` отключает BM25, делает pure-cosine.
        ``after``/``before`` — ISO-8601 strings (FastAPI декодирует в
        datetime).
        """
        return self._post(
            "/dialogue/search",
            {
                "agent_id": agent_id,
                "query": query,
                "session_id": session_id,
                "after": after,
                "before": before,
                "semantic_only": semantic_only,
                "limit": limit if limit is not None else 10,
            },
            timeout=self._long_timeout,
            wrap_for_llm=True,
        )

    def dialogue_recent(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        before: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """POST /dialogue/recent — chronological retrieval (oldest first)."""
        return self._post(
            "/dialogue/recent",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "before": before,
                "limit": limit if limit is not None else 20,
            },
            wrap_for_llm=True,
        )

    def dialogue_prepare_summary(
        self,
        agent_id: str,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """POST /dialogue/prepare_summary — transcript для summarizer'а."""
        return self._post(
            "/dialogue/prepare_summary",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "limit": limit if limit is not None else 200,
            },
            timeout=self._long_timeout,
            wrap_for_llm=True,
        )

    # ── file-ingest (волна 28) ─────────────────────────────────────────

    def ingest_document(
        self,
        agent_id: str,
        path: str,
        *,
        source_ref: str | None = None,
        visibility: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_hash: str | None = None,
    ) -> dict[str, Any]:
        """POST /ingest_document — file-ingest pipeline (волна 28 +
        Defect-fix A).

        Daemon читает файл по absolute path, парсит, режет на chunks,
        embed'ит, INSERT'ит document + chunks. В memories пишется
        tail-memory с маркером акта архивации (Defect-fix A; IAmBook
        §V). Идемпотентен по SHA256 file bytes.
        """
        return self._post(
            "/ingest_document",
            {
                "agent_id": agent_id,
                "path": path,
                "source_ref": source_ref,
                "visibility": visibility,
                "metadata": metadata or {},
                "content_hash": content_hash,
            },
            timeout=self._long_timeout,
        )

    # ── reinterpret (волна 22) ─────────────────────────────────────────

    def reinterpret(
        self,
        agent_id: str,
        memory_id: str,
        new_understanding_text: str,
        *,
        weight: float | None = None,
    ) -> dict[str, Any]:
        """POST /reinterpret — explicit reinterpret memory (enqueue-only).

        404/409 не raise'ятся: route возвращает structured detail,
        client'ом разворачиваем в `{status, ...}` body как для 202.
        Это симметрия с in-process tool path'ом — caller LLM получает
        одинаковую structured shape независимо от status code.
        """
        url = f"{self._base_url}/reinterpret"
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "memory_id": memory_id,
            "new_understanding_text": new_understanding_text,
        }
        if weight is not None:
            payload["weight"] = float(weight)
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            log.warning("styx-core POST /reinterpret failed: %s", exc)
            raise

        # 200 — happy path; 404/409 — structured statuses (memory_not_found
        # / cooldown / already_pending). Иначе — обычная raise_for_status.
        if resp.status_code in (404, 409):
            try:
                body = resp.json()
            except ValueError:
                body = {}
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, dict):
                return detail
            return {"status": "unknown_error", "detail": body}
        return _parse_response("/reinterpret", resp)

    # ── internals ──────────────────────────────────────────────────────

    def _get(self, path: str, *, auth: bool = True) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.get(url, timeout=self._timeout)
        except requests.RequestException as exc:
            log.warning("styx-core GET %s failed: %s", path, exc)
            raise

        return _parse_response(path, resp)

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        wrap_for_llm: bool = False,
    ) -> dict[str, Any]:
        """POST к styx-core.

        ``wrap_for_llm`` (волна 30, Phase D) → header
        ``X-Wrap-For-LLM: 1``. Используется LLM-facing методами
        (``recall``, ``search_archive``, ``dialogue_*``) — core
        возвращает дополнительное поле ``llm_text`` с pre-rendered
        обёрткой ``<styx-{channel}>...</styx-{channel}>``. Caller'ы
        в ``providers/memory.py`` затем подставляют ``llm_text`` как
        tool result content вместо собственного render'а.
        """
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if wrap_for_llm:
            headers["X-Wrap-For-LLM"] = "1"
        try:
            resp = self._session.post(
                url,
                json=payload,
                timeout=timeout or self._timeout,
                headers=headers or None,
            )
        except requests.RequestException as exc:
            log.warning("styx-core POST %s failed: %s", path, exc)
            raise

        return _parse_response(path, resp)


def _parse_response(path: str, resp: "requests.Response") -> dict[str, Any]:
    """Парсит JSON-ответ; raises HTTPError при non-2xx."""
    if resp.status_code == 204:
        return {}
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not (200 <= resp.status_code < 300):
        log.warning(
            "styx-core %s returned %d: %s",
            path,
            resp.status_code,
            body if body else resp.text[:200],
        )
        resp.raise_for_status()
    return body if isinstance(body, dict) else {}
