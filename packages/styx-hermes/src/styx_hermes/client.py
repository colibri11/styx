"""HTTP клиент для styx-core daemon.

Тонкий synchronous wrapper над ``requests.Session``. Все вызовы — sync,
потому что Hermes invoke'ит plugin methods синхронно.

Конфигурация:
- ``base_url`` — обычно ``STYX_DAEMON_URL`` (default ``http://127.0.0.1:8788``)
- ``token`` — ``STYX_HTTP_TOKEN`` (если задан, daemon на non-loopback'е).
- ``social_token`` — отдельный ``STYX_SOCIAL_TOKEN`` для явных social routes.

Контракт endpoint'ов — ``.design/host-agnostic-split-v1.md`` § 6.
"""

from __future__ import annotations

import hashlib
import hmac
import json
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

# Core's causal observer may spend up to eight seconds in the local model and
# still needs time to acquire the per-agent lock and commit its evidence.  Keep
# this below Hermes' 30 second post-hook budget while avoiding the ordinary
# five second transport timeout.
AFFECT_TIMEOUT_S = 20.0


class StyxCoreClient:
    """Sync HTTP клиент к styx-core daemon."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        social_token: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        long_timeout_s: float = LONG_TIMEOUT_S,
        affect_timeout_s: float = AFFECT_TIMEOUT_S,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("STYX_DAEMON_URL", "http://127.0.0.1:8788")
        ).rstrip("/")
        self._token = token if token is not None else os.environ.get("STYX_HTTP_TOKEN")
        self._social_token = (
            social_token
            if social_token is not None
            else os.environ.get("STYX_SOCIAL_TOKEN")
        )
        self._timeout = timeout_s
        self._long_timeout = long_timeout_s
        self._affect_timeout = affect_timeout_s
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
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/sync_turn",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "tool_calls": tool_calls,
                "idempotency_key": idempotency_key,
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
                "messages": messages or [],
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

    def cognition_preturn(
        self,
        agent_id: str,
        *,
        host_key: str | None = None,
        parent_host_key: str | None = None,
        session_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        query: str | None = None,
        token_budget: int | None = None,
        model: str | None = None,
        platform: str | None = None,
        planned_execution_provenance: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one fenced pre-cognitive envelope for a Hermes turn."""
        return self._post(
            "/cognition/preturn",
            {
                "agent_id": agent_id,
                "host_key": host_key,
                "parent_host_key": parent_host_key,
                "session_id": session_id,
                "messages": messages or [],
                "query": query,
                "token_budget": token_budget,
                "model": model,
                "platform": platform,
                "planned_execution_provenance": planned_execution_provenance,
                "extra": extra or {},
            },
            timeout=self._long_timeout,
        )

    def cognition_commit(
        self,
        agent_id: str,
        *,
        session_id: str | None,
        host_key: str,
        parent_host_key: str | None,
        snapshot_token: str | None,
        status: str,
        user_message: str,
        assistant_response: str,
        conversation_history: list[dict[str, Any]] | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        consequences: list[dict[str, Any]] | None = None,
        model: str | None = None,
        platform: str | None = None,
        execution_provenance: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit the finalized channel projection under stable host lineage."""
        return self._post(
            "/cognition/commit",
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "host_key": host_key,
                "parent_host_key": parent_host_key,
                "snapshot_token": snapshot_token,
                "status": status,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "conversation_history": conversation_history or [],
                "tool_events": tool_events or [],
                "consequences": consequences or [],
                "model": model,
                "platform": platform,
                "execution_provenance": execution_provenance,
                "extra": extra or {},
            },
            timeout=self._affect_timeout,
        )

    def cognition_ready_claim(
        self,
        agent_id: str,
        *,
        consumer_id: str,
        after_generation: int = 0,
        limit: int = 1,
        wait_ms: int = 0,
    ) -> dict[str, Any]:
        """Claim content-free wake candidates; the caller owns scheduling policy."""
        return self._post(
            "/cognition/ready-events/claim",
            {
                "agent_id": agent_id,
                "consumer_id": consumer_id,
                "after_generation": after_generation,
                "limit": limit,
                "wait_ms": wait_ms,
            },
            timeout=max(self._long_timeout, wait_ms / 1000 + 5.0),
        )

    def cognition_ready_signal(
        self, agent_id: str, *, signal_generation: int
    ) -> dict[str, Any]:
        return self._post(
            "/cognition/ready-events/signal",
            {"agent_id": agent_id, "signal_generation": signal_generation},
        )

    def cognition_ready_resolve(
        self,
        agent_id: str,
        *,
        consumer_id: str,
        claim_token: str,
        outcome: str,
        snapshot_token: str | None = None,
        policy_reason: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/cognition/ready-events/resolve",
            {
                "agent_id": agent_id,
                "consumer_id": consumer_id,
                "claim_token": claim_token,
                "outcome": outcome,
                "snapshot_token": snapshot_token,
                "policy_reason": policy_reason,
            },
        )

    def cognition_observe(
        self,
        agent_id: str,
        *,
        source_id: str,
        source_stream: str,
        source_sequence: int,
        observation_key: str,
        difference_kind: str,
        content: str,
        salience: float,
        confidence: float,
        reducer_name: str,
        reducer_version: str,
        action_ref: dict[str, Any] | None = None,
        source_observed_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish one independently observed, pre-reduced difference.

        The terminal Hermes hook intentionally does not call this method for
        same-act tool results.  It is an explicit connector/sensor surface.
        """
        return self._post(
            "/cognition/observations",
            {
                "agent_id": agent_id,
                "source_id": source_id,
                "source_stream": source_stream,
                "source_sequence": source_sequence,
                "observation_key": observation_key,
                "difference_kind": difference_kind,
                "content": content,
                "salience": salience,
                "confidence": confidence,
                "reducer_name": reducer_name,
                "reducer_version": reducer_version,
                "action_ref": action_ref,
                "source_observed_at": source_observed_at,
                "metadata": metadata or {},
            },
            timeout=self._long_timeout,
        )

    def observe_affective_turn(
        self,
        agent_id: str,
        *,
        idempotency_key: str,
        turn_id: str,
        session_id: str | None = None,
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
        model: str | None = None,
        platform: str | None = None,
    ) -> dict[str, Any]:
        """Forward a finalized Hermes turn to the core affect seam.

        The endpoint is additive.  Calling a pre-feature core may return 404;
        ``post_llm_call`` owns fail-open handling so completed turns are never
        failed by a mixed-version deployment.
        """
        return self._post(
            "/affect/observe_turn",
            {
                "agent_id": agent_id,
                "idempotency_key": idempotency_key,
                "turn_id": turn_id,
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "conversation_history": conversation_history or [],
                "tool_events": tool_events or [],
                "task_id": task_id,
                "model": model,
                "platform": platform,
            },
            timeout=self._affect_timeout,
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

    # ── scoped social evidence (wave 42) ──────────────────────────────

    def social_create_actor(
        self,
        agent_id: str,
        *,
        identity_namespace: str,
        actor_key: str,
        actor_kind: str,
        identity_evidence_hash: str,
        private_label: str | None = None,
        attestation_principal_id: str | None = None,
    ) -> dict[str, Any]:
        """Register one owner-scoped actor coordinate, without attesting it."""
        return self._post(
            "/social/actors",
            {
                "agent_id": agent_id,
                "identity_namespace": identity_namespace,
                "actor_key": actor_key,
                "actor_kind": actor_kind,
                "private_label": private_label,
                "identity_evidence_hash": identity_evidence_hash,
                "attestation_principal_id": attestation_principal_id,
            },
            social_auth=True,
        )

    def social_create_scope(
        self,
        agent_id: str,
        *,
        scope_key: str,
        protocol_id: str,
        protocol_version: str,
        policy_hash: str,
    ) -> dict[str, Any]:
        """Create an empty local social scope; no membership is inferred."""
        return self._post(
            "/social/scopes",
            {
                "agent_id": agent_id,
                "scope_key": scope_key,
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "policy_hash": policy_hash,
            },
            social_auth=True,
        )

    def social_record_encounter(
        self,
        agent_id: str,
        *,
        encounter_key: str,
        scope_id: str,
        observer_actor_id: str,
        encountered_actor_id: str,
        direction: str,
        channel_kind: str,
        evidence_hash: str,
        confidence: float,
        source_act_id: str | None = None,
        source_observation_id: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Record an explicit encounter; this never creates an attestation."""
        return self._post(
            "/social/encounters",
            {
                "agent_id": agent_id,
                "encounter_key": encounter_key,
                "scope_id": scope_id,
                "observer_actor_id": observer_actor_id,
                "encountered_actor_id": encountered_actor_id,
                "direction": direction,
                "channel_kind": channel_kind,
                "source_act_id": source_act_id,
                "source_observation_id": source_observation_id,
                "summary": summary,
                "evidence_hash": evidence_hash,
                "confidence": confidence,
            },
            social_auth=True,
        )

    def social_attest(
        self,
        agent_id: str,
        *,
        scope_id: str,
        issuer_actor_id: str,
        subject_actor_id: str,
        attestation_key: str,
        verdict: str,
        protocol_id: str,
        protocol_version: str,
        source_act_id: str,
        trust_level: str,
        attestation_kind: str = "direct",
        source_action_ordinal: int | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
        signature_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit one explicit issuer act; no text classifier calls this API."""
        return self._social_attestation(
            "/social/attestations",
            agent_id=agent_id,
            scope_id=scope_id,
            issuer_actor_id=issuer_actor_id,
            subject_actor_id=subject_actor_id,
            attestation_key=attestation_key,
            verdict=verdict,
            protocol_id=protocol_id,
            protocol_version=protocol_version,
            source_act_id=source_act_id,
            trust_level=trust_level,
            attestation_kind=attestation_kind,
            source_action_ordinal=source_action_ordinal,
            evidence_refs=evidence_refs,
            signature_metadata=signature_metadata,
            supersedes_attestation_id=None,
        )

    def social_revise_attestation(
        self,
        agent_id: str,
        *,
        supersedes_attestation_id: str,
        scope_id: str,
        issuer_actor_id: str,
        subject_actor_id: str,
        attestation_key: str,
        verdict: str,
        protocol_id: str,
        protocol_version: str,
        source_act_id: str,
        trust_level: str,
        attestation_kind: str = "direct",
        source_action_ordinal: int | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
        signature_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a revision; the superseded attestation is never edited."""
        return self._social_attestation(
            "/social/attestations/revise",
            agent_id=agent_id,
            scope_id=scope_id,
            issuer_actor_id=issuer_actor_id,
            subject_actor_id=subject_actor_id,
            attestation_key=attestation_key,
            verdict=verdict,
            protocol_id=protocol_id,
            protocol_version=protocol_version,
            source_act_id=source_act_id,
            trust_level=trust_level,
            attestation_kind=attestation_kind,
            source_action_ordinal=source_action_ordinal,
            evidence_refs=evidence_refs,
            signature_metadata=signature_metadata,
            supersedes_attestation_id=supersedes_attestation_id,
        )

    def _social_attestation(
        self,
        path: str,
        *,
        agent_id: str,
        scope_id: str,
        issuer_actor_id: str,
        subject_actor_id: str,
        attestation_key: str,
        verdict: str,
        protocol_id: str,
        protocol_version: str,
        source_act_id: str,
        trust_level: str,
        attestation_kind: str,
        source_action_ordinal: int | None,
        evidence_refs: list[dict[str, Any]] | None,
        signature_metadata: dict[str, Any] | None,
        supersedes_attestation_id: str | None,
    ) -> dict[str, Any]:
        return self._post(
            path,
            {
                "agent_id": agent_id,
                "scope_id": scope_id,
                "issuer_actor_id": issuer_actor_id,
                "subject_actor_id": subject_actor_id,
                "attestation_key": attestation_key,
                "attestation_kind": attestation_kind,
                "verdict": verdict,
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "source_act_id": source_act_id,
                "source_action_ordinal": source_action_ordinal,
                "evidence_refs": evidence_refs or [],
                "trust_level": trust_level,
                "signature_metadata": signature_metadata or {},
                "supersedes_attestation_id": supersedes_attestation_id,
            },
            social_auth=True,
        )

    def social_dissolve_scope(
        self,
        agent_id: str,
        *,
        scope_id: str,
    ) -> dict[str, Any]:
        return self._post(
            "/social/scopes/dissolve",
            {"agent_id": agent_id, "scope_id": scope_id},
            social_auth=True,
        )

    def social_create_grant(
        self,
        agent_id: str,
        *,
        grant_key: str,
        scope_id: str,
        grantee_principal_id: str,
        capability: str,
        evidence_class: str,
        evidence_id: str | None = None,
        actor_a_id: str | None = None,
        actor_b_id: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/social/grants",
            {
                "agent_id": agent_id,
                "grant_key": grant_key,
                "scope_id": scope_id,
                "grantee_principal_id": grantee_principal_id,
                "capability": capability,
                "evidence_class": evidence_class,
                "evidence_id": evidence_id,
                "actor_a_id": actor_a_id,
                "actor_b_id": actor_b_id,
                "expires_at": expires_at,
            },
            social_auth=True,
        )

    def social_query(
        self,
        agent_id: str,
        *,
        scope_id: str,
        actor_a_id: str,
        actor_b_id: str,
    ) -> dict[str, Any]:
        return self._post(
            "/social/query",
            {
                "agent_id": agent_id,
                "scope_id": scope_id,
                "actor_a_id": actor_a_id,
                "actor_b_id": actor_b_id,
            },
            social_auth=True,
        )

    def social_revoke_grant(
        self,
        agent_id: str,
        *,
        revocation_key: str,
        grant_id: str,
    ) -> dict[str, Any]:
        return self._post(
            "/social/grants/revoke",
            {
                "agent_id": agent_id,
                "revocation_key": revocation_key,
                "grant_id": grant_id,
            },
            social_auth=True,
        )

    def social_explain(
        self,
        agent_id: str,
        *,
        scope_id: str,
    ) -> dict[str, Any]:
        """Read bounded audit coordinates, never private evidence content."""
        return self._post(
            "/social/explain",
            {
                "agent_id": agent_id,
                "scope_id": scope_id,
            },
            social_auth=True,
        )

    def social_deliver(
        self,
        agent_id: str,
        *,
        delivery_key: str,
        scope_id: str,
        receiving_agent_id: str,
        evidence_class: str,
        evidence_id: str,
    ) -> dict[str, Any]:
        """Explicitly bridge one granted social event into an observation."""
        return self._post(
            "/social/deliver",
            {
                "agent_id": agent_id,
                "delivery_key": delivery_key,
                "scope_id": scope_id,
                "evidence_class": evidence_class,
                "evidence_id": evidence_id,
                "receiving_agent_id": receiving_agent_id,
            },
            social_auth=True,
        )

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
        social_auth: bool = False,
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
        if social_auth and self._social_token:
            # The ordinary daemon bearer does not authorize cross-agent
            # social access.  Keep this principal credential per-request so
            # it cannot leak to health, cognition, recall or other routes.
            headers["X-Styx-Social-Token"] = self._social_token
        signed_body: bytes | None = None
        if (
            social_auth
            and path in {"/social/attestations", "/social/attestations/revise"}
            and payload.get("trust_level") == "verified"
            and self._social_token
        ):
            signed_body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["X-Styx-Social-Signature"] = hmac.new(
                self._social_token.encode("utf-8"), signed_body, hashlib.sha256
            ).hexdigest()
        try:
            request_kwargs: dict[str, Any] = {
                "timeout": timeout or self._timeout,
                "headers": headers or None,
            }
            if signed_body is None:
                request_kwargs["json"] = payload
            else:
                request_kwargs["data"] = signed_body
            resp = self._session.post(url, **request_kwargs)
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
        if path.startswith("/social/"):
            log.warning("styx-core %s returned %d", path, resp.status_code)
        else:
            log.warning(
                "styx-core %s returned %d: %s",
                path,
                resp.status_code,
                body if body else resp.text[:200],
            )
        resp.raise_for_status()
    return body if isinstance(body, dict) else {}
