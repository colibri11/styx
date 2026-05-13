"""StyxMemoryCore — host-agnostic ядро Styx memory provider.

Долгая память агента в Postgres + pgvector:

- ``initialize`` подключает Postgres, регистрирует session.
- ``sync_turn`` пишет user/assistant пару как два отдельных message.
- ``handle_recall`` — обработка вызова styx_recall tool.
- ``get_tool_schemas`` возвращает schema для styx_recall.

Не наследуется от Hermes ABC. Hermes-обёртка
``StyxMemoryProvider(MemoryProvider)`` живёт в styx-hermes и проксирует
вызовы по HTTP в core daemon, который держит state и вызывает этот класс.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psycopg

if TYPE_CHECKING:
    from styx.engine.store_routing import StoreRoutingConfig

from styx import turn_state
from styx.config import StyxConfig, load as load_config
from styx.embedding import EmbeddingClient, EmbeddingError, make_embedding_client
from styx.emotional.sentiment import (
    SentimentClient,
    make_sentiment_client,
    scale_hot_vad_delta,
)
from styx.emotional.state import append_emotional_state
from styx.providers.recall_tracker import RecallTracker
from styx.storage.queries import AgentScopedQueries
from styx.storage.recall import format_recall_text, recall_full
from styx.storage.recall_config import (
    DEFAULT_RECALL_CONFIG,
    RecallConfig,
    resolve_recall_config,
)

log = logging.getLogger(__name__)


class StyxMemoryCore:
    """Host-agnostic ядро вокруг Postgres+pgvector storage Styx.

    Per-agent: каждый экземпляр привязан к одному ``agent_id``,
    зафиксированному в конструкторе. Один core daemon обслуживает
    несколько ``StyxMemoryCore`` параллельно (по одному на agent_id).
    """

    def __init__(self, agent_id: str = "") -> None:
        self._agent_id: str = (agent_id or "").strip()
        self._config: StyxConfig | None = None
        self._conn: psycopg.Connection | None = None
        self._queries: AgentScopedQueries | None = None
        self._embedding: EmbeddingClient | None = None
        self._sentiment: SentimentClient | None = None
        self._recall_tracker: RecallTracker = RecallTracker()
        self._session_id: uuid.UUID | None = None
        self._write_lock = threading.Lock()
        # Применяется в initialize после load_config; до initialize
        # get_tool_schemas работает на дефолте.
        self._recall_config: RecallConfig = DEFAULT_RECALL_CONFIG

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -- identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "styx-memory"

    def is_available(self) -> bool:
        from styx.config import is_available as _avail
        return _avail()

    # -- lifecycle -------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        # Фикс #4: повторный initialize не должен утекать старое соединение.
        if self._conn is not None:
            self.shutdown()

        # agent_id может быть задан в конструкторе (HTTP path) или через
        # kwargs.agent_identity (legacy path). Хотя бы один источник должен
        # дать non-empty agent_id, иначе RuntimeError.
        identity = (kwargs.get("agent_identity") or "").strip()
        if identity and not self._agent_id:
            self._agent_id = identity
        if not self._agent_id:
            raise RuntimeError(
                "StyxMemoryCore.initialize requires non-empty agent_id "
                "(в __init__ или 'agent_identity' kwarg)"
            )

        hermes_home = kwargs.get("hermes_home")

        self._config = load_config(hermes_home)
        self._session_id = _coerce_session_id(session_id)
        self._recall_config = _build_recall_config(self._config)

        self._conn = psycopg.connect(self._config.database_url)
        self._queries = AgentScopedQueries(self._conn, self._agent_id)
        self._embedding = make_embedding_client(
            base_url=self._config.ollama_url,
            model=self._config.embedding_model,
            dim=self._config.embedding_dim,
            timeout=self._config.embedding_timeout_s,
        )

        # Sentiment hot-path (волна 7d). Может быть выключен через
        # STYX_SENTIMENT_ENABLED=0 для тестов / debug'а.
        if self._config.sentiment_enabled:
            self._sentiment = make_sentiment_client(
                base_url=self._config.llm_url,
                model=self._config.llm_model,
                timeout_s=self._config.sentiment_timeout_s,
            )
        else:
            self._sentiment = None

        if self._session_id is not None:
            self._queries.upsert_session(self._session_id)
            self._conn.commit()

        # Transport state (per-agent через core API). Core-сторона
        # держит prompt_cache_key для transport-классов, которые в
        # styx-hermes наследуют от Hermes ABC.
        from styx.engine.transport import configure as configure_transport
        configure_transport(self._agent_id)

        # Волна 14 (10a): temporal isolation — TurnStateManager под
        # snapshot fence в recall'ах. TTL глобальный для daemon —
        # один configure на process.
        turn_state.configure(ttl_s=self._config.turn_state_ttl_s)

        # Salient bridge (волна 9). ContextEngine достаёт queries/embed
        # через per-agent handle, инжектит recall'нутые memories в
        # каждый compress(). Отключается через STYX_SALIENT_ENABLED=0.
        if self._config.salient_enabled:
            from styx.engine import salient_bridge
            salient_bridge.configure(
                self._agent_id,
                queries=self._queries,
                embed_client=self._embedding,
                recall_config=self._recall_config,
                timeout_s=self._config.salient_timeout_s,
                min_query_len=self._config.salient_min_query_len,
            )

        # Focus tracker / drift detection (волна 10). Кэширует salient
        # block на эпоху, инвалидирует на смене темы. Зависит от
        # salient_enabled (без него focus_tracker бесполезен — кэшировать
        # нечего). STYX_DRIFT_ENABLED=0 → fallback на волна-9 поведение.
        if self._config.salient_enabled and self._config.drift_enabled:
            from styx.engine import focus_tracker
            focus_tracker.configure(
                self._agent_id,
                window_size=self._config.focus_window_size,
                drift_threshold=self._config.drift_threshold,
            )

        # Hot-tier (волна 11). In-process store memory items, прошедших
        # через recall_full недавно (TTL 5 мин default). recall_full
        # supplement'ит результат items'ами из hot до filter+dedup+slice.
        # disabled → state не configure'ится, supplement пуст, put no-op.
        if self._config.hot_tier_enabled:
            from styx.engine import hot_tier
            hot_tier.configure(
                self._agent_id,
                ttl_s=self._config.hot_tier_ttl_s,
                lru_bound=self._config.hot_tier_lru_bound,
            )

        # Eviction relevance-aware (волна 12). compress читает handle,
        # ранжирует middle-сообщения по cosine к focus centroid'у и
        # keep'ит top-K между head и tail. disabled → handle is None,
        # apply_relevance_eviction возвращает [] (recency-only).
        if self._config.eviction_relevance_enabled:
            from styx.engine import eviction_relevance_bridge
            eviction_relevance_bridge.configure(
                self._agent_id,
                queries=self._queries,
                keep_k=self._config.eviction_relevance_keep_k,
                threshold=self._config.eviction_relevance_threshold,
            )

        # Pre-LLM focus inject (волна 15). Multi-channel framework для
        # инжекта в user message через Hermes pre_llm_call hook. Hook
        # регистрируется в plugin.py::register; здесь только configure
        # framework state'а (handle + channels list).
        if self._config.pre_llm_inject_enabled:
            from styx.engine import pre_llm_inject
            from styx.engine.pre_llm_channels.peer_vad import channel_peer_vad
            handle = pre_llm_inject.ChannelHandle(
                queries=self._queries,
                peer_vad_enabled=self._config.peer_vad_enabled,
                peer_vad_min_norm=self._config.peer_vad_min_norm,
                peer_vad_ttl_s=self._config.peer_vad_ttl_s,
            )
            pre_llm_inject.configure(
                self._agent_id,
                handle=handle,
                channels=[("peer_vad", channel_peer_vad)],
                enabled=True,
            )

        # Working set persistence (волна 13). Restore'им focus_tracker и
        # hot_tier из БД (если state не stale), запускаем background
        # daemon-thread для periodic save'а. Идёт ПОСЛЕ всех configure'ов
        # — restore требует _STATE'ы уже инициализированы. Disabled →
        # load skip, save-thread не стартует, shutdown не flush'ит.
        if self._config.working_set_persistence_enabled:
            from styx.engine import (
                focus_tracker as _ft,
                hot_tier as _ht,
                working_set_persistence as _wsp,
            )
            try:
                snapshot = _wsp.load(
                    self._conn,
                    agent_id=self._agent_id,
                    ttl_s=self._config.working_set_ttl_s,
                    hot_ttl_s=self._config.hot_tier_ttl_s,
                    embedding_dim=self._config.embedding_dim,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open
                log.warning("working_set_persistence load failed: %s", exc)
                snapshot = None
            if snapshot is not None:
                if snapshot.focus is not None:
                    _ft.restore(
                        self._agent_id,
                        window=snapshot.focus.window,
                        cached_salient=snapshot.focus.cached_salient,
                        epoch_id=snapshot.focus.epoch_id,
                    )
                if snapshot.hot is not None:
                    _ht.restore(self._agent_id, snapshot.hot)

            agent_id_local = self._agent_id

            def _snapshot_fn() -> tuple[
                tuple[list[list[float]], dict | None, int] | None,
                list[Any],
            ]:
                return (
                    _ft.snapshot(agent_id_local),
                    _ht.snapshot(agent_id_local),
                )

            _wsp.start(
                self._agent_id,
                dsn=self._config.database_url,
                embedding_dim=self._config.embedding_dim,
                interval_s=self._config.working_set_save_interval_s,
                write_lock=self._write_lock,
                snapshot_fn=_snapshot_fn,
            )

        log.info(
            "StyxMemoryCore initialized agent_id=%s session_id=%s",
            self._agent_id, self._session_id,
        )

    def shutdown(self) -> None:
        # Working set persistence (волна 13) останавливаем ДО acquire
        # write_lock'а: save-thread может в этот момент держать write_lock
        # внутри _tick'а; если shutdown держит lock и ждёт join thread'а —
        # deadlock. После stop() final flush делается синхронно из main
        # thread'а уже с остановленным save-thread'ом.
        if (
            self._config is not None
            and self._config.working_set_persistence_enabled
        ):
            from styx.engine import working_set_persistence
            working_set_persistence.stop(self._agent_id)

        # Фикс #12: берём lock чтобы не гонять с активным sync_turn в другом потоке.
        with self._write_lock:
            from styx.engine import (
                eviction_relevance_bridge,
                focus_tracker,
                hot_tier,
                pre_llm_inject,
                salient_bridge,
                transport,
            )
            pre_llm_inject.reset(self._agent_id)
            eviction_relevance_bridge.reset(self._agent_id)
            hot_tier.reset(self._agent_id)
            focus_tracker.reset(self._agent_id)
            salient_bridge.reset(self._agent_id)
            transport.reset(self._agent_id)
            turn_state.reset(self._agent_id)
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
                    self._queries = None
                    self._embedding = None
                    self._sentiment = None

    # -- system prompt + recall ------------------------------------------

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # v1: retrieval не выполняется. Suffix композиция в волне 3
        # делает stable+recent без обращения к long-tier.
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    # -- write path ------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if self._queries is None:
            log.warning("sync_turn до initialize — пропуск")
            return

        target_session = _coerce_session_id(session_id) if session_id else self._session_id
        inserted_ids: list[tuple[uuid.UUID, str]] = []

        # Фикс #13: upsert_session + оба insert_message — единая транзакция.
        # Embed-after-commit (волна 7) идёт ОТДЕЛЬНОЙ транзакцией ниже,
        # чтобы Ollama latency (~120ms на turn по замеру 2026-04-30) не
        # держал row-lock на memories во время вызова HTTP.
        with self._write_lock:
            if self._queries is None:
                log.warning("sync_turn: queries обнулены во время ожидания lock — пропуск")
                return
            if target_session is not None:
                self._queries.upsert_session(target_session)
            if user_content:
                mid = self._queries.insert_message(
                    role="user",
                    content=user_content,
                    session_id=target_session,
                )
                inserted_ids.append((mid, user_content))
            if assistant_content:
                mid = self._queries.insert_message(
                    role="assistant",
                    content=assistant_content,
                    session_id=target_session,
                )
                inserted_ids.append((mid, assistant_content))
            self._conn.commit()  # type: ignore[union-attr]

            # Embed-after-commit (sync, decisions § 17.A1).
            # Любая ошибка Ollama → лог error, embedding остаётся NULL,
            # memory не подтягивается в recall до следующего успешного
            # embed (отдельный re-embed CLI — волна 7e).
            if self._embedding is not None:
                for mid, content in inserted_ids:
                    try:
                        vec = self._embedding.embed(content)
                    except EmbeddingError as exc:
                        log.warning(
                            "embed-after-commit упал для memory %s: %s",
                            mid, exc,
                        )
                        continue
                    try:
                        self._queries.update_embedding(mid, vec)
                        self._conn.commit()  # type: ignore[union-attr]
                    except Exception as exc:  # pragma: no cover — defensive
                        log.warning(
                            "update_embedding упал для memory %s: %s",
                            mid, exc,
                        )
                        self._conn.rollback()  # type: ignore[union-attr]
                        continue

                    # Auto-link для dialogue ряда (волна 18).
                    # cross-agent: dialogue реплика связывается со
                    # similarly-themed subjective memories других
                    # агентов. Fail-open — auto-link errors не должны
                    # ронять sync_turn.
                    try:
                        self._auto_link_dialogue(mid, vec)
                        self._conn.commit()  # type: ignore[union-attr]
                    except Exception as exc:  # noqa: BLE001 — fail-open
                        log.warning(
                            "sync_turn auto_link упал для memory %s: %s",
                            mid, exc,
                        )
                        try:
                            self._conn.rollback()  # type: ignore[union-attr]
                        except Exception:
                            pass

            # Recall classifier enqueue (волна 7c, ADR § 20). Если в этом
            # turn'е был styx_recall и assistant написал содержательный
            # ответ — enqueue'им классификацию. Триггер собственный, не
            # port (memorybox skeleton). Идёт ДО sentiment'а потому что
            # sentiment может занять до 800ms; classifier-enqueue —
            # быстрый INSERT, не блокирует.
            if (
                self._config is not None
                and target_session is not None
                and assistant_content
                and len(assistant_content) >= self._config.classifier_min_assistant_length
            ):
                buffer_ids = self._recall_tracker.take(target_session)
                if buffer_ids:
                    max_per_turn = self._config.classifier_max_recall_events_per_turn
                    selected = buffer_ids[-max_per_turn:]
                    truncated_reply = assistant_content[:80_000]
                    try:
                        self._queries.enqueue_classification(
                            recall_event_ids=selected,
                            llm_output_text=truncated_reply,
                        )
                        self._conn.commit()  # type: ignore[union-attr]
                    except Exception as exc:  # noqa: BLE001 — fail-open
                        log.warning("classifier enqueue упал: %s", exc)
                        try:
                            self._conn.rollback()  # type: ignore[union-attr]
                        except Exception:
                            pass

            # Sentiment hot-path (волна 7d). Только на user_content
            # (peer-реплика). Fail-open: extract_vad возвращает None при
            # любой ошибке/skip'е, append'им только если получили VAD.
            # Идёт после embed'а — embed критичнее для recall.
            if self._sentiment is not None and user_content:
                vad = self._sentiment.extract_vad(user_content)
                if vad is not None:
                    delta = scale_hot_vad_delta(vad)
                    try:
                        # Волна 15: пишем raw VAD в metadata.hot_vad для
                        # peer_vad канала pre_llm_inject. Без этого канал
                        # видел бы только аккумулированное состояние
                        # (base + delta), не «острый» peer-сигнал.
                        append_emotional_state(
                            self._conn,  # type: ignore[arg-type]
                            self._agent_id,
                            delta,
                            source="hot_sentiment",
                            metadata={
                                "hot_vad": [
                                    vad.valence, vad.arousal, vad.dominance,
                                ],
                            },
                        )
                        self._conn.commit()  # type: ignore[union-attr]
                        self._sentiment.increment_applied()
                    except Exception as exc:  # pragma: no cover — defensive
                        log.warning(
                            "emotional_state append упал: %s", exc
                        )
                        try:
                            self._conn.rollback()  # type: ignore[union-attr]
                        except Exception:
                            pass

            # Волна 14 (10a): natural marker конца turn'а. Все sync_turn
            # writes (memories, embedding, classifier-enqueue, emotional
            # state) committed → следующий compress / handle_tool_call
            # откроет новый turn с свежим cycle_start, увидит batch-
            # memories появившиеся между этим turn'ом и предыдущим.
            turn_state.close(self._agent_id)

    def ingest_single_message(
        self,
        *,
        role: str,
        content: str,
        session_id: str = "",
    ) -> uuid.UUID | None:
        """Raw insert + sync embed для одного message (волна 26 Phase B).

        Без полного sync_turn-pipeline (gatekeeper / auto-link /
        classifier-enqueue / sentiment не применяются). Используется через
        ``POST /context/ingest`` когда OpenClaw runtime отдаёт реплики по
        одной — пара (user, assistant) недоступна, поэтому кусочки stack'а,
        требующие парности, тут не работают. Полный stack — через
        ``ingest_batch_pairwise`` / ``sync_turn``.

        Возвращает memory_id если запись прошла; ``None`` — если queries
        обнулены (core не initialized) или content пуст.
        """
        if self._queries is None:
            log.warning("ingest_single_message до initialize — пропуск")
            return None
        if not content:
            return None

        target_session = _coerce_session_id(session_id) if session_id else self._session_id

        with self._write_lock:
            if self._queries is None:
                log.warning("ingest_single_message: queries обнулены — пропуск")
                return None
            if target_session is not None:
                self._queries.upsert_session(target_session)
            mid = self._queries.insert_message(
                role=role,
                content=content,
                session_id=target_session,
            )
            self._conn.commit()  # type: ignore[union-attr]

            # Embed-after-commit (sync). Любая ошибка Ollama → лог,
            # embedding остаётся NULL, recall не подтянет до re-embed.
            if self._embedding is not None:
                try:
                    vec = self._embedding.embed(content)
                except EmbeddingError as exc:
                    log.warning(
                        "embed-after-commit упал для memory %s: %s",
                        mid, exc,
                    )
                    return mid
                try:
                    self._queries.update_embedding(mid, vec)
                    self._conn.commit()  # type: ignore[union-attr]
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning(
                        "update_embedding упал для memory %s: %s",
                        mid, exc,
                    )
                    try:
                        self._conn.rollback()  # type: ignore[union-attr]
                    except Exception:
                        pass

        return mid

    # -- subjective writes (волна 17) ------------------------------------

    def memory_store(
        self,
        *,
        content: str,
        kind: str = "note",
        kind_src: str = "subjective",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance_provisional: float | None = None,
    ) -> "MemoryStoreOutcome":
        """Subjective write entry: insert + sync embed + selective gatekeeper.

        Каждое решение gatekeeper'а (skip/merge/supersede/store) применяется
        в одной транзакции вместе с insert'ом. Caller получает action +
        memory_id (None для skip и для merge — поглощён existing).

        Raises ``RuntimeError`` если provider не initialize'нут или если
        embedding-call фатально упал. Constraint violation на content >
        2400 chars пробрасывается из Postgres.
        """
        if (
            self._queries is None
            or self._embedding is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError("StyxMemoryCore.memory_store: provider не initialize'нут")

        sid = _coerce_session_id(session_id) if session_id else None
        role = _role_for_kind(kind)

        # Store-routing (волна 19): длинный content разделяется на
        # chunks в documents/chunks; в memories пишется tail-memory с
        # archive_ref. CHECK constraint memories_content_length_check
        # (≤ 2400) обходится через tail-summary ≤ summary_chars.
        # Gatekeeper пропускается (D6 в waves/19); auto-link
        # применяется к tail-memory.
        routing = self._config.store_routing_config()
        if routing.enabled and len(content) > routing.limit:
            return self._memory_store_routed(
                content=content,
                kind=kind,
                kind_src=kind_src,
                role=role,
                session_id=sid,
                metadata=metadata,
                importance_provisional=importance_provisional,
                routing=routing,
            )

        with self._write_lock:
            mid = self._queries.insert_memory(
                role=role,
                content=content,
                kind=kind,
                kind_src=kind_src,
                session_id=sid,
                metadata=metadata or {},
                importance_provisional=importance_provisional,
            )

            try:
                vec = self._embedding.embed(content)
            except EmbeddingError as exc:
                # Сохраняем ряд как есть, без gatekeeper'а — он только
                # на ряды с embedding'ом. reembed CLI / следующий
                # запрос подберут.
                log.warning("memory_store: embed failed для %s: %s", mid, exc)
                self._conn.commit()
                return MemoryStoreOutcome(action="store", memory_id=str(mid))

            self._queries.update_embedding(mid, vec)

            gk_config = self._config.gatekeeper_config()
            if not gk_config.enabled:
                self._conn.commit()
                return MemoryStoreOutcome(action="store", memory_id=str(mid))

            from styx.engine.selective_gatekeeper import Action, decide
            from styx.observability.logging import log_event

            candidates = self._queries.find_gatekeeper_candidates(
                vec,
                max_cosine_distance=1.0 - gk_config.supersede_threshold,
                exclude_id=mid,
            )
            decision = decide(content, candidates, config=gk_config)
            log_event(
                log, "selective_decision",
                agent_id=str(self._agent_id),
                memory_id=str(mid),
                action=decision.action.value,
                existing_id=(
                    str(decision.existing_id)
                    if decision.existing_id else None
                ),
                similarity=decision.similarity,
                levenshtein_ratio=decision.levenshtein_ratio,
                source="memory_store",
            )

            if decision.action == Action.STORE:
                self._auto_link_after_subjective_write(mid, vec)
                self._conn.commit()
                return MemoryStoreOutcome(
                    action="store", memory_id=str(mid),
                    similarity=decision.similarity,
                )
            if decision.action == Action.SKIP:
                self._queries.apply_gatekeeper_skip(mid)
                self._conn.commit()
                return MemoryStoreOutcome(action="skip")
            if decision.action == Action.MERGE:
                assert decision.existing_id is not None
                self._queries.apply_gatekeeper_merge(
                    new_id=mid, existing_id=decision.existing_id,
                    new_content=content, new_embedding=vec,
                )
                self._conn.commit()
                return MemoryStoreOutcome(
                    action="merge",
                    existing_id=str(decision.existing_id),
                    similarity=decision.similarity,
                )
            # SUPERSEDE
            assert decision.existing_id is not None
            self._queries.apply_gatekeeper_supersede(
                new_id=mid, existing_id=decision.existing_id,
                new_embedding=vec,
            )
            self._auto_link_after_subjective_write(mid, vec)
            self._conn.commit()
            return MemoryStoreOutcome(
                action="supersede", memory_id=str(mid),
                existing_id=str(decision.existing_id),
                similarity=decision.similarity,
            )

    # -- reinterpret enqueue (волна 22) -----------------------------------

    def reinterpret_enqueue(
        self,
        *,
        memory_id: str,
        new_understanding_text: str,
        weight: float | None = None,
    ) -> "ReinterpretEnqueueOutcome":
        """Enqueue reinterpret_merge task. Apply отдельно через
        `reinterpret_apply_sweeper` под write-gate'ом.

        Возвращает discriminated result:
        - ``status='memory_not_found'``  — memory не найдена под этим agent_id.
        - ``status='cooldown'``          — последняя revision < 24h.
        - ``status='already_pending'``   — есть pending_sleep application.
        - ``status='queued'``            — task поставлен; task_id +
                                            application_id populated.

        Используется HTTP route /reinterpret + in-process Hermes-tool
        wrapper. Не делает polling task'а — apply через sweeper.
        """
        from styx.engine.reinterpret import reinterpret_cooldown
        from styx.storage.queries import enqueue_llm_task
        from styx.workers.handlers.reinterpret_merge import (
            REINTERPRET_MERGE_TASK_TYPE,
        )

        if (
            self._queries is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError(
                "StyxMemoryCore.reinterpret_enqueue: provider не initialize'нут"
            )
        if not self._config.reinterpret_enabled:
            raise RuntimeError(
                "reinterpret disabled (STYX_REINTERPRET_ENABLED=0)"
            )

        try:
            mid = uuid.UUID(memory_id)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"memory_id must be UUID, got {memory_id!r}"
            ) from exc
        if not new_understanding_text or len(new_understanding_text) > 2400:
            raise ValueError(
                "new_understanding_text — строка 1..2400"
            )
        if weight is not None and not (0.0 <= float(weight) <= 1.0):
            raise ValueError(f"weight={weight} вне [0, 1]")

        with self._write_lock:
            if self._queries is None or self._conn is None:
                raise RuntimeError("provider shut down mid-call")

            if not self._queries.memory_exists(mid):
                self._conn.commit()
                return ReinterpretEnqueueOutcome(
                    status="memory_not_found", memory_id=str(mid),
                )

            check = reinterpret_cooldown(
                self._queries, mid,
                cooldown_s=self._config.reinterpret_cooldown_s,
            )
            if not check.ok:
                self._conn.commit()
                if check.reason == "pending":
                    return ReinterpretEnqueueOutcome(
                        status="already_pending",
                        memory_id=str(mid),
                        pending_application_id=check.pending_application_id,
                    )
                # reason == 'recent'
                return ReinterpretEnqueueOutcome(
                    status="cooldown",
                    memory_id=str(mid),
                    last_reinterpreted_at=(
                        check.last_at.isoformat() if check.last_at else None
                    ),
                    next_available_at=(
                        check.next_at.isoformat() if check.next_at else None
                    ),
                )

            payload: dict[str, Any] = {
                "agent_id": self._agent_id,
                "new_understanding_text": new_understanding_text,
            }
            if weight is not None:
                payload["weight"] = float(weight)

            task_id = enqueue_llm_task(
                self._conn,
                task_type=REINTERPRET_MERGE_TASK_TYPE,
                payload=payload,
                memory_id=mid,
            )
            app_id = self._queries.insert_reinterpret_application(
                task_id=task_id, memory_id=mid,
            )
            self._conn.commit()
            return ReinterpretEnqueueOutcome(
                status="queued",
                memory_id=str(mid),
                task_id=str(task_id),
                application_id=app_id,
            )

    # -- ingest experience (волна 23) -----------------------------------

    def ingest_experience(
        self,
        *,
        content: str,
        kind: str = "note",
        kind_src: str = "experience_intake",
        metadata: dict[str, Any] | None = None,
        importance_provisional: float | None = None,
        content_hash: str | None = None,
        pipeline_id: str | None = None,
        pipeline_version: str | None = None,
        content_ref: dict[str, Any] | None = None,
    ) -> "IngestOutcome":
        """Pipeline ingest entry: idempotent INSERT через content_hash.

        Hash priority (D3 в waves/23):
          1. ``content_hash`` явный — pipeline сам контролирует.
          2. Auto-compute из (pipeline_id, pipeline_version, content_ref)
             если все три заданы и content_ref не пуст.
          3. Иначе ``None`` — partial UNIQUE индекс игнорирует, idempotency
             не применяется (каждый INSERT новый ряд).

        Без gatekeeper'а / auto-link'а / store-routing'а — pipeline-канал
        не subjective. Длинные доки (> 2400) → ValueError; pipeline
        разбивает сам (или OpenClaw plugin track будет).

        Не повторяет побочные эффекты при дедупликации: embedding и
        metadata из второго вызова игнорируются (existing ряд возвращается
        как есть).
        """
        if (
            self._queries is None
            or self._embedding is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError(
                "StyxMemoryCore.ingest_experience: provider не initialize'нут"
            )
        if not self._config.ingest_api_enabled:
            raise RuntimeError(
                "ingest API disabled (STYX_INGEST_API_ENABLED=0)"
            )

        if kind not in _VALID_KINDS:
            raise ValueError(f"ingest_experience: неизвестный kind={kind!r}")
        if not content or len(content) > 2400:
            raise ValueError(
                "ingest_experience: content — строка 1..2400"
            )

        # Resolve content_hash. Explicit > auto-compute > None.
        from styx.engine.ingest_hash import (
            compute_content_hash,
            is_content_ref_empty,
        )
        used_hash: str | None = None
        if content_hash is not None and content_hash.strip():
            used_hash = content_hash.strip()
        elif (
            pipeline_id
            and pipeline_version
            and not is_content_ref_empty(content_ref)
        ):
            used_hash = compute_content_hash(
                pipeline_id=pipeline_id,
                pipeline_version=pipeline_version,
                content_ref=content_ref or {},
            )

        # Enrich metadata: pipeline source + content_ref для трассируемости.
        # User metadata имеет priority — не затираем явно переданные ключи.
        enriched: dict[str, Any] = dict(metadata or {})
        if pipeline_id or pipeline_version:
            source: dict[str, Any] = {}
            if pipeline_id:
                source["pipeline_id"] = pipeline_id
            if pipeline_version:
                source["pipeline_version"] = pipeline_version
            enriched.setdefault("source", source)
        if content_ref and not is_content_ref_empty(content_ref):
            enriched.setdefault("content_ref", content_ref)

        with self._write_lock:
            # Embed inline. Если fail — INSERT без embedding (recall не
            # найдёт до reembed'а; idempotency не ломается, hash от
            # tuple-payload).
            try:
                vec: list[float] | None = self._embedding.embed(content)
            except EmbeddingError as exc:
                log.warning(
                    "ingest_experience: embed failed: %s", exc
                )
                vec = None

            memory_id, deduplicated = self._queries.ingest_upsert_memory(
                content=content,
                kind=kind,
                kind_src=kind_src,
                content_hash=used_hash,
                embedding=vec,
                metadata=enriched,
                role="system",
                importance_provisional=importance_provisional,
            )
            self._conn.commit()
            return IngestOutcome(
                memory_id=str(memory_id),
                deduplicated=deduplicated,
                used_hash=used_hash,
            )

    # -- file-ingest pipeline (волна 28) --------------------------------

    def ingest_document(
        self,
        *,
        path: str,
        source_ref: str | None = None,
        visibility: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_hash: str | None = None,
    ) -> "IngestDocumentOutcome":
        """File-ingest entry: parse → chunks → embed → INSERT document.

        Pipeline channel (как ``ingest_experience``): без gatekeeper'а
        / auto-link'а / classifier'а / tail-memory (D5/D13 в waves/28).
        Документ доступен только через ``search_archive``.

        Path-mode (D1): core читает файл с диска по абсолютному пути,
        валидирует path под whitelist (``STYX_INGEST_DOC_ROOTS``) +
        size guard.

        Raises:
            ValueError: path invalid / unsupported extension / mime
                mismatch / encrypted PDF / empty document / chunker
                degenerate input.
            RuntimeError: provider не инициализирован / ingest_doc
                disabled через config.
        """
        if (
            self._queries is None
            or self._embedding is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError(
                "StyxMemoryCore.ingest_document: provider не initialize'нут"
            )
        if not self._config.ingest_doc_enabled:
            raise RuntimeError(
                "ingest_document disabled (STYX_INGEST_DOC_ENABLED=0)"
            )

        from styx.engine.document_ingest import ingest_document

        doc_cfg = self._config.document_ingest_config()
        store_cfg = self._config.store_routing_config()

        with self._write_lock:
            try:
                result = ingest_document(
                    self._queries,
                    self._embedding,
                    raw_path=path,
                    config=doc_cfg,
                    store_routing=store_cfg,
                    source_ref=source_ref,
                    visibility=visibility,
                    metadata=metadata,
                    content_hash=content_hash,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            return IngestDocumentOutcome(
                document_id=str(result.document_id),
                deduplicated=result.deduplicated,
                chunks_count=result.chunks_count,
                mime_type=result.mime_type,
                original_name=result.original_name,
                size_bytes=result.size_bytes,
                char_count=result.char_count,
                content_hash=result.content_hash,
            )

    # -- dialogue tools (волна 24) --------------------------------------

    def dialogue_save(
        self,
        *,
        role: str,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Explicit ad-hoc save одной реплики (D5 в waves/24).

        Не триггерит auto-link / classifier / sentiment — только
        ``upsert_session`` (если задан) → ``insert_message`` → embed-
        after-commit. Для полного pipeline — ``sync_turn``.

        ``role`` ∈ {'user','assistant'}. CHECK constraint в БД
        защищает обход.

        Embed fail handling: warning + INSERT без embedding (search
        не найдёт до reembed'а). Идемпотентности нет — повторный
        save того же content получит новый ряд (D21).
        """
        if (
            self._queries is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError(
                "StyxMemoryCore.dialogue_save: provider не initialize'нут"
            )
        if not self._config.dialogue_api_enabled:
            raise RuntimeError(
                "dialogue API disabled (STYX_DIALOGUE_API_ENABLED=0)"
            )
        if role not in ("user", "assistant"):
            raise ValueError(
                f"dialogue_save: role должен быть 'user'|'assistant', "
                f"получен {role!r}"
            )
        if not content or len(content) > 2400:
            raise ValueError("dialogue_save: content — строка 1..2400")

        sid = _coerce_session_id(session_id) if session_id else None

        with self._write_lock:
            if self._queries is None:
                raise RuntimeError(
                    "dialogue_save: queries обнулены во время ожидания lock"
                )
            if sid is not None:
                self._queries.upsert_session(sid)
            memory_id = self._queries.insert_message(
                role=role,
                content=content,
                session_id=sid,
                metadata=metadata,
            )
            self._conn.commit()

            # Embed-after-commit (как sync_turn). Без auto-link /
            # classifier / sentiment — D5.
            if self._embedding is not None:
                try:
                    vec = self._embedding.embed(content)
                except EmbeddingError as exc:
                    log.warning(
                        "dialogue_save embed failed for %s: %s",
                        memory_id, exc,
                    )
                    vec = None
                if vec is not None:
                    try:
                        self._queries.update_embedding(memory_id, vec)
                        self._conn.commit()
                    except Exception as exc:  # pragma: no cover — defensive
                        log.warning(
                            "dialogue_save update_embedding упал: %s", exc
                        )
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
            return memory_id

    def dialogue_search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        after: Any | None = None,
        before: Any | None = None,
        semantic_only: bool = False,
        limit: int = 10,
    ) -> list["DialogueSearchOutcome"]:
        """Hybrid (default) либо pure-vector search.

        Embed'ит ``query`` через core embedding client; передаёт
        ``query_text=query`` для FTS hybrid если ``semantic_only=False``,
        иначе ``None`` → pure-vector mode.
        """
        if (
            self._queries is None
            or self._embedding is None
            or self._config is None
        ):
            raise RuntimeError(
                "StyxMemoryCore.dialogue_search: provider не initialize'нут"
            )
        if not self._config.dialogue_api_enabled:
            raise RuntimeError(
                "dialogue API disabled (STYX_DIALOGUE_API_ENABLED=0)"
            )
        query = query.strip()
        if not query:
            raise ValueError("dialogue_search: query пустой")

        sid = _coerce_session_id(session_id) if session_id else None

        try:
            qvec = self._embedding.embed(query)
        except EmbeddingError as exc:
            raise RuntimeError(
                f"dialogue_search: embed failed: {exc}"
            ) from exc

        hits = self._queries.dialogue_search(
            query_vector=qvec,
            query_text=None if semantic_only else query,
            limit=limit,
            session_id=sid,
            after=after,
            before=before,
        )
        return [
            DialogueSearchOutcome(
                memory_id=str(h.memory_id),
                role=h.role,
                content=h.content,
                score=float(h.score),
                created_at=h.created_at,
                session_id=str(h.session_id) if h.session_id else None,
            )
            for h in hits
        ]

    def dialogue_recent(
        self,
        *,
        session_id: str | None = None,
        before: Any | None = None,
        limit: int = 20,
    ) -> list["DialogueRecentOutcome"]:
        """Последние ``limit`` реплик в chronological order (oldest first)."""
        if self._queries is None or self._config is None:
            raise RuntimeError(
                "StyxMemoryCore.dialogue_recent: provider не initialize'нут"
            )
        if not self._config.dialogue_api_enabled:
            raise RuntimeError(
                "dialogue API disabled (STYX_DIALOGUE_API_ENABLED=0)"
            )
        sid = _coerce_session_id(session_id) if session_id else None
        rows = self._queries.dialogue_recent(
            limit=limit,
            session_id=sid,
            before=before,
        )
        # DESC by seq → reverse для chronological output.
        return [
            DialogueRecentOutcome(
                memory_id=str(r.memory_id),
                role=r.role,
                content=r.content,
                created_at=r.created_at,
                session_id=str(r.session_id) if r.session_id else None,
            )
            for r in reversed(rows)
        ]

    def dialogue_list_sessions(
        self, *, limit: int = 10,
    ) -> list["DialogueSessionOutcome"]:
        """List sessions с counts + first/last_at."""
        if self._queries is None or self._config is None:
            raise RuntimeError(
                "StyxMemoryCore.dialogue_list_sessions: "
                "provider не initialize'нут"
            )
        if not self._config.dialogue_api_enabled:
            raise RuntimeError(
                "dialogue API disabled (STYX_DIALOGUE_API_ENABLED=0)"
            )
        sessions = self._queries.dialogue_list_sessions(limit=limit)
        return [
            DialogueSessionOutcome(
                session_id=str(s.session_id),
                message_count=s.message_count,
                first_message_at=s.first_message_at,
                last_message_at=s.last_message_at,
            )
            for s in sessions
        ]

    def dialogue_prepare_summary(
        self, *, session_id: str, limit: int = 200,
    ) -> "DialogueSummaryOutcome":
        """Готовит transcript для summarizer-агента (D9 в waves/24).

        Пустая session → empty transcript, message_count=0, both
        timestamps None. Не 404.
        """
        if self._queries is None or self._config is None:
            raise RuntimeError(
                "StyxMemoryCore.dialogue_prepare_summary: "
                "provider не initialize'нут"
            )
        if not self._config.dialogue_api_enabled:
            raise RuntimeError(
                "dialogue API disabled (STYX_DIALOGUE_API_ENABLED=0)"
            )
        sid = _coerce_session_id(session_id)
        if sid is None:
            raise ValueError("dialogue_prepare_summary: session_id обязателен")
        rows = self._queries.dialogue_prepare_summary(
            session_id=sid, limit=limit,
        )
        if not rows:
            return DialogueSummaryOutcome(
                session_id=str(sid),
                message_count=0,
                first_message_at=None,
                last_message_at=None,
                transcript="",
            )

        from styx.engine.dialogue_format import format_transcript_line

        lines = [
            format_transcript_line(r.role, r.content, r.created_at)
            for r in rows
        ]
        return DialogueSummaryOutcome(
            session_id=str(sid),
            message_count=len(rows),
            first_message_at=rows[0].created_at,
            last_message_at=rows[-1].created_at,
            transcript="\n".join(lines),
        )

    # -- explain / analytics / confirm_usage (волна 25) -----------------

    def _require_explain_ready(self, op: str) -> None:
        """Гард: explain endpoint'ы требуют queries + config + (для
        embedding-flows) embedding client."""
        if (
            self._queries is None
            or self._config is None
            or self._conn is None
        ):
            raise RuntimeError(
                f"StyxMemoryCore.{op}: provider не initialize'нут"
            )
        if not self._config.explain_api_enabled:
            raise RuntimeError(
                "explain API disabled (STYX_EXPLAIN_API_ENABLED=0)"
            )

    def explain_decompose(
        self,
        *,
        memory_id: str,
        query: str,
        top_k_limit: int = 10,
        min_score: float | None = None,
    ) -> "ExplainDecomposeOutcome":
        """11-факторный breakdown скоринга (memory_id, query) — port
        memorybox `explainDecompose` (waves/25 D4)."""
        self._require_explain_ready("explain_decompose")
        if self._embedding is None:
            raise RuntimeError(
                "StyxMemoryCore.explain_decompose: embedding client не готов"
            )
        assert self._queries is not None and self._conn is not None
        try:
            mid = uuid.UUID(memory_id)
        except ValueError as exc:
            raise ValueError(
                f"explain_decompose: memory_id не UUID — {memory_id!r}"
            ) from exc
        query = query.strip()
        if not query:
            raise ValueError("explain_decompose: query пустой")

        try:
            qvec = self._embedding.embed(query)
        except EmbeddingError as exc:
            raise RuntimeError(
                f"explain_decompose: embed failed: {exc}"
            ) from exc

        from styx.emotional.baseline import read_baseline_for_scoring
        from styx.engine.explain import build_factors_block
        from styx.storage.scoring import (
            BuildFactorExprsOptions,
            EmotionalBaseline,
            build_factor_exprs,
        )

        usage_p75 = self._queries.compute_agent_usage_p75()
        baseline_obj = read_baseline_for_scoring(self._conn, self._agent_id)
        baseline = (
            EmotionalBaseline(
                valence=baseline_obj.valence,
                arousal=baseline_obj.arousal,
                dominance=baseline_obj.dominance,
            )
            if baseline_obj is not None
            else None
        )

        row = self._queries.explain_decompose_target(
            memory_id=mid,
            query_vector=qvec,
            query_text=query,
            usage_norm_p75=usage_p75,
            emotional_baseline=baseline,
        )
        if row is None:
            raise LookupError(
                f"explain_decompose: memory not found: {memory_id}"
            )

        # factor_meta — те же FactorExprs которыми SQL построен.
        factor_meta = build_factor_exprs(
            {"text_query": query},
            BuildFactorExprsOptions(
                text_query_param_index=2,
                table_alias="m",
                usage_norm_p75=usage_p75,
                emotional_baseline=baseline,
            ),
        )
        factors_block = build_factors_block(row, factor_meta, decay_config=None)

        final_score = float(row.get("final_score") or 0.0)
        is_superseded = row.get("superseded_by") is not None

        not_returned: dict[str, Any] | None = None
        return_reason: str | None = None
        rank: int | None = None

        if is_superseded:
            not_returned = {
                "code": "superseded",
                "description": f"memory superseded by {row['superseded_by']}",
                "superseded_by": str(row["superseded_by"]),
            }
        else:
            rank = self._queries.explain_decompose_rank(
                target_score=final_score,
                query_vector=qvec,
                query_text=query,
                usage_norm_p75=usage_p75,
                emotional_baseline=baseline,
            )
            if min_score is not None and final_score < min_score:
                not_returned = {
                    "code": "below_min_score",
                    "description": (
                        f"final_score {final_score:.3f} < "
                        f"min_score {min_score}"
                    ),
                    "actual_score": final_score,
                    "required_min_score": float(min_score),
                }
            elif rank is not None and rank > top_k_limit:
                not_returned = {
                    "code": "outside_top_k",
                    "description": (
                        f"rank {rank} exceeds top_k limit {top_k_limit}"
                    ),
                    "actual_rank": rank,
                    "top_k_limit": top_k_limit,
                }
            else:
                return_reason = (
                    "top_k_with_min_score"
                    if min_score is not None
                    else "top_k"
                )

        from datetime import datetime, timezone
        return ExplainDecomposeOutcome(
            mode="decompose",
            memory_id=str(row["id"]),
            kind=str(row["kind"]),
            query=query,
            final_score=final_score,
            rank_in_result_set=rank,
            top_k_limit=top_k_limit,
            would_be_returned=not_returned is None,
            return_reason=return_reason,
            not_returned_because=not_returned,
            factors=factors_block,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def explain_lifetime(
        self,
        *,
        memory_id: str,
        include_recall_history: bool = True,
        recall_history_limit: int = 10,
        prune_min_relevance: float | None = None,
    ) -> "ExplainLifetimeOutcome":
        """Lifecycle trace для memory: importance lifecycle, access,
        recall history, co-retrieval links, decay projections."""
        self._require_explain_ready("explain_lifetime")
        assert self._queries is not None
        try:
            mid = uuid.UUID(memory_id)
        except ValueError as exc:
            raise ValueError(
                f"explain_lifetime: memory_id не UUID — {memory_id!r}"
            ) from exc

        row = self._queries.explain_lifetime_main(memory_id=mid)
        if row is None:
            raise LookupError(
                f"explain_lifetime: memory not found: {memory_id}"
            )

        from styx.engine.explain import (
            hex_query_hash_short,
            lifecycle_multiplier,
            lifetime_decay_projections,
            truncate_preview,
        )

        kind = str(row["kind"])
        age_days = float(row["age_days"] or 0.0)
        importance_final_raw = row.get("importance_final")
        importance_final = (
            None
            if importance_final_raw is None
            else float(importance_final_raw)
        )
        importance_provisional = float(row.get("importance_provisional") or 0.5)
        relevance = float(row.get("relevance") or 1.0)

        decay_block = lifetime_decay_projections(
            kind=kind,
            age_days=age_days,
            importance_final=importance_final,
            relevance=relevance,
            decay_config=None,
            prune_min_relevance=prune_min_relevance,
        )

        history_rows: list[dict[str, Any]] | None = None
        if include_recall_history:
            raw_history = self._queries.explain_lifetime_recall_history(
                memory_id=mid, limit=recall_history_limit,
            )
            history_rows = [
                {
                    "matched_at": _iso(r["matched_at"]),
                    "query_hash": hex_query_hash_short(r.get("query_hash")),
                    "match_score": float(r.get("match_score") or 0.0),
                }
                for r in raw_history
            ]

        co_links_raw = self._queries.explain_lifetime_co_retrieval(
            memory_id=mid, limit=20,
        )
        co_links = []
        for link in co_links_raw:
            meta = link.get("metadata") or {}
            last_re = meta.get("last_reinforced") if isinstance(meta, dict) else None
            co_links.append({
                "target_memory_id": str(link["target_id"]),
                "target_preview": truncate_preview(
                    link.get("target_content")
                ),
                "weight": float(link.get("weight") or 1.0),
                "last_reinforced": last_re if last_re else None,
            })

        from datetime import datetime, timezone
        return ExplainLifetimeOutcome(
            mode="lifetime",
            memory_id=str(row["id"]),
            content_preview=truncate_preview(row.get("content")),
            kind=kind,
            agent_id=str(row["agent_id"]),
            visibility=str(row.get("visibility") or "shared"),
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
            age_days=age_days,
            importance={
                "provisional": importance_provisional,
                "final": importance_final,
                "effective": (
                    importance_final
                    if importance_final is not None
                    else importance_provisional
                ),
                "source": (
                    "final" if importance_final is not None else "provisional"
                ),
                "llm_task_status": row.get("llm_task_status"),
                "llm_task_created_at": _iso_or_none(
                    row.get("llm_task_created_at")
                ),
                "llm_task_id": (
                    str(row["llm_task_id"])
                    if row.get("llm_task_id") is not None
                    else None
                ),
                "llm_task_version": (
                    None
                    if row.get("llm_task_version") is None
                    else int(row["llm_task_version"])
                ),
            },
            lifecycle={
                "current_state": row.get("lifecycle") or "fresh",
                "multiplier": lifecycle_multiplier(row.get("lifecycle")),
            },
            access={
                "access_count": int(row.get("access_count") or 0),
                "last_accessed_at": _iso_or_none(row.get("last_accessed_at")),
                "unique_query_count": int(row.get("unique_query_count") or 0),
                "recall_score_sum": float(row.get("recall_score_sum") or 0.0),
                "total_recall_events": int(
                    row.get("total_recall_events") or 0
                ),
                "avg_match_score": (
                    None
                    if row.get("avg_match_score") is None
                    else float(row["avg_match_score"])
                ),
            },
            relevance={
                "current": relevance,
                "started_at": 1.0,
                "growth_pattern": "Hebbian on access",
            },
            usefulness={
                "current": float(row.get("usefulness") or 0.0),
                "last_updated_via": "feedback hook / keyword overlap",
            },
            decay=decay_block,
            recall_history=history_rows,
            co_retrieval_links=co_links,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def explain_topk(
        self,
        *,
        query: str,
        limit: int = 10,
        kinds: list[str] | None = None,
        after: Any | None = None,
        before: Any | None = None,
        min_score: float | None = None,
        include_factors: bool = True,
    ) -> "ExplainTopKOutcome":
        """Top-K с factor breakdown'ом каждого — port memorybox
        `explainTopK` (waves/25 D6)."""
        self._require_explain_ready("explain_topk")
        if self._embedding is None:
            raise RuntimeError(
                "StyxMemoryCore.explain_topk: embedding client не готов"
            )
        assert self._queries is not None and self._conn is not None
        query = query.strip()
        if not query:
            raise ValueError("explain_topk: query пустой")

        try:
            qvec = self._embedding.embed(query)
        except EmbeddingError as exc:
            raise RuntimeError(
                f"explain_topk: embed failed: {exc}"
            ) from exc

        from styx.emotional.baseline import read_baseline_for_scoring
        from styx.engine.explain import build_factors_block, truncate_preview
        from styx.storage.scoring import (
            BuildFactorExprsOptions,
            EmotionalBaseline,
            build_factor_exprs,
        )

        usage_p75 = self._queries.compute_agent_usage_p75()
        baseline_obj = read_baseline_for_scoring(self._conn, self._agent_id)
        baseline = (
            EmotionalBaseline(
                valence=baseline_obj.valence,
                arousal=baseline_obj.arousal,
                dominance=baseline_obj.dominance,
            )
            if baseline_obj is not None
            else None
        )

        rows, total = self._queries.explain_topk(
            query_vector=qvec,
            query_text=query,
            limit=limit,
            kinds=kinds,
            after=after,
            before=before,
            usage_norm_p75=usage_p75,
            emotional_baseline=baseline,
        )

        # min_score post-filter — рейтинг нумеруется относительно
        # отфильтрованного набора (memorybox semantics).
        if min_score is not None:
            rows = [
                r for r in rows
                if float(r.get("final_score") or 0.0) >= min_score
            ]

        factor_meta = build_factor_exprs(
            {"text_query": query},
            BuildFactorExprsOptions(
                text_query_param_index=2,
                table_alias=None,
                usage_norm_p75=usage_p75,
                emotional_baseline=baseline,
            ),
        )

        # llm_tasks статус для importance_factor блока (если показываем).
        llm_task_map: dict[uuid.UUID, dict[str, Any]] = {}
        if include_factors and rows:
            ids = [r["id"] for r in rows if r.get("id") is not None]
            llm_task_map = self._queries.explain_topk_llm_tasks(
                memory_ids=ids,
            )

        items: list[dict[str, Any]] = []
        for idx, r in enumerate(rows):
            mid = r.get("id")
            llm_info = llm_task_map.get(mid, {}) if mid else {}
            enriched = dict(r)
            enriched["llm_task_status"] = llm_info.get("status")
            enriched["llm_task_created_at"] = llm_info.get("created_at")
            item: dict[str, Any] = {
                "memory_id": str(mid) if mid else None,
                "kind": str(r.get("kind") or ""),
                "content_preview": truncate_preview(r.get("content")),
                "final_score": float(r.get("final_score") or 0.0),
                "rank": idx + 1,
                "factors": (
                    build_factors_block(enriched, factor_meta, decay_config=None)
                    if include_factors
                    else None
                ),
            }
            items.append(item)

        from datetime import datetime, timezone
        return ExplainTopKOutcome(
            mode="top_k",
            query=query,
            limit=limit,
            total_candidates_considered=total,
            items=items,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_analytics(self) -> "AnalyticsOutcome":
        """Per-agent counts + global totals + pending indexing.

        Caller-scoped (D3): один агент в `agents`, без `display_name`.
        """
        self._require_explain_ready("get_analytics")
        assert self._queries is not None
        raw = self._queries.analytics_for_agent()
        return AnalyticsOutcome(
            agents=raw["agents"],
            global_totals=raw["global"],
            pending_indexing=raw["pending_indexing"],
        )

    def confirm_usage(
        self, *, memory_ids: list[str],
    ) -> "ConfirmUsageOutcome":
        """Explicit `used_in_output=true` для recall_event'ов памятей.

        Cross-agent guard: memory_id чужого агента → попадает в
        `missing` массив.

        Идемпотентно: повторный call возвращает то же `updated`
        (RETURNING row не зависит от факта изменения значения).
        Дубликаты в input collapsed на Python-стороне.
        """
        self._require_explain_ready("confirm_usage")
        assert self._queries is not None and self._conn is not None
        if not memory_ids:
            raise ValueError("confirm_usage: memory_ids пустой")
        # Dedupe input (D7).
        seen: list[str] = []
        seen_set: set[str] = set()
        for mid in memory_ids:
            if mid in seen_set:
                continue
            seen_set.add(mid)
            seen.append(mid)

        parsed: list[uuid.UUID] = []
        for mid in seen:
            try:
                parsed.append(uuid.UUID(mid))
            except ValueError as exc:
                raise ValueError(
                    f"confirm_usage: memory_id не UUID — {mid!r}"
                ) from exc

        with self._write_lock:
            matched = self._queries.confirm_usage_update(
                memory_ids=parsed,
            )
            self._conn.commit()

        matched_strs = {str(m) for m in matched}
        missing = [m for m in seen if m not in matched_strs]
        return ConfirmUsageOutcome(
            updated=len(matched_strs),
            requested=len(seen),
            missing=missing,
        )

    def _auto_link_after_subjective_write(
        self, memory_id: uuid.UUID, embedding: list[float],
    ) -> None:
        """Auto-link helper для memory_store STORE/SUPERSEDE ветки (волна 18).

        Изолирован в helper'е чтобы оба call-site'а в memory_store
        выглядели одинаково. Не делает commit — caller commit'ит после.
        """
        if self._config is None or self._queries is None:
            return
        from styx.engine.auto_link import auto_link_after_store
        auto_link_after_store(
            self._queries,
            memory_id=memory_id,
            embedding=embedding,
            config=self._config.auto_link_config(),
            agent_id=self._agent_id,
            source="memory_store",
        )

    def _auto_link_dialogue(
        self, memory_id: uuid.UUID, embedding: list[float],
    ) -> None:
        """Auto-link helper для sync_turn dialogue ряда (волна 18).

        В отличие от subjective writers, source = 'sync_turn'. Зовётся
        для каждого user/assistant ряда после embed-after-commit.
        Не делает commit — caller commit'ит после.
        """
        if self._config is None or self._queries is None:
            return
        from styx.engine.auto_link import auto_link_after_store
        auto_link_after_store(
            self._queries,
            memory_id=memory_id,
            embedding=embedding,
            config=self._config.auto_link_config(),
            agent_id=self._agent_id,
            source="sync_turn",
        )

    def _memory_store_routed(
        self,
        *,
        content: str,
        kind: str,
        kind_src: str,
        role: str,
        session_id: uuid.UUID | None,
        metadata: dict[str, Any] | None,
        importance_provisional: float | None,
        routing: "StoreRoutingConfig",
    ) -> "MemoryStoreOutcome":
        """Store-routing path (волна 19): content > limit → documents +
        chunks + tail-memory.

        - Gatekeeper НЕ применяется (D6 в waves/19) — skip/merge ветки
          сломали бы archive_ref consistency.
        - Auto-link применяется для tail-memory через
          _auto_link_after_subjective_write — consistency с design
          intent волны 18.
        - Транзакция: route_long_content + auto-link + commit. На
          embed-fail внутри route_long_content rollback атомарен.
        """
        assert self._queries is not None and self._embedding is not None
        assert self._conn is not None
        from styx.engine.store_routing import route_long_content
        from styx.observability.logging import log_event

        # Если caller передал kind_src='subjective' (default) — конвертируем
        # в 'subjective_tail' для tail-memory. Иначе сохраняем (например,
        # явно пришёл 'dialogue_batch_consolidation' от handler'а).
        tail_kind_src = (
            "subjective_tail" if kind_src == "subjective" else kind_src
        )

        with self._write_lock:
            try:
                result = route_long_content(
                    self._queries,
                    self._embedding,
                    content=content,
                    kind=kind,
                    kind_src=tail_kind_src,
                    role=role,
                    session_id=session_id,
                    metadata=metadata or {},
                    importance_provisional=importance_provisional,
                    config=routing,
                    source="memory_store",
                )
            except Exception:
                self._conn.rollback()
                raise

            self._auto_link_after_subjective_write(
                result.tail_memory_id, result.summary_embedding,
            )
            self._conn.commit()

            log_event(
                log, "store_routing",
                agent_id=str(self._agent_id),
                tail_memory_id=str(result.tail_memory_id),
                document_id=str(result.document_id),
                chunks_count=result.chunks_count,
                content_length=len(content),
                source="memory_store",
            )

        return MemoryStoreOutcome(
            action="store",
            memory_id=str(result.tail_memory_id),
            routed=True,
            document_id=str(result.document_id),
            chunks_count=result.chunks_count,
        )

    # -- tools (recall — волна 7) ----------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        memory_limit = self._recall_config.full.memory_limit
        # Fallback на defaults — get_tool_schemas вызывается до initialize
        # (volna 7 contract; legacy test поверхность).
        if self._config is not None:
            archive_cfg = self._config.search_archive_config()
        else:
            from styx.engine.search_archive import SearchArchiveConfig
            archive_cfg = SearchArchiveConfig()
        return [
            {
                "name": "styx_recall",
                "description": (
                    "Recall up to N memories from long-tier storage by "
                    "semantic+keyword similarity. Returns memories ranked by "
                    "composite score (vector similarity, recency, importance, "
                    "decay, lifecycle). Use when you need context that's "
                    "outside the current conversation window."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Topic, question, or keywords describing what "
                                "you want to recall."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of memories to return "
                                f"(default {memory_limit})."
                            ),
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "styx_reinterpret",
                "description": (
                    "Переосмыслить существующую memory: добавить координату "
                    "смысла, не переписывая историю. memory_id сохраняется, "
                    "граф цел. Применяется когда новое понимание встроилось в "
                    "прежнее. Не для исправления опечаток (используй "
                    "memory_store + supersede) и не для противоречий "
                    "(используй новую memory). Cooldown 24h на memory. Apply "
                    "deferred — переосмысление применится после закрытия "
                    "текущего turn'а (обычно 30-90s)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "UUID memory которую переосмысляешь.",
                        },
                        "new_understanding_text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2400,
                            "description": (
                                "Что добавилось в понимании. 1-3 "
                                "предложения. На русском, в первом лице "
                                "если оригинал в первом лице. "
                                "LLM-handler склеит prev+new в один "
                                "merged_text, не дополнение через запятую."
                            ),
                        },
                        "weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Опц. вес нового понимания при blend "
                                "embedding'ов (default 0.5). 0.5 = "
                                "равноправный микс; больше = новое "
                                "сильнее тянет recall к себе."
                            ),
                        },
                    },
                    "required": ["memory_id", "new_understanding_text"],
                },
            },
            {
                "name": "styx_search_archive",
                "description": (
                    "Pull-channel into long-form archive (documents and "
                    "dialogue history) for this agent. FTS+vector hybrid "
                    "search. Returns stitched document regions, raw chunks, "
                    "or dialogue replies — depending on `scope`. NOT "
                    "auto-injected into context — caller uses results in "
                    "reasoning explicitly. Use for citations, fact-lookups, "
                    "or recovering text that was offloaded from the active "
                    "memory tier."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["documents", "chunks", "dialogue", "all"],
                            "description": (
                                "documents = stitched document regions; "
                                "chunks = individual chunk hits (no stitching); "
                                "dialogue = past user/assistant turns; "
                                "all = fair-share interleave of documents+dialogue."
                            ),
                            "default": "all",
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum results (default "
                                f"{archive_cfg.default_limit}, max "
                                f"{archive_cfg.max_limit})."
                            ),
                            "minimum": 1,
                            "maximum": archive_cfg.max_limit,
                        },
                        "date_from": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO-8601 lower bound (optional).",
                        },
                        "date_to": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO-8601 upper bound (optional).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "styx_dialogue_search",
                "description": (
                    "Search past user/assistant replies from this agent's "
                    "dialogue history. FTS+vector hybrid by default; pure "
                    "vector if `semantic_only=true`. Filter by session_id "
                    "or date range. Diff with styx_search_archive scope="
                    "'dialogue': this tool exposes session_id/before/after "
                    "filters and a pure-vector mode useful when keywords "
                    "don't match the corpus. Use to recover specific past "
                    "exchanges; not auto-injected into context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": (
                                "Optional UUID — restrict to one session."
                            ),
                        },
                        "after": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO-8601 lower bound (optional).",
                        },
                        "before": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO-8601 upper bound (optional).",
                        },
                        "semantic_only": {
                            "type": "boolean",
                            "description": (
                                "If true — pure cosine similarity (default "
                                "false → hybrid FTS+vector)."
                            ),
                            "default": False,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max replies returned (default 10).",
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "styx_dialogue_recent",
                "description": (
                    "Chronological retrieval of past user/assistant replies "
                    "(oldest first). No semantic search — pure ordering by "
                    "time. Use to reconstruct what was said recently in this "
                    "or another session. Filter by session_id (one session) "
                    "or before (cutoff timestamp). Replies of role "
                    "tool/system/summary are excluded."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": (
                                "Optional UUID — restrict to one session."
                            ),
                        },
                        "before": {
                            "type": "string",
                            "format": "date-time",
                            "description": (
                                "Optional ISO-8601 cutoff — exclude replies "
                                "after this timestamp."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max replies returned (default 20).",
                            "minimum": 1,
                            "maximum": 200,
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "styx_dialogue_prepare_summary",
                "description": (
                    "Build a chronological transcript of one session for "
                    "summarization. Returns formatted lines `[YYYY-MM-DD "
                    "HH:MM:SS] Human/Agent: content` plus message_count and "
                    "first/last timestamps. Use when you want to summarize "
                    "or analyze a past session. Empty session — empty "
                    "transcript, message_count=0 (not an error)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": (
                                "UUID of the session to prepare. Required."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Max replies (default 200, max 1000). LLM "
                                "should chunk if session is huge."
                            ),
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "styx_ingest_document",
                "description": (
                    "Archive a document file (PDF, DOCX, XLSX, Markdown, "
                    "plain text) into Styx for later hybrid retrieval via "
                    "styx_search_archive. Path-mode: caller passes an "
                    "absolute path; core reads the file, parses it, chunks, "
                    "embeds, and stores document + chunks. Does NOT inject "
                    "into recall — pull-only archive. Idempotent by SHA256 "
                    "of file bytes: repeated calls return existing "
                    "document_id with deduplicated=true. Use when the user "
                    "attaches a file or asks you to read a document; not "
                    "for short pasted text (use styx_store or "
                    "styx_ingest_experience)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Absolute path to the file. Supported "
                                "extensions: .pdf, .docx, .xlsx, .md, "
                                ".markdown, .txt, .text."
                            ),
                            "minLength": 1,
                        },
                        "source_ref": {
                            "type": "string",
                            "description": (
                                "Optional source reference (URL, ticket "
                                "id, channel:message) for traceability."
                            ),
                            "maxLength": 512,
                        },
                        "visibility": {
                            "type": "string",
                            "description": (
                                "Optional visibility label "
                                "('private'/'shared'). Cosmetic in wave 28."
                            ),
                            "maxLength": 32,
                        },
                        "metadata": {
                            "type": "object",
                            "description": (
                                "Arbitrary metadata stored in "
                                "documents.metadata JSONB."
                            ),
                        },
                        "content_hash": {
                            "type": "string",
                            "description": (
                                "Optional explicit hash override; core "
                                "computes SHA256(file_bytes) otherwise."
                            ),
                            "maxLength": 256,
                        },
                    },
                    "required": ["path"],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        if tool_name == "styx_search_archive":
            return self._handle_search_archive(args)
        if tool_name == "styx_reinterpret":
            return self._handle_reinterpret(args)
        if tool_name == "styx_dialogue_search":
            return self._handle_dialogue_search(args)
        if tool_name == "styx_dialogue_recent":
            return self._handle_dialogue_recent(args)
        if tool_name == "styx_dialogue_prepare_summary":
            return self._handle_dialogue_prepare_summary(args)
        if tool_name != "styx_recall":
            return json.dumps({"error": f"unknown tool: {tool_name}"})

        if self._queries is None or self._embedding is None:
            return json.dumps(
                {"error": "styx_recall called before initialize"}
            )

        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "styx_recall: query is required"})

        limit_arg = args.get("limit")
        full_cfg = self._recall_config.full
        if isinstance(limit_arg, int) and 1 <= limit_arg <= 20:
            from dataclasses import replace
            full_cfg = replace(full_cfg, memory_limit=limit_arg)

        session_id_arg = kwargs.get("session_id") or ""
        session_uuid = _coerce_session_id(session_id_arg) if session_id_arg else None
        session_str = str(session_uuid) if session_uuid else None

        # Волна 14 (10a): snapshot fence. observe() либо открывает turn
        # (первый recall в активном turn'е), либо возвращает existing
        # cycle_start (sticky). Если build_salient_block в compress'е
        # уже observe'нул — мы получим тот же snapshot.
        snapshot = turn_state.observe(self._agent_id)

        with self._write_lock:
            if self._queries is None or self._embedding is None:
                return json.dumps({"error": "provider shut down mid-call"})
            result = recall_full(
                queries=self._queries,
                embed_client=self._embedding,
                query=query,
                full_config=full_cfg,
                session_id=session_str,
                snapshot=snapshot,
            )

            # Hebbian co-retrieval reinforcement (волна 21).
            # На всех C(N, 2) парах top-K results bump'ит weight ребра
            # 'co_retrieved'. Sync (D2 в waves/21): один UPSERT per pair,
            # ~5-10ms на K=10. В той же транзакции что recall_event'ы
            # из recall_full → коммитим всё вместе.
            if (
                self._config is not None
                and self._config.hebbian_enabled
                and len(result.memories) >= 2
            ):
                from styx.engine.hebbian import reinforce_co_retrieval
                try:
                    reinforce_co_retrieval(
                        self._queries,
                        memory_ids=[h.id for h in result.memories],
                        config=self._config.hebbian_config(),
                        agent_id=self._agent_id,
                    )
                except Exception as exc:  # noqa: BLE001 — fail-open
                    log.warning(
                        "hebbian reinforcement упал: %s", exc,
                    )
                    try:
                        self._conn.rollback()  # type: ignore[union-attr]
                    except Exception:
                        pass

            self._conn.commit()  # type: ignore[union-attr]

        # Запомнить recall_event_ids для последующего classifier'а
        # (волна 7c). Тащим все ids — sync_turn возьмёт last 20.
        if self._session_id is not None:
            ids = [
                hit.recall_event_id
                for hit in result.memories
                if hit.recall_event_id is not None
            ]
            if ids:
                self._recall_tracker.append(self._session_id, ids)

        text = format_recall_text(result)
        return json.dumps(
            {
                "memories_text": text,
                "count": len(result.memories),
                "queried_count": result.queried_count,
                "duplicates_removed": result.internal_duplicates_removed,
            }
        )

    def _handle_search_archive(self, args: dict[str, Any]) -> str:
        """Tool dispatch для styx_search_archive (волна 20).

        in-process путь — для callers, которые не ходят через HTTP
        (например OpenClaw plugin in same process). Hermes plugin
        идёт по client.search_archive напрямую через HTTP, минуя core
        handle_tool_call (паттерн как со styx_recall)."""
        if self._queries is None or self._embedding is None or self._config is None:
            return json.dumps(
                {"error": "styx_search_archive called before initialize"}
            )

        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps(
                {"error": "styx_search_archive: query is required"}
            )

        scope = args.get("scope") or "all"
        if scope not in ("documents", "chunks", "dialogue", "all"):
            return json.dumps(
                {"error": f"styx_search_archive: invalid scope {scope!r}"}
            )

        limit_arg = args.get("limit")
        limit: int | None = None
        if isinstance(limit_arg, int) and limit_arg > 0:
            limit = limit_arg

        from styx.engine import search_archive as _engine

        cfg = self._config.search_archive_config()
        common = dict(
            queries=self._queries,
            embedder=self._embedding,
            query=query,
            limit=limit,
            config=cfg,
        )
        if scope == "documents":
            resp = _engine.search_documents(**common)
        elif scope == "chunks":
            resp = _engine.search_chunks(**common)
        elif scope == "dialogue":
            resp = _engine.search_dialogue(**common)
        else:
            resp = _engine.search_all(**common)

        return json.dumps(
            {
                "results": [
                    {
                        "scope": r.scope,
                        "text": r.text,
                        "snippet": r.snippet,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_position": r.chunk_position,
                        "chunk_positions": (
                            list(r.chunk_positions) if r.chunk_positions else None
                        ),
                        "char_start": r.char_start,
                        "char_end": r.char_end,
                        "memory_id": r.memory_id,
                        "role": r.role,
                        "created_at": r.created_at,
                    }
                    for r in resp.results
                ],
                "total_matched": resp.total_matched,
            }
        )

    def _handle_reinterpret(self, args: dict[str, Any]) -> str:
        """Tool dispatch для styx_reinterpret (волна 22).

        in-process путь — для callers без HTTP. Hermes plugin идёт
        через client.reinterpret напрямую (см. providers/memory в
        styx-hermes), минуя core handle_tool_call.
        """
        if (
            self._queries is None
            or self._config is None
            or self._conn is None
        ):
            return json.dumps(
                {"error": "styx_reinterpret called before initialize"}
            )
        if not self._config.reinterpret_enabled:
            return json.dumps(
                {"error": "styx_reinterpret disabled"}
            )

        memory_id = args.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            return json.dumps(
                {"error": "styx_reinterpret: memory_id is required"}
            )
        text = args.get("new_understanding_text")
        if not isinstance(text, str) or not text:
            return json.dumps(
                {
                    "error": (
                        "styx_reinterpret: new_understanding_text is required"
                    )
                }
            )
        weight = args.get("weight")
        if weight is not None and not isinstance(weight, (int, float)):
            return json.dumps(
                {"error": "styx_reinterpret: weight must be number"}
            )

        try:
            outcome = self.reinterpret_enqueue(
                memory_id=memory_id,
                new_understanding_text=text,
                weight=float(weight) if weight is not None else None,
            )
        except (ValueError, RuntimeError) as exc:
            return json.dumps(
                {"error": f"styx_reinterpret: {exc}"}
            )

        # Сворачиваем outcome в structured tool response.
        out: dict[str, Any] = {"status": outcome.status}
        if outcome.memory_id is not None:
            out["memory_id"] = outcome.memory_id
        if outcome.task_id is not None:
            out["task_id"] = outcome.task_id
        if outcome.application_id is not None:
            out["application_id"] = outcome.application_id
        if outcome.last_reinterpreted_at is not None:
            out["last_reinterpreted_at"] = outcome.last_reinterpreted_at
        if outcome.next_available_at is not None:
            out["next_available_at"] = outcome.next_available_at
        if outcome.pending_application_id is not None:
            out["pending_application_id"] = outcome.pending_application_id
        if outcome.status == "queued":
            out["message"] = (
                "переосмысление поставлено в очередь, применится "
                "после ~30-60s"
            )
        return json.dumps(out)

    # -- dialogue tool dispatch (волна 24 follow-up) --------------------

    def _handle_dialogue_search(self, args: dict[str, Any]) -> str:
        """Tool dispatch для styx_dialogue_search.

        in-process путь через `dialogue_search`. Hermes plugin тоже
        ходит через client.dialogue_search HTTP, минуя core
        handle_tool_call — паттерн как со styx_search_archive.
        """
        if (
            self._queries is None
            or self._embedding is None
            or self._config is None
        ):
            return json.dumps(
                {"error": "styx_dialogue_search called before initialize"}
            )
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return json.dumps(
                {"error": "styx_dialogue_search: query is required"}
            )
        limit_arg = args.get("limit")
        limit = (
            int(limit_arg)
            if isinstance(limit_arg, int) and 1 <= limit_arg <= 50
            else 10
        )
        try:
            results = self.dialogue_search(
                query=query,
                session_id=args.get("session_id"),
                after=args.get("after"),
                before=args.get("before"),
                semantic_only=bool(args.get("semantic_only", False)),
                limit=limit,
            )
        except (ValueError, RuntimeError) as exc:
            return json.dumps(
                {"error": f"styx_dialogue_search: {exc}"}
            )
        return json.dumps({
            "results": [
                {
                    "memory_id": r.memory_id,
                    "role": r.role,
                    "content": r.content,
                    "score": r.score,
                    "created_at": (
                        r.created_at.isoformat()
                        if hasattr(r.created_at, "isoformat")
                        else r.created_at
                    ),
                    "session_id": r.session_id,
                }
                for r in results
            ],
        })

    def _handle_dialogue_recent(self, args: dict[str, Any]) -> str:
        """Tool dispatch для styx_dialogue_recent."""
        if self._queries is None or self._config is None:
            return json.dumps(
                {"error": "styx_dialogue_recent called before initialize"}
            )
        limit_arg = args.get("limit")
        limit = (
            int(limit_arg)
            if isinstance(limit_arg, int) and 1 <= limit_arg <= 200
            else 20
        )
        try:
            rows = self.dialogue_recent(
                session_id=args.get("session_id"),
                before=args.get("before"),
                limit=limit,
            )
        except (ValueError, RuntimeError) as exc:
            return json.dumps(
                {"error": f"styx_dialogue_recent: {exc}"}
            )
        return json.dumps({
            "rows": [
                {
                    "memory_id": r.memory_id,
                    "role": r.role,
                    "content": r.content,
                    "created_at": (
                        r.created_at.isoformat()
                        if hasattr(r.created_at, "isoformat")
                        else r.created_at
                    ),
                    "session_id": r.session_id,
                }
                for r in rows
            ],
        })

    def _handle_dialogue_prepare_summary(self, args: dict[str, Any]) -> str:
        """Tool dispatch для styx_dialogue_prepare_summary."""
        if self._queries is None or self._config is None:
            return json.dumps(
                {"error": "styx_dialogue_prepare_summary called before initialize"}
            )
        session_id = args.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return json.dumps(
                {"error": "styx_dialogue_prepare_summary: session_id is required"}
            )
        limit_arg = args.get("limit")
        limit = (
            int(limit_arg)
            if isinstance(limit_arg, int) and 1 <= limit_arg <= 1000
            else 200
        )
        try:
            outcome = self.dialogue_prepare_summary(
                session_id=session_id, limit=limit,
            )
        except (ValueError, RuntimeError) as exc:
            return json.dumps(
                {"error": f"styx_dialogue_prepare_summary: {exc}"}
            )
        return json.dumps({
            "session_id": outcome.session_id,
            "message_count": outcome.message_count,
            "first_message_at": (
                outcome.first_message_at.isoformat()
                if outcome.first_message_at is not None
                and hasattr(outcome.first_message_at, "isoformat")
                else outcome.first_message_at
            ),
            "last_message_at": (
                outcome.last_message_at.isoformat()
                if outcome.last_message_at is not None
                and hasattr(outcome.last_message_at, "isoformat")
                else outcome.last_message_at
            ),
            "transcript": outcome.transcript,
        })

    # -- setup wizard ----------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "database_url",
                "description": "PostgreSQL DSN (postgresql://user:pwd@host:port/db)",
                "secret": True,
                "required": True,
                "env_var": "STYX_DATABASE_URL",
            },
            {
                "key": "ollama_url",
                "description": "Ollama endpoint для эмбеддингов",
                "default": "http://ollama:11434",
            },
            {
                "key": "embedding_model",
                "description": "Имя embedding-модели в Ollama",
                "default": "embeddinggemma:300m-qat-q8_0",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        import json
        from pathlib import Path

        from styx.config import CONFIG_FILENAME

        path = Path(hermes_home) / CONFIG_FILENAME
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        # Фикс #11: DSN содержит пароль — файл должен быть читаем только владельцем.
        os.chmod(path, 0o600)

    # -- introspection (для тестов / отладки) ---------------------------

    @property
    def queries(self) -> AgentScopedQueries:
        with self._write_lock:
            if self._queries is None:
                raise RuntimeError("provider не инициализирован")
            return self._queries


# Hermes session_id — произвольная строка (например, '20260430_043717_6fb431').
# Наша таблица sessions.id — uuid PK. UUID5 даёт детерминированный,
# стабильный mapping любой строки в UUID, без привязки к UUID-формату на стороне
# Hermes. Использует NAMESPACE_DNS — это произвольный namespace, важна только
# стабильность.
_STYX_SESSION_NAMESPACE = uuid.NAMESPACE_DNS


def _iso(value: Any) -> str:
    """Datetime-like → ISO8601 string. None → пустая строка."""
    if value is None:
        return ""
    try:
        return value.isoformat()  # type: ignore[no-any-return]
    except AttributeError:
        return str(value)


def _iso_or_none(value: Any) -> str | None:
    """Datetime-like → ISO8601 string. None → None."""
    if value is None:
        return None
    try:
        return value.isoformat()  # type: ignore[no-any-return]
    except AttributeError:
        return str(value)


def _coerce_session_id(value: str | None) -> uuid.UUID | None:
    """Конвертирует Hermes session_id (строка) в стабильный UUID.

    - Пустое/None → None
    - Уже UUID-строка → uuid.UUID(value)
    - Произвольная строка → uuid.uuid5(_STYX_SESSION_NAMESPACE, value)
    """
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(_STYX_SESSION_NAMESPACE, value)


# ── memory_store helpers (волна 17) ─────────────────────────────────


@dataclass(frozen=True)
class MemoryStoreOutcome:
    """Результат /memory_store + StyxMemoryCore.memory_store.

    - ``action='store'``     — memory создан, ``memory_id`` set.
    - ``action='merge'``     — поглощён существующим, ``memory_id=None``,
                               ``existing_id`` указывает на сохранившийся ряд.
    - ``action='supersede'`` — новый создан с ``superseded_by``-связью на
                               старого; ``memory_id`` set, ``existing_id`` —
                               supersededный ряд.
    - ``action='skip'``      — отсечено noise filter'ом, ``memory_id=None``.

    При store-routing'е (волна 19, len(content) > store_routing_limit):
    - ``routed=True`` → tail-memory создан и связан с document'ом.
    - ``memory_id`` указывает на tail-memory.
    - ``document_id`` / ``chunks_count`` — реквизиты archive-стороны.
    Action остаётся ``'store'`` (gatekeeper для tail-memory не
    применяется — D6 в waves/19).
    """

    action: str  # store | merge | supersede | skip
    memory_id: str | None = None
    existing_id: str | None = None
    similarity: float | None = None
    routed: bool = False
    document_id: str | None = None
    chunks_count: int | None = None


# ── reinterpret enqueue outcome (волна 22) ─────────────────────────


@dataclass(frozen=True)
class ReinterpretEnqueueOutcome:
    """Результат `StyxMemoryCore.reinterpret_enqueue` (HTTP route +
    in-process Hermes-tool).

    `status` discriminator — один из:
    - ``queued``           — task поставлен; task_id + application_id populated.
    - ``cooldown``         — последняя revision < 24h; last_at/next_at set.
    - ``already_pending``  — есть pending_sleep application; pending_application_id set.
    - ``memory_not_found`` — memory нет под этим agent_id.
    """

    status: str
    memory_id: str | None = None
    task_id: str | None = None
    application_id: int | None = None
    last_reinterpreted_at: str | None = None
    next_available_at: str | None = None
    pending_application_id: int | None = None


# ── ingest experience outcome (волна 23) ───────────────────────────


@dataclass(frozen=True)
class IngestOutcome:
    """Результат ``StyxMemoryCore.ingest_experience``.

    - ``deduplicated=False`` — новый ряд создан, ``memory_id`` свежий.
    - ``deduplicated=True``  — повторный ingest того же payload'а от
                               того же агента; ``memory_id`` указывает
                               на existing ряд (без побочных эффектов).

    ``used_hash`` — content_hash, который реально использовался при
    INSERT'е (explicit | auto-computed | None если идемпотентность не
    применима). Полезно pipeline'у для логирования / verification.
    """

    memory_id: str
    deduplicated: bool
    used_hash: str | None


@dataclass(frozen=True)
class IngestDocumentOutcome:
    """Результат ``StyxMemoryCore.ingest_document`` (волна 28).

    Pipeline channel — НЕ создаёт tail-memory (D5 в waves/28). Документ
    доступен через ``search_archive`` (pull-only).

    - ``deduplicated=False`` — новый document INSERT'нут, chunks
      посчитаны, embedding'и сохранены.
    - ``deduplicated=True``  — повторный ingest того же файла (matched
      SHA256 content_hash); ``document_id`` указывает на existing
      ряд, parser не вызывался, chunks_count=0.

    ``content_hash`` — SHA256 от file bytes либо explicit hash из
    request (D9 в waves/28).
    """

    document_id: str
    deduplicated: bool
    chunks_count: int
    mime_type: str
    original_name: str
    size_bytes: int
    char_count: int
    content_hash: str


# ── dialogue outcomes (волна 24) ────────────────────────────────────


@dataclass(frozen=True)
class DialogueSearchOutcome:
    memory_id: str
    role: str
    content: str
    score: float
    created_at: Any
    session_id: str | None


@dataclass(frozen=True)
class DialogueRecentOutcome:
    memory_id: str
    role: str
    content: str
    created_at: Any
    session_id: str | None


@dataclass(frozen=True)
class DialogueSessionOutcome:
    session_id: str
    message_count: int
    first_message_at: Any
    last_message_at: Any


@dataclass(frozen=True)
class DialogueSummaryOutcome:
    session_id: str
    message_count: int
    first_message_at: Any
    last_message_at: Any
    transcript: str


# ── explain / analytics / confirm_usage outcomes (волна 25) ─────────


@dataclass(frozen=True)
class ExplainDecomposeOutcome:
    """Результат ``StyxMemoryCore.explain_decompose``.

    ``factors`` — FactorsBlock dict (engine.explain.build_factors_block).
    ``not_returned_because`` либо ``return_reason`` — взаимоисключающие.
    """

    mode: str
    memory_id: str
    kind: str
    query: str
    final_score: float
    rank_in_result_set: int | None
    top_k_limit: int
    would_be_returned: bool
    return_reason: str | None
    not_returned_because: dict[str, Any] | None
    factors: dict[str, Any]
    computed_at: str


@dataclass(frozen=True)
class ExplainLifetimeOutcome:
    mode: str
    memory_id: str
    content_preview: str
    kind: str
    agent_id: str
    visibility: str
    created_at: str
    updated_at: str
    age_days: float
    importance: dict[str, Any]
    lifecycle: dict[str, Any]
    access: dict[str, Any]
    relevance: dict[str, Any]
    usefulness: dict[str, Any]
    decay: dict[str, Any]
    recall_history: list[dict[str, Any]] | None
    co_retrieval_links: list[dict[str, Any]]
    computed_at: str


@dataclass(frozen=True)
class ExplainTopKOutcome:
    mode: str
    query: str
    limit: int
    total_candidates_considered: int
    items: list[dict[str, Any]]
    computed_at: str


@dataclass(frozen=True)
class AnalyticsOutcome:
    """Per-agent counts + global totals + pending indexing."""

    agents: list[dict[str, Any]]
    global_totals: dict[str, Any]
    pending_indexing: dict[str, Any]


@dataclass(frozen=True)
class ConfirmUsageOutcome:
    updated: int
    requested: int
    missing: list[str]


# Все subjective kinds → role='summary'. CHECK memories_role_check
# разрешает {user, assistant, tool, system, summary}; для subjective
# memories 'summary' семантически ближе всего (агентская заметка о
# действительности, не реплика и не tool output). insert_batch_memory
# использует то же mapping для episode-summaries.
_VALID_KINDS: frozenset[str] = frozenset({
    "fact", "episode", "decision", "concept", "note",
})


def _role_for_kind(kind: str) -> str:
    """Validate kind и вернуть role для subjective write'а (волна 17).

    Пять валидных kinds (CHECK constraint memories_kind_check):
    fact / episode / decision / concept / note. Все маппятся на
    role='summary' — единственный CHECK-разрешённый role не для
    dialogue capture / tool output / system-bootstrap.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"memory_store: неизвестный kind={kind!r}")
    return "summary"


def _build_recall_config(config: StyxConfig) -> RecallConfig:
    """Собирает RecallConfig поверх дефолтов из StyxConfig override-полей.

    None в config = «не override», берётся из DEFAULT_RECALL_CONFIG.
    Применяется в initialize; до initialize self._recall_config держит
    дефолт для get_tool_schemas.
    """
    partial: dict[str, dict[str, float]] = {}
    if config.recall_min_score is not None:
        partial.setdefault("full", {})["min_score"] = config.recall_min_score
    if config.recall_dialogue_min_score is not None:
        partial.setdefault("companion", {}).setdefault("dialogue", {})[
            "min_score"
        ] = config.recall_dialogue_min_score
    return resolve_recall_config(partial) if partial else DEFAULT_RECALL_CONFIG
