# Configuration

Полная карта ENV-переменных Styx. Все toggle'ы безопасно отключаются
(default — production-ready); пороги откалиброваны на
`embeddinggemma:300m-qat-q8_0` (768-dim) + `qwen3:4b-local`.

В таблицах ниже значение `default` — то, что применяется если переменная
не задана. Empty/unset поведение разное для разных типов:
- `bool` toggle'ы: `=0` → off, `=1`/unset → on (если default on).
- `int` / `float`: unset → default. Пустая строка обычно эквивалентна
  unset.

## Connectivity

| Var | Default | Назначение |
|---|---|---|
| `STYX_DATABASE_URL` | (required) | PostgreSQL DSN. Без него daemon не стартует. |
| `DATABASE_URL` | — | Fallback для `STYX_DATABASE_URL`. |
| `STYX_OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint для embeddings. |
| `STYX_LLM_URL` | = `STYX_OLLAMA_URL` | Endpoint chat LLM для workers (qwen3 на той же Ollama). |
| `STYX_HTTP_BIND` | `127.0.0.1` | Address binding daemon'а. |
| `STYX_HTTP_PORT` | `8788` | Port HTTP API. |
| `STYX_HTTP_TOKEN` | — | Bearer-token для всех non-healthz routes. **Обязателен** если `STYX_HTTP_BIND` не loopback (daemon откажется стартовать иначе). |
| `STYX_DAEMON_URL` | `http://127.0.0.1:8788` | URL daemon'а для plugin'ов. |
| `STYX_PG_POOL_MIN` | `1` | Минимум connection'ов в pool. |
| `STYX_PG_POOL_MAX` | `4` | Максимум connection'ов. |
| `STYX_SESSION_NAMESPACE` | `default` | Префикс для derived `agent_id` (изоляция инсталляций). |

## Embeddings

| Var | Default | Назначение |
|---|---|---|
| `STYX_EMBEDDING_MODEL` | `embeddinggemma:300m-qat-q8_0` | Имя модели в Ollama. |
| `STYX_EMBEDDING_DIM` | `768` | Размерность вектора. Должна совпадать с моделью. |
| `STYX_EMBEDDING_TIMEOUT_S` | `30` | Таймаут embed-запроса. |

## LLM workers

| Var | Default | Назначение |
|---|---|---|
| `STYX_LLM_MODEL` | `qwen3:4b-local` | Модель для importance / classifier / consolidation handlers. |
| `STYX_LLM_TIMEOUT_S` | `60` | Таймаут chat-запроса. |
| `STYX_LLM_MAX_ATTEMPTS` | `2` | Retry на transient errors. |
| `STYX_LLM_RATE_LIMIT_CAPACITY` | `4` | Token-bucket capacity (защита Ollama от наплыва). |
| `STYX_LLM_RATE_LIMIT_REFILL_PER_S` | `1.0` | Refill rate token-bucket'а. |

## Recall

| Var | Default | Назначение |
|---|---|---|
| `STYX_RECALL_MIN_SCORE` | `0.32` | Composite-score cutoff в `recall_full` (откалиброван под embeddinggemma). |
| `STYX_RECALL_DIALOGUE_MIN_SCORE` | `0.6` | Отдельный порог для companion-mode dialogue recall (зарезервировано). |

## Salient injection (ContextEngine)

| Var | Default | Назначение |
|---|---|---|
| `STYX_SALIENT_ENABLED` | on | `=0` отключает inject салиента в `compress()` / `assemble()`. Bridge не configure'ится, payload возвращается head+tail. |
| `STYX_SALIENT_TIMEOUT_S` | `1.0` | Общий timeout на recall внутри `build_salient_block`. Превышение → skip salient (fail-open). |
| `STYX_SALIENT_MIN_QUERY_LEN` | `20` | Минимум длины last user message для запуска recall'а. |

## Drift detection (focus tracker)

| Var | Default | Назначение |
|---|---|---|
| `STYX_DRIFT_ENABLED` | on | `=0` отключает кэш salient'а — fresh recall каждый turn. |
| `STYX_DRIFT_THRESHOLD` | `0.4` | Cosine ниже = drift detected. |
| `STYX_FOCUS_WINDOW_SIZE` | `3` | K последних user-embed'ов в sliding centroid'е. |

## Pre-LLM inject (Hermes hook)

| Var | Default | Назначение |
|---|---|---|
| `STYX_PRE_LLM_INJECT_ENABLED` | on | `=0` отключает hook целиком. |
| `STYX_PRE_LLM_PEER_VAD_ENABLED` | on | `=0` отключает только peer VAD channel. |
| `STYX_PEER_VAD_MIN_NORM` | `0.2` | VAD с нормой ниже считается слишком нейтральным — skip. |
| `STYX_PEER_VAD_TTL_S` | `60.0` | Максимальный возраст hot_sentiment записи для inject'а. |

## Sentiment / emotional baseline

| Var | Default | Назначение |
|---|---|---|
| `STYX_SENTIMENT_ENABLED` | on | `=0` отключает hot-path VAD-extraction в `sync_turn`. |
| `STYX_SENTIMENT_TIMEOUT_S` | `0.8` | Timeout `extract_vad` в hot-path. |
| `STYX_EMOTIONAL_TICK_INTERVAL_S` | `60` | Интервал periodic-task `emotional_tick` (decay + baseline EMA). |
| `STYX_BATCH_SENTIMENT_ENABLED` | on | `=0` отключает apply batch sentiment в `dialogue_batch_consolidation`. |

## Classifier (post-hoc usage)

| Var | Default | Назначение |
|---|---|---|
| `STYX_CLASSIFIER_MIN_ASSISTANT_LENGTH` | `50` | Ниже этой длины assistant_content classifier не enqueue'ится. |
| `STYX_CLASSIFIER_MAX_RECALL_EVENTS_PER_TURN` | `20` | Лимит ids в payload одного classifier-task'а. |

## Hot-tier

| Var | Default | Назначение |
|---|---|---|
| `STYX_HOT_TIER_ENABLED` | on | `=0` отключает hot-tier — recall работает только с БД. |
| `STYX_HOT_TIER_TTL_S` | `300` | TTL записи в hot (5 мин). |
| `STYX_HOT_TIER_LRU_BOUND` | `100` | Максимум items в hot per process. |

## Eviction relevance

| Var | Default | Назначение |
|---|---|---|
| `STYX_EVICTION_RELEVANCE_ENABLED` | on | `=0` — eviction чисто recency-based (head + tail без middle). |
| `STYX_EVICTION_RELEVANCE_KEEP_K` | `2` | Сколько top pair-групп из middle сохраняется по cosine к focus centroid'у. |
| `STYX_EVICTION_RELEVANCE_THRESHOLD` | `0.4` | Floor cosine — группы ниже отбрасываются (симметрично drift threshold'у). |

## Working set persistence

| Var | Default | Назначение |
|---|---|---|
| `STYX_WORKING_SET_PERSISTENCE_ENABLED` | on | `=0` — focus_tracker / hot_tier не load'ятся при initialize, save-thread не стартует. |
| `STYX_WORKING_SET_SAVE_INTERVAL_S` | `30` | Период save tick в background thread'е. |
| `STYX_WORKING_SET_TTL_S` | `86400` | TTL persisted state'а — past возраста state дропается на load (cold start). |
| `STYX_TURN_STATE_TTL_S` | `60.0` | TTL inactivity для temporal isolation turn'а (safety net). |

## Selective gatekeeper (subjective writes)

| Var | Default | Назначение |
|---|---|---|
| `STYX_SELECTIVE_ENABLED` | on | `=0` — каждый writer пишет напрямую (action=store, без skip/merge/supersede). |
| `STYX_SELECTIVE_MERGE_THRESHOLD` | `0.92` | Cosine ≥ → merge в existing. |
| `STYX_SELECTIVE_SUPERSEDE_THRESHOLD` | `0.85` | Cosine ≥ + Levenshtein ratio > threshold → supersede. |
| `STYX_SELECTIVE_LEVENSHTEIN_THRESHOLD` | `0.3` | Минимум Levenshtein ratio для supersede ветки. |
| `STYX_SELECTIVE_NOISE_FILTER` | on | `=0` отключает фильтр коротких писаний. |
| `STYX_SELECTIVE_NOISE_MIN_LENGTH` | `10` | Content < этой длины → skip (если noise filter on). |

## Auto-link

| Var | Default | Назначение |
|---|---|---|
| `STYX_AUTO_LINK_ENABLED` | on | `=0` — никаких `related_to` рёбер при INSERT. |
| `STYX_AUTO_LINK_MAX_DISTANCE` | `0.25` | Cosine distance ≤ (= similarity ≥ 0.75) — кандидат для auto-link. |
| `STYX_AUTO_LINK_MAX_LINKS` | `3` | До скольких соседей подключаем на один INSERT. |

## Hebbian co-retrieval + relation decay

| Var | Default | Назначение |
|---|---|---|
| `STYX_HEBBIAN_ENABLED` | on | `=0` отключает создание `co_retrieved` рёбер при recall. |
| `STYX_HEBBIAN_INITIAL_WEIGHT` | `1.1` | Начальный weight нового co_retrieved ребра. |
| `STYX_HEBBIAN_WEIGHT_BUMP` | `0.1` | На сколько повышается weight при повторном recall'е. |
| `STYX_HEBBIAN_WEIGHT_MAX` | `2.0` | Cap weight'а. |
| `STYX_RELATION_DECAY_ENABLED` | on | `=0` отключает periodic decay. |
| `STYX_RELATION_DECAY_INTERVAL_S` | `3600` | Интервал sweep'а (1 час). |
| `STYX_RELATION_DECAY_RATE` | `0.05` | На сколько падает weight idle-ребра за tick. |
| `STYX_RELATION_DECAY_IDLE_DAYS` | `14` | Сколько дней без reinforce → idle. |

## Reinterpret

| Var | Default | Назначение |
|---|---|---|
| `STYX_REINTERPRET_ENABLED` | on | `=0` отключает route + apply-sweeper. |
| `STYX_REINTERPRET_APPLY_TICK_S` | `30` | Period sweeper'а. |
| `STYX_REINTERPRET_COOLDOWN_S` | `86400` | Cooldown на memory (24h). |
| `STYX_REINTERPRET_BLEND_WEIGHT` | `0.5` | Default weight для embedding blend. |

## Memory consolidation (cluster N→1)

| Var | Default | Назначение |
|---|---|---|
| `STYX_MEMORY_CONSOLIDATION_ENABLED` | on | `=0` отключает scheduler + apply-sweeper. |
| `STYX_MEMORY_CONSOLIDATION_TICK_S` | `3600` | Интервал scheduler tick'а. |
| `STYX_MEMORY_CONSOLIDATION_APPLY_TICK_S` | `30` | Интервал apply-sweeper'а. |
| `STYX_MEMORY_CONSOLIDATION_COSINE` | `0.88` | Cluster cosine threshold. |
| `STYX_MEMORY_CONSOLIDATION_MIN_SIZE` | `3` | Минимум memories в cluster'е. |
| `STYX_MEMORY_CONSOLIDATION_MAX_SIZE` | `8` | Максимум. |
| `STYX_MEMORY_CONSOLIDATION_WINDOW_DAYS` | `7` | Окно поиска кандидатов. |
| `STYX_MEMORY_CONSOLIDATION_WINDOW_TAIL_H` | `24` | Дополнительный tail-window. |
| `STYX_MEMORY_CONSOLIDATION_COOLDOWN_H` | `23` | Cooldown между consolidation'ами одного агента. |

## Dialogue batch consolidation

| Var | Default | Назначение |
|---|---|---|
| `STYX_BATCH_CONSOLIDATION_ENABLED` | on | `=0` отключает scheduler. |
| `STYX_BATCH_TICK_INTERVAL_S` | `60` | Период scheduler tick'а. |
| `STYX_BATCH_TRIGGER_MESSAGES` | `20` | Порог новых user/assistant memories для триггера batch'а. |
| `STYX_BATCH_TRIGGER_INTERVAL_S` | `1200` | Минимум секунд между двумя batch'ами одного агента (20 мин). |
| `STYX_BATCH_PERIOD_GAP_S` | `21600` | Пауза с последней реплики, после которой batch идёт без overlap'а (6 ч). |

## Lifecycle sweep

| Var | Default | Назначение |
|---|---|---|
| `STYX_SWEEP_INTERVAL_S` | `3600` | Интервал consolidation sweep (1 час). |
| `STYX_SWEEP_LOCK_TIMEOUT_S` | `1800` | Per-task statement_timeout (защита от длинных запросов). |

## Store-routing (long content → documents+chunks)

| Var | Default | Назначение |
|---|---|---|
| `STYX_STORE_ROUTING_ENABLED` | on | `=0` — длинный content падает с CHECK violation как до волны 19. |
| `STYX_STORE_ROUTING_LIMIT` | `2400` | Boundary chars; ≤ limit идёт в `memories`, > limit — в documents+chunks. |
| `STYX_STORE_ROUTING_SUMMARY_CHARS` | `1500` | Длина tail-memory summary. |
| `STYX_CHUNK_SIZE` | `1600` | Размер chunk'а в symbols. |
| `STYX_CHUNK_OVERLAP` | `320` | Overlap между chunks для smoothing границ. |

## Search archive

| Var | Default | Назначение |
|---|---|---|
| `STYX_SEARCH_ARCHIVE_DEFAULT_LIMIT` | `10` | Default limit response'а. |
| `STYX_SEARCH_ARCHIVE_MAX_LIMIT` | `50` | Hard cap. |
| `STYX_SEARCH_ARCHIVE_K_CANDIDATES_FACTOR` | `8` | Сколько chunks подтянуть для stitching'а (factor × limit). |
| `STYX_SEARCH_ARCHIVE_K_CANDIDATES_MIN` | `80` | Floor на k_candidates. |

## Ingest API (pipelines)

| Var | Default | Назначение |
|---|---|---|
| `STYX_INGEST_API_ENABLED` | on | `=0` → `POST /ingest_experience` отвечает 503. |
| `STYX_INGEST_DOC_ENABLED` | on | `=0` → `POST /ingest_document` отвечает 503. |
| `STYX_INGEST_DOC_ROOTS` | (empty) | Colon-separated whitelist абсолютных директорий. Production: `/var/lib/styx/docs:/var/lib/styx/uploads`. Empty = lab mode (любой path). |
| `STYX_INGEST_DOC_MAX_BYTES` | `52428800` | Hard cap размера файла (50 MiB). |

## Dialogue routes

| Var | Default | Назначение |
|---|---|---|
| `STYX_DIALOGUE_API_ENABLED` | on | `=0` → все 5 `/dialogue/*` routes отвечают 503. |

## Explain / analytics / confirm_usage

| Var | Default | Назначение |
|---|---|---|
| `STYX_EXPLAIN_API_ENABLED` | on | `=0` → все 5 observability routes отвечают 503. |

## Memory markers (taxonomy)

| Var | Default | Назначение |
|---|---|---|
| `STYX_TAG_SALIENT` | `salient` | Имя channel'а в обёртке `<styx-{name}>`. |
| `STYX_TAG_RECALL` | `recall` | Override `<styx-recall>` тега. |
| `STYX_TAG_ARCHIVE` | `archive` | Override `<styx-archive>` тега. |
| `STYX_TAG_DIALOGUE` | `dialogue` | Override `<styx-dialogue>` тега. |
| `STYX_TAG_RELATIONS` | `relations` | Override `<styx-relations>` тега. |
| `STYX_TAG_EXPLAIN` | `explain` | Override `<styx-explain>` тега. |
| `STYX_TAG_WORKING_SET` | `working-set` | Override `<styx-working-set>` тега. |
| `STYX_FAMILY_BLOCK_RE` | (computed) | Regex для wholesale sanitize всех `<styx-*>` блоков. Обычно не override'ится. |
| `STYX_LEGACY_BLOCK_RE` | (computed) | Regex для legacy `<styx>` (без channel). Sanitize backwards-compat. |

## Healthz / readyz

| Var | Default | Назначение |
|---|---|---|
| `STYX_HEALTHZ_PORT` | `8788` | Port для `/healthz` + `/readyz`. `=0` отключает HTTP-сервер healthz. |
| `STYX_HEALTHZ_BIND` | `127.0.0.1` | Bind interface (в Docker — `0.0.0.0`). |
| `STYX_HEALTHZ_LIVENESS_THRESHOLD_S` | `120` | Возраст последней main-loop итерации старше → `/healthz` 503. |
| `STYX_HEALTHZ_READINESS_THRESHOLD_S` | `300` | Возраст drain-progress старше → `/readyz` 503. |

## Logging

| Var | Default | Назначение |
|---|---|---|
| `STYX_LOG_FORMAT` | `text` | `text` (для dev) или `json` (для production / vector / fluentbit). |
| `STYX_LOG_LEVEL` | `INFO` | `INFO` / `DEBUG` / `WARNING`. |
| `STYX_WIRE_LOG_RAW` | off | `=1` записывает payload-slice в wire-log на INFO. Только debug, privacy off. |

## Hermes integration

| Var | Default | Назначение |
|---|---|---|
| `STYX_ALLOW_HERMES_HOME` | off | `=1` снимает path-traversal guard в `styx-hermes-setup` для нестандартных путей (Docker, `/opt/...`). |
| `HERMES_PATH` | — | Путь к Hermes checkout'у для dev/test (если Hermes не установлен через pip). |

## Версия

| Var | Default | Назначение |
|---|---|---|
| `STYX_VERSION` | (auto) | Override version string в `/healthz` response. Обычно подбирается из package metadata. |
