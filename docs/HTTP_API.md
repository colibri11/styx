# Styx HTTP API

Контракт REST API styx-core daemon. Поддерживается через FastAPI, OpenAPI
schema автогенерируется на `GET /openapi.json`, интерактивные docs —
`GET /docs`.

## Auth

- `STYX_HTTP_TOKEN` ENV — bearer token. Если задан, все non-healthz endpoint'ы
  требуют `Authorization: Bearer <token>`, иначе 401.
- Loopback rule: если `STYX_HTTP_BIND` не loopback (127.0.0.1, localhost, ::1)
  и `STYX_HTTP_TOKEN` пустой — daemon **не стартует**. Защита от
  случайно открытого endpoint'а.
- `/healthz`, `/readyz` — без auth всегда (для probe).

## Opt-in LLM wrap (`?wrap_for_llm=1` / `X-Wrap-For-LLM: 1`)

Волна 30 (memory markers). 11 LLM-facing endpoint'ов (см. список
ниже) поддерживают опциональную обёртку результата pre-rendered
строкой с маркером таксономии: `<styx-{channel}>...</styx-{channel}>`.
Caller активирует обёртку одним из эквивалентных способов:

- query-параметр `?wrap_for_llm=1` (значения `0` / `1`),
- header `X-Wrap-For-LLM: 1` (значения `1` / `true` / `yes` / `on`).

Когда флаг установлен, response получает дополнительное поле
`llm_text: str` — pre-rendered обёртка вокруг JSON-сериализации
остального payload'а. Default response (без флага) сохраняет старый
shape — `llm_text` отсутствует или равен `null`. Не-LLM consumers
(CLI, тесты, host-side провайдеры) ничего не теряют.

| Channel | Endpoint(s) |
|---|---|
| `recall` | `POST /recall` |
| `archive` | `POST /search_archive` |
| `dialogue` | `POST /dialogue/{save,search,recent,sessions,prepare_summary}` |
| `relations` | `POST /relations/query`, `POST /graph/traverse` |
| `explain` | `POST /explain/{decompose,lifetime,topK}` |

Пример:

```
$ curl -s -X POST http://127.0.0.1:8788/recall \
    -H 'content-type: application/json' \
    -d '{"agent_id":"agent_demo","query":"что мы решили про X"}' | jq .
{
  "memories": [...],
  "queried_count": 12,
  "internal_duplicates_removed": 1,
  "elapsed_ms": 187,
  "llm_text": null
}

$ curl -s -X POST 'http://127.0.0.1:8788/recall?wrap_for_llm=1' \
    -H 'content-type: application/json' \
    -d '{"agent_id":"agent_demo","query":"что мы решили про X"}' | jq .
{
  "memories": [...],
  "queried_count": 12,
  "internal_duplicates_removed": 1,
  "elapsed_ms": 192,
  "llm_text": "<styx-recall>\n{\n  \"memories\": [...],\n  ...\n}\n</styx-recall>"
}
```

Plugin'ы (OpenClaw `extensions/styx`, Hermes `packages/styx-hermes`)
устанавливают header автоматически для всех LLM-facing paths и
подставляют `llm_text` напрямую как content tool result'а — LLM
видит маркер источника инжекта.

**Sanitize counter** (`/analytics`) различает per-tag breakdown
вырезанных входящих блоков:

```
"styx_sanitized_blocks_total": 0,
"styx_sanitized_blocks_by_tag": {
  "salient": 0,    "recall": 0,    "archive": 0,
  "dialogue": 0,   "relations": 0, "explain": 0,
  "working-set": 0, "legacy": 0
}
```

Ненулевое значение — leak assembled view / tool result в historical
messages (transcript echo, snapshot replay, native memory dump).
Per-tag breakdown указывает на источник.

## Endpoint'ы

### `GET /healthz`

Liveness probe — процесс жив, Postgres ping ok.

**Response 200:**
```json
{
  "status": "ok",
  "uptime_s": 12.3,
  "postgres": "ok",
  "version": "<version>"
}
```

**Response 503:** `status="down"`, `postgres="down"` если Postgres недоступен.

### `GET /readyz`

Readiness — Ollama ping + drain progress (если worker pool запущен).

**Response 200:**
```json
{
  "status": "ok",
  "uptime_s": 12.3,
  "last_drain_progress_age_s": 0.5,
  "ollama": "ok",
  "queue": {"processed": 4, "failed": 0},
  "version": "<version>"
}
```

**Response 503** если Ollama недоступна.

### `POST /agent/initialize`

Регистрирует agent_id в registry, configure'ит module-global state
(salient_bridge, focus_tracker, hot_tier, eviction_relevance_bridge,
pre_llm_inject), restore'ит working_set из БД.

Идемпотентный (Q15): повторный вызов для уже зарегистрированного
agent_id обновляет session_id и tools, не пересоздаёт state.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "session_id": "20260504_103000_abc123",
  "agent_identity": "agent_demo",
  "platform": "cli",
  "model": "zai/glm-5.1"
}
```

**Response 200:**
```json
{
  "agent_id": "agent_demo",
  "tools": [
    {
      "name": "styx_recall",
      "description": "Recall up to N memories from long-tier storage...",
      "parameters": {"type": "object", "properties": {...}, "required": ["query"]}
    }
  ]
}
```

### `POST /agent/shutdown`

Сбрасывает state агента, flush working_set, освобождает connection.
Идемпотентный (unknown agent_id → 204 без ошибки).

**Request:** `{"agent_id": "agent_demo"}`

**Response 204:** No body.

### `POST /sync_turn`

Запись turn'а — INSERT memory, embedding-after-commit, sentiment hot-path,
importance enqueue, classifier enqueue.

**Defect-fix B — split больших реплик:** реплика (user/assistant)
длиннее `STYX_MESSAGE_SPLIT_PART_CHARS` (default 2000) режется на N
рядов дневника того же role/session, каждый ≤ лимита (CHECK
constraint `memories_content_length_check`, 2400). Все ряды остаются
в `memories` (дневник = речь целиком; IAmBook §V — в архив части НЕ
уходят). Группа помечается в metadata `msg_group` (uuid) / `part`
(индекс) / `parts` (всего); композиция (`StyxComposer`,
`recent_messages`) пересобирает группу обратно в один блок. Inline-
embed ограничен `STYX_MESSAGE_SPLIT_INLINE_EMBED_CAP` (default 4) —
остаток embed'ит re-embed CLI / воркер. Реплика ≤ лимита → один ряд,
поведение не меняется.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "session_id": "20260504_103000_abc123",
  "user_content": "Привет",
  "assistant_content": "Привет, что нового?",
  "ts": "2026-05-04T10:00:00Z"
}
```

**Response 200:**
```json
{
  "memory_ids": [],
  "recall_event_ids": []
}
```

(Поля `memory_ids` / `recall_event_ids` зарезервированы; текущая
реализация возвращает пустые списки — расширится когда `MemoryCore.sync_turn`
научится возвращать вставленные ids.)

### `POST /recall`

On-demand recall — embed query, search_similar, format. Используется
`styx_recall` tool на стороне обёртки (Hermes/OpenClaw).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "query": "что мы обсуждали про Hermes",
  "limit": 5
}
```

**Response 200:**
```json
{
  "memories": [
    {
      "id": "uuid",
      "content": "...",
      "score": 0.45,
      "role": "user",
      "created_at": "2026-05-01T10:00:00Z"
    }
  ],
  "queried_count": 12,
  "internal_duplicates_removed": 2,
  "elapsed_ms": 31
}
```

### `POST /context/build`

Принимает messages list, возвращает compose'нутый context. Внутри —
`StyxComposer.compress()` со всем pipeline (salient inject, focus
update, eviction relevance).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "messages": [{"role": "user", "content": "..."}],
  "current_tokens": 12000,
  "focus_topic": null
}
```

**Response 200:**
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "compression_count": 1,
  "salient_injected": true
}
```

### `POST /pre_llm_inject`

Multi-channel framework: возвращает строки каналов для inject'а в user
message. Сейчас единственный канал — self_state (волна 35).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "session_id": "20260504_103000_abc123",
  "user_message": "...",
  "is_first_turn": false,
  "model": "zai/glm-5.1",
  "platform": "cli"
}
```

**Response 200:**
```json
{
  "context": "Тебе сейчас воодушевлённо и уверенно."
}
```

или `{"context": null}` если каналы вернули None.

### `GET /agent_state?agent_id=agent_demo`

Snapshot эмоционального состояния (instant, baseline, mood).

**Response 200:**
```json
{
  "agent_id": "agent_demo",
  "instant": {"valence": 0.2, "arousal": 0.5, "dominance": 0.1},
  "baseline": {"valence": 0.15, "arousal": 0.4, "dominance": 0.05},
  "mood": null
}
```

`mood` derivation отложен — поле зарезервировано.

### `POST /memory_store`

Subjective write через selective gatekeeper (волна 17). Каждое решение
gatekeeper'а — `skip` / `merge` / `supersede` / `store` — применяется
синхронно: insert + sync embed + apply, всё в одной транзакции.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "content": "напоминание про релиз пятницы",
  "kind": "note",
  "kind_src": "subjective",
  "session_id": "...",
  "metadata": {},
  "importance_provisional": null
}
```

- `content` — 1..500_000 chars. Короткий (≤ `STYX_STORE_ROUTING_LIMIT`,
  default 2400) идёт через gatekeeper в `memories` с CHECK ≤ 2400.
  Длинный (> limit) роутится в `documents` + `chunks` с tail-memory
  ≤ summary_chars (волна 19).
- `kind` — fact / episode / decision / concept / note (CHECK constraint).
  Mapping kind→role: всё → `summary`.
- `kind_src` — subjective / subjective_tail / experience_intake /
  dialogue_consolidation_daily / dialogue_batch_consolidation
  (CHECK constraint).

**Response 200 (короткий content):**
```json
{
  "action": "store",
  "memory_id": "uuid-новой-записи",
  "existing_id": null,
  "similarity": 0.42,
  "routed": false,
  "document_id": null,
  "chunks_count": null
}
```

**Response 200 (длинный content, store-routing — волна 19):**
```json
{
  "action": "store",
  "memory_id": "uuid-tail-memory",
  "existing_id": null,
  "similarity": null,
  "routed": true,
  "document_id": "uuid-virtual-document",
  "chunks_count": 3
}
```

Семантика `action`:
- `store` — memory создан, `memory_id` populated. При `routed=true`
  это tail-memory с `archive_ref` на `document_id`; gatekeeper
  пропущен (волна 19, ADR § 35.5).
- `merge` — поглощён existing (тот получил длинный новый content +
  embedding); `memory_id=null`, `existing_id` указывает на сохранившийся
  ряд.
- `supersede` — новый создан, `existing_id` — старый ряд (получил
  `superseded_by=new_id`); INSERT relation `supersedes`.
- `skip` — отсечено noise filter'ом (`content < STYX_SELECTIVE_NOISE_MIN_LENGTH`),
  `memory_id=null`.

При `routed=true`: `memory_id` указывает на tail-memory, `document_id`
— на virtual document'а в таблице `documents`, `chunks_count` — сколько
chunks (с embedding'ами) лежит в `chunks`. Auto-link применяется к
tail-memory; gatekeeper — нет.

**Errors:**
- 422 — content вне 1..500_000 (Pydantic upper-bound), либо
  неизвестный `kind`.
- 500 — CHECK violation `memories_content_length_check` если
  `STYX_STORE_ROUTING_ENABLED=0` И content > 2400 (routing отключён,
  ряд не помещается в legacy схему).
- 404 — `agent_id` не зарегистрирован.

**ENV (selective gatekeeper):**
- `STYX_SELECTIVE_ENABLED` (default `true`) — главный toggle. False →
  каждый writer пишет как раньше (action=store).
- `STYX_SELECTIVE_MERGE_THRESHOLD` (0.92), `_SUPERSEDE_THRESHOLD`
  (0.85), `_LEVENSHTEIN_THRESHOLD` (0.3).
- `STYX_SELECTIVE_NOISE_FILTER` (true), `_NOISE_MIN_LENGTH` (10).

**Auto-link (волна 18):** на ветках `store` и `supersede` после
gatekeeper'а сразу же находятся ближайшие соседи по embedding'у (cosine
distance ≤ 0.25, до 3 штук) и в `relations` пишется `related_to` ребро
от `memory_id` к каждому соседу. На `merge`/`skip` — нет (нет нового
ряда). Cross-agent: соседей ищет без agent_id фильтра — общий пул
знаний между всеми агентами в одном PG. Идемпотентно через UNIQUE
constraint в `relations`.

Также auto-link срабатывает для каждого user/assistant ряда в
`/sync_turn` после embed-after-commit — dialogue → subjective memory
связи через тот же `related_to`.

**ENV (auto-link):**
- `STYX_AUTO_LINK_ENABLED` (default `true`) — главный toggle.
- `STYX_AUTO_LINK_MAX_DISTANCE` (0.25 → similarity ≥ 0.75),
  `_MAX_LINKS` (3).

**ENV (store-routing — волна 19):**
- `STYX_STORE_ROUTING_ENABLED` (default `true`) — главный toggle. False →
  длинный content падает с CHECK violation как до волны 19.
- `STYX_STORE_ROUTING_LIMIT` (default `2400`) — boundary chars; ≤ limit
  идёт в `memories` напрямую, > limit — в documents+chunks.
- `STYX_CHUNK_SIZE` (1600), `STYX_CHUNK_OVERLAP` (320) — параметры
  chunker'а.
- `STYX_STORE_ROUTING_SUMMARY_CHARS` (1500) — длина tail-memory
  summary; truncate стратегия с word-boundary, маркер `…` при
  truncate'е.

### `POST /relations/query`

Плоский фильтр-запрос по таблице `relations` (волна 21). Cross-agent:
без фильтра по `agent_id` (общий пул знаний между всеми агентами в
одном PG).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "source_type": "memory",
  "source_id": "uuid-source",
  "target_type": "memory",
  "target_id": "uuid-target",
  "relation": "related_to",
  "limit": 50
}
```

Все фильтры опциональны. `limit` ∈ [1, 500] (default 50).

**Response 200:** `{"rows": [...]}` со списком `{id, source_type,
source_id, target_type, target_id, relation, weight, metadata,
created_at}`.

### `POST /graph/traverse`

Recursive CTE traversal от `entity_id`, depth ≤ 3, limit ≤ 20
(волна 21). Cross-agent — видит соседей всех агентов через рёбра
auto-link / Hebbian / supersedes / manual links.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "entity_id": "uuid",
  "entity_type": "memory",  
  "depth": 2,
  "relation_filter": "related_to",
  "limit": 20
}
```

`relation_filter` (опциональный) применяется в каждой ветке CTE — не
пропускает через рёбра другого типа между depth-уровнями. Сейчас
`entity_type` — только `memory`; после волны 19 расширится.

**Response 200:**
```json
{
  "root": {"id": "...", "type": "memory", "content_preview": "...", ...},
  "nodes": [
    {"id": "...", "type": "memory", "relation": "related_to",
     "direction": "outgoing", "depth": 1, "weight": 1.0,
     "content_preview": "..."},
    ...
  ]
}
```

**Errors:**
- 404 — entity не найден.
- 422 — depth/limit вне допустимых границ.

### `POST /link`

Manual edge insert (волна 21). Идемпотентен через UNIQUE constraint
(миграция 0004). Primary use — pipeline ingest (`/ingest_experience`
в волне 23) и debug.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "source_type": "memory",
  "source_id": "uuid",
  "target_type": "memory",
  "target_id": "uuid",
  "relation": "custom_label",
  "weight": 1.0,
  "metadata": {}
}
```

**Response 200:** `{"created": true}` если ребро создано, `false` если
уже существовало (no-op через `ON CONFLICT DO NOTHING`).

**Hebbian co-retrieval (волна 21):** на любом recall'е через
`/recall` или styx_recall tool с N≥2 results автоматически создаются
рёбра `relation='co_retrieved'` для всех C(N,2) пар. Initial weight
1.1; повторный recall тех же ids bump'ит weight на 0.1 до cap 2.0.
Cold links (last_reinforced > 14d) decay'ят в periodic-task на 0.05
раз в час до floor 1.0.

**ENV (Hebbian + decay):**
- `STYX_HEBBIAN_ENABLED` (default `true`) — главный toggle.
- `STYX_HEBBIAN_WEIGHT_BUMP` (0.1), `_INITIAL_WEIGHT` (1.1),
  `_WEIGHT_MAX` (2.0).
- `STYX_RELATION_DECAY_ENABLED` (default `true`),
  `_INTERVAL_S` (3600), `_RATE` (0.05), `_IDLE_DAYS` (14).

### `POST /search_archive`

Pull-канал к архиву (волна 20). FTS+vector hybrid query поверх
`documents`/`chunks` (миграции 0005+0006) и `memories WHERE role IN
('user','assistant')`. Pull-only — никогда не инжектится в context;
вызов через explicit tool на стороне Hermes-плагина (`styx_search_archive`)
либо OpenClaw-плагина.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "query": "что мы решили про gatekeeper",
  "scope": "all",
  "limit": 10,
  "date_from": "2026-04-01T00:00:00Z",
  "date_to": "2026-05-05T23:59:59Z",
  "snapshot_cycle_start": null
}
```

`scope` ∈ `documents` / `chunks` / `dialogue` / `all` (default `all`).

- `documents` — chunks search'нуты по hybrid score, затем grouped по
  `document_id` и stitched в regions (overlap из chunker'а удаляется).
  Region.score = max chunk score across stitched.
- `chunks` — top-K индивидуальных chunks без stitching'а (для точечных
  citation snippet'ов).
- `dialogue` — top-K реплик из `memories WHERE role IN
  ('user','assistant')`.
- `all` — fair-share interleave **`documents`+`dialogue`** (НЕ
  chunks): `[doc_0, dlg_0, doc_1, dlg_1, ...]`, slice `limit`.

`limit` — clamp `[1, search_archive_max_limit]`; default из
`STYX_SEARCH_ARCHIVE_DEFAULT_LIMIT` (10). `search_archive_max_limit`
— 50.

`snapshot_cycle_start: datetime | null` — опц. temporal isolation.
Если передан, фильтр `chunks.created_at <= cycle_start` и
`memories.created_at <= cycle_start`. По default не передаётся.

**Response 200:**
```json
{
  "results": [
    {
      "scope": "document",
      "text": "склеенный текст региона",
      "snippet": "первые 300 chars",
      "score": 0.87,
      "document_id": "uuid",
      "chunk_position": null,
      "chunk_positions": [0, 1, 2],
      "char_start": 0, "char_end": 1500,
      "memory_id": null, "role": null, "created_at": null
    },
    {
      "scope": "dialogue",
      "text": "реплика",
      "snippet": "реплика", "score": 0.62,
      "document_id": null, "chunk_position": null,
      "chunk_positions": null, "char_start": null, "char_end": null,
      "memory_id": "uuid", "role": "user",
      "created_at": "2026-05-04T18:30:00+00:00"
    }
  ],
  "total_matched": 2
}
```

`scope` поле — discriminator. Поля, нерелевантные для данного
scope, остаются `null`. `snippet = text[:300]` без highlight'а.

**Hybrid ranking:** `vector_weight × (1 - cosine) + bm25_weight ×
ts_rank(content_tsv, plainto_tsquery('simple', query), 32)`. Веса
адаптивные через `compute_weights(query)`: ≤ 2 words → 0.8/0.2; ≤ 4
words → 0.7/0.3; > 4 words → 0.6/0.4.

**agent_id isolation:** strict через `AgentScopedQueries`. Для
chunks — JOIN на `documents` по `agent_id`. Для dialogue —
`memories.agent_id` фильтр + `superseded_by IS NULL`.

**Errors:**
- 422 — invalid `scope` (не из 4 enum'ов) или Pydantic validation.
- 503 — agent core / embedder не initialized.

**ENV:**
- `STYX_SEARCH_ARCHIVE_DEFAULT_LIMIT` (10)
- `STYX_SEARCH_ARCHIVE_MAX_LIMIT` (50)
- `STYX_SEARCH_ARCHIVE_K_CANDIDATES_FACTOR` (8) — для `documents`-scope
  тянем `max(limit*factor, k_candidates_min)` chunks для stitching'а.
- `STYX_SEARCH_ARCHIVE_K_CANDIDATES_MIN` (80) — floor на k_candidates.

**Hermes tool wrapper:** `styx_search_archive` (через
`get_tool_schemas()` + `handle_tool_call`). LLM-видимые параметры:
`query`, `scope`, `limit`, `date_from`, `date_to`.
`snapshot_cycle_start` НЕ exposed в tool schema (internal parameter
для in-process callers).

### `POST /reinterpret`

Explicit reinterpret memory (волна 22). Агент явно переосмысляет
существующую memory: добавляет координату смысла, не переписывая
историю. memory_id сохраняется, граф цел. Cooldown 24h на memory.

**Enqueue-only:** route не ждёт LLM. Apply применяется через
`reinterpret_apply_sweeper` (раз в 30s) после закрытия turn'а агента
(write-gate из волны 14). Worst-case latency apply'а: ~30-90s.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "memory_id": "uuid",
  "new_understanding_text": "Что добавилось в понимании. 1-3 предложения.",
  "weight": 0.5
}
```

`weight` — опциональный blend weight в [0, 1] для embedding'ов.
None → default `STYX_REINTERPRET_BLEND_WEIGHT` (0.5). 0.5 =
равноправный микс prev+next; больше = новое сильнее тянет recall.

**Response 200 (queued):**
```json
{
  "status": "queued",
  "memory_id": "uuid",
  "task_id": "uuid",
  "application_id": 7,
  "message": "reinterpret queued; will apply once current turn closes and the sweeper runs (~30-60s)"
}
```

**Response 404 (memory_not_found):**
```json
{
  "detail": {
    "status": "memory_not_found",
    "memory_id": "uuid"
  }
}
```

**Response 409 (cooldown):**
```json
{
  "detail": {
    "status": "cooldown",
    "memory_id": "uuid",
    "next_available_at": "2026-05-06T12:00:00+00:00",
    "last_reinterpreted_at": "2026-05-05T12:00:00+00:00"
  }
}
```

**Response 409 (already_pending):** `pending_application_id`
populated; для memory уже есть pending_sleep application — повторный
enqueue заблокирован partial UNIQUE индексом.

**Response 422:** validation error (Pydantic) или невалидный UUID.
**Response 503:** core не initialized или `STYX_REINTERPRET_ENABLED=0`.

**Apply pipeline:**
1. HTTP enqueue → INSERT `llm_tasks` (`reinterpret_merge` task_type)
   + INSERT `reinterpret_applications` (status='pending_sleep')
   атомарно.
2. Worker drain task → handler читает memory.{content, embedding},
   LLM call qwen3:4b-local → `merged_text` или `skip`. Если skip=False:
   embed merged_text → `blend_embeddings(prev, next, weight)` +
   L2-norm. Result сохраняется в `llm_tasks.result`.
3. `reinterpret_apply_sweeper` (раз в 30s) — per-agent loop:
   `is_active(agent_id)` → fast-path skip; иначе для каждого
   pending_sleep + done-task: UPDATE memory.{content, embedding} +
   INSERT `memory_reinterpretations` (audit revision) + UPDATE
   applications status='applied'. Per-row `conn.transaction()` —
   atomic.

**ENV:**
- `STYX_REINTERPRET_ENABLED` (1) — включает HTTP route + apply-sweeper.
- `STYX_REINTERPRET_APPLY_TICK_S` (30) — period sweeper'а.
- `STYX_REINTERPRET_COOLDOWN_S` (86400) — cooldown 24h на memory.
- `STYX_REINTERPRET_BLEND_WEIGHT` (0.5) — default weight.

**Hermes tool wrapper:** `styx_reinterpret` (через
`get_tool_schemas()` + `handle_tool_call`). LLM-видимые параметры:
`memory_id`, `new_understanding_text`, `weight`. Возвращает
structured `{status, ...}` JSON, симметрично HTTP-ответу.

### `POST /ingest_experience`

Внешний канал для pipelines (волна 23) — AudioBox и future ingest
workers пишут в память агента через HTTP, не лезут в Postgres
напрямую. Идемпотентен по `content_hash`: повторный ingest того же
payload'а от того же агента возвращает существующий `memory_id` с
`deduplicated: true`, без побочных эффектов (embedding/metadata из
второго вызова игнорируются).

**Pipeline-канал — не subjective:**
- Selective gatekeeper НЕ применяется (gatekeeper только для
  subjective writers, D11 в waves/17).
- Auto-link НЕ применяется (для structural edges есть `POST /link`).
- Store-routing НЕ применяется (content ≤ 2400 chars; длинные доки —
  через OpenClaw plugin track).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "content": "транскрипт записи",
  "kind": "fact",
  "kind_src": "experience_intake",
  "metadata": {"any": "user fields"},
  "importance_provisional": 0.5,
  "content_hash": "abc...64hex...",
  "pipeline_id": "audiobox",
  "pipeline_version": "v1.2",
  "content_ref": {"file_path": "/recordings/a.wav"}
}
```

`content` — 1..2400 chars (CHECK constraint
`memories_content_length_check`). `kind` — одно из
`{fact, episode, decision, concept, note}`. `kind_src` default
`"experience_intake"`.

**Hash priority:**
1. **Explicit `content_hash`** — pipeline сам контролирует политику.
2. **Auto-compute** из `(pipeline_id, pipeline_version, content_ref)`
   если все три заданы и `content_ref` не пуст. Hash =
   `sha256(canonicalize({pipeline_id, pipeline_version, content_ref}))`,
   ключи объектов сортируются рекурсивно.
3. **Иначе** — `content_hash=null`, partial UNIQUE индекс
   `memories_agent_content_hash_uniq` игнорирует NULL, idempotency не
   применяется (каждый INSERT новый ряд).

**Response 200:**
```json
{
  "memory_id": "uuid",
  "deduplicated": false,
  "content_hash": "<hex>|null"
}
```

`deduplicated=true` означает что ряд с этим `(agent_id, content_hash)`
уже существовал — `memory_id` указывает на existing.

**Atomic UPSERT pattern:**
```sql
INSERT INTO memories (agent_id, role, kind, kind_src, content,
                     embedding, metadata, content_hash)
VALUES (...)
ON CONFLICT (agent_id, content_hash) WHERE content_hash IS NOT NULL
   DO NOTHING
RETURNING id
```

При пустом RETURNING — SELECT existing по `(agent_id, content_hash)`.
ON CONFLICT vs SELECT+INSERT — race-safe: параллельные ingest'ы
одного payload'а не дадут unique_violation 23505.

**metadata enrichment:** `pipeline_id` / `pipeline_version` /
`content_ref` (если заданы) автомерж'аются в memory.metadata для
трассируемости:
```json
{
  ...user_metadata,
  "source": {"pipeline_id": "audiobox", "pipeline_version": "v1.2"},
  "content_ref": {"file_path": "/recordings/a.wav"}
}
```
User metadata имеет priority (явно переданные ключи не затираются).

**agent_id isolation:** UNIQUE constraint partial по
`(agent_id, content_hash)` — два агента с одинаковым hash имеют
независимые ряды.

**Errors:**
- 422 — `content` > 2400 / unknown `kind` / Pydantic validation.
- 503 — `STYX_INGEST_API_ENABLED=0` либо provider не initialize'нут.

**ENV:**
- `STYX_INGEST_API_ENABLED` (default `true`) — главный toggle.

**Hermes wrapper:** нет (pipeline-канал, LLM не вызывает; для
explicit edges используется `POST /link` который тоже не имеет
wrapper'а, симметрия).

### `POST /ingest_document`

File-ingest pipeline (волна 28 + Defect-fix A) — core читает файл по
абсолютному пути, парсит (PDF/DOCX/XLSX/Markdown/text), режет на
chunks через `route_long_content`, embed'ит, INSERT'ит document +
chunks. Документ-артефакт уходит в архив (`documents`+`chunks`),
доступен через `/search_archive`.

**Defect-fix A — маркер акта вместо «no tail-memory»:** документ ≠
память (IAmBook §V). В `memories` пишется tail-memory с **маркером
акта архивации** — «я положил в архив документ такого-то рода»
(тип / происхождение / о чём / ссылка `styx://store/<id>`), БЕЗ
содержания документа. Маркер — это след акта в линии `я`. Раньше
(волна 28) tail-memory не создавалась вовсе.

**Defect-fix A — async ingest большого документа:** документ, который
chunker делит больше чем на `STYX_DOCUMENT_INGEST_ASYNC_CHUNK_
THRESHOLD` chunks (default 12), ingest'ится async — chunks
INSERT'ятся с `embedding=NULL`, embedding выполняется в worker pool
(task `document_chunk_embed`), endpoint возвращается быстро
(`chunks_embedded_inline=false`). Маркер акта embed'ится всегда
inline (один embed — дёшево).

**Lightweight plugin pattern (directive 2026-05-12):** plugin шлёт
один POST с маленьким JSON `{path}`, core делает всё (disk read +
parse + chunk + embed + INSERT). Multipart upload не используется.

**Pipeline-канал — не subjective** (симметрично `/ingest_experience`):
- Selective gatekeeper НЕ применяется.
- Auto-link НЕ применяется.
- Classifier-enqueue / sentiment hot-path НЕ применяются.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "path": "/var/lib/styx/uploads/spec.pdf",
  "source_ref": "support-ticket-#142",
  "visibility": "private",
  "metadata": {"uploaded_by": "agent_demo"},
  "content_hash": "optional-explicit-hash-override"
}
```

`path` — обязательный абсолютный путь. Поддерживаемые расширения:
`.pdf` / `.docx` / `.xlsx` / `.md` / `.markdown` / `.txt` / `.text`.
Остальные поля опциональны.

**Response 200:**
```json
{
  "document_id": "uuid",
  "deduplicated": false,
  "chunks_count": 3,
  "mime_type": "application/pdf",
  "original_name": "spec.pdf",
  "size_bytes": 12345,
  "char_count": 5000,
  "content_hash": "sha256-hex-64-chars",
  "act_marker_memory_id": "uuid",
  "chunks_embedded_inline": true
}
```

`deduplicated=true` — повторный ingest того же файла (matched
`(agent_id, content_hash)` через partial UNIQUE index
`uq_documents_agent_content_hash`); `document_id` указывает на
existing ряд, `chunks_count=0` (нет новых INSERT'ов),
`act_marker_memory_id=null` (новый маркер не создаётся — акт уже
зафиксирован при первом ingest'е).

`act_marker_memory_id` — id tail-memory с маркером акта архивации
(Defect-fix A); `null` при dedup. `chunks_embedded_inline=false` —
большой документ, chunks embed'ятся async в worker pool;
`/search_archive` найдёт документ после того как async-задача
отработает.

**Path validation pipeline (D10):**
1. Absolute path (`Path.is_absolute()`).
2. Resolve symlinks strictly (`Path.resolve(strict=True)`) — кидает
   если файл не существует.
3. Если `STYX_INGEST_DOC_ROOTS` непуст — resolved path должен
   начинаться с одного из roots (защита от symlink escape).
4. Size guard: `STYX_INGEST_DOC_MAX_BYTES` (default 50 MiB).

**mime detection (D11):** Расширение определяет canonical mime;
первые 8 байт проверяются как magic bytes для binary форматов
(PDF `%PDF-`, DOCX/XLSX `PK\x03\x04`). Mismatch → 422.

**content_hash:** SHA256 от file bytes (streaming через 1 MiB
блоки). Explicit override через request возможен. Partial UNIQUE
`uq_documents_agent_content_hash` WHERE NOT NULL обеспечивает
идемпотентность.

**Errors:**
- 422 — relative path, file not found, not a regular file, outside
  allowed roots, file too large, unsupported extension, mime
  mismatch, encrypted PDF, empty document (image-only PDF / blank
  file), malformed PDF/DOCX/XLSX.
- 503 — `STYX_INGEST_DOC_ENABLED=0` либо provider не initialize'нут.

**ENV:**
- `STYX_INGEST_DOC_ENABLED` (default `true`) — главный toggle.
- `STYX_INGEST_DOC_ROOTS` (default empty = lab mode) — colon-
  separated whitelist абсолютных директорий. Production:
  `STYX_INGEST_DOC_ROOTS=/var/lib/styx/docs:/var/lib/styx/uploads`.
- `STYX_INGEST_DOC_MAX_BYTES` (default `52428800` = 50 MiB).
- `STYX_DOCUMENT_INGEST_ASYNC_CHUNK_THRESHOLD` (default `12`) —
  документ с числом chunks больше порога ingest'ится async
  (Defect-fix A).

**Plugin wrappers (D14):** OpenClaw plugin `styx_ingest_document`
tool (factory + impl + lazy HTTP call). Hermes plugin —
`_handle_ingest_document` в provider'е + tool schema через core's
`get_tool_schemas`. Оба тонкие HTTP-обёртки к этому endpoint'у.

### `POST /dialogue/save`

Explicit ad-hoc save одной реплики (волна 24). Пишет в `memories`
с `role IN ('user','assistant')`. Auto-capture идёт через
`/sync_turn` — этот route для programmatic callers (OpenClaw plugin
context-engine `dialogue_save` tool).

**Pipeline-канал — не subjective:**
- Auto-link НЕ применяется (D5 в waves/24, симметрия с `/ingest_experience`).
- Classifier-enqueue / sentiment hot-path НЕ применяются (только
  для natural turn'а через `/sync_turn`).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "role": "user",
  "content": "явно сохранённая реплика",
  "session_id": "uuid|null",
  "metadata": {"hermes_session_idx": 12}
}
```

`role` — `'user'` либо `'assistant'`. `content` — 1..2400 chars.
`session_id` опц.: если задан — `upsert_session` идемпотентно перед
INSERT'ом, иначе FK NULL (session-list этот ряд не увидит).

**Response 200:**
```json
{"memory_id": "uuid"}
```

**Errors:**
- 422 — bad role / `content` > 2400 / Pydantic validation.
- 503 — `STYX_DIALOGUE_API_ENABLED=0` либо provider не initialize'нут.

### `POST /dialogue/search`

Hybrid (FTS+vector) либо pure-vector search поверх
`memories WHERE role IN ('user','assistant')` под `agent_id`
(cross-agent НЕТ, D13 в waves/24).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "query": "postgres tuning",
  "session_id": "uuid|null",
  "after": "2026-04-01T00:00:00Z",
  "before": "2026-05-01T00:00:00Z",
  "semantic_only": false,
  "limit": 10
}
```

`semantic_only=false` (default) — hybrid через
`compute_weights(query)` (те же веса что `search_archive` dialogue-
scope). `semantic_only=true` — pure cosine, score = `1 - distance`
∈ [0..1]. `limit` clamp [1..50], default 10.

**Response 200:**
```json
{
  "results": [
    {
      "memory_id": "uuid",
      "role": "user",
      "content": "postgres performance tuning",
      "score": 0.91,
      "created_at": "2026-04-15T12:34:56Z",
      "session_id": "uuid|null"
    }
  ]
}
```

**Errors:**
- 422 — empty `query` / bad limit.
- 503 — disabled / not initialized.

### `POST /dialogue/recent`

Последние N реплик в chronological order (oldest first после
internal reverse). Без vector ranking'а — pure `ORDER BY seq DESC
LIMIT` + reverse в Python.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "session_id": "uuid|null",
  "before": "2026-05-01T00:00:00Z",
  "limit": 20
}
```

`limit` clamp [1..200], default 20.

**Response 200:**
```json
{
  "rows": [
    {
      "memory_id": "uuid",
      "role": "user",
      "content": "первая",
      "created_at": "2026-05-01T10:00:00Z",
      "session_id": "uuid|null"
    }
  ]
}
```

`role IN ('tool','system','summary')` отфильтровываются (SQL фильтр).

### `POST /dialogue/sessions`

List of sessions с counts + first/last_message_at, ORDER DESC по
last_message_at. Реплики без `session_id` (FK NULL) не учитываются.

**Request:**
```json
{"agent_id": "agent_demo", "limit": 10}
```

`limit` clamp [1..100], default 10.

**Response 200:**
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "message_count": 42,
      "first_message_at": "2026-05-01T10:00:00Z",
      "last_message_at": "2026-05-01T18:00:00Z"
    }
  ]
}
```

### `POST /dialogue/prepare_summary`

Готовит chronological transcript конкретной session для
summarizer-агента (LLM-driven summary — за пределами скоупа этого
route'а; здесь только подготовка текста).

**Request:**
```json
{"agent_id": "agent_demo", "session_id": "uuid", "limit": 200}
```

`session_id` обязателен (D9 в waves/24). `limit` clamp [1..1000],
default 200.

**Response 200:**
```json
{
  "session_id": "uuid",
  "message_count": 42,
  "first_message_at": "2026-05-01T10:00:00Z",
  "last_message_at": "2026-05-01T18:00:00Z",
  "transcript": "[2026-05-01 10:00:00] Human: ...\n[2026-05-01 10:00:30] Agent: ..."
}
```

Формат строк: `[YYYY-MM-DD HH:MM:SS] Speaker: content` (UTC,
без миллисекунд). Speaker mapping: `user → Human`, `assistant →
Agent`. Реплики `tool`/`system`/`summary` не попадают (SQL фильтр).

**Empty session:**
```json
{
  "session_id": "uuid",
  "message_count": 0,
  "first_message_at": null,
  "last_message_at": null,
  "transcript": ""
}
```

Не 404 — пустая session валидна (могла быть только что создана).

**ENV для всей dialogue surface:**
- `STYX_DIALOGUE_API_ENABLED` (default `true`) — главный toggle.
  При `false` все 5 routes отвечают 503.

**Hermes wrappers:**

Read-only три обёрнуты как LLM-tools в Styx-Hermes plugin (волна 24
follow-up):

- `styx_dialogue_search` — wrapper над `POST /dialogue/search`.
- `styx_dialogue_recent` — wrapper над `POST /dialogue/recent`.
- `styx_dialogue_prepare_summary` — wrapper над `POST /dialogue/prepare_summary`.

`POST /dialogue/save` — без Hermes wrapper'а: Hermes capture идёт
через `/sync_turn`; explicit save — для plugin-канала (симметрия
с `/ingest_experience`).

`POST /dialogue/sessions` — без Hermes wrapper'а: административная
поверхность без LLM use case (list of UUIDs без UI бесполезен LLM).

OpenClaw plugin track в будущем зарегистрирует все 5 routes как
LLM-tools напрямую через HTTP.

### `POST /explain/decompose`

11-факторный breakdown скоринга для `(memory_id, query)`. Возвращает
`final_score`, `rank_in_result_set` и блок factors с объяснением
каждого множителя (base_match, recency_boost, frequency_boost,
lifecycle_factor, feedback_factor, importance_factor, diversity_bonus,
usage_factor, decay_factor, relevance, emotional_resonance).

**Request:**
```json
{
  "agent_id": "agent_demo",
  "memory_id": "uuid",
  "query": "что мы обсуждали",
  "top_k_limit": 10,
  "min_score": 0.45
}
```

`top_k_limit` clamp [1..200], default 10. `min_score` опц. — если
задан и `final_score < min_score`, возвращается
`not_returned_because.code = "below_min_score"`.

**Response 200:**
```json
{
  "mode": "decompose",
  "memory_id": "uuid",
  "kind": "fact",
  "query": "что мы обсуждали",
  "final_score": 0.523,
  "rank_in_result_set": 3,
  "top_k_limit": 10,
  "would_be_returned": true,
  "return_reason": "top_k",
  "not_returned_because": null,
  "factors": {
    "base_match": {"value": 0.7, "mode": "vector", ...},
    "recency_boost": {"value": 1.3, "rule": "< 1 day", "age_days": 0.4},
    "decay_factor": {"value": 0.99, "lambda_base": 0.005, ...},
    "...": "..."
  },
  "computed_at": "2026-05-05T12:00:00Z"
}
```

`not_returned_because.code` ∈ `{superseded, below_min_score, outside_top_k}`.
`return_reason` ∈ `{top_k, top_k_with_min_score}` либо `null`.

**Errors:**
- 404 — memory_id не найдено для этого agent_id (cross-agent scope).
- 422 — bad memory_id (не uuid) / empty query.
- 503 — `STYX_EXPLAIN_API_ENABLED=0`.

### `POST /explain/lifetime`

Lifecycle trace memory: importance lifecycle (provisional/final/llm_task
status), lifecycle state, access stats, recall_history, co-retrieval
links (relation='co_retrieved' Hebbian), decay projections.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "memory_id": "uuid",
  "include_recall_history": true,
  "recall_history_limit": 10,
  "prune_min_relevance": 0.1
}
```

`prune_min_relevance` опц. — если задан, считается
`decay.estimated_days_to_prune_threshold` (через закрытое выражение
`d = ln(relevance/threshold)/effective_lambda - age_days`).

**Response 200:**
```json
{
  "mode": "lifetime",
  "memory_id": "uuid",
  "content_preview": "...",
  "kind": "fact",
  "agent_id": "agent_demo",
  "visibility": "private",
  "created_at": "2026-04-30T10:00:00Z",
  "updated_at": "2026-04-30T10:00:00Z",
  "age_days": 5.2,
  "importance": {
    "provisional": 0.5,
    "final": 0.7,
    "effective": 0.7,
    "source": "final",
    "llm_task_status": "completed",
    "...": "..."
  },
  "lifecycle": {"current_state": "fresh", "multiplier": 1.0},
  "access": {
    "access_count": 3,
    "total_recall_events": 5,
    "avg_match_score": 0.62,
    "...": "..."
  },
  "decay": {
    "lambda_base": 0.005,
    "effective_lambda": 0.00325,
    "current_decay_factor": 0.983,
    "projected_decay_in_30d": 0.890,
    "estimated_days_to_prune_threshold": 211.5,
    "...": "..."
  },
  "recall_history": [
    {"matched_at": "...", "query_hash": "0xdeadbeef...", "match_score": 0.62}
  ],
  "co_retrieval_links": [
    {"target_memory_id": "uuid", "target_preview": "...", "weight": 1.5, "...": "..."}
  ],
  "computed_at": "2026-05-05T12:00:00Z"
}
```

**Errors:**
- 404 — memory не найдено для agent_id.
- 422 — bad memory_id.
- 503 — disabled.

### `POST /explain/topK`

Top-K кандидатов для query'а с factor breakdown'ом. Возвращает items в score-DESC,
nu`ranks` 1..N + total_candidates_considered.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "query": "postgres tuning",
  "limit": 10,
  "kinds": ["fact", "decision"],
  "after": "2026-04-01T00:00:00Z",
  "before": null,
  "min_score": null,
  "include_factors": true
}
```

`limit` clamp [1..50], default 10. `kinds` опц. (subset валидных
kind'ов: fact/episode/decision/concept/note). `after`/`before` опц.
по `created_at`. `include_factors=false` — items без factor blocks
(быстрее).

**Response 200:**
```json
{
  "mode": "top_k",
  "query": "postgres tuning",
  "limit": 10,
  "total_candidates_considered": 137,
  "items": [
    {
      "memory_id": "uuid",
      "kind": "fact",
      "content_preview": "...",
      "final_score": 0.812,
      "rank": 1,
      "factors": {"base_match": {...}, "...": "..."}
    }
  ],
  "computed_at": "2026-05-05T12:00:00Z"
}
```

**Errors:**
- 422 — limit > 50 / empty query.
- 503 — disabled.

### `GET /analytics?agent_id=…`

Per-agent counts + global totals + pending indexing. Caller-scoped:
один агент в `agents`, без enumeration других (audit-fix v0.7).

`agent_id` обязателен в query string.

**Response 200:**
```json
{
  "agents": [{
    "agent_id": "agent_demo",
    "display_name": null,
    "memories_count": 1247,
    "memories_by_kind": {"fact": 800, "episode": 300, "decision": 147},
    "documents_count": 12,
    "chunks_count": 348,
    "dialogue_messages_count": 4521,
    "total_storage_bytes": 24518
  }],
  "global": {
    "total_memories": 1247,
    "total_documents": 12,
    "total_chunks": 348,
    "total_dialogue_messages": 4521,
    "total_relations": 8214,
    "total_storage_bytes": 24518,
    "database_size_bytes": 158000000
  },
  "pending_indexing": {
    "dialogue_messages": 0,
    "memories": 5,
    "chunks": 0,
    "oldest_pending_at": "2026-05-04T15:00:00Z"
  }
}
```

`display_name` всегда `null` — Styx не имеет таблицы `agents`.

`dialogue_messages` в `pending_indexing` всегда 0 — Styx не имеет
отдельной `dialogue_messages` таблицы (реплики живут в `memories`,
их pending count учитывается в `memories`).

`total_relations` — глобальный (cross-agent) — relations это shared
knowledge graph (§ 33.2 / § 34.1).

**Errors:**
- 422 — отсутствует `agent_id` query param.
- 503 — disabled.

### `POST /confirm_usage`

Explicit `used_in_output=true` маркер для последних recall_event'ов
указанных memory_ids. Cross-agent guard через JOIN на
`memories.agent_id` — recall_event чужого агента не апдейтится,
memory_id попадает в response.missing.

**Request:**
```json
{
  "agent_id": "agent_demo",
  "memory_ids": ["uuid1", "uuid2", "uuid3"]
}
```

`memory_ids` capped [1..100] в Pydantic. Дубликаты collapsed на
Python-стороне.

**Response 200:**
```json
{
  "updated": 2,
  "requested": 3,
  "missing": ["uuid3"]
}
```

`updated` — number of memory_ids которые имели подходящий recall_event
и были UPDATE'нуты. `missing` — те что не нашлись (либо memory не
существует, либо чужого агента, либо не было recall_event'а).

Идемпотентно — повторный call с тем же payload'ом возвращает то же
`updated` (RETURNING row не зависит от факта изменения значения).

**ENV для всей explain/analytics/confirm_usage surface:**
- `STYX_EXPLAIN_API_ENABLED` (default `true`) — главный toggle.
  При `false` все 5 routes отвечают 503.

**Hermes wrappers:** none (D10 в waves/25). Это observability surface
для оператора (CLI / Postman / future Grafana dashboard), не для LLM.

**Errors:**
- 422 — empty memory_ids / > 100 / bad uuid.
- 503 — disabled.

### `POST /maintenance/reembed`

Backfill / re-embed `memories.embedding` через HTTP (волна 31) — тот же
`run_reembed`, что и CLI `styx reembed` (волна 7e). Нужен host-агенту в
контейнере, который ходит к Styx только по HTTP (docker.sock не
примонтирован → CLI/`docker exec` недоступны): агент сам добивает
NULL-хвосты, а не только наблюдает счётчик `pending_indexing.memories`
в `/analytics`.

Sync-хендлер (FastAPI оффлоудит в threadpool) — rate-limited backfill-loop
не блокирует event loop. Конкурентность ограничена session-level advisory
lock'ом (key `9876543211`, отдельный от sweep `9876543210`): параллельные
вызовы не дублируют embed и не давят на Ollama — второй получит
`skipped: true`.

**Scope — только memories.** chunks (`chunks.embedding IS NULL`) — отдельный
follow-up.

**Request** (все поля опциональны, паритет CLI):
```json
{
  "mode": "null_only",
  "agent_id": "agent_demo",
  "limit": null,
  "dry_run": false,
  "batch_size": 50,
  "rate_per_second": 5.0
}
```

- `mode` — `"null_only"` (default; только `embedding IS NULL`) | `"all"`
  (полный re-embed, например после смены модели).
- `agent_id` — `str | null`; `null` означает все агенты.
- `limit` — `int ≥ 0 | null`; ограничивает число обработанных рядов.
- `dry_run` — `bool` (default `false`); `true` возвращает `would_process`
  без UPDATE.
- `batch_size` — `int ≥ 1` (default 50) — cursor-pagination batch.
- `rate_per_second` — `float > 0` (default 5.0) — token-bucket лимит на
  embed-вызовы к Ollama.

**Response 200:**
```json
{
  "processed": 79,
  "failed": 0,
  "would_process": 0,
  "dry_run": false,
  "elapsed_ms": 15234,
  "skipped": false
}
```

- `processed` — сколько рядов перезаписано embedding'ом (0 при `dry_run`
  или `skipped`).
- `failed` — embed/UPDATE упали на этих рядах (continue, не abort).
- `would_process` — прогноз при `dry_run` (иначе 0).
- `skipped` — `true` если advisory lock занят другим instance'ом; backfill
  не запускался.
- `elapsed_ms` — wall-clock хендлера (connect + embed-loop).

**Auth:** `Depends(require_auth)` — Bearer как у `/ingest_experience`
(mutating route, защита не слабее).

**Hermes wrapper:** нет (operational surface, не LLM-facing).

**Errors:**
- 401 — нет / неверный Bearer (при заданном `http_token`).
- 422 — Pydantic validation (например `rate_per_second: 0`, `mode` вне
  enum, `limit < 0`) либо `ValueError` из `run_reembed`.
- 503 — Postgres недоступен (`psycopg.OperationalError` при connect).

## Коды ошибок

- **400** — Pydantic schema validation fail / business validation
  (например, пустой `query` в `/recall`).
- **401** — auth fail (missing / invalid bearer).
- **404** — `agent_id` не зарегистрирован (нужен `/agent/initialize` first).
- **422** — Pydantic schema validation fail (FastAPI default).
- **500** — внутренняя ошибка (psycopg, etc.).
- **503** — daemon недоступен / Postgres down / Ollama down (на healthz/readyz).

## Examples

```bash
# Liveness
curl -s http://127.0.0.1:8788/healthz | jq .

# Initialize agent (loopback, без token)
curl -sX POST http://127.0.0.1:8788/agent/initialize \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent_demo","session_id":"sid-1","agent_identity":"agent_demo"}' \
  | jq .

# С bearer token
curl -sX POST http://styx-daemon:8788/recall \
  -H "Authorization: Bearer $STYX_HTTP_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent_demo","query":"что мы обсуждали"}' \
  | jq .
```

## Schema discovery

Полный JSON-Schema:
```bash
curl -s http://127.0.0.1:8788/openapi.json
```

Интерактивный Swagger UI: `http://127.0.0.1:8788/docs`.
