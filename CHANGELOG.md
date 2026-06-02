# Changelog

Все значимые изменения в Styx документируются здесь. Формат —
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).

## [1.0.2] — 2026-06-02

Патч-релиз. Совместимость плагина `styx-hermes` с Hermes Agent **v0.15.2**
(образ v2026.5.29.2). Адаптер валидировался против v0.11/0.13; v0.15.2
добавил `api_mode` в ABC `ContextEngine.update_model` и зовёт его на старте
агента **без** try/except — HARD-BREAK. `styx-core` daemon без изменений
(остаётся 1.0.1).

### Fixed

- **HARD-BREAK инициализации под Hermes v0.15.2.**
  `StyxContextEngine.update_model` теперь принимает `api_mode` + `**kwargs`
  (v0.15.2 зовёт `update_model(api_mode=…)` без try/except на старте агента,
  `agent_init.py:1441/:1458`). `**kwargs` также на `compress` (поглощает
  host-овый `compress(force=…)` — иначе `focus_topic` терялся бы через
  degraded TypeError-retry путь Hermes) и защитно на
  `should_compress`/`update_from_response`/`on_session_reset`.
- **`StyxCodexTransport`** больше не ставит `prompt_cache_key` на путях
  GitHub Models / xAI Responses — гейтит по `is_github_responses` /
  `is_xai_responses`, как Hermes-default (`codex.py:158`).

### Changed

- Пин `styx-core` в `styx-hermes` поднят до `==1.0.1`.
- `_agent_session.set_session` логирует warning при замене session на
  другой `agent_id` (one-process-one-agent, Q20; замена остаётся намеренной).
- **Деплой:** `docker/styx-bootstrap.sh`, `styx-hermes-setup` и
  `docs/DEPLOYMENT.md` выставляют/документируют `context.engine: styx` —
  без него Hermes v0.15.2 берёт встроенный compressor: Styx-движок
  зарегистрирован, но **не выбран**.

### Tests

- Engine-тесты транспортов (`test_transport`, `test_codex_transport`)
  переписаны под host-agnostic split (классы из `styx_hermes.engine.transport`,
  `agent_id` через `_agent_session`); добавлен regression-тест
  `StyxContextEngine` (`api_mode` HARD-BREAK + `compress(force=)`); удалён
  pre-split `test_e2e_smoke.py` (рушил коллекцию всего suite; intent покрыт
  `integration/test_real_plugin_discovery.py`).

Валидация: host 159 passed / 6 infra-skip / 0 failed; hermes-drift 0/46
против v0.15.2; полный Docker-integration smoke против Hermes v0.15.2 —
**PASS** (live `hermes chat` активировал styx-движок без `update_model`
TypeError; transports + hook + daemon-pipeline отработали).

## [1.0.1] — 2026-05-20

Патч-релиз. Устранён боевой инцидент `memories_content_length_check`:
документ, большое сообщение и память разведены по правильным каналам
вместо общего turn-write-path.

### Fixed

- **Боевой инцидент `memories_content_length_check`.** User-реплика
  15090 символов (документ-Markdown с вложением), приехавшая
  turn-каналом, роняла запись по CHECK constraint'у и оставляла
  daemon-соединение в aborted-state до рестарта. Исправлено в трёх
  слоях:
  - **rollback-guard**: блок insert+commit в `sync_turn` /
    `ingest_single_message` обёрнут в try/except — при любой ошибке
    `rollback()`, соединение остаётся рабочим.
  - **core-инвариант**: `insert_message` **и** `insert_memory` бросают
    `ContentTooLongError` при `content` длиннее лимита — симметричная
    страховка до CheckViolation на обоих write-path'ах.
  - **split больших реплик дневника**: реплика (user/assistant)
    длиннее `STYX_MESSAGE_SPLIT_PART_CHARS` режется на N рядов
    `memories` того же role/session (дневник = речь целиком; IAmBook
    §V), группа помечается `msg_group`/`part`/`parts`; `StyxComposer`
    и `recent_messages` пересобирают группу обратно в один блок,
    группа не режется на границе `LIMIT`.
- **Перехват вложений не теряет данные при сбое ingest.** Если
  `/ingest_document` для media-вложения не отработал (файл не
  резолвится / TTL-cleanup / endpoint упал), OpenClaw plugin
  оставляет маркер вложения в turn-тексте вместо вырезания —
  вложение деградирует в обычное сообщение дневника, а не исчезает
  молча.
- **Валидация `message_split_part_chars`.** `config.load()` отвергает
  на старте конфигурацию, где `STYX_MESSAGE_SPLIT_PART_CHARS`
  не строго меньше лимита `memories.content` (2400) — иначе сплиттер
  выдавал бы части сверх CHECK constraint'а. Fail-fast, не clamp.

### Changed

- **Документ ≠ память (IAmBook §V).** `/ingest_document` теперь
  создаёт tail-memory с **маркером акта** архивации («я положил в
  архив документ такого-то рода» — тип/происхождение/о чём/ссылка),
  вместо «no tail-memory» волны 28. Содержание документа в память не
  входит — только акт.
- **Async ingest больших документов.** Документ с числом chunks
  больше `STYX_DOCUMENT_INGEST_ASYNC_CHUNK_THRESHOLD` ingest'ится
  через worker pool (новый handler `document_chunk_embed`): chunks
  INSERT'ятся с `embedding=NULL`, endpoint возвращается быстро.
- **OpenClaw plugin** перехватывает media-вложения turn'а
  (`media://inbound/...`) и шлёт документ documents-каналом
  (`/ingest_document`), в turn-текст подставляя только ссылку —
  документ не едет turn-каналом как 15K-символьный текст. Требует
  общего media-root между styx-daemon и OpenClaw-хостом
  (см. `docs/DEPLOYMENT.md` § 4.4).

## [1.0.0] — 2026-05-13

Первый публичный релиз.

Styx — инженерная реализация Locus: непрерывной среды между обращениями
к LLM, в которой разворачивается линия `я` агента. Подробности
концепции и архитектуры — в [`README.md`](README.md).

В релиз входит:

- **`packages/styx-core/`** — host-agnostic ядро + HTTP API daemon
  (FastAPI). Long-tier memory (PostgreSQL + pgvector), three-tier
  composer (active suffix / hot / long), background workers
  (importance scoring, lifecycle, classifier, sentiment, dialogue
  consolidation, reinterpret apply, memory consolidation,
  relation decay).
- **`packages/styx-hermes/`** — Hermes Agent plugin. Тонкий HTTP
  клиент к daemon'у. Регистрирует MemoryProvider, ContextEngine,
  Transport, `pre_llm_call` hook.
- **`extensions/styx/`** — OpenClaw plugin (TypeScript). 17 LLM
  tools, ContextEngine lifecycle bridge, 4 LLM-runbook skills.
- HTTP API (~30 endpoint'ов): lifecycle, composer, recall,
  search_archive, dialogue (5 routes), relations + graph traverse,
  reinterpret, ingest (experience + document), explain (3 modes)
  + analytics + confirm_usage. Контракт —
  [`docs/HTTP_API.md`](docs/HTTP_API.md).
- Memory markers taxonomy (`<styx-{salient,recall,archive,
  dialogue,relations,explain,working-set}>`) для различения
  source-канала инжекта в LLM-input'е.
- Document pipeline (PDF / DOCX / XLSX / Markdown / plain text)
  через pure-Python parsers.

Технологический стек: Python 3.11+, PostgreSQL 18 + pgvector,
Ollama (`embeddinggemma:300m-qat-q8_0` + `qwen3:4b-local`),
FastAPI, TypeScript.

Совместимость: Hermes Agent v2026.4.30+ (тестировалось до
v2026.5.7), OpenClaw v2026.5.3+.
