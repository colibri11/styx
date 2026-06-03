# Changelog

Все значимые изменения в Styx документируются здесь. Формат —
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).
`styx-core` и `styx-hermes` версионируются независимо — heading'и помечают
пакет где это неоднозначно (`[1.0.2]`/`[1.0.3]` ниже — релизы `styx-hermes`,
`styx-core` тогда оставался на 1.0.1).

## [styx-core 1.0.2] — 2026-06-02

Минорный релиз `styx-core`. Новый HTTP-эндпоинт `POST /maintenance/reembed`
(волна 31) — backfill `memories.embedding` через HTTP. `styx-hermes` без
изменений (остаётся 1.0.3).

### Added

- **HTTP reembed-эндпоинт** `POST /maintenance/reembed` — host-агент в
  контейнере (ходит к Styx только по HTTP, docker.sock не примонтирован)
  сам добивает `embedding IS NULL`, а не только наблюдает счётчик
  `pending_indexing.memories` в `/analytics`. Дёргает тот же `run_reembed`
  (`commands/reembed.py`, волна 7e), что и CLI `styx reembed` — без
  дублирования логики. Sync-хендлер (FastAPI threadpool) не блокирует event
  loop на rate-limited embed-loop. Конкурентность — session-level advisory
  lock (key `9876543211`, отдельный от sweep `9876543210`): параллельный
  вызов получает `skipped: true`. Auth: `Depends(require_auth)` (Bearer как
  у `/ingest_experience`). Параметры — паритет CLI: `mode`
  (`null_only`/`all`), `agent_id`, `limit`, `dry_run`, `batch_size`,
  `rate_per_second`. Scope — только memories (chunks — follow-up).
  Контракт — `docs/HTTP_API.md` § `POST /maintenance/reembed`.

## [1.0.3] — 2026-06-02

Патч-релиз деплоя. Обёртка `styx-hermes` запускалась как entrypoint-wrapper
с `exec gateway run`, но официальный образ **`nousresearch/hermes-agent`**
(s6-overlay) НЕ держит `gateway` бинарём на PATH — его поднимает
s6-супервизор, а docker CMD маршрутизирует `main-wrapper.sh` в
`s6-setuidgid hermes hermes gateway run`. На s6-образе `exec gateway`
давал `exec: gateway: not found` → exit 127 → crash-loop. `styx-core`
daemon без изменений (остаётся 1.0.1) — `Dockerfile.styx-daemon` это не
затрагивает (slim-образ без s6).

### Fixed

- **HARD-BREAK деплоя на официальном s6-overlay образе Hermes.**
  Bootstrap переведён с entrypoint-wrapper на нативный **s6 cont-init.d
  hook** (`docker/styx-bootstrap.sh` → `/etc/cont-init.d/30-styx-bootstrap`).
  Скрипт делает только setup (styx-hermes-setup) + патч `config.yaml`
  (`memory.provider`/`plugins.enabled`/`context.engine`), **без `exec
  gateway`** — gateway поднимает сам s6. Хук выполняется как root в s6
  init ПОСЛЕ штатных `01-hermes-setup`/`015-supervise-perms`/
  `02-reconcile-profiles` (префикс `30-`), а setup+patch работают под
  `s6-setuidgid hermes` (файлы shim'а и config — владелец `hermes`).
  Shebang `#!/command/with-contenv sh` (восстановление контейнерного env).
- **Robustness: тихо-сломанный bootstrap на персистентном volume.**
  s6-overlay по дефолту (`S6_BEHAVIOUR_IF_STAGE2_FAILS=0`) НЕ халтит
  контейнер при падении cont-init — а `styx-hermes-setup --force` (rmtree
  shim'а) падал с EACCES, если HERMES_HOME-volume накопил root-owned
  `__pycache__` от прежних прогонов: контейнер поднимался «healthy», но
  shim не доустановлен → memory-провайдер молча неактивен → `sync_turn`
  не вызывался. Хук (бежит как root) теперь chown'ит `plugins/styx-memory`
  на hermes ПЕРЕД setup → rmtree всегда проходит. Healthcheck test-стека
  усилен: проверяет наличие `plugins/styx-memory/__init__.py` + патч
  `styx-memory` в config — половинчатый bootstrap виден как unhealthy.

### Changed

- **Деплой:** `docker/Dockerfile.styx-hermes` — base-образ по умолчанию
  `nousresearch/hermes-agent:v2026.5.29.2` (s6-overlay), bootstrap кладётся
  в `/etc/cont-init.d/30-styx-bootstrap` (вместо `/usr/local/bin/styx-bootstrap`).
- **Деплой:** `command` обёртки в проде — **дефолтный образа**
  `["gateway","run"]` (styx-bootstrap из `command` убран); в
  `docker/docker-compose.test.yml` test-сервис держится живым через
  `["sleep","infinity"]` (cont-init hook отрабатывает setup до CMD).
- **Docs:** `docs/DEPLOYMENT.md` синхронизирован — шапка `styx-core 1.0.1 /
  styx-hermes 1.0.3`, §5 `hermes ask`→`hermes chat -q` (в v2026.5.29.2 нет
  subcommand `ask`), новый §7.1 про s6 cont-init bootstrap, требование
  s6-образа для `HERMES_IMAGE` и `STYX_DATABASE_URL` у обёртки.

### Tests

- Smoke против официального s6-образа `nousresearch/hermes-agent:v2026.5.29.2`:
  контейнер стартует без crash-loop (нет `exec: gateway: not found`),
  `→ gateway is now running under s6 supervision`, `Using context engine:
  styx`, `StyxMemoryProvider initialized` (без `update_model TypeError`);
  config пропатчен идемпотентно (рестарт не задваивает `styx` в
  `plugins.enabled`), shim установлен с владельцем `hermes`.
- **Живой end-to-end** (стек postgres + styx-daemon + hermes-styx против
  официального s6-образа, провайдер z.ai `glm-4.6`): полный `hermes chat`
  turn → context engine styx → memory-провайдер styx-memory → `POST
  /agent/initialize 200` + `POST /sync_turn 200` (+ `auto_link`) → запись
  в `memories` (счётчик инкрементится). Robustness-харднинг проверен
  репродукцией: умышленно загрязнённый volume (root-owned `__pycache__`)
  ронял cont-init (exit 1) до фикса — после chown-харднинга cont-init
  `exited 0`, shim переустановлен (владелец hermes), контейнер healthy.

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
