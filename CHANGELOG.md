# Changelog

Все значимые изменения в Styx документируются здесь. Формат —
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).
`styx-core` и `styx-hermes` версионируются независимо — heading'и помечают
пакет где это неоднозначно (`[1.0.2]`/`[1.0.3]` ниже — релизы `styx-hermes`,
`styx-core` тогда оставался на 1.0.1).

## [styx-core 1.0.8] — 2026-07-01

Волна 35 follow-up: переформулировка 3 фраз `OCTANTS` (ADR § 59).

### Изменено

- **3 из 8 фраз в `channel_self_state` заменены** — читались коряво в
  шаблоне «Тебе сейчас X»: «оживлённо и уверенно» → «воодушевлённо и
  уверенно»; «мягко и расслабленно» → «умиротворённо и расслабленно»;
  «сдержанно и тяжело» → «тяжело и непреклонно». Остальные 5 не тронуты.
- `docs/HTTP_API.md` — попутно исправлен пропуск волны 35: пример ответа
  `/pre_llm_inject` всё ещё показывал старый `"Peer прозвучал: ..."`.

## [styx-core 1.0.7] — 2026-07-01

Волна 35: self-state expression channel (ADR § 58). `styx-hermes` без
изменений — чисто server-side, wire-протокол к Hermes не менялся.

### Изменено

- **Канал `channel_peer_vad` заменён на `channel_self_state`.** Раньше
  канал читал сырой VAD последней peer-реплики и генерил «Peer прозвучал:
  X» — описание собеседника. Теперь читает накопленное состояние агента
  (`emotional.state.read_last_state` — уже с decay и K_HOT-резонансом
  peer'а) и говорит от лица агента: «Тебе сейчас X». Закрывает открытый
  вопрос Q6/§21.1 (волна 7d) со ссылкой на IAmBook §29/§IX.
- **`dialogue_batch_consolidation.py`** — добавлен явно демаркированный
  user-only VAD-блок в промпт (усиление role-separation поверх уже
  существующей инструкции).
- Конфиг `peer_vad_enabled`/`peer_vad_min_norm`/`peer_vad_ttl_s`
  (`STYX_PEER_VAD_*`) удалён; новый `self_state_enabled`/
  `self_state_min_norm`/`self_state_max_age_s` (`STYX_SELF_STATE_*`).
  **Прод-деплой:** если Hermes-агент выставлял старые `STYX_PEER_VAD_*`
  ENV — переключиться на новые имена.

### Исправлено

- **User-only VAD-блок в batch-path не был ограничен по размеру** и
  приклеивался целиком к промпту каждого chunk'а — на большом backlog'е
  (окно > 85k символов) это давало HTTP 400 "exceeds available context
  size" на каждом chunk-вызове, причём неудачный tick не продвигал окно
  вперёд, гарантируя повторный и всё более тяжёлый отказ (потенциальный
  перманентный стопор consolidation). Добавлена граница
  `USER_ONLY_VAD_MAX_CHARS=18000` с обрезкой по хвосту.

## [styx-core 1.0.6] — 2026-07-01

Compat-доводка: `recall_memory_limit` — настраиваемый лимит memories в

Compat-доводка: `recall_memory_limit` — настраиваемый лимит memories в
salient-блоке (ADR § 57). `styx-hermes` без изменений.

### Добавлено

- **`StyxConfig.recall_memory_limit`** (+ `STYX_RECALL_MEMORY_LIMIT`) —
  override для `FullRecallConfig.memory_limit` (сколько memories попадает
  в salient-блок, инжектируемый через `prefetch()`/`build_salient_block()`
  каждый turn). Дефолт остаётся `6`. Fail-fast валидация диапазона `[1,
  20]` на этапе `load()` — по прецеденту `message_split_part_chars`; те же
  границы, что уже использует per-call override тула `styx_recall`.

## [styx-hermes 1.0.8] — 2026-06-15

Архитектурный фикс роли Styx в Hermes. `styx-core` без изменений (1.0.5).

### Изменено

- **Hermes-плагин больше НЕ регистрируется как context engine.** Прежний
  дизайн подменял собой штатный компрессор Hermes (`agent.context_compressor`)
  и гнал `should_compress()→True` каждый ход — Styx, будучи лишь малой
  частью контекстного окна, не имеет права управлять компрессией всего
  окна. Правильная роль Styx в Hermes — **memory-provider**: память
  подмешивается каналами provider'а — `prefetch` / `system_prompt_block`
  (per-turn, в system-prompt/инпут, messages-пары не трогает) + tools +
  `on_pre_compress` (Styx отдаёт, что сохранить в summary). Компрессию
  всего окна ведёт встроенный компрессор Hermes сам.
- **Устранена заглушка `[Result from earlier conversation …]` вместо
  результатов Styx.** Её ставил встроенный компрессор Hermes, когда
  Styx-движок не был выбран (стабил tool-результаты). Теперь память едет
  каналом инъекции и под компрессию `messages` не попадает.
- **`styx-hermes-setup --attach` больше не ставит `context.engine: styx`**
  и снимает его при ре-attach (config возвращается к штатному компрессору
  Hermes). Подключение Styx = `memory.provider: styx-memory` +
  `plugins.enabled += styx`.

### Удалено

- `styx_hermes.engine.context.StyxContextEngine` (и его тесты) — класс
  больше не используется.

## [styx-core 1.0.5] — 2026-06-15

Defect-fix `styx-core`. `styx-hermes` бампается следом (пин зависимости):
1.0.6 → 1.0.7.

### Исправлено

- **Salient-блок разрывал tool-пару при компрессии в середине
  tool-loop'а.** `StyxComposer.compress()` вставлял salient (role=user)
  через `_inject_salient_block(..., len(body)-1)` — на позицию ПЕРЕД
  последним сообщением окна, без учёта tool-пар. Когда `compress()`
  звался в середине tool-loop'а (до финального текста модели), хвост
  окна был tool-результатом, и salient вклинивался между
  `assistant(tool_calls)` и его `tool(result)`. Дальше (вне core)
  результат отрывался/дропался, осиротевший tool_call на следующем
  `/context/build` затыкался `STUB_TOOL_RESULT` (`"[Result from earlier
  conversation — see context summary above]"`) — симптом «заглушка
  вместо результата тула» в проде. Фикс: pair-aware позиция вставки
  (`_safe_salient_insert_index`) — для обычного хвоста (user/assistant-
  текст) позиция не меняется (cache-намерение вол. 26.5 сохранено),
  сдвиг влево только когда хвост — tool-группа. Найдено в боевой
  эксплуатации, воспроизведено на стенде. styx-core 1.0.4 → 1.0.5;
  styx-hermes 1.0.6 → 1.0.7 (бамп пина `styx-core==1.0.5`).

## [1.0.6] — 2026-06-14

Релиз `styx-hermes` — defect-fix routing tool-call'ов. `styx-core` без
изменений (остаётся 1.0.4).

### Исправлено

- `StyxMemoryProvider.get_tool_schemas()` теперь отдаёт статический каталог
  ядра до `initialize()` (fallback на `StyxMemoryCore.get_tool_schemas()`
  — чистый, без БД/HTTP). Раньше до init метод возвращал пустой список:
  Hermes строит routing-индекс `_tool_to_provider` в
  `MemoryManager.add_provider()` ДО `initialize()` (`agent_init.py:1101`
  vs `:1144`), поэтому индекс получался пустым и каждый `styx_*` tool-call
  падал в `Unknown tool` — при том что схема к моменту инжекта в модель
  (`agent_init.py:1176`, уже после init) была видна. Найдено live-e2e на
  Hermes v0.16.0.

## [1.0.5] — 2026-06-10

Релиз `styx-hermes` — совместимость с **Hermes Agent v0.16.0** (тег/образ
`v2026.6.5`). Compat-валидация по образцу § 47: ABI-breaks для внешнего
context engine / memory provider / transport **не найдены**, код адаптера
не менялся. `styx-core` без изменений (остаётся 1.0.4).

### Проверено против Hermes v0.16.0

- Изменения Hermes 0.15.2 → 0.16.0 на наших поверхностях: новый
  **опциональный** метод движка `should_defer_preflight_to_real_usage()`
  (base-default False, вызов через `getattr`-fallback — `StyxContextEngine`
  наследует базу, безопасно); статический fallback компакции
  `_build_static_fallback_summary` (внутренний, plugin `compress()` не
  затрагивает); `_EngineCollector.register_command` (slash-команды движков,
  опционально); synthetic package namespace для user-installed memory
  providers. Сигнатуры `update_model(api_mode=)` / `compress()` /
  discovery-пути — без изменений.
- **Батарея:** drift-sentinel **55/55** (4 якоря переехали — выверены
  вручную, +9 новых якорей 0.16.0); host-suite **174 passed / 6 infra-skip**;
  Docker in-container полная **180 passed / 0 skipped** + core-suite
  **1461 passed / 12 design-skip** (Postgres + Ollama); live e2e против
  официального образа `nousresearch/hermes-agent:v2026.6.5` — boot s6 чистый,
  `registered context engine: styx`, полный `hermes chat` turn (z.ai glm-5.1)
  → `POST /sync_turn 200`, обе реплики легли в `memories`.

### Исправлено

- **Протухший пин `styx-core==1.0.3`** в `pyproject.toml` styx-hermes —
  конфликтовал с актуальным styx-core 1.0.4 (бамп волны 34). Теперь `==1.0.4`.
- **`__version__ = "1.0.0"`** в `styx_hermes/__init__.py` отставал от
  pyproject с самого split'а (тот же класс дефекта, что healthz-фикс
  styx-core 1.0.4). Теперь синхронизирован: `1.0.5`.

### Изменено

- Дефолтный базовый образ test-стека и `Dockerfile.styx-hermes`:
  `nousresearch/hermes-agent:v2026.5.29.2` → **`v2026.6.5`**
  (`docker-compose.test.yml`, `ARG HERMES_IMAGE`).
- `docs/DEPLOYMENT.md`: примеры выверены против v2026.6.5; пример `healthz`
  отражает styx-core 1.0.4.

## [styx-core 1.0.4] — 2026-06-09

Defect-fix `styx-core` (волна 34). Боевой инцидент в продакшене 2026-06-09
(agent_id=main): `POST /memory_store` с `session_id`, которого нет
в `sessions`, ронял запись по `ForeignKeyViolation` → **HTTP 500, память
потеряна**, а постоянное per-agent соединение оставалось в aborted-state →
следующие запросы агента падали `InFailedSqlTransaction` до ближайшего
guarded-вызова. `styx-hermes` без изменений (остаётся 1.0.4).
**Закрывает запаркованный follow-up** из ревью § 46 («Аудит rollback-guard'ов
всех commit-сайтов»).

### Исправлено

- **FK→NULL деградация на пути `memory_store`.** Перед insert'ом
  `memory_store` и `_memory_store_routed` зовут новый
  `AgentScopedQueries.session_exists()` (`SELECT 1 FROM sessions WHERE id=%s`):
  если строки сессии нет — `log.warning` + деградация `session_id → NULL`
  (память сохраняется; NULL `session_id` штатен — FK `ON DELETE SET NULL`,
  877 рядов main уже такие). Выбор pre-check, а не catch-FK-retry:
  детерминизм (нет повторной вставки в aborted-транзакцию), стоимость —
  один indexed lookup до дорогого `embed()`.
- **Отравление постоянного соединения при ошибке записи.** Любой сбой
  write-блока на постоянном `self._conn` больше не оставляет соединение
  в aborted-state — единый rollback-guard откатывает транзакцию и
  пробрасывает исключение (route отдаёт 500, соединение остаётся рабочим
  для следующих запросов того же агента).

### Изменено

- **Хелпер `_guarded_write(label)`** (context-manager на `StyxMemoryCore`)
  инкапсулирует прежний ad-hoc паттерн `try/except → rollback → re-raise`
  (как в `sync_turn`/`ingest_single_message`).
- **Sweep всех write-входов на постоянном соединении под единый guard.**
  Обёрнуты ранее незащищённые (`memory_store`, `reinterpret_enqueue`,
  `ingest_experience`, `dialogue_save` 1-й блок, `confirm_usage`,
  `initialize` init-upsert), расширен частичный (`_memory_store_routed` —
  guard теперь на `auto_link`+`commit`), мигрированы уже-guarded ради
  единого паттерна (`sync_turn`, `ingest_single_message`, `ingest_document`,
  `handle_tool_call`).
- **Два HTTP-роута, коммитившие постоянный conn в обход `memory.py`**
  (найдены ревью), обёрнуты в `core._guarded_write(...)`: `POST /recall`
  (`recall.py`, recall_event commit) и `POST /link` (`relations.py`);
  `/link` дополнительно получил `session.write_lock` (guard зовёт rollback
  на shared conn → нужна сериализация).
- **Best-effort embed-after-commit фаза оставлена swallow** (НЕ re-raise):
  2-й embed-блок `dialogue_save` сохраняет локальный
  `try/except → rollback → swallow` (как `sync_turn`/`ingest_single_message`) —
  сообщение уже durably закоммичено в 1-м блоке, re-raise здесь был бы
  регрессией.

## [1.0.4] — 2026-06-03

Релиз `styx-hermes` (волна 33, multi-agent Hermes provisioning, этап 1).
Чистая установка Styx больше НЕ подключает ни одного профиля — образ
оставляет Styx установленным, но не подключённым ни к одному профилю
(все профили на штатной памяти Hermes); подключение к Styx стало отдельным
явным идемпотентным шагом. `styx-core`
без изменений (остаётся 1.0.3).

### Добавлено

- **Bundled styx-memory shim в образ.** Shim кладётся в образ как
  bundled-каталог `/opt/hermes/plugins/memory/styx-memory/` → memory
  discovery находит его из любого профиля без per-profile/per-base
  установки. Host-деплои продолжают пользоваться `styx-hermes-setup`
  shim-install (`--force`) — этот путь без изменений.
- **Идемпотентная attach-команда** `styx-hermes-setup --attach
  [--profile <name>]` — подключает Hermes-профиль к Styx патчем его
  `config.yaml`: `memory.provider: styx-memory`, `plugins.enabled += styx`
  (без дублей, существующие сохранены), `context.engine: styx`. Target:
  база (без `--profile`) — `<hermes_home>/config.yaml`; именованный —
  `<hermes_home>/profiles/<name>/config.yaml`. Перед патчем — файловый
  бэкап `config.yaml.bak.<ts>` (дословная копия исходных байт), только
  когда патч реально меняет файл; идемпотентный повтор — no-op, exit 0,
  без бэкапа. Нет `config.yaml` → чистая английская ошибка + nonzero
  exit, config НЕ создаётся.
- Объявлена прямая зависимость **`pyyaml>=6.0`** (attach читает/пишет
  `config.yaml`); раньше тянулась транзитивно.

### Деплой

- **Не подключён по умолчанию.** Cont-init авто-патч базы при старте убран из
  образа (`docker/styx-bootstrap.sh` ретайрен); подключение базы — теперь
  тоже явный `styx-hermes-setup --attach` (без `--profile`).

## [styx-core 1.0.3] — 2026-06-03

Минорный релиз `styx-core`. Новая админская CLI-команда
`styx rename-agent <old> <new>` (волна 32) — переименование `agent_id`
по всем agent-scoped таблицам. `styx-hermes` без изменений (остаётся 1.0.3).

### Added

- **CLI `styx rename-agent <old> <new>`** — переименование `agent_id`
  во всех таблицах со столбцом `agent_id` (этап 1 миграции agent-a/agent-b
  memorybox→styx: приведение `agent_id` к именам Hermes-профилей).
  Список таблиц **schema-driven** из `information_schema.columns`
  (не хардкод — инвариант волны 27: пропущенная таблица = расщеплённое
  `я`). Одна транзакция на все `UPDATE` (атомарность «всё-или-ничего»);
  UUID'ы (`memory_id`/`document_id`/`session_id`, граф
  `source_id/target_id`) не трогаются — эмбеддинги, граф, переосмысления,
  эмоц-снапшоты остаются байт-в-байт, cross-agent рёбра сохраняются.
  Existence-гард (`old` обязан существовать) + collision-refuse (`new`
  не должен иметь данных — слияние агентов не поддержано). `--dry-run`
  считает per-table counts без записи; `--yes` пропускает интерактивный
  confirm (для `docker exec`). Имена таблиц через `psycopg.sql.Identifier`
  (не f-string), значения — параметры. Логика — `commands/rename_agent.py`
  (`run_rename_agent`, caller-owned conn, не коммитит сам — как `reembed`).
  Операционная оговорка: прямая запись в БД, выполнять на неактивном
  агенте / рестартить daemon после (in-memory state осиротеет).

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
