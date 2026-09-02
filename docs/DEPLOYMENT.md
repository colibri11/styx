# Styx — Deployment Runbook

Инструкция по установке Styx в production (актуальные версии пакетов —
[`CHANGELOG.md`](../CHANGELOG.md)). Архитектура изменилась относительно 0.1.0 — теперь это два пакета и
три процесса:

- **`styx-core`** — host-agnostic ядро + HTTP API daemon
- **`styx-hermes`** — тонкий plugin для Hermes Agent (HTTP клиент)
- **PostgreSQL** + **Ollama** — внешние deps, как и были

Полный архитектурный контекст — `.design/host-agnostic-split-v1.md`
(если есть в чекауте) и `docs/HTTP_API.md`.

## 0. Breaking changes при апгрейде

### Текущий переход: cognitive continuity и migration 0009

Core и оба host adapter'а нужно обновлять как один rollout. Миграция
`0009_cognitive_continuity.sql`:

- классифицирует `memories` в `dialogue`, `external_evidence` и
  `subjective_trace`, добавляет обязательный guard `line_eligible`;
- создаёт append-only cognitive acts, ordered actions, consequence inbox,
  fenced snapshots и отдельную typed `memory_lineage`;
- создаёт versioned `line_state`/`will_projections`; любое изменение live
  eligible subjective trace транзакционно помечает projection dirty;
- сохраняет UUID и archive/search поверхности существующих rows.

Новый host path — `/cognition/preturn` перед model call и ровно один
`/cognition/commit` после завершения tool loop. Commit несёт finalized channel
projection, ordered `call|result|error`, `host_key`, declared parent и
`snapshot_token`. Принимается до 32 явных consequences; каждый допустимый
`result`/`error` tool event также становится consequence, общий валидируемый
bound — 96. Core отклоняет переполнение, а не молча обрезает причинный feedback.
Consequences выдаются snapshot'у по lease, подтверждаются только terminal
act'ом с этим token и после истечения lease снова доступны, если snapshot был
брошен. Поэтому доставка recoverable at-least-once, а acknowledgement и retry
того же act идемпотентны; поздний commit истёкшего snapshot не подтверждает
повторную presentation, exactly-once при падении host не обещается. Legacy
writers используются только если новый endpoint вернул `404`; не на timeout,
auth, validation или `5xx`.

Рекомендуемый порядок rollout:

1. Сделать обычный backup PostgreSQL и остановить host writes либо перевести
   gateway'и в maintenance.
2. Обновить checkout/пакеты, затем выполнить `.venv/bin/styx migrate` один раз.
3. Перезапустить `styx-daemon`; проверить `/healthz` и наличие обоих cognition
   paths в `/openapi.json`.
4. Обновить `styx-hermes` и OpenClaw plugin, затем перезапустить host
   процессы. Не оставлять mixed-version fallback постоянным режимом.
5. Выполнить два последовательных хода с tool result. Для Hermes передавать
   физический `host_key` уже в preturn; OpenClaw v2026.8.2 сам доставляет
   accepted advancement key через durable `commitTurn` outbox. Второй preturn
   должен получить consequence первого, а terminal commit — вернуть ненулевой
   `acknowledged_consequences`. Повтор commit с тем же `host_key` должен вернуть
   `duplicate=true` без новых rows.

OpenClaw-плагин требует v2026.8.2: `assemble` сохраняет rich messages и строит
session-fenced snapshot, а `commitTurn` атомарно и идемпотентно продвигает
принятый transcript turn. Hooks `agent_end`/`before_prompt_build` для finality
не используются.

Схема additive, но откат только binaries на старый writer семантически
небезопасен: legacy writer не задаёт новые domain/eligibility поля явно.
Для rollback возвращайте согласованный snapshot БД либо сохраняйте новый core
до завершения обратной миграции данных.

### Исторический переход: causal affect continuity и migration 0008

Миграция `0008_emotional_evidence.sql` добавила `emotional_events` и
provenance-колонки. Она не объявляет старые rows достоверным evidence: legacy
provenance остаётся `NULL`.

- `STYX_AFFECTIVE_TRANSITION_ENABLED=0` отключает causal observer. Только в
  этом режиме `STYX_SENTIMENT_ENABLED` возвращает peer-only rollback path.
- `STYX_AFFECTIVE_TRANSITION_TIMEOUT_S` (default `8.0`) ограничивает один
  fail-open model-call. Ошибка observer не роняет durable cognitive act.
- Для OpenClaw требуется версия не ниже `2026.8.2`; старые hook permissions
  для causal finality больше не нужны.

### Историческое изменение 1.0.7

**styx-core 1.0.7 (волна 35, ADR § 58, 2026-07-01) — переименование
pre-LLM emotional-канала.** Если на этом деплое (например, действующий
Hermes-профиль) когда-либо выставлялись любые из этих ENV —
**они больше не читаются, вообще:**

- `STYX_PRE_LLM_PEER_VAD_ENABLED`
- `STYX_PEER_VAD_MIN_NORM`
- `STYX_PEER_VAD_TTL_S`

Канал `channel_peer_vad` был полностью заменён на `channel_self_state`.
В 1.1.0 его собственная семантика дополнительно изменена: вместо описательной
эмоциональной фразы он передаёт cognitive posture. Это не deprecated-алиас — старые имена не работают ни
как fallback, ни как no-op-предупреждение, они просто ничего не
конфигурируют. Эквиваленты для миграции конфига:

| Было (не читается) | Стало |
|---|---|
| `STYX_PRE_LLM_PEER_VAD_ENABLED` | `STYX_SELF_STATE_ENABLED` |
| `STYX_PEER_VAD_MIN_NORM` | `STYX_SELF_STATE_MIN_NORM` (тот же дефолт `0.2`) |
| `STYX_PEER_VAD_TTL_S` (default `60.0`, «окно свежести реакции») | `STYX_SELF_STATE_MAX_AGE_S` (default `900.0`, **другая семантика** — safety net на мёртвый `styx-worker`, не freshness-окно; см. `docs/CONFIGURATION.md`) |

Если ни одна из старых переменных не выставлялась явно (типичный
случай — все три жили на дефолтах) — апгрейд безопасен, действие не
требуется. Полный список текущих ENV — `docs/CONFIGURATION.md`.

## 1. Prerequisites

| Компонент | Версия | Зачем |
|---|---|---|
| Hermes Agent | точный tag `v2026.8.31`, package `0.21.0` | Проверенный canonical cognition host-контракт |
| PostgreSQL | 18+ с расширением `pgvector` | Long-tier memories + working_set persistence |
| Ollama | актуальная | Embedding (`embeddinggemma:300m-qat-q8_0`) + LLM workers (`qwen3:4b-local`) |
| Python | 3.11+ | Для `styx-core` daemon (минимальный slim Python) |
| `uv` | 0.5+ | Установка пакетов (стандарт workspace) |

Postgres инициализирован:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Ollama имеет обе модели:
```
ollama pull embeddinggemma:300m-qat-q8_0
ollama pull qwen3:4b-local
```

## 2. Install

### 2.1. Hermes Agent

Для canonical `/cognition/{preturn,commit}` deployment установите точно Hermes
Agent tag `v2026.8.31` (package `0.21.0`). Docker-путь ниже пинит официальный
образ `nousresearch/hermes-agent:v2026.8.31`; host-install должен быть закреплён
на том же релизе, а не следовать `main` или непинованному installer channel.

Upstream installer поддерживает pin через `--branch`; до attach Styx проверьте
его canonical version output:
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
  | bash -s -- --branch v2026.8.31
hermes --version
# expected package version in output: 0.21.0
```

После установки `hermes` доступен в `$PATH`, `~/.hermes/` создан как
`$HERMES_HOME`.

Более старые Hermes не являются canonical cognition deployment. Их можно
временно оставить только как документированный degraded legacy/mixed-version
lane: legacy preturn/sync/affect surfaces не дают атомарный fenced envelope,
recoverable consequence lease и полный terminal action feedback. Успешный
legacy fallback не считается проверкой семантики Wave 37.

### 2.2. Styx (workspace из git checkout)

```bash
cd /path/to/styx
uv sync
```

`uv sync` ставит и `styx-core`, и `styx-hermes` как editable из workspace.
Проверить что обе команды доступны:
```bash
.venv/bin/styx --help
.venv/bin/styx-hermes-setup --help
```

### 2.3. Hermes integration shim

Скопировать `styx-memory` shim в `$HERMES_HOME/plugins/`:
```bash
.venv/bin/styx-hermes-setup --hermes-home ~/.hermes
```

Это путь для **host-деплоя** (без Docker). В Docker-образе shim **bundled**
в `/opt/hermes/plugins/memory/styx-memory/` (см. § 7.1), отдельная установка
не нужна.

General plugin (`styx`) подхватывается через entry-point `hermes_agent.plugins`
из pip-установки — отдельного shim'а не требует.

### 2.4. Hermes config.yaml

```yaml
memory:
  provider: styx-memory

plugins:
  enabled:
    - styx
```

**`context.engine` ставить НЕ нужно.** Styx в Hermes — **memory-provider**:
он подмешивает память (`prefetch`/`system_prompt_block` per-turn + tools +
`on_pre_compress`), а компрессию всего окна ведёт сам Hermes своим штатным
компрессором. Styx context engine'ом компрессор Hermes НЕ подменяет (прежний
дизайн делал это и приводил к заглушке `[Result from earlier conversation …]`
вместо результатов — устранено в styx-hermes 1.0.8). Подключение Styx — это
ровно два ключа: `memory.provider: styx-memory` + `plugins.enabled += styx`.

Эти ключи на профиль выставляет идемпотентная команда
`styx-hermes-setup --attach` (см. § 4.7) — это **предпочтительный** способ
подключения профиля; ручная правка YAML — альтернатива. (`--attach` также
снимает legacy `context.engine: styx`, оставшийся от прежних версий.) Под
текущей моделью config **не патчится автоматически** при старте контейнера
(авто-bootstrap ретайрен, см. § 7.1): чистая установка оставляет все профили
на штатной памяти Hermes, пока их явно не подключат через `--attach`.

## 3. Database setup

```bash
export STYX_DATABASE_URL="postgresql://user:pass@host:5432/styx"
.venv/bin/styx migrate
```

Идемпотентно — повторный вызов ничего не пересоздаёт.

## 4. Three-process deploy

### 4.1. styx-daemon

Запускается отдельным процессом (systemd unit / docker container):
```bash
export STYX_DATABASE_URL="postgresql://user:pass@host:5432/styx"
export STYX_OLLAMA_URL="http://ollama:11434"
export STYX_HTTP_BIND="127.0.0.1"   # или 0.0.0.0 для удалённого Hermes
export STYX_HTTP_TOKEN="$(openssl rand -hex 32)"  # обязательно если bind != loopback
.venv/bin/styx daemon run
```

Проверка: `curl http://127.0.0.1:8788/healthz`.

Loopback rule: если `STYX_HTTP_BIND` не loopback и `STYX_HTTP_TOKEN`
пустой — daemon **не стартует**.

### 4.2. Hermes (с styx plugin)

```bash
export STYX_DAEMON_URL="http://127.0.0.1:8788"
export STYX_HTTP_TOKEN="<тот же что у daemon>"
hermes
```

`styx_hermes` plugin при старте сделает `POST /agent/initialize` к daemon'у.
Если daemon недоступен — Hermes упадёт с понятной ошибкой.

### 4.3. Postgres + Ollama

Стандартный hosting. Доступ из обоих процессов (Hermes + daemon) на:
- Postgres: `STYX_DATABASE_URL`
- Ollama: `STYX_OLLAMA_URL`

### 4.4. Общий media-root (documents-канал OpenClaw)

Обязательно при использовании OpenClaw plugin'а с документами-вложениями
(fix cdc5221 часть A — documents-канал).

Когда OpenClaw runtime получает вложение, он сохраняет файл в свой
media-store (`<config-dir>/media/inbound/<id>`) и в turn-текст
вставляет маркер `[media attached: media://inbound/<id>]`. Styx
OpenClaw plugin (`media-attachments.ts`) перехватывает маркер,
резолвит его в **абсолютный путь файла** и шлёт этот путь в
`POST /ingest_document` (path-mode) к styx-daemon. Демон **читает
файл с диска сам** — он не получает байты по HTTP.

Отсюда два жёстких требования к deploy:

1. **Общий media-root по идентичному абсолютному пути.** styx-daemon
   и OpenClaw-хост обязаны видеть директорию media-store по одному и
   тому же абсолютному пути. Если они на одной машине / в одном
   контейнере — условие выполнено само. Если в разных контейнерах —
   смонтируйте один volume в оба по идентичному `target` (см.
   `docker/docker-compose.test.yml`: named volume `styx-openclaw-media`
   → `/home/node/.openclaw/media` в openclaw-сервисах и в styx-daemon).
   Если пути расходятся — `/ingest_document` вернёт `422 file not
   found`.

2. **Media-root в `STYX_INGEST_DOC_ROOTS`.** Если whitelist непуст,
   директория media-store обязана быть в нём (resolved path
   проверяется через `relative_to`). Иначе — `422 path outside
   allowed roots`. Подробности про сам whitelist —
   `docs/CONFIGURATION.md` § «Ingest API (pipelines)».

happy-path documents-канала зависит **от обоих** условий. При сбое
ingest'а вложение деградирует безопасно: маркер остаётся в turn-тексте,
реплика уходит дневник-каналом как обычное сообщение (нарезается при
длине свыше лимита) — документ не теряется молча, но и не
архивируется. Чтобы документы реально попадали в архив
(`documents`+`chunks`), оба условия должны быть выполнены.

### 4.5. Периодический reembed (обязательный шаг эксплуатации)

Styx **не добивает `embedding IS NULL` автоматически** — воркера-комплитера
в daemon нет (reembed по дизайну — on-demand утилита). Хвосты длинных ходов
(inline embed-after-commit ограничен `message_split_inline_embed_cap`,
default 4) копятся с `embedding = NULL` и **не находятся семантическим
поиском**, пока их не реембедить. Авто-драйвер внутри daemon — open
follow-up; **пока backfill драйвится снаружи, и это надо настроить при
деплое.**

Поставь периодический прогон внешним планировщиком (cron / systemd timer),
например раз в 10 минут:

```bash
# host-агент / любой клиент с доступом к daemon по HTTP (styx-core ≥ 1.0.2):
*/10 * * * * curl -fsS -X POST http://127.0.0.1:8788/maintenance/reembed \
  -H "Authorization: Bearer $STYX_HTTP_TOKEN" \
  -H "Content-Type: application/json" -d '{"mode":"null_only"}' >/dev/null

# либо на хосте с доступом к процессу/контейнеру daemon'а — тем же cron'ом:
# */10 * * * * /path/to/.venv/bin/styx reembed
```

Перехлёст расписания безопасен: параллельные вызовы взаимоисключаются
session-level advisory lock'ом → лишний вернёт `{"skipped":true}`, двойного
embed'а нет. `--all` / `{"mode":"all"}` — полный re-embed после смены
embedding-модели (тяжелее, гонять вручную). Контракт эндпоинта —
`docs/HTTP_API.md` § `POST /maintenance/reembed`.

### 4.6. Переименование агента (`styx rename-agent`)

Админская миграционная операция (styx-core ≥ 1.0.3): переименовать
`agent_id` по всем agent-scoped таблицам. Нужна, в частности, при
переносе agent-scoped памяти в Styx, чтобы накопленные данные смотрели под
именем Hermes-профиля (`agent_identity` = имя профиля), — выравнивание
`agent_id` под Hermes-профили перед миграцией.

```bash
# сухой прогон — counts по таблицам, БД не трогается:
.venv/bin/styx rename-agent old-id new-profile-name --dry-run

# реальное переименование (одна транзакция; --yes для docker exec / скриптов):
docker exec -e STYX_DATABASE_URL="$STYX_DATABASE_URL" styx-daemon \
  styx rename-agent old-id new-profile-name --yes
```

Список таблиц вычисляется из `information_schema` в рантайме (новые
agent_id-таблицы покрываются автоматически). UUID'ы не меняются —
эмбеддинги, граф и переосмысления остаются нетронутыми; cross-agent
рёбра сохраняются. `new` не должен уже существовать (collision-refuse —
слияние агентов не поддержано).

**Операционная оговорка:** это прямая запись в БД. Если daemon в этот
момент держит in-memory state агента `old`
(focus_tracker / hot_tier / working_set), он осиротеет — выполняй на
**неактивном** агенте либо рестартни daemon после. Для агентов, ещё не
обслуживаемых styx-daemon (например на старом фронтенде), это безопасно.

### 4.7. Multi-agent Hermes — привязка профилей к Styx

**Модель.** Каждый Hermes-профиль = отдельный агент. `agent_id` в общем
styx-daemon = имя профиля (базовый профиль → `default`, именованный → его
имя). Один gateway-процесс на профиль; styx-daemon общий на все профили.
Чистая установка Styx на Hermes оставляет Styx установленным, но **не
подключённым ни к одному профилю** (включая базовый): bundled memory-shim
(§ 7.1) делает `StyxMemoryProvider` обнаружимым из любого профиля, но ни один
профиль его не использует — все работают на штатной памяти Hermes, пока их
явно не привяжут командой ниже. Авто-патча config больше нет (§ 7.1).

**Предусловие.** В env hermes-контейнера должны быть `STYX_DATABASE_URL`
(гейт выбора провайдера, § 7.1), `STYX_DAEMON_URL` и `STYX_HTTP_TOKEN`
(§ 4.2). Без `STYX_DATABASE_URL` провайдер не выберется даже после attach.
Эти переменные **контейнер-уровневые** — общие для всех профилей; `--clone`
копирует per-profile `.env` (идентичность профиля), но connection-vars Styx
берутся из env контейнера, не из профильного `.env`.

Все команды ниже — формат `docker exec -u hermes <hermes-container>
/opt/hermes/.venv/bin/styx-hermes-setup …`. В test-стеке
(`docker/docker-compose.test.yml`) hermes-контейнер — `styx-test-hermes`,
postgres — `styx-test-postgres`, `HERMES_HOME=/opt/data`. В произвольном
деплое имена контейнеров узнать через `docker compose -f <file> ps`.

**Применение attach — test-стек vs прод.**
- *Test-стек*: hermes-контейнер живёт как `sleep infinity`, постоянного
  gateway нет — профиль прогоняется разовым `hermes chat` (читает config
  заново при каждом запуске), поэтому attach действует **со следующего
  `hermes chat`, без рестарта**.
- *Прод* (`command=["gateway","run"]`): на каждый профиль свой long-running
  gateway (s6). Живой gateway переписывает `config.yaml` из памяти
  (`save_config_value`) → после attach **рестартнуть gateway этого профиля**.
  В одно-контейнерном test-стеке рестарт контейнера задел бы все профили; в
  проде рестартится только сервис нужного профиля.

**1. Подключить БАЗОВЫЙ профиль:**
```bash
docker exec -u hermes <hermes> /opt/hermes/.venv/bin/styx-hermes-setup \
  --attach --hermes-home /opt/data
```
Ожидаемый stdout (дословно):
```
attached '<name>' to Styx:
  memory.provider = styx-memory
  plugins.enabled += styx
  (context.engine не выставляется — компрессию ведёт сам Hermes)
  config: <path>/config.yaml
  backup: <path>/config.yaml.bak.<YYYYMMDD-HHMMSS>
Restart the profile gateway / container to apply (a running gateway reverts config edits).
```
Применить: в test-стеке — следующий `hermes chat` (рестарт не нужен); в
проде — **рестартнуть gateway этого профиля** (иначе живой gateway перезапишет
config из памяти и откатит attach). См. «Применение attach» выше.

**2. Подключить ИМЕНОВАННЫЙ профиль.** Профилю сперва нужен `config.yaml`.
Голый `hermes profile create <name>` его **НЕ создаёт** → attach выдаст
ошибку (см. п. 4). Создавай с наследованием config от активного профиля:
```bash
docker exec -u hermes -e HOME=/opt/data <hermes> \
  /opt/hermes/.venv/bin/hermes profile create <name> --clone
```
(`--clone` копирует `config.yaml`/`.env`/`SOUL.md` из активного профиля.)
Затем:
```bash
docker exec -u hermes <hermes> /opt/hermes/.venv/bin/styx-hermes-setup \
  --attach --profile <name> --hermes-home /opt/data
```
Применить как в п. 1 (test-стек: следующий `hermes chat` под профилем; прод:
рестарт gateway профиля). Если клонируешь от **уже подключённой** базы — клон
уже несёт styx-config, attach будет no-op (см. п. 3); клонируешь от
неподключённого профиля — attach патчит config.

**3. Идемпотентность.** Повторный attach уже подключённого профиля:
```
'<name>' already attached to Styx; no changes
```
EXIT 0, новый бэкап **не создаётся**.

**4. Ошибка «нет config»** (EXIT 1, дословно):
```
Config not found for profile '<name>' at <path>/config.yaml. The profile has no config.yaml yet — initialize it first (run the profile once), then re-run --attach.
```
Что делать: создать профиль через `--clone` (п. 2) **либо** один раз
запустить профиль, чтобы Hermes лениво создал config, затем повторить attach.

**5. Бэкап и откат.** Каждый реальный патч пишет `config.yaml.bak.<ts>` —
дословную копию исходника. attach переписывает весь `config.yaml` через YAML
round-trip, поэтому **комментарии / форматирование штатного config
теряются**; дословный исходник остаётся в `.bak`. Detach вручную:
восстановить из `.bak.<ts>` + рестарт.

**6. Выравнивание agent_id под существующую линию `я`.** Если имя профиля
отличается от уже накопленного styx-`agent_id` (миграция агента в Styx),
привести id командой `styx rename-agent <old> <new>` (§ 4.6) — чтобы профиль
смотрел на накопленную память, а не на пустого агента.

**7. Верификация.** Прогнать профиль (создаёт его `agent_id` в БД) — `hermes
chat` под `HERMES_HOME` этого профиля. `agent_identity` (=`agent_id`)
выводится из `HERMES_HOME`: `/opt/data`→`default`,
`/opt/data/profiles/<name>`→`<name>`:
```bash
# именованный профиль agent-a (agent_id=agent-a):
docker exec -u hermes -e HOME=/opt/data/profiles/agent-a \
  -e HERMES_HOME=/opt/data/profiles/agent-a styx-test-hermes \
  /opt/hermes/.venv/bin/hermes chat -Q -q "ping" -m <model>
```
Затем проверить, что профиль появился отдельным `agent_id`:
```bash
docker exec styx-test-postgres psql -U postgres -d styx -c \
  "select agent_id, count(*) from memories group by agent_id order by agent_id;"
```
Базовый профиль → `default`, именованный → его имя. Контент изолирован
per-agent.

**8. Чистая установка.** Чтобы воспроизвести чистый старт (Styx ни к одному
профилю не подключён), нужен **чистый `HERMES_HOME` volume**: переиспользуемый
volume может нести
styx-привязку от прошлых сессий (config уже пропатчен).

## 5. Validation

```bash
# Daemon живой
curl -s http://127.0.0.1:8788/healthz | jq .
# {"status":"ok","postgres":"ok","version":"<version>",...}

# Inspect API schema
curl -s http://127.0.0.1:8788/openapi.json | jq '.paths | keys'
curl -fsS http://127.0.0.1:8788/openapi.json | jq -e \
  '.paths["/cognition/preturn"] and .paths["/cognition/commit"]'

# Из Hermes-process (one-shot turn; с v2026.5.29.2 subcommand — chat, не ask;
# выверено против v2026.6.19)
hermes chat -q "Привет"  # в логах: "StyxMemoryProvider initialized" + "Styx pre_llm_call hook зарегистрирован"
```

После двух последовательных validation-turn'ов проверьте журнал без вывода
содержимого сообщений:

```sql
SELECT count(*) AS acts,
       count(*) FILTER (WHERE status = 'completed') AS completed
FROM cognitive_acts
WHERE agent_id = 'agent_demo';

SELECT status, count(*)
FROM cognitive_consequences
WHERE agent_id = 'agent_demo'
GROUP BY status ORDER BY status;
```

Не включайте `channel_input`, `channel_output`, action/consequence content или
provider credentials в deploy-логи и тикеты.

## 6. Troubleshooting

### Daemon не стартует: `STYX_HTTP_BIND=0.0.0.0 != loopback требует STYX_HTTP_TOKEN`

Это loopback rule. Поставь `STYX_HTTP_TOKEN`:
```bash
export STYX_HTTP_TOKEN="$(openssl rand -hex 32)"
```

Или binding на 127.0.0.1 (если Hermes на той же машине).

### Hermes падает: `styx-core daemon недоступен`

Проверь что daemon запущен и доступен по `STYX_DAEMON_URL`:
```bash
curl -fsS http://127.0.0.1:8788/healthz
```

### `/healthz` отдаёт `503 postgres=down`

Daemon не может подключиться к `STYX_DATABASE_URL`. Проверь DSN, доступ
к Postgres и применённость миграций (`styx migrate`).

### `/readyz` отдаёт `503 ollama=down`

`STYX_OLLAMA_URL` недоступен или Ollama не отвечает. Liveness (`/healthz`)
независим от Ollama.

### `/analytics` показывает растущий `pending_indexing.memories > 0`

Это NULL-хвосты длинных ходов (embedding не добит). Само не рассосётся:
авто-комплитера нет. Если счётчик растёт и не падает — **периодический
reembed не настроен**, см. § 4.5. Разовый прогон — `styx reembed` или
`POST /maintenance/reembed`.

## 7. Docker

См. `docker/docker-compose.test.yml` — готовый стек для integration-тестов
и быстрого деплоя. Сервисы: `postgres`, `styx-daemon`, `hermes-styx`
(Hermes-фронт) + `openclaw-gateway`, `openclaw-cli` (OpenClaw-track).
Собирается:

```bash
docker compose -f docker/docker-compose.test.yml --env-file .env up -d --build
```

`.env` содержит `OLLAMA_HOST_IP`, `STYX_HTTP_TOKEN`, провайдерские ключи
для Hermes и воспроизводимые host pins. Canonical defaults:

```dotenv
HERMES_IMAGE=nousresearch/hermes-agent:v2026.8.31
OPENCLAW_IMAGE=ghcr.io/openclaw/openclaw:2026.8.2
```

OpenClaw `2026.8.2` совпадает с plugin SDK в `extensions/styx/package-lock.json`.
Override `OPENCLAW_IMAGE=ghcr.io/openclaw/openclaw:latest` разрешён только как
отдельный latest-compat lane: его результат документируется отдельно и не
заменяет canonical pin. Аналогично, override `HERMES_IMAGE` не считается
canonical cognition deployment без отдельной compat-валидации exact tag/package.

### 7.1. hermes-styx обёртка на s6-overlay образе

Базовый образ `nousresearch/hermes-agent` — **s6-overlay**. `gateway` в нём
не самостоятельный бинарь на PATH: его поднимает s6-супервизор, а docker
CMD маршрутизируется `main-wrapper.sh` в `s6-setuidgid hermes hermes
gateway run`. Поэтому:

- **Styx ставится в образ** (pip: `styx-core` + `styx-hermes`), а
  **bundled memory-shim** копируется в `/opt/hermes/plugins/memory/styx-memory/`
  (Dockerfile `COPY`) → `StyxMemoryProvider` **обнаружим из любого профиля**.
  Bundled-каталог приоритетнее user-`$HERMES_HOME/plugins/`. Bundled-shim
  через `COPY` работает на любой базе, но runtime Hermes рассчитан на s6.
- **Авто-attach'а НЕТ.** В чистом образе shim обнаружим, но
  **ни один профиль не подключён**, включая базовый. Все профили работают на
  штатной памяти Hermes, пока их явно не подключат. Авто-патч `config.yaml`
  при старте контейнера ретайрен (cont-init `30-styx-bootstrap` удалён,
  `docker/styx-bootstrap.sh` ретайрен). Привязка профиля — явный
  идемпотентный шаг `styx-hermes-setup --attach` (см. § 4.7).
- **`STYX_DATABASE_URL` обязателен в env hermes-контейнера** (или `styx.json`
  с `database_url`): Hermes выбирает memory-провайдер по
  `StyxMemoryProvider.is_available()`, который гейтит по
  `STYX_DATABASE_URL`/`DATABASE_URL`, а **НЕ** по `STYX_DAEMON_URL`. Без него
  провайдер не выберется **даже после attach** (хотя обёртка ходит в daemon
  только по HTTP).
- **`HERMES_IMAGE` — exact `nousresearch/hermes-agent:v2026.8.31`** для
  canonical contract; override обязан оставаться s6-overlay образом и является
  отдельным compat lane (нужно для маршрутизации gateway-CMD через
  `main-wrapper.sh`).
- **Production Compose обязан задать `command: ["gateway","run"]`**:
  официальный образ v2026.8.31 не задаёт Docker CMD. В test-стеке команда —
  `["sleep","infinity"]` (контейнер живёт для
  `docker compose exec ... pytest` / `docker exec ... styx-hermes-setup`).
