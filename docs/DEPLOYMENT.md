# Styx — Deployment Runbook (styx-core 1.0.2 / styx-hermes 1.0.3)

Инструкция по установке Styx в production (`styx-core` 1.0.2, `styx-hermes`
1.0.3). Архитектура изменилась относительно 0.1.0 — теперь это два пакета и
три процесса:

- **`styx-core`** — host-agnostic ядро + HTTP API daemon
- **`styx-hermes`** — тонкий plugin для Hermes Agent (HTTP клиент)
- **PostgreSQL** + **Ollama** — внешние deps, как и были

Полный архитектурный контекст — `.design/host-agnostic-split-v1.md`
(если есть в чекауте) и `docs/HTTP_API.md`.

## 1. Prerequisites

| Компонент | Версия | Зачем |
|---|---|---|
| Hermes Agent | `v2026.4.30+` | Host-фреймворк для plugin'а |
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

Официальный installer (если не установлен):
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

После установки `hermes` доступен в `$PATH`, `~/.hermes/` создан как
`$HERMES_HOME`.

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

General plugin (`styx`) подхватывается через entry-point `hermes_agent.plugins`
из pip-установки — отдельного shim'а не требует.

### 2.4. Hermes config.yaml

```yaml
context:
  engine: styx

memory:
  provider: styx-memory

plugins:
  enabled:
    - styx
```

**`context.engine: styx` обязателен.** Без него Hermes (≥ v0.15.2) берёт
встроенный compressor: Styx-движок регистрируется через entry-point, но
**не выбирается**, и `ContextEngine` Styx'а не работает (recall/инъекция
геометрии не происходит). `plugins.enabled += styx` включает плагин, но
не выбирает движок — это разные ключи.

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

## 5. Validation

```bash
# Daemon живой
curl -s http://127.0.0.1:8788/healthz | jq .
# {"status":"ok","postgres":"ok","version":"1.0.2",...}

# Inspect API schema
curl -s http://127.0.0.1:8788/openapi.json | jq '.paths | keys'

# Из Hermes-process (one-shot turn; в v2026.5.29.2 subcommand — chat, не ask)
hermes chat -q "Привет"  # в логах: "Using context engine: styx" + "StyxMemoryProvider initialized"
```

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
для Hermes.

### 7.1. hermes-styx обёртка на s6-overlay образе

Базовый образ `nousresearch/hermes-agent` — **s6-overlay**. `gateway` в нём
не самостоятельный бинарь на PATH: его поднимает s6-супервизор, а docker
CMD маршрутизируется `main-wrapper.sh` в `s6-setuidgid hermes hermes
gateway run`. Поэтому:

- **Bootstrap Styx — это s6 cont-init.d hook** (`docker/styx-bootstrap.sh`
  → `/etc/cont-init.d/30-styx-bootstrap` в `Dockerfile.styx-hermes`): он
  делает setup (`styx-hermes-setup`) + патч `config.yaml`
  (`memory.provider`/`plugins.enabled`/`context.engine`) при старте
  контейнера, ДО запуска gateway. НЕ перехватывает запуск через
  `exec gateway` (на s6 это даёт `exec: gateway: not found` → exit 127).
- **`command` обёртки в проде — дефолтный образа `["gateway","run"]`**
  (styx-bootstrap из command убран). В test-стеке — `["sleep","infinity"]`
  (контейнер живёт для `docker compose exec ... pytest`; cont-init hook
  всё равно отрабатывает setup).
- **`HERMES_IMAGE` обязан быть s6-overlay образом.** На не-s6 базе
  cont-init.d не выполнится → bootstrap молча не отработает.
- **`STYX_DATABASE_URL` нужен у hermes-styx-обёртки** (или styx.json с
  `database_url`): Hermes выбирает memory-провайдер по
  `StyxMemoryProvider.is_available()`, который гейтит по
  `STYX_DATABASE_URL`/`DATABASE_URL`, а НЕ по `STYX_DAEMON_URL`. Без него
  провайдер не выберется и `sync_turn` не вызовется (хотя обёртка ходит в
  daemon только по HTTP).
