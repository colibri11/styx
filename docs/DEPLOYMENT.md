# Styx — Deployment Runbook (v1.0.0)

Инструкция по установке Styx 1.0.0 в production. Архитектура изменилась
относительно 0.1.0 — теперь это два пакета и три процесса:

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
memory:
  provider: styx-memory

plugins:
  enabled:
    - styx
```

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

## 5. Validation

```bash
# Daemon живой
curl -s http://127.0.0.1:8788/healthz | jq .
# {"status":"ok","postgres":"ok","version":"1.0.0",...}

# Inspect API schema
curl -s http://127.0.0.1:8788/openapi.json | jq '.paths | keys'

# Из Hermes-process
hermes ask "Привет"  # должен работать; в логах появится "StyxMemoryProvider initialized"
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

## 7. Docker

См. `docker/docker-compose.test.yml` — три сервиса (postgres, styx-daemon,
hermes-styx), готовый стек для integration-тестов и быстрого деплоя.
Собирается:

```bash
docker compose -f docker/docker-compose.test.yml --env-file .env up -d --build
```

`.env` содержит `OLLAMA_HOST_IP`, `STYX_HTTP_TOKEN`, провайдерские ключи
для Hermes.
