# styx — OpenClaw plugin (TypeScript)

OpenClaw plugin под `~/.openclaw/plugins/styx/`. Подключает Styx-core
(FastAPI HTTP API daemon) к OpenClaw gateway: hooks (`message:preprocessed`
/ `message:sent` → `POST /sync_turn`), tools (`api.registerTool`),
context engine (`api.registerContextEngine`).

См. дизайн в `.design/NEXT.md` (раздел «Следующая задача — OpenClaw
plugin track») и будущий `.design/openclaw-plugin-v1.md`.

## Связь с docker-стиком

Эта папка bind-mount'ится в openclaw-контейнеры как
`/home/node/.openclaw/plugins/styx:rw`:
- `openclaw-gateway` — runtime
- `openclaw-cli`     — для `openclaw plugins install/inspect/...`

То есть редактирование `extensions/styx/src/...` на хосте сразу видно
обоим контейнерам без rebuild'а.

## Структура

```
extensions/styx/
├── openclaw.plugin.json   # манифест (id, capabilities, entry, skills)
├── package.json           # зависимости + build/test скрипты
├── src/
│   ├── index.ts           # definePluginEntry — entry point
│   ├── client.ts          # HTTP клиент к styx-daemon (8788)
│   ├── context-engine.ts  # bootstrap/ingest/assemble/compact/afterTurn/dispose
│   └── tools/             # 16 tools (recall/store/search_archive/...)
├── skills/                # LLM runbook'и (мини-волна 26.6)
│   ├── styx-capture/SKILL.md      # когда вызывать styx_store
│   ├── styx-recall/SKILL.md       # когда explicit query (после automatic block)
│   └── styx-reinterpret/SKILL.md  # переосмысление как weighted blend
├── dist/                  # tsc output
└── test/                  # vitest unit tests (host integration в styx-core)
```

## Skills (LLM runbook'и)

Манифест содержит `"skills": ["./skills"]` — OpenClaw runtime подгружает четыре SKILL.md в системный промпт когда description matches запрос пользователя. Это **дисциплина использования tool'ов для LLM**, не код:

- **styx-capture** — когда вызывать `styx_store`, что **не** делать (дублирование dialogue, pre-check duplicates, фрагменты помещающиеся в текущий turn).
- **styx-recall** — главный принцип: «прочитай automatic ContextEngine block в системном промпте первым, не дублируй через explicit recall». Дальше — два канала (line of `я` через `styx_recall` vs archive через `styx_search_archive`), 5 dialogue tools, knowledge graph, debugging через `styx_explain`.
- **styx-reinterpret** — Styx-уникальная операция «переосмысление через blend embeddings» ([IAmBook §V](https://github.com/colibri11/IAm/blob/main/IAmBook_EN.md)). Когда reinterpret vs supersede vs correction, choosing weight, что апплай deferred 30-90s.
- **styx-ingest** — `styx_ingest_document`: file → archive (PDF/DOCX/XLSX/Markdown через core-парсеры, волна 28). Pull-only архив, tail-memory не создаётся.

Скиллы консервативны: описывают только actually-implemented поведение волн 17-28 + отсылки к ADR. Не выдумывают параметры — каждый field соответствует tool factory'у в `src/tools/*.ts`.

## Sanity-check (после `docker compose up -d --wait`)

```bash
# Plugin виден в gateway
docker compose -f docker/docker-compose.test.yml exec openclaw-cli \
  node /app/dist/index.js plugins inspect styx --runtime --json

# styx-daemon доступен из openclaw сети
docker compose -f docker/docker-compose.test.yml exec openclaw-cli \
  node -e "fetch('http://styx-daemon:8788/healthz').then(r=>r.json()).then(console.log)"
```
