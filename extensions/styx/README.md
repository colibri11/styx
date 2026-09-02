# styx — OpenClaw plugin (TypeScript)

OpenClaw plugin под `~/.openclaw/plugins/styx/`. Подключает Styx-core
(FastAPI HTTP API daemon) к OpenClaw gateway: tools (`api.registerTool`) и
durable context engine (`api.registerContextEngine`).

Требуется **OpenClaw >= 2026.8.2**. Плагин собирается и проверяется с точно
этой версией из lock-файла. Он объявляет current-turn fence и atomic
idempotent advancement, поэтому OpenClaw не деградирует persistent turn к
legacy context engine.

`ContextEngine.assemble` вызывает `/cognition/preturn` и добавляет один
`<styx-cognitive-continuity>` block в порядке: whole-line causal carrier →
continuity freshness → current posture → frozen observations → reconstructed
subjective traces.
OpenClaw AgentMessage возвращаются без нормализации: tool calls, multimodal
parts и provider metadata остаются у host. Встроенный memory prompt OpenClaw
добавляется ровно один раз рядом со Styx block.

Принятый durable turn записывает `ContextEngine.commitTurn`, который OpenClaw
вызывает через SQLite outbox и повторяет после restart. Core атомарно выбирает
последний live snapshot только той же session, выводит parent из последнего
принятого session act и дедуплицирует по advancement key. `agent_end` и prompt
hooks не являются владельцами finality и плагином не регистрируются.

Observations выдаются по recoverable lease: fallback-повтор того же prompt
получает точный сохранённый envelope, а accepted `commitTurn` подтверждает
только snapshot своей session. Истёкший или отсутствующий snapshot не
заменяется данными другой session.

Plugin передаёт bounded finalized projection до 64 ordered
`call|result|error` events суммарно. Result/error, уже увиденный внутри акта,
остаётся same-act journal event и не возвращается следующему акту как новое
наблюдение. Client предоставляет явный `cognitionObserve` только sensor/
connector-интеграциям для предварительно редуцированных post-act различий.
Commit создаёт durable reduction outcome; canonical reducer
асинхронно выводит 0..4 evidence-bound residues и перестраивает causal
carrier. Повтор advancement key с изменённым bounded request получает `409`.

Архитектурный контракт — в корневом `README.md` и `docs/HTTP_API.md`.

## Связь с docker-стиком

Эта папка bind-mount'ится в openclaw-контейнеры как
`/home/node/.openclaw/plugins/styx:rw`:
- `openclaw-gateway` — runtime
- `openclaw-cli`     — для `openclaw plugins install/inspect/...`

После `npm run build` изменения в `extensions/styx/dist/...` сразу видны
обоим контейнерам без rebuild Docker image.

## Структура

```
extensions/styx/
├── openclaw.plugin.json   # манифест (id, capabilities, entry, skills)
├── package.json           # зависимости + build/test скрипты
├── src/
│   ├── index.ts           # definePluginEntry — entry point
│   ├── client.ts          # HTTP клиент к styx-daemon (8788)
│   ├── context-engine.ts  # assemble/commitTurn/compact/maintenance lifecycle
│   ├── hooks/agent-end.ts # bounded message/tool projection helpers
│   └── tools/             # 17 tools (recall/store/search_archive/...)
├── skills/                # LLM runbook'и (мини-волна 26.6)
│   ├── styx-capture/SKILL.md      # когда вызывать styx_store
│   ├── styx-recall/SKILL.md       # когда explicit query (после automatic block)
│   └── styx-reinterpret/SKILL.md  # переосмысление как weighted blend
├── dist/                  # tsc output
└── scripts/               # contract tests (host integration в styx-core)
```

## Skills (LLM runbook'и)

Манифест содержит `"skills": ["./skills"]` — OpenClaw runtime подгружает четыре SKILL.md в системный промпт когда description matches запрос пользователя. Это **инженерная дисциплина использования tool'ов для LLM**, а не онтологическое описание системы:

- **styx-capture** — когда вызывать `styx_store`, что **не** делать (дублирование dialogue, pre-check duplicates, фрагменты помещающиеся в текущий turn).
- **styx-recall** — сначала прочитать fenced cognition envelope, не дублировать его explicit recall. Дальше — subjective traces, cited archive/dialogue evidence, knowledge graph и scoring diagnostics.
- **styx-reinterpret** — weighted blend embeddings как текущая инженерная политика Styx, а не норма трактата. Когда reinterpret vs supersede vs correction, choosing weight, что апплай deferred 30-90s.
- **styx-ingest** — `styx_ingest_document`: file → archive
  (PDF/DOCX/XLSX/Markdown через core-парсеры, волна 28). Тело остаётся
  pull-only архивом; создаётся одна короткая act-marker memory с
  `memory_domain=external_evidence`, `line_eligible=false`, а dedup retry её
  не дублирует.

Онтологический источник — [IAmBook.md](https://github.com/colibri11/IAm/blob/main/IAmBook.md), прикладные границы — [IAmPhilosophyOfSilicon.md](https://github.com/colibri11/IAm/blob/main/IAmPhilosophyOfSilicon.md). Скиллы описывают только actually-implemented инженерное поведение; каждый field соответствует tool factory в `src/tools/*.ts`.

## Sanity-check (после `docker compose up -d --wait`)

```bash
# Plugin виден в gateway
docker compose -f docker/docker-compose.test.yml exec openclaw-cli \
  node /app/dist/index.js plugins inspect styx --runtime --json

# styx-daemon доступен из openclaw сети
docker compose -f docker/docker-compose.test.yml exec openclaw-cli \
  node -e "fetch('http://styx-daemon:8788/healthz').then(r=>r.json()).then(console.log)"
```
