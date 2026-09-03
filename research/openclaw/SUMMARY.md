# OpenClaw integration research

Актуальный проверенный snapshot: **OpenClaw `v2026.8.2`**, commit
`0965053fe6b9341776df147a6934b7485c60b5ca` (2026-08-31). Markdown-копии
`docs_*.md` синхронизированы напрямую с официальным репозиторием
[`openclaw/openclaw`](https://github.com/openclaw/openclaw/tree/v2026.8.2/docs).
Они являются vendor reference, а не документацией поведения Styx.

## Текущий вывод для Styx

Styx регистрируется как **context-engine plugin**, а не как memory-slot plugin:

- manifest: `kind: "context-engine"`;
- runtime: `api.registerContextEngine("styx", factory)`;
- пользовательская memory-поверхность: 17 явных `styx_*` tools через
  `api.registerTool`;
- минимальная и проверенная версия host: OpenClaw `2026.8.2`.

Это разделяет две ответственности: OpenClaw владеет выполнением принятого
turn'а и его durable delivery, а Styx владеет своей agent-scoped памятью,
fenced cognitive snapshot и атомарным advancement в daemon/PostgreSQL.

## Контракт context engine

| OpenClaw lifecycle | Поведение Styx |
|---|---|
| `bootstrap` | Инициализирует связку OpenClaw agent/session с core daemon. |
| `ingest` / `ingestBatch` | Не создают отдельную финальную запись turn'а; остаются no-op, чтобы не конкурировать с accepted-turn finality. |
| `assemble` | Вызывает `/cognition/preturn`, сохраняет session-fenced snapshot и добавляет один `<styx-cognitive-continuity>` block, не нормализуя host messages. |
| `commitTurn` | Принимает только успешно завершённый host turn и идемпотентно вызывает `/cognition/commit` по durable `advancementKey`. |
| `compact` | Либо использует Styx compaction, либо явно делегирует штатному runtime в debug-режиме. |
| `afterTurn` | Выполняет только post-turn maintenance; не владеет finality. |
| `dispose` | Очищает bounded transient coordination state. |

`transcriptSemantics` объявляет current-turn fence и atomic idempotent
advancement. OpenClaw хранит retryable commit в SQLite outbox и вызывает
`commitTurn` только для принятого успешного turn'а. Поэтому `agent_end`,
`message_sent` и prompt hooks не используются как второй владелец записи.

Если надёжный `agentId` не выводится из `sessionTarget`, `runtimeContext`,
`sessionKey` или `agentDir`, engine работает в passthrough. Выбирать чужую
agent-scoped память по догадке нельзя.

## Prompt, сообщения и compaction

- `assemble()` получает host messages и token budget до model call.
- Styx возвращает `promptAuthority: "assembled"` и собственную оценку токенов.
- Встроенный OpenClaw memory prompt добавляется один раз через
  `buildMemorySystemPromptAddition(...)` рядом со Styx envelope.
- Text, tool-call, multimodal parts и provider metadata остаются в исходных
  OpenClaw `AgentMessage`; плоская bounded projection формируется только для
  записи в core.
- При `ownsCompaction: true` Styx отвечает за compaction lifecycle. При
  `false` используется явный runtime delegate; это режим диагностики, а не
  второй параллельный compactor.

## Tools и capability boundaries

17 имён объявлены одновременно в `openclaw.plugin.json` и runtime-регистрации:

`styx_store`, `styx_recall`, `styx_search_archive`, `styx_reinterpret`,
`styx_ingest_experience`, `styx_ingest_document`, `styx_dialogue_save`,
`styx_dialogue_search`, `styx_dialogue_recent`, `styx_dialogue_sessions`,
`styx_dialogue_prepare_summary`, `styx_relations_query`,
`styx_graph_traverse`, `styx_analytics`, `styx_explain`,
`styx_confirm_usage`, `styx_link`.

Плагин не регистрирует OpenClaw `memory` capability и не занимает
`plugins.slots.memory`. Social client methods также не являются tools и не
вызываются lifecycle автоматически: для них нужен отдельный явный credential.

## Проверяемые источники

Для текущей интеграции в первую очередь нужны:

- `docs_concepts_context-engine.md` — lifecycle, transcript semantics,
  `commitTurn`, compaction ownership и failure isolation;
- `docs_plugins_architecture.md` и `docs_plugins_building-plugins.md` —
  регистрация plugin/tool surfaces;
- `docs_plugins_manifest.md` — manifest contracts и version floor;
- `docs_plugins_sdk-overview.md`, `docs_plugins_sdk-runtime.md` и
  `docs_plugins_sdk-testing.md` — стабильные SDK entrypoints/runtime/tests;
- `docs_concepts_messages.md` и `docs_plugins_message-presentation.md` —
  сохранение shapes сообщений и presentation boundaries;
- `docs_plugins_hooks.md` — только для проверки, что hooks не становятся
  конкурентным owner'ом accepted-turn finality.

Страницы `concepts/openclaw-sdk`, `plugins/agent-tools`,
`plugins/building-extensions` и `plugins/sdk-channel-turn` отсутствуют в
официальном tag `v2026.8.2`; их старые локальные копии удалены. Остальные
Markdown-файлы зеркалируют существующие paths этого tag.

## Локальные источники истины Styx

Vendor snapshot помогает проверить ABI, но фактическое поведение интеграции
задают:

- `extensions/styx/openclaw.plugin.json`;
- `extensions/styx/package.json`;
- `extensions/styx/index.ts`;
- `extensions/styx/src/context-engine.ts`;
- `extensions/styx/src/client.ts`;
- `extensions/styx/src/tools/`;
- `extensions/styx/scripts/` и OpenClaw integration tests в `styx-core`.

Оставшиеся `*.ts`/`*.json` в `research/openclaw/` — исторический материал
первичного исследования. Они не используются как current ABI reference; для
этого служат официальный tag, lock-файл плагина и проверяемые локальные
контрактные тесты.
