# OpenClaw plugin track — research summary

Snapshot 2026-05-05. Все артефакты в `research/openclaw/`. Это
промежуточная сводка из docs.openclaw.ai (Mintlify) и репо
`openclaw/openclaw` (TS monorepo, MIT, ~368k stars).

## TL;DR — три роли для Styx

1. **Memory plugin** (`kind: "memory"`) — `api.registerMemoryCapability`
   + `memory_search/get/recall/store/forget` tools. Так живут
   `memory-core` и `memory-lancedb`.
2. **Context engine plugin** (`kind: "context-engine"`) —
   `api.registerContextEngine(id, factory)` с lifecycle
   `ingest` / `assemble` / `compact` / `afterTurn`. **Это и есть
   «полный доступ к динамической части контекстного окна»** —
   `assemble()` возвращает `{messages, systemPromptAddition,
   estimatedTokens, promptAuthority}`.
3. **Hybrid** — один TS-плагин регистрирует обе capability сразу
   плюс 6-9 styx_* tools и hooks `message_received` / `message_sent` /
   `agent_end` для `sync_turn`-style записи.

`plugins.slots.{memory, contextEngine}` — exclusive слоты, одновременно
активный плагин один на слот. Hybrid plugin нормально владеет обоими.

## Архитектурная карта

| Уровень | Кто owns | Контракт |
|---|---|---|
| **Capabilities** (typed contracts) | core | `registerProvider`, `registerChannel`, `registerMemoryCapability`, `registerContextEngine`, `registerSpeechProvider`, etc. |
| **Tools** | plugin | `api.registerTool(factory, { names })` — factory `(ctx) => AgentTool \| null` |
| **Commands** (slash) | plugin | `api.registerCommand({ name, handler })` |
| **Hooks** | plugin | `api.on("hook_name", handler, { priority, timeoutMs })` |
| **Hosts** | plugin | `api.registerService`, `registerHttpRoute`, `registerCli`, `registerGatewayMethod` |

Plugin runs **in-process**, не sandboxed. Один OpenClaw process =
shared memory с core. Crash в плагине = crash gateway.

## Регистрация Memory Plugin

```ts
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

export default definePluginEntry({
  id: "styx",
  name: "Styx",
  description: "PostgreSQL+pgvector dynamic context orchestrator",
  kind: "memory",
  configSchema: {...},
  register(api) {
    api.registerMemoryCapability({
      promptBuilder,             // (ctx) => prompt section
      flushPlanResolver,         // pre-compaction memory flush plan
      runtime: {                 // MemoryPluginRuntime interface
        async getMemorySearchManager(params) {...},
        resolveMemoryBackendConfig(params) {...},
        async closeAllMemorySearchManagers() {...},
      },
      publicArtifacts: { listArtifacts },  // optional
    });

    api.registerMemoryEmbeddingProvider(adapter);  // optional

    api.registerTool(
      (ctx) => createLazyMemorySearchTool(ctx),
      { names: ["memory_search"] }
    );
    // + memory_get, и Styx-специфичные: styx_recall, styx_search_archive,
    //   styx_dialogue_*, etc.

    api.registerCommand({ name: "styx", handler });
    api.registerCli((reg) => {...}, { descriptors: [...] });
  },
});
```

Manifest `openclaw.plugin.json`:
```json
{
  "id": "styx",
  "kind": "memory",
  "activation": { "onStartup": true },
  "contracts": {
    "tools": ["memory_search", "memory_get", "styx_recall", ...],
    "memoryEmbeddingProviders": ["styx-ollama"]
  },
  "configSchema": {...},
  "uiHints": {...}
}
```

## Регистрация Context Engine

```ts
api.registerContextEngine("styx", (ctx) => ({
  info: { id: "styx", name: "Styx", ownsCompaction: true },

  async ingest({ sessionId, message, isHeartbeat }) {
    // POST /sync_turn эквивалент — через styx-core daemon
  },

  async assemble({ sessionId, messages, tokenBudget,
                   availableTools, citationsMode }) {
    // Это где Styx показывает своё working_set + tail + salient
    return {
      messages: assembled,
      estimatedTokens,
      systemPromptAddition,    // memory prompt section
      promptAuthority: "assembled",
    };
  },

  async compact({ sessionId, force }) {
    // Styx own compaction (или delegateCompactionToRuntime для legacy)
    return { ok: true, compacted: true };
  },

  async afterTurn({ sessionId, messages }) {
    // post-turn bookkeeping (background drift, sweep ticks)
  },

  // optional:
  async bootstrap({...}) {},
  async ingestBatch({...}) {},
  async prepareSubagentSpawn({...}) {},
  async onSubagentEnded({...}) {},
  dispose() {},
}));
```

`ownsCompaction: true` отключает Pi's built-in auto-compaction. Styx
тогда полностью owns `/compact`, overflow recovery, proactive
compaction в `afterTurn()`.

`assemble().systemPromptAddition` строится через
`buildMemorySystemPromptAddition(...)` из `openclaw/plugin-sdk/core` —
конвертирует active memory prompt sections в готовый prepend.

## Hooks для sync_turn

NEXT.md называл `message:preprocessed` / `message:sent`. Реальные
имена с **подчёркиваниями**:

| Hook | Что отдаёт | Когда |
|---|---|---|
| `message_received` | `content`, `senderId`, `threadId`, `bodyForAgent`, `metadata` | inbound, до agent routing |
| `message_sending` | `content` (rewrite), `cancel: true` (terminal) | outbound, перед channel |
| `message_sent` | success/failure | observe-only, после delivery |
| `agent_turn_prepare` | drained next-turn injections | перед prompt hooks |
| `before_prompt_build` | session messages | для prompt mutation (alt к context engine) |
| `agent_end` | final messages, success | post-turn observation |

Non-bundled плагины должны включить
`plugins.entries.styx.hooks.allowConversationAccess: true` чтобы
получить `llm_input/output, before_agent_finalize, agent_end`.

Если Styx работает как context engine, **hooks для sync_turn не
нужны** — `ingest()` lifecycle покрывает запись каждой реплики
автоматически. Hooks остаются для cross-cutting (например, drift
detection на `agent_end`).

## Tool factory

```ts
api.registerTool(
  (ctx: OpenClawPluginToolContext) => {
    if (!shouldEnable(ctx)) return null;  // dynamic disable
    return {
      label: "Memory Search",
      name: "memory_search",
      description: "...",
      parameters: { type: "object", ... } as TSchema,
      execute: async (toolCallId, params, signal, onUpdate) => {
        return await styxClient.recall(params);
      },
    };
  },
  { names: ["memory_search"] }   // в manifest contracts.tools
);
```

`ctx` даёт `agentId`, `sessionKey`, `sandboxed`, `runtimeConfig`,
`getRuntimeConfig`, `config`. Lazy-load: реальный код через
`await import("./src/tools.js")` только при первом execute.

## Memory model OpenClaw vs Styx

OpenClaw native memory = **plain Markdown в workspace**:
- `MEMORY.md` — long-term
- `memory/YYYY-MM-DD.md` — daily notes
- `DREAMS.md` — dream diary

Backends:
- **builtin** SQLite (default)
- **QMD** local sidecar
- **LanceDB** (bundled plugin) — vector
- **Honcho** external service

Styx — PostgreSQL+pgvector с structured memories, knowledge graph,
hot/long tier, dialogue, ingest API. **Полностью замещает memory-core
плюс делает то, что не делает ни один из bundled** (graph traversal,
reinterpret, explain, analytics, ingest experience).

## Что критично для design-doc волны 26

### Открытые вопросы для пользователя

1. **Роль Styx-плагина**: Memory only / ContextEngine only / Hybrid?
   - **Memory only** → классическая роль "active memory plugin",
     `memory_search/get` стандартные tools, sync_turn через hooks.
     Минимально invasive, OpenClaw legacy context engine остаётся.
   - **ContextEngine only** → Styx полностью owns контекстное окно
     (assemble/compact). Но тогда `memory_search/get` не работают
     стандартным способом, у LLM нет recall tools (только через
     systemPromptAddition).
   - **Hybrid** → max power: memory tools + полный контроль над
     окном. Это совпадает с original Styx vision «дирижёр динамической
     части контекстного окна».
2. **Tool naming**: использовать стандартные OpenClaw имена
   (`memory_search`, `memory_get`, `memory_recall`) для совместимости
   или Styx-специфичные (`styx_recall`, `styx_search_archive`)?
   Hermes-wrappers сейчас именованы `styx_*`. Возможен mix:
   `memory_search` маппится на recall, остальные — `styx_*`.
3. **Subset of 15 styx_* tools для LLM**: NEXT.md ADR § 40.9 сказал
   что explain/analytics/confirm_usage без LLM wrapper'а
   (observability surface). Hermes сейчас экспортирует 6 schemas:
   `styx_recall + styx_search_archive + styx_reinterpret +
   styx_dialogue_{search,recent,prepare_summary}`. Остальные
   (`styx_ingest_experience`, `styx_relations_query`,
   `styx_graph_traverse`, `styx_link`) — экспортировать в OpenClaw
   или нет?
4. **Volume scope волны 26**: skeleton-only (manifest +
   bare-minimum register + 1-2 tools для smoke) → волна 27 full
   parity, или сразу полный 15-tool parity в одну волну?

### Что делает context engine особенным

Из всех плагин-расширений именно `registerContextEngine` даёт
**полный доступ ко всему окну** — на каждый model run engine
получает messages array и tokenBudget, возвращает финальный array.
Это превращает Styx из «memory backend» в **орchestrator контекстного
окна**, что и было оригинальной задумкой.

Без context engine Styx работал бы как memory-lancedb: tools `memory_recall`
для LLM-driven recall + `systemPromptAddition` через
`MemoryCapability.promptBuilder` для inline injection. Context engine
открывает доступ к compact, message rewriting, salient inject в
произвольное место истории, working set state — всё, что есть в
текущем `engine/context.py::StyxComposer`.

## Что не прочитано

- `docs_plugins_manifest.md` (87KB) — полная manifest spec
- `docs_plugins_building-plugins.md` (18KB) — first-plugin guide
- `docs_plugins_sdk-setup.md` (25KB) — packaging
- `docs_plugins_architecture-internals.md` (59KB) — internals
- `docs_plugins_sdk-runtime.md` (22KB) — runtime helpers
- `ext_active-memory_index.ts` (42KB tok) — реальный полный пример
- `src_plugin-sdk_provider-tools.ts` (13KB) — tool type definitions

После решения по роли + scope можно дочитать только релевантное.

## Скачанные артефакты

```
research/openclaw/
├── SUMMARY.md (этот файл)
├── docs_plugins_*.md           — основные plugin docs
├── docs_concepts_*.md          — context-engine, active-memory, memory
├── docs_automation_hooks.md    — operator hooks (HOOK.md)
├── docs_tools_plugin.md        — end-user plugin install
├── docs_index.json             — Mintlify nav
├── src_plugin-sdk_*.ts         — TS interface definitions
├── src_memory-host-*.ts        — memory-host helpers
├── src_memory-core-engine-runtime.ts
├── ext_active-memory_*         — active-memory plugin source
├── ext_memory-core_*           — memory-core plugin source
├── ext_memory-lancedb_*        — memory-lancedb plugin source
├── sdk_*.ts                    — packages/plugin-sdk re-exports
└── sdk_pkg_contract.ts         — packages/plugin-package-contract
```
