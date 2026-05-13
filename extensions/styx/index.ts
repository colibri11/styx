// Styx — OpenClaw plugin entry (волна 26 Phase D).
//
// Регистрирует ContextEngine + 16 LLM tools (`contracts.tools` в
// manifest matches). Конфигурация — `daemonUrl`, `httpToken`,
// `agentMapping`, `requestTimeoutMs`, `logging`, `ownsCompaction`.
//
// Tool factories импортируются eagerly (`./src/tools/index.js`),
// но их implementation модули (*-impl.ts) лениво подгружаются внутри
// первого `execute` через `import()` — minimum cold-start cost для
// runtime'а который перебирает 16 factories при load.

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import type { OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";

import { resolveAgentId } from "./src/agent-mapping.js";
import { createStyxClient, type StyxLogger } from "./src/client.js";
import { createStyxContextEngine } from "./src/context-engine.js";
import { createBeforePromptBuildHook } from "./src/hooks/before-prompt-build.js";
import {
  createAnalyticsTool,
  createConfirmUsageTool,
  createDialoguePrepareSummaryTool,
  createDialogueRecentTool,
  createDialogueSaveTool,
  createDialogueSearchTool,
  createDialogueSessionsTool,
  createExplainTool,
  createGraphTraverseTool,
  createIngestDocumentTool,
  createIngestExperienceTool,
  createLinkTool,
  createRecallTool,
  createReinterpretTool,
  createRelationsQueryTool,
  createSearchArchiveTool,
  createStoreTool,
  type StyxToolFactoryParams,
} from "./src/tools/index.js";

type PluginConfig = {
  daemonUrl?: string;
  httpToken?: string;
  agentMapping?: Record<string, string>;
  requestTimeoutMs?: number;
  logging?: boolean;
  ownsCompaction?: boolean;
};

export default definePluginEntry({
  id: "styx",
  name: "Styx",
  description:
    "Locus — динамический оркестратор контекстного окна на PostgreSQL+pgvector",
  kind: "context-engine",
  register(api) {
    const apiAny = api as unknown as Record<string, unknown>;
    const cfg = ((apiAny["pluginConfig"] as PluginConfig | undefined) ?? {}) as PluginConfig;
    const logger = (apiAny["logger"] as StyxLogger | undefined) ?? console;

    const client = createStyxClient({
      baseUrl: cfg.daemonUrl ?? "http://127.0.0.1:8788",
      httpToken: cfg.httpToken,
      timeoutMs: cfg.requestTimeoutMs ?? 30_000,
      logger,
    });

    const ownsCompaction = cfg.ownsCompaction ?? true;

    logger.info?.(
      `[styx] registering context engine + 17 tools (daemon=${cfg.daemonUrl ?? "http://127.0.0.1:8788"}, ownsCompaction=${ownsCompaction})`,
    );

    const registerCE = (api as unknown as {
      registerContextEngine?: (
        id: string,
        factory: (ctx: Record<string, unknown>) => unknown,
      ) => void;
    }).registerContextEngine;

    if (typeof registerCE !== "function") {
      logger.error?.(
        "[styx] api.registerContextEngine отсутствует — несовместимая версия openclaw",
      );
      return;
    }

    // Adapter для resolveAgentId — фиксируем mapping/client лексически
    // и отдаём в виде функции, ожидаемой ContextEngine + всеми tools.
    const resolve = (openclawAgentId: string) =>
      resolveAgentId({
        openclawAgentId,
        mapping: cfg.agentMapping,
        client,
      });

    registerCE("styx", (ctx) =>
      createStyxContextEngine({
        client,
        ctx,
        logger,
        resolveAgentId: resolve,
        ownsCompaction,
      }),
    );

    // Tools. SDK 2026.5.7 экспортирует registerTool с TSchema-generic
    // (`OpenClawPluginToolFactory` + `OpenClawPluginToolOptions` в
    // `dist/plugin-sdk/src/plugins/tool-types.d.ts`), но
    // `OpenClawPluginToolOptions` не реэкспортируется через корневой
    // entry-point `openclaw/plugin-sdk/plugin-entry`. Используем loose
    // typing через cast c local-описанным opts shape — symmetria с
    // registerCE выше; работает для inline JSON schemas (`as const`).
    const registerTool = (api as unknown as {
      registerTool?: (
        factory: (ctx: OpenClawPluginToolContext) => unknown,
        opts?: { names?: string[] },
      ) => void;
    }).registerTool;

    if (typeof registerTool !== "function") {
      logger.warn?.(
        "[styx] api.registerTool отсутствует — tools не зарегистрированы (несовместимая версия openclaw)",
      );
      return;
    }

    // Factory params одинаковые для всех 16; меняется только toolCtx
    // на каждый registerTool вызов (runtime передаёт его в factory).
    const baseParams = (toolCtx: OpenClawPluginToolContext): StyxToolFactoryParams => ({
      client,
      toolCtx,
      logger,
      resolveAgentId: resolve,
    });

    registerTool((tc) => createStoreTool(baseParams(tc)), {
      names: ["styx_store"],
    });
    registerTool((tc) => createRecallTool(baseParams(tc)), {
      names: ["styx_recall"],
    });
    registerTool((tc) => createSearchArchiveTool(baseParams(tc)), {
      names: ["styx_search_archive"],
    });
    registerTool((tc) => createReinterpretTool(baseParams(tc)), {
      names: ["styx_reinterpret"],
    });
    registerTool((tc) => createIngestExperienceTool(baseParams(tc)), {
      names: ["styx_ingest_experience"],
    });
    registerTool((tc) => createIngestDocumentTool(baseParams(tc)), {
      names: ["styx_ingest_document"],
    });
    registerTool((tc) => createDialogueSaveTool(baseParams(tc)), {
      names: ["styx_dialogue_save"],
    });
    registerTool((tc) => createDialogueSearchTool(baseParams(tc)), {
      names: ["styx_dialogue_search"],
    });
    registerTool((tc) => createDialogueRecentTool(baseParams(tc)), {
      names: ["styx_dialogue_recent"],
    });
    registerTool((tc) => createDialogueSessionsTool(baseParams(tc)), {
      names: ["styx_dialogue_sessions"],
    });
    registerTool((tc) => createDialoguePrepareSummaryTool(baseParams(tc)), {
      names: ["styx_dialogue_prepare_summary"],
    });
    registerTool((tc) => createRelationsQueryTool(baseParams(tc)), {
      names: ["styx_relations_query"],
    });
    registerTool((tc) => createGraphTraverseTool(baseParams(tc)), {
      names: ["styx_graph_traverse"],
    });
    registerTool((tc) => createAnalyticsTool(baseParams(tc)), {
      names: ["styx_analytics"],
    });
    registerTool((tc) => createExplainTool(baseParams(tc)), {
      names: ["styx_explain"],
    });
    registerTool((tc) => createConfirmUsageTool(baseParams(tc)), {
      names: ["styx_confirm_usage"],
    });
    registerTool((tc) => createLinkTool(baseParams(tc)), {
      names: ["styx_link"],
    });

    // Hook before_prompt_build (мини-волна 26.8) — канал доставки
    // salient в pi-embedded runner (openai-codex backend). Lifecycle
    // ContextEngine.assemble игнорируется этим runner'ом; вместо
    // этого pi-embedded склеивает baseSystemPrompt с
    // `appendSystemContext` от hook'а через `joinPresentTextSegments`
    // (cf. OpenClaw checkout 2026.4.15
    // `src/agents/pi-embedded-runner/run/attempt.thread-helpers.ts:23`).
    //
    // `api.on("before_prompt_build", handler)` — typed hook API из
    // plugin SDK (`PluginHookName` enum в `hook-types.d.ts:553`).
    // Loose-typed cast симметричен с registerCE / registerTool выше
    // (SDK типизация недоступна через корневой entry-point).
    //
    // Graceful degradation: если SDK старая и не имеет `.on()` —
    // warn + skip (engine + tools зарегистрированы выше, plugin
    // продолжит работу в non-embedded path через assemble lifecycle).
    const apiOn = (api as unknown as {
      on?: (
        hookName: string,
        handler: unknown,
        opts?: { priority?: number; timeoutMs?: number },
      ) => void;
    }).on;

    if (typeof apiOn === "function") {
      const beforePromptBuild = createBeforePromptBuildHook({
        client,
        logger,
        resolveAgentId: resolve,
      });
      apiOn("before_prompt_build", beforePromptBuild);
      logger.info?.(
        "[styx] registered before_prompt_build hook (pi-embedded salient delivery)",
      );
    } else {
      logger.warn?.(
        "[styx] api.on отсутствует — before_prompt_build hook не зарегистрирован (несовместимая версия openclaw)",
      );
    }
  },
});
