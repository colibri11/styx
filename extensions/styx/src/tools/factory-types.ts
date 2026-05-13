// Общие типы для всех tool factory'ев Styx.
//
// Каждая factory принимает один и тот же набор параметров, чтобы entry
// (`index.ts::register`) мог одинаково подавать их 16 раз. Implementation
// модули (`*-impl.ts`) принимают расширенный объект с per-call данными
// (toolCallId, toolParams, signal, onUpdate).

import type { OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";

import type { StyxClient, StyxLogger } from "../client.js";
import type { ResolveAgentId } from "../context-engine.js";

export type StyxToolFactoryParams = {
  client: StyxClient;
  toolCtx: OpenClawPluginToolContext;
  logger: StyxLogger;
  resolveAgentId: ResolveAgentId;
};

export type StyxToolExecuteParams<TInput = Record<string, unknown>> =
  StyxToolFactoryParams & {
    toolCallId: string;
    toolParams: TInput;
    signal?: AbortSignal;
    onUpdate?: unknown;
  };
