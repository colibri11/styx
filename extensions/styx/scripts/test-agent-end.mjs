import assert from "node:assert/strict";

import {
  MAX_ATTACHMENT_INGESTS,
  MAX_ATTACHMENT_MARKERS,
  MAX_SOURCE_MESSAGES,
  MAX_SERIALIZED_DEPTH,
  MAX_SERIALIZED_ITEMS,
  buildToolEvents,
  createAgentEndHook,
  deriveTurnIdentity,
  extractBoundedMessageContent,
  extractFinalTurnMessages,
} from "../dist/src/hooks/agent-end.js";
import { createBeforePromptBuildHook } from "../dist/src/hooks/before-prompt-build.js";
import {
  TerminalTurnBarrier,
  terminalScopeKey,
} from "../dist/src/hooks/terminal-barrier.js";
import { extractMediaAttachments } from "../dist/src/media-attachments.js";
import { createStyxContextEngine } from "../dist/src/context-engine.js";

const calls = [];
const barrier = new TerminalTurnBarrier();
const client = {
  async contextBootstrap(payload) { calls.push(["bootstrap", payload]); },
  async affectObserveTurn(payload) {
    calls.push(["affect", payload]);
    return { accepted: true, duplicate: false, reason: null };
  },
  async contextIngestBatch(payload) { calls.push(["ingest", payload]); },
};
const hook = createAgentEndHook({
  client,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: barrier,
});
const ctx = {
  agentId: "main",
  sessionId: "session-1",
  modelId: "model-1",
};
const messages = [
  { role: "user", content: "same text", timestamp: 1 },
  {
    role: "assistant",
    content: [
      { type: "toolCall", id: "call-1", name: "lookup", arguments: { q: "x" } },
    ],
    timestamp: 2,
  },
  {
    role: "toolResult",
    toolCallId: "call-1",
    toolName: "lookup",
    content: [{ type: "text", text: "result" }],
    timestamp: 3,
  },
  { role: "assistant", content: [{ type: "text", text: "same answer" }], timestamp: 4 },
];

await hook({ runId: "run-1", messages, success: true }, ctx);
assert.equal(calls.filter(([kind]) => kind === "affect").length, 1);
assert.equal(calls.filter(([kind]) => kind === "ingest").length, 1);
assert.equal(calls.find(([kind]) => kind === "affect")[1].tool_events.length, 2);

// Re-delivery of the same terminal run is locally duplicate-safe; core also
// receives the same idempotency key if a process restart causes a retry.
await hook({ runId: "run-1", messages, success: true }, ctx);
assert.equal(calls.filter(([kind]) => kind === "affect").length, 1);
assert.equal(calls.filter(([kind]) => kind === "ingest").length, 1);

// Identical content in a distinct host run is a distinct legitimate turn.
await hook({ runId: "run-2", messages, success: true }, ctx);
assert.equal(calls.filter(([kind]) => kind === "affect").length, 2);
assert.equal(calls.filter(([kind]) => kind === "ingest").length, 2);
const affectCalls = calls.filter(([kind]) => kind === "affect");
assert.notEqual(affectCalls[0][1].idempotency_key, affectCalls[1][1].idempotency_key);
const ingestCalls = calls.filter(([kind]) => kind === "ingest");
assert.match(ingestCalls[0][1].idempotency_key, /^openclaw:.+:dialogue$/);

// Failed/non-final runs do not capture dialogue or affect.
await hook({ runId: "run-3", messages, success: false }, ctx);
assert.equal(calls.filter(([kind]) => kind === "affect").length, 2);

// Preprocessing is source-bounded before the final-turn scan.
const oversized = [
  { role: "user", content: "too old", timestamp: -1 },
  ...Array.from({ length: MAX_SOURCE_MESSAGES }, (_, index) => ({
    role: "assistant",
    content: `a-${index}`,
    timestamp: index,
  })),
];
const boundedTail = extractFinalTurnMessages(oversized);
assert.equal(boundedTail.length, 1);
assert.equal(boundedTail[0].content, `a-${MAX_SOURCE_MESSAGES - 1}`);
assert.equal(
  extractBoundedMessageContent(
    Array.from({ length: 128 }, () => "x".repeat(100_000)),
    4_000,
  ).length,
  4_000,
);

// Native pi tool-call/result shapes are represented causally.
assert.deepEqual(buildToolEvents(messages).map((event) => event.kind), ["call", "result"]);

// Tool arguments are projected by depth/item count before JSON serialization.
let hugeNested = "leaf";
for (let depth = 0; depth < MAX_SERIALIZED_DEPTH + 20; depth++) {
  hugeNested = { child: hugeNested };
}
const hugeArgs = Object.fromEntries(
  Array.from({ length: MAX_SERIALIZED_ITEMS + 500 }, (_, index) => [
    `key-${index}`, hugeNested,
  ]),
);
const projectedEvents = buildToolEvents([{
  role: "assistant",
  tool_calls: [{ id: "huge", function: { name: "tool", arguments: hugeArgs } }],
}]);
assert.equal(projectedEvents.length, 1);
assert.ok(projectedEvents[0].content.length <= 4_000);
assert.match(projectedEvents[0].content, /bounded/);

// Fallback identity uses host message coordinates, not dialogue content.
const turn = extractFinalTurnMessages(messages);
assert.match(deriveTurnIdentity({ messages, success: true }, ctx, turn), /^messages:/);

// The terminal promise is installed synchronously and duplicate concurrent
// delivery of the same run shares one flight. The next prompt waits for it
// before fetching state.
const ordering = [];
let releaseAffect;
const deferredClient = {
  async contextBootstrap() { ordering.push("bootstrap"); },
  async affectObserveTurn() {
    ordering.push("affect:start");
    await new Promise((resolve) => { releaseAffect = resolve; });
    ordering.push("affect:end");
    return { accepted: true, duplicate: false, reason: null };
  },
  async contextIngestBatch() { ordering.push("ingest"); },
  async contextAssemble() {
    ordering.push("assemble");
    return { messages: [], estimated_tokens: 0, system_prompt_addition: "state" };
  },
};
const deferredBarrier = new TerminalTurnBarrier();
const deferredHook = createAgentEndHook({
  client: deferredClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: deferredBarrier,
});
const pendingOne = deferredHook({ runId: "run-barrier", messages, success: true }, ctx);
const pendingTwo = deferredHook({ runId: "run-barrier", messages, success: true }, ctx);
const beforeHook = createBeforePromptBuildHook({
  client: deferredClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: deferredBarrier,
  terminalDrainTimeoutMs: 1_000,
});
const promptPending = beforeHook(
  { messages },
  { agentId: "main", sessionId: "session-1", modelId: "model-1" },
);
await new Promise((resolve) => setTimeout(resolve, 0));
assert.deepEqual(ordering, ["bootstrap", "affect:start"]);
releaseAffect();
await Promise.all([pendingOne, pendingTwo, promptPending]);
assert.deepEqual(ordering, [
  "bootstrap",
  "affect:start",
  "affect:end",
  "ingest",
  "bootstrap",
  "assemble",
]);
assert.equal(ordering.filter((item) => item === "affect:start").length, 1);

// A stuck terminal observer cannot wedge the next model pass.
let releaseSlow;
const timeoutBarrier = new TerminalTurnBarrier();
const slow = timeoutBarrier.track(
  terminalScopeKey("main", "session-timeout"),
  "run:slow",
  async () => new Promise((resolve) => { releaseSlow = resolve; }),
);
const timeoutWarnings = [];
const timeoutCalls = [];
const timeoutBeforeHook = createBeforePromptBuildHook({
  client: {
    async contextBootstrap() { timeoutCalls.push("bootstrap"); },
    async contextAssemble() {
      timeoutCalls.push("assemble");
      return { messages: [], estimated_tokens: 0 };
    },
  },
  logger: { warn(message) { timeoutWarnings.push(message); } },
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: timeoutBarrier,
  terminalDrainTimeoutMs: 5,
});
await timeoutBeforeHook(
  { messages: [] },
  { agentId: "main", sessionId: "session-timeout" },
);
assert.deepEqual(timeoutCalls, ["bootstrap", "assemble"]);
assert.match(timeoutWarnings[0], /timed out.*assembling fail-open/);
assert.deepEqual(
  await timeoutBarrier.drain(
    terminalScopeKey("main", "session-timeout"),
    5,
  ),
  { completed: true, pending: 0 },
);
releaseSlow();
await slow;

// Prompt timeout does not permit a later terminal turn to overtake the slow
// one: the prompt wait set and the per-scope causal ordering tail are separate.
const serializedBarrier = new TerminalTurnBarrier();
const serializedOrder = [];
let releaseFirst;
const first = serializedBarrier.track("scope", "turn-1", async () => {
  serializedOrder.push("n:start");
  await new Promise((resolve) => { releaseFirst = resolve; });
  serializedOrder.push("n:commit");
});
assert.deepEqual(
  await serializedBarrier.drain("scope", 5),
  { completed: false, pending: 1 },
);
const second = serializedBarrier.track("scope", "turn-2", async () => {
  serializedOrder.push("n+1:commit");
});
await new Promise((resolve) => setTimeout(resolve, 0));
assert.deepEqual(serializedOrder, ["n:start"]);
releaseFirst();
await Promise.all([first, second]);
assert.deepEqual(serializedOrder, ["n:start", "n:commit", "n+1:commit"]);

// The terminal budget is propagated as AbortSignal to core calls. Affect is
// committed first; a hung later phase cannot multiply per-call timeouts.
const deadlineOrder = [];
const deadlineHook = createAgentEndHook({
  client: {
    async contextBootstrap() { deadlineOrder.push("bootstrap"); },
    async affectObserveTurn() {
      deadlineOrder.push("affect");
      return { accepted: true, duplicate: false, reason: null };
    },
    async contextIngestBatch(_payload, options) {
      deadlineOrder.push("ingest:start");
      await new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => {
          deadlineOrder.push("ingest:aborted");
          reject(new Error("aborted"));
        }, { once: true });
      });
    },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
  terminalWorkBudgetMs: 20,
});
const deadlineStarted = Date.now();
await deadlineHook(
  { runId: "deadline-run", messages, success: true },
  ctx,
);
assert.ok(Date.now() - deadlineStarted < 250);
assert.deepEqual(deadlineOrder, [
  "bootstrap", "affect", "ingest:start", "ingest:aborted",
]);

// Attachment marker discovery used by agent_end is capped before traversal;
// network ingest has a stricter independent budget.
const attachmentText = Array.from(
  { length: MAX_ATTACHMENT_MARKERS + 100 },
  (_, index) => `[media attached: media://inbound/file-${index}.pdf]`,
).join("\n");
assert.equal(
  extractMediaAttachments(attachmentText, MAX_ATTACHMENT_MARKERS).length,
  MAX_ATTACHMENT_MARKERS,
);
assert.ok(MAX_ATTACHMENT_INGESTS < MAX_ATTACHMENT_MARKERS);

// ContextEngine.afterTurn performs maintenance only; an intermediate tool-loop
// lifecycle callback cannot observe affect or capture finalized dialogue.
const maintenanceCalls = [];
const maintenanceClient = {
  async contextBootstrap(payload) { maintenanceCalls.push(["bootstrap", payload]); },
  async contextAfterTurn(payload) { maintenanceCalls.push(["maintenance", payload]); },
  async affectObserveTurn(payload) { maintenanceCalls.push(["affect", payload]); },
  async contextIngestBatch(payload) { maintenanceCalls.push(["ingest", payload]); },
};
const engine = createStyxContextEngine({
  client: maintenanceClient,
  ctx: {},
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  ownsCompaction: true,
});
await engine.afterTurn({
  sessionId: "session-1",
  sessionKey: "agent:main:session:session-1",
  messages,
});
assert.deepEqual(
  maintenanceCalls.map(([kind]) => kind),
  ["bootstrap", "maintenance"],
);

console.log("agent_end hook contract tests passed");
