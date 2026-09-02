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
import { StyxHttpError } from "../dist/src/client.js";

const calls = [];
const barrier = new TerminalTurnBarrier();
const client = {
  async contextBootstrap(payload) { calls.push(["bootstrap", payload]); },
  async cognitionCommit(payload) {
    calls.push(["commit", payload]);
    return { committed: true, duplicate: false, act_id: "act", line_version: 1,
      consequence_ids: [], memory_ids: [] };
  },
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
assert.equal(calls.filter(([kind]) => kind === "commit").length, 1);
assert.equal(calls.filter(([kind]) => kind === "ingest").length, 0);
assert.equal(calls.find(([kind]) => kind === "commit")[1].tool_events.length, 2);
assert.deepEqual(
  calls.find(([kind]) => kind === "commit")[1].tool_events.map((event) => [
    event.kind, event.tool_event_id, event.name,
  ]),
  [["call", "call-1", "lookup"], ["result", "call-1", "lookup"]],
);

// Re-delivery of the same terminal run is locally duplicate-safe; core also
// receives the same idempotency key if a process restart causes a retry.
await hook({ runId: "run-1", messages, success: true }, ctx);
assert.equal(calls.filter(([kind]) => kind === "commit").length, 1);

// Identical content in a distinct host run is a distinct legitimate turn.
await hook({ runId: "run-2", messages, success: true }, ctx);
assert.equal(calls.filter(([kind]) => kind === "commit").length, 2);
const commitCalls = calls.filter(([kind]) => kind === "commit");
assert.notEqual(commitCalls[0][1].host_key, commitCalls[1][1].host_key);
assert.equal(commitCalls[1][1].parent_host_key, commitCalls[0][1].host_key);

// Failed runs are durable terminal acts, not silently missing ancestry.
await hook({ runId: "run-3", messages, success: false }, ctx);
assert.equal(calls.filter(([kind]) => kind === "commit").length, 3);
assert.equal(calls.filter(([kind]) => kind === "commit")[2][1].status, "failed");

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

// The shared core contract accepts at most 64 ordered tool events. Host
// extraction keeps the newest events within that one combined budget;
// result/error do not get a separate unbounded allowance.
const boundaryTrajectory = Array.from({ length: 33 }, (_, index) => [
  {
    role: "assistant",
    tool_calls: [{
      id: `call-${index}`,
      function: { name: "tool", arguments: "{}" },
    }],
  },
  {
    role: "toolResult",
    toolCallId: `call-${index}`,
    toolName: "tool",
    content: `result-${index}`,
    isError: index === 32,
  },
]).flat();
const boundedTrajectory = buildToolEvents(boundaryTrajectory);
assert.equal(boundedTrajectory.length, 64);
assert.deepEqual(
  boundedTrajectory.slice(0, 2).map((event) => [event.kind, event.tool_event_id]),
  [["call", "call-1"], ["result", "call-1"]],
);
assert.deepEqual(
  boundedTrajectory.slice(-2).map((event) => [event.kind, event.tool_event_id]),
  [["call", "call-32"], ["error", "call-32"]],
);

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
  async cognitionCommit() {
    ordering.push("commit:start");
    await new Promise((resolve) => { releaseAffect = resolve; });
    ordering.push("commit:end");
    return { committed: true, duplicate: false };
  },
  async cognitionPreturn() {
    ordering.push("preturn");
    return { messages: [], snapshot_token: "snapshot-next",
      system_prompt_addition: "state" };
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
assert.deepEqual(ordering, ["bootstrap", "commit:start"]);
releaseAffect();
await Promise.all([pendingOne, pendingTwo, promptPending]);
assert.deepEqual(ordering, [
  "bootstrap",
  "commit:start",
  "commit:end",
  "bootstrap",
  "preturn",
]);
assert.equal(ordering.filter((item) => item === "commit:start").length, 1);

// Preturn and terminal commit share one snapshot fence and stable host ancestry.
// The preturn adapter bounds source count/content before JSON transport.
const fencedCalls = [];
const fencedBarrier = new TerminalTurnBarrier();
const fencedClient = {
  async contextBootstrap() {},
  async cognitionPreturn(payload) {
    fencedCalls.push(["preturn", payload]);
    return { messages: [], line_version: 7, snapshot_token: `snap-${fencedCalls.length}`,
      system_prompt_addition: "<styx-cognitive-continuity>{}</styx-cognitive-continuity>" };
  },
  async cognitionCommit(payload) {
    fencedCalls.push(["commit", payload]);
    return { committed: true, duplicate: false };
  },
};
const fencedBefore = createBeforePromptBuildHook({
  client: fencedClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: fencedBarrier,
  terminalDrainTimeoutMs: 1_000,
});
const fencedEnd = createAgentEndHook({
  client: fencedClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: fencedBarrier,
});
const hugePreturnMessages = Array.from({ length: 400 }, (_, index) => ({
  role: index % 2 ? "assistant" : "user",
  content: `${index}:` + "x".repeat(20_000),
}));
await fencedBefore(
  { prompt: "q".repeat(30_000), messages: hugePreturnMessages },
  { ...ctx, runId: "fenced-1" },
);
const firstPreturn = fencedCalls.find(([kind]) => kind === "preturn")[1];
assert.equal(firstPreturn.messages.length, 256);
assert.ok(firstPreturn.messages.every((item) => item.content.length <= 4_000));
assert.equal(firstPreturn.query.length, 20_000);
assert.equal(typeof firstPreturn.extra.current_event, "object");
await fencedEnd({ runId: "fenced-1", messages, success: true }, ctx);
await fencedBefore(
  { prompt: "next", messages },
  { ...ctx, runId: "fenced-2" },
);
await fencedEnd({ runId: "fenced-2", messages, success: true }, ctx);
const fencedCommits = fencedCalls.filter(([kind]) => kind === "commit");
const fencedPreturns = fencedCalls.filter(([kind]) => kind === "preturn");
assert.equal(fencedCommits[0][1].snapshot_token, "snap-1");
assert.equal(firstPreturn.host_key, fencedCommits[0][1].host_key);
assert.equal(firstPreturn.parent_host_key, undefined);
assert.equal(fencedCommits[0][1].parent_host_key, null);
assert.equal(
  fencedPreturns[1][1].parent_host_key,
  fencedCommits[0][1].host_key,
);
assert.equal(fencedCommits[1][1].snapshot_token, "snap-3");
assert.equal(fencedCommits[1][1].parent_host_key, fencedCommits[0][1].host_key);

// Explicit turn identity is the same physical coordinate on both host seams
// when OpenClaw does not expose a run id.
const explicitTurnCalls = [];
const explicitTurnBarrier = new TerminalTurnBarrier();
const explicitTurnClient = {
  async contextBootstrap() {},
  async cognitionPreturn(payload) {
    explicitTurnCalls.push(["preturn", payload]);
    return { messages: [], snapshot_token: "turn-snapshot" };
  },
  async cognitionCommit(payload) {
    explicitTurnCalls.push(["commit", payload]);
    return { committed: true, duplicate: false };
  },
};
await createBeforePromptBuildHook({
  client: explicitTurnClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: explicitTurnBarrier,
  terminalDrainTimeoutMs: 1_000,
})(
  { turnId: "turn-only", messages },
  { ...ctx, runId: undefined },
);
await createAgentEndHook({
  client: explicitTurnClient,
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: explicitTurnBarrier,
})(
  { turnId: "turn-only", messages, success: true },
  { ...ctx, runId: undefined },
);
assert.equal(
  explicitTurnCalls.find(([kind]) => kind === "preturn")[1].host_key,
  explicitTurnCalls.find(([kind]) => kind === "commit")[1].host_key,
);

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
    async cognitionPreturn() {
      timeoutCalls.push("preturn");
      return { messages: [], snapshot_token: "timeout-snapshot" };
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
assert.deepEqual(timeoutCalls, ["bootstrap", "preturn"]);
assert.match(timeoutWarnings[0], /timed out.*predecessor-pending/);
assert.deepEqual(
  await timeoutBarrier.drain(
    terminalScopeKey("main", "session-timeout"),
    5,
  ),
  { completed: false, pending: 1, pendingIdentities: ["run:slow"] },
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
  { completed: false, pending: 1, pendingIdentities: ["turn-1"] },
);
const second = serializedBarrier.track("scope", "turn-2", async () => {
  serializedOrder.push("n+1:commit");
});
await new Promise((resolve) => setTimeout(resolve, 0));
assert.deepEqual(serializedOrder, ["n:start"]);
releaseFirst();
await Promise.all([first, second]);
assert.deepEqual(serializedOrder, ["n:start", "n:commit", "n+1:commit"]);

// The terminal budget is propagated as AbortSignal to the single commit.
const deadlineOrder = [];
const deadlineHook = createAgentEndHook({
  client: {
    async contextBootstrap() { deadlineOrder.push("bootstrap"); },
    async cognitionCommit(_payload, options) {
      deadlineOrder.push("commit:start");
      await new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => {
          deadlineOrder.push("commit:aborted");
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
  "bootstrap", "commit:start", "commit:aborted",
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

// Declaration precedes awaited resolution: an early failure remains the
// declared predecessor, and retry keeps the original parent coordinate.
const earlyBarrier = new TerminalTurnBarrier();
const earlyCommits = [];
let failResolve = true;
const earlyHook = createAgentEndHook({
  client: {
    async contextBootstrap() {},
    async cognitionCommit(payload) {
      earlyCommits.push(payload);
      return { committed: true, duplicate: false };
    },
  },
  logger: {},
  async resolveAgentId() {
    if (failResolve) throw new Error("mapping unavailable");
    return "styx-agent";
  },
  terminalBarrier: earlyBarrier,
});
await earlyHook({ runId: "early", messages, success: true }, ctx);
failResolve = false;
await earlyHook({ runId: "next", messages, success: true }, ctx);
assert.match(earlyCommits[0].parent_host_key, /^openclaw:/);
const nextParent = earlyCommits[0].parent_host_key;
await earlyHook({ runId: "early", messages, success: true }, ctx);
assert.equal(earlyCommits[1].parent_host_key, null);
assert.equal(earlyCommits[1].host_key, nextParent);

const bootstrapBarrier = new TerminalTurnBarrier();
const bootstrapCommits = [];
let failBootstrap = true;
const bootstrapHook = createAgentEndHook({
  client: {
    async contextBootstrap() {
      if (failBootstrap) throw new Error("bootstrap unavailable");
    },
    async cognitionCommit(payload) {
      bootstrapCommits.push(payload);
      return { committed: true, duplicate: false };
    },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: bootstrapBarrier,
});
await bootstrapHook({ runId: "bootstrap-failed", messages, success: true }, ctx);
failBootstrap = false;
await bootstrapHook({ runId: "bootstrap-next", messages, success: true }, ctx);
assert.match(bootstrapCommits[0].parent_host_key, /^openclaw:/);

// Snapshot fences are act-keyed. Unkeyed/cancelled and expired snapshots can
// never drift forward onto the following physical act.
const staleBarrier = new TerminalTurnBarrier();
const originalNow = Date.now;
let fakeNow = 1_000;
Date.now = () => fakeNow;
staleBarrier.rememberSnapshot("scope", "unkeyed", null);
staleBarrier.rememberSnapshot("scope", "stale", "run:old");
fakeNow += 121_000;
assert.equal(
  staleBarrier.declareAct("scope", "run:new", "act:new").snapshotToken,
  null,
);
assert.equal(
  staleBarrier.declareAct("scope", "run:old", "act:old").snapshotToken,
  null,
);
Date.now = originalNow;

// Current OpenClaw lifecycle: assemble creates one session-fenced preturn while
// preserving rich AgentMessage objects; accepted finality is commitTurn.
const lifecycleCalls = [];
let lifecycleDuplicate = false;
const lifecycleClient = {
  async contextBootstrap(payload) { lifecycleCalls.push(["bootstrap", payload]); },
  async cognitionPreturn(payload) {
    lifecycleCalls.push(["preturn", payload]);
    return {
      messages: [{ role: "user", content: "normalized-by-core" }],
      line_version: 1,
      snapshot_token: "lifecycle-snapshot",
      system_prompt_addition: "one-injection",
    };
  },
  async cognitionCommit(payload) {
    lifecycleCalls.push(["commit", payload]);
    return { committed: true, duplicate: lifecycleDuplicate };
  },
  async contextIngestBatch(payload) { lifecycleCalls.push(["ingest", payload]); },
};
const lifecycleEngine = createStyxContextEngine({
  client: lifecycleClient,
  ctx: {},
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  ownsCompaction: true,
});
const lifecycleOpts = {
  sessionId: "session-lifecycle",
  sessionKey: "agent:main:session:session-lifecycle",
  messages,
  prompt: "same text",
  model: "model-1",
};
const assembled = await lifecycleEngine.assemble(lifecycleOpts);
assert.equal(assembled.messages, messages);
assert.equal(assembled.messages[1].content, messages[1].content);
assert.equal(assembled.systemPromptAddition, "one-injection");
assert.equal(lifecycleCalls.filter(([kind]) => kind === "preturn").length, 1);
assert.equal(
  Object.hasOwn(lifecycleCalls.find(([kind]) => kind === "preturn")[1], "host_key"),
  false,
);
assert.deepEqual(await lifecycleEngine.ingestBatch(lifecycleOpts), { ingestedCount: 0 });
const durableTurn = {
  advancementKey: "logical-turn-1",
  admission: {
    logicalTurnId: "logical-turn-1",
    entryId: "user-entry-1",
    generation: 3,
  },
  terminal: { entryId: "assistant-entry-1", generation: 3 },
  messages,
  sessionId: "session-lifecycle",
  sessionKey: "agent:main:session:session-lifecycle",
  sessionTarget: {
    agentId: "main",
    sessionId: "session-lifecycle",
    sessionKey: "agent:main:session:session-lifecycle",
    storePath: "/state/sessions.sqlite",
  },
};
assert.deepEqual(await lifecycleEngine.commitTurn(durableTurn), { status: "committed" });
assert.equal(lifecycleCalls.filter(([kind]) => kind === "commit").length, 1);
assert.equal(lifecycleCalls.filter(([kind]) => kind === "ingest").length, 0);
assert.equal(
  lifecycleCalls.find(([kind]) => kind === "commit")[1].snapshot_policy,
  "latest_session",
);
assert.equal(
  lifecycleCalls.find(([kind]) => kind === "commit")[1].parent_policy,
  "latest_session",
);
assert.equal(
  lifecycleCalls.find(([kind]) => kind === "commit")[1].extra.logical_turn_id,
  "logical-turn-1",
);
lifecycleDuplicate = true;
assert.deepEqual(await lifecycleEngine.commitTurn(durableTurn), { status: "duplicate" });
assert.equal(
  lifecycleCalls.at(-1)[1].host_key,
  lifecycleCalls.find(([kind]) => kind === "commit")[1].host_key,
);
assert.equal(
  lifecycleEngine.info.transcriptSemantics.currentTurnFence,
  "before-current-turn-entry-v1",
);
assert.equal(
  lifecycleEngine.info.transcriptSemantics.turnAdvancementIdempotency,
  "atomic-idempotent-v1",
);
await assert.rejects(
  lifecycleEngine.commitTurn({
    ...durableTurn,
    sessionTarget: { ...durableTurn.sessionTarget, sessionId: "other-session" },
  }),
  /sessionTarget conflicts with sessionId/,
);

// Mixed-version failed turns degrade to diary-only: no legacy affect write.
const legacyCalls = [];
const legacyHook = createAgentEndHook({
  client: {
    async contextBootstrap() {},
    async cognitionCommit() {
      throw new StyxHttpError(404, "not found", "");
    },
    async affectObserveTurn(payload) { legacyCalls.push(["affect", payload]); },
    async syncTurn(payload) { legacyCalls.push(["sync", payload]); },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
});
await legacyHook({ runId: "legacy-failed", messages, success: false }, ctx);
assert.equal(legacyCalls.filter(([kind]) => kind === "affect").length, 0);
assert.equal(legacyCalls.filter(([kind]) => kind === "sync").length, 1);
assert.equal(
  legacyCalls.find(([kind]) => kind === "sync")[1].idempotency_key,
  "openclaw:8756638d149f0b768f95b3002a3ca9511cb6e4215154dd6964da53c8428d76b1",
);

// Legacy affect and diary are independent guarded phases. A rejected affect
// write never suppresses the direct /sync_turn diary, and a retry only
// repeats the failed phase.
const partialLegacyCalls = [];
let affectFails = true;
const partialLegacyHook = createAgentEndHook({
  client: {
    async contextBootstrap() {},
    async cognitionCommit() { throw new StyxHttpError(404, "not found", ""); },
    async affectObserveTurn(payload) {
      partialLegacyCalls.push(["affect", payload]);
      if (affectFails) throw new Error("affect unavailable");
    },
    async syncTurn(payload) { partialLegacyCalls.push(["sync", payload]); },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
});
await partialLegacyHook({ runId: "legacy-partial", messages, success: true }, ctx);
assert.equal(partialLegacyCalls.filter(([kind]) => kind === "affect").length, 1);
assert.equal(partialLegacyCalls.filter(([kind]) => kind === "sync").length, 1);
const legacyCanonicalKey = partialLegacyCalls.find(([kind]) => kind === "sync")[1]
  .idempotency_key;
assert.equal(
  partialLegacyCalls.find(([kind]) => kind === "affect")[1].idempotency_key,
  legacyCanonicalKey,
);
affectFails = false;
await partialLegacyHook({ runId: "legacy-partial", messages, success: true }, ctx);
assert.equal(partialLegacyCalls.filter(([kind]) => kind === "affect").length, 2);
assert.equal(partialLegacyCalls.filter(([kind]) => kind === "sync").length, 1);

// A later v2 adapter retry uses the exact key already written by /sync_turn,
// so core deduplicates the mixed-version transition rather than rehashing it.
const laterCanonical = [];
const laterCanonicalHook = createAgentEndHook({
  client: {
    async contextBootstrap() {},
    async cognitionCommit(payload) {
      laterCanonical.push(payload);
      return { committed: false, duplicate: true };
    },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
});
await laterCanonicalHook({ runId: "legacy-partial", messages, success: true }, ctx);
assert.equal(laterCanonical[0].host_key, legacyCanonicalKey);

// Completion suppression is tenant-scoped: the same run id in distinct
// agent/session scopes is two real acts, while same-scope delivery is once.
const scopedCommits = [];
const scopedHook = createAgentEndHook({
  client: {
    async contextBootstrap() {},
    async cognitionCommit(payload) {
      scopedCommits.push(payload);
      return { committed: true, duplicate: false };
    },
  },
  logger: {},
  async resolveAgentId(openclawAgentId) { return `styx-${openclawAgentId}`; },
  terminalBarrier: new TerminalTurnBarrier(),
});
await scopedHook(
  { runId: "shared-run", messages, success: true },
  { ...ctx, agentId: "agent-a", sessionId: "session-a" },
);
await scopedHook(
  { runId: "shared-run", messages, success: true },
  { ...ctx, agentId: "agent-b", sessionId: "session-b" },
);
await scopedHook(
  { runId: "shared-run", messages, success: true },
  { ...ctx, agentId: "agent-a", sessionId: "session-a" },
);
assert.equal(scopedCommits.length, 2);

// Compatibility fallback is status-exact: 404 delegates once; all other
// failures remain fail-open and never trigger the legacy endpoint.
const fallbackCalls = [];
const fallbackBefore = createBeforePromptBuildHook({
  client: {
    async contextBootstrap() {},
    async cognitionPreturn() {
      throw new StyxHttpError(404, "not found", "");
    },
    async contextAssemble(payload) {
      fallbackCalls.push(payload);
      return { messages: [], estimated_tokens: 0, system_prompt_addition: "legacy" };
    },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
  terminalDrainTimeoutMs: 100,
});
assert.deepEqual(
  await fallbackBefore(
    { prompt: "q", messages },
    { ...ctx, sessionId: "fallback", runId: "fallback-run" },
  ),
  { appendSystemContext: "legacy" },
);
assert.equal(fallbackCalls.length, 1);
let unexpectedLegacy = false;
const non404Before = createBeforePromptBuildHook({
  client: {
    async contextBootstrap() {},
    async cognitionPreturn() { throw new Error("network"); },
    async contextAssemble() { unexpectedLegacy = true; },
  },
  logger: {},
  async resolveAgentId() { return "styx-agent"; },
  terminalBarrier: new TerminalTurnBarrier(),
  terminalDrainTimeoutMs: 100,
});
assert.equal(
  await non404Before(
    { prompt: "q", messages },
    { ...ctx, sessionId: "non404", runId: "non404-run" },
  ),
  undefined,
);
assert.equal(unexpectedLegacy, false);

// All core-bound OpenClaw coordinates use the same bounded hash normalizer.
const hugeCalls = [];
const hugeId = "identifier-" + "x".repeat(5_000);
const hugeHook = createAgentEndHook({
  client: {
    async contextBootstrap(payload) { hugeCalls.push(["bootstrap", payload]); },
    async cognitionCommit(payload) {
      hugeCalls.push(["commit", payload]);
      return { committed: true, duplicate: false };
    },
  },
  logger: {},
  async resolveAgentId() { return hugeId; },
  terminalBarrier: new TerminalTurnBarrier(),
});
await hugeHook(
  { runId: hugeId, messages, success: true },
  { agentId: hugeId, sessionId: hugeId, modelId: hugeId },
);
const hugeBootstrap = hugeCalls.find(([kind]) => kind === "bootstrap")[1];
const hugeCommit = hugeCalls.find(([kind]) => kind === "commit")[1];
assert.equal(hugeBootstrap.agent_id, hugeCommit.agent_id);
assert.equal(hugeBootstrap.session_id, hugeCommit.session_id);
assert.ok(hugeCommit.agent_id.length <= 256 && hugeCommit.agent_id.includes("sha256"));
assert.ok(hugeCommit.session_id.length <= 256 && hugeCommit.session_id.includes("sha256"));
assert.ok(hugeCommit.model.length <= 512 && hugeCommit.model.includes("sha256"));

console.log("OpenClaw extension contract tests passed");
