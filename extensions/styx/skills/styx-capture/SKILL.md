---
name: styx-capture
description: "Persist an explicit durable trace via styx_store when a decision, fact, episode, or integrated understanding should remain available across turns. NOT for raw dialogue, external pipeline evidence, code that can be re-read, ephemeral task state, or material that has not participated in cognition."
---

# Styx Capture

Save a structured subjective trace to Styx. `styx_store` is an explicit legacy-compatible incorporation surface: it preserves useful material but cannot by itself prove its cognitive provenance. Prefer content that actually affected a decision or understanding, and keep external evidence in archive/evidence channels until it has participated in cognition.

## What Styx already captures automatically

- **Every accepted dialogue turn** lands in the diary (`memories` with `role IN ('user','assistant')`) through host finality (`cognition/commit`; legacy hosts use `sync_turn`). You do **not** need `styx_dialogue_save` for ordinary turns — calling it on a turn that already happened produces a duplicate.
- **The selective gatekeeper** (wave 17) compares every `styx_store` write against recent neighbours by cosine and Levenshtein. It will **merge** a near-identical write into the existing memory, **supersede** an older formulation when the new one substantially refines it, or **store** a fresh entry. Do not pre-check for duplicates yourself; store the value and let the gatekeeper decide. This is the single most common LLM mistake to avoid.
- **Provisional importance** is computed from kind / role / metadata at write time; an LLM worker later refines it into `importance_final`. The grace period keeps unscored memories alive.
- **Auto-link** (wave 18) attaches new memories to recent dialogue turns and other memories along semantic distance. You do not need to manually link routine context — `styx_link` is for explicit causal/structural edges only (see below).
- **Store-routing for long content** — if your `content` exceeds 2400 characters Styx automatically routes it to `documents` + `chunks` (with embeddings) and keeps a tail-summary memory with an `archive_ref`. The long form will be searchable through `styx_search_archive`; the tail will live as a normal memory in `styx_recall`. This means: do not pre-truncate or pre-chunk. Pass the full content; Styx will do the right thing.

## When to call `styx_store`

Call it when the conversation surfaced a **structured fragment that would otherwise be lost between the lines of the diary**:

- **`fact`** — verified information about the user, project, system, or environment that is not already visible in code or docs.
- **`decision`** — a choice made with rationale; record both *what* was chosen and *why* (and what was rejected).
- **`episode`** — a notable event: debugging session outcome, incident, deployment, breakthrough, session summary.
- **`concept`** — a reusable pattern, principle, or domain abstraction that future conversations will reach for.
- **`note`** — loose capture for things you want to come back to but do not fit the other kinds. This is also the default when you omit `kind`.

## How to call it

```
styx_store({
  content: "self-contained description with the why, not just the what",
  kind: "decision",
  importance_provisional: 0.85,   // optional [0..1]; default 0.5
  metadata: {
    source: "chat 2026-05-10",
    tags: ["architecture", "v1"]
  }
})
```

Parameter notes:
- **`agent_id` is not a parameter** here. Styx infers the caller scope from the OpenClaw session; you do not pass it.
- **`visibility` is not a parameter**. Styx writes are caller-scoped; cross-agent sharing happens through the knowledge graph (`styx_link`), not through visibility flags.
- **`importance_provisional`** is the named field for the importance hint.

### Writing good content

- **Self-contained.** A future reader who never saw this chat must still understand the entry. Expand pronouns and references. "We chose the new path because the old one hit a real failure mode" is useless without the path and the failure mode.
- **Why over what.** "Switched to PG advisory locks because the two-instance sweep collision was a real production incident" beats "Using advisory locks".
- **One memory, one idea.** Do not bundle unrelated facts. The selective gatekeeper rewards atomic entries — bundles defeat dedup and supersession.
- **Use the user's terminology** for domain concepts. The diary already accumulates their phrasing; keep your structured memories aligned to it so recall finds both.
- **Preserve the source perspective.** First person is appropriate only when the trace really records the current agent's decision or understanding; never convert external evidence into a first-person claim.

### The `importance_provisional` hint

This is the main lever you have over scoring. Use it sparingly but deliberately:

- `0.9+` — landmark decision, strong preference, architectural invariant.
- `0.7` — typical fact worth keeping for months (default zone for decisions).
- `0.5` — ordinary note (and the implicit default if you omit it).
- `0.3-` — short-lived context you will likely not need after the week.

Do not fight the formula with hints. The provisional score combines per-kind base, role bonus, supersede context, and your hint; an LLM worker later replaces it with an `importance_final` based on actual content. Hints set the starting point and bias scoring during the grace period; they are not a replacement for the formula.

### Supersession

If the user *corrects* a prior memory or updates a decision, store the new version normally. The gatekeeper compares it against recent entries and either merges (cosine > merge threshold), supersedes (close but distinct phrasing), or keeps a fresh record. You do not delete anything; supersession sets `superseded_by` automatically and the old version becomes background — the recall pipeline gives it less weight without losing it.

### Refinement vs. supersession vs. correction

This is a frequent confusion point:

- **Correction / contradiction** ("I was wrong about X, the truth is Y") → `styx_store` again with the corrected statement. The gatekeeper supersedes the old version. Old version stays as background.
- **Refinement / new coordinate** ("I now understand X more deeply, with this additional facet") → `styx_reinterpret` (see the `styx-reinterpret` skill). The memory_id stays the same, the graph stays intact, the meaning shifts via weighted-average blending of embeddings. Use this when the new understanding *integrates* with the old one rather than replacing it.
- **Pure typo / phrasing fix** → `styx_store` with the corrected text; let the gatekeeper supersede.

If you cannot tell which one applies, default to `styx_store` (correction / supersession). Reinterpret has a 24h cooldown per memory, so do not casually reach for it.

## When NOT to call `styx_store`

- **Ordinary dialogue turns.** OpenClaw `commitTurn` already captured the accepted turn into the diary. Manual writes duplicate and pollute. Call `styx_dialogue_save` *only* for replies that arrived from outside the OpenClaw turn loop and need to be injected into the diary as `user`/`assistant` (rare, e.g. a telegram bridge). Never for what just happened in this conversation.
- **Small talk, greetings, confirmations.** The gatekeeper will filter most, but the call still wastes gatekeeper work and turns.
- **Anything already in the current conversation** that the agent can re-read directly. Styx is for what would otherwise be lost between turns or sessions.
- **Anything visible in code or docs.** The agent can read those. Storing them produces stale shadows that drift from source-of-truth.
- **Transient task state** (current step, todo, progress). Use the OpenClaw task list, not memory.
- **Raw logs or verbose tool output.** Summarise first, store the summary as an `episode`.
- **Pipeline ingest** (telegram bot, scheduled job, audio pipeline producing material). That is `styx_ingest_experience` — idempotent by `content_hash`, does not go through gatekeeper merge/supersede semantics, scoped for machine-driven flow.
- **A thought you want to come back to within this turn.** That is what your reasoning is for; storing then immediately re-recalling is a wasted round trip.

## Related write tools

- `styx_dialogue_save` — explicit one-off diary write. See `styx-recall` skill for when this is justified (very rare).
- `styx_ingest_experience` — pipeline-channel ingest. Idempotent by `content_hash`, no gatekeeper merge/supersede. Use for material from external automated channels, not for interactive writes from the LLM.
- `styx_reinterpret` — refine a memory's meaning without losing its identity. See the `styx-reinterpret` skill.
- `styx_link` — explicit edge in the knowledge graph (`memory ↔ memory`, `memory ↔ document`, `memory ↔ dialogue`, etc.). Auto-link covers routine semantic neighbourhood; use `styx_link` only for relations that auto-link cannot infer (e.g. "decision X *was caused by* incident Y" — a structural causal link). `relation` is open vocabulary; sensible types: `related_to`, `discussed_in`, `mentions`, `caused_by`, `derived_from`, `supersedes`. Idempotent by `(source, target, relation)` UNIQUE — repeat calls return `created: false`.

## Markers in your input

When you act on something the user said vs something Styx remembered, you need
to tell them apart. Anything between `<styx-*>...</styx-*>` tags is a
Styx-injected fragment, not a user message. The canonical automatic preturn is
the fenced `<styx-cognitive-continuity>` envelope; read it first to decide
whether the durable fragment already exists. `<styx-salient>` is the legacy
automatic recall marker, while `<styx-recall>` is an explicit-tool result.
Full taxonomy and decision logic is in the `styx-recall` skill under "How to
read markers in your input".
