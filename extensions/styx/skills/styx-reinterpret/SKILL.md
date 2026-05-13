---
name: styx-reinterpret
description: "Refine the meaning of an existing memory via styx_reinterpret — add a new coordinate of understanding without rewriting history. Use when: (1) a new understanding has integrated into a prior one — same fact, deeper grasp, sharper formulation, (2) you want recall to lean toward the refined understanding without losing the trace of the original, (3) you want the memory_id and its graph edges preserved (relations, recall_history, lifecycle stay intact). NOT for: contradictions or corrections (use styx_store and let the gatekeeper supersede), typo fixes (use styx_store), bundling several refinements into one call (one memory, one new coordinate), or refining a memory you have not actually re-engaged with — reinterpret is for *integrated* understanding, not speculation."
---

# Styx Reinterpret

`styx_reinterpret` is the Styx-native channel for the conceptual operation called "переосмысление" in the foundational treatise on which Styx is built ([IAmBook §V][iambook]): **the meaning of a memory shifts in vector space through weighted averaging of embeddings, not through overwriting the text**. The previous understanding remains as a softer signal in the background; the new coordinate pulls recall toward the refined meaning. The memory's identity (its UUID, its place in the graph, its recall history) is preserved. This is the operation that distinguishes a living line of `я` from a versioned database of statements.

[iambook]: https://github.com/colibri11/IAm/blob/main/IAmBook_EN.md

## When reinterpret is the right operation

The operation makes sense exactly when the new understanding **integrates** with the prior one. The old memory said something, you now understand it better — same object, sharper view. Concrete signals:

- The user reframes their own earlier statement and you see why the reframing is right.
- You worked through an ambiguity from an old memory and now know which reading was correct.
- A vague concept memory crystallised after weeks of incidental references — you can now state the same idea more precisely.
- Two adjacent memories converged into a single understanding; you keep one, reinterpret the other to mark the convergence.

In every case the test is: "is the *new* phrasing a refinement of the *old* one, such that I want recall to find this idea slightly differently next time, but I still want the original anchor preserved as background?" If yes → reinterpret. If no → either `styx_store` (new memory or supersession of the old) or do nothing.

## When reinterpret is the wrong operation

- **The user said the prior memory is wrong.** That is contradiction, not refinement. Use `styx_store` with the corrected statement; let the gatekeeper supersede the old. The old version stays as background by being marked `superseded_by`, *not* by being blended.
- **You are fixing a typo or a phrasing slip.** Same — `styx_store` with the fixed text; gatekeeper supersedes.
- **You want to add several distinct refinements at once.** One reinterpret per memory per call, and one new coordinate per refinement. Bundling muddles the blend.
- **You have not actually re-engaged with the memory.** Reinterpret applies to memories whose meaning shifted *in your trajectory*. Reinterpreting an arbitrary memory you found via recall but did not work with is speculation; it pulls the embedding for no good reason.
- **The memory is ≤24h old or you reinterpreted it recently.** There is a per-memory cooldown of 24h; a second call within the window returns `status=cooldown` and does nothing.
- **You just stored the memory in this turn.** Same idea — let it settle, observe how it gets recalled, refine on a later turn if needed.

## How it works under the hood (so you can reason about it)

You do not need to know the implementation to use the tool, but it helps to understand the semantics:

1. The new understanding text is embedded.
2. The memory's existing embedding is **blended** with the new one via weighted average — `weight=0.5` is an equal mix; higher weight pulls toward the new understanding more strongly. This is the `previous_embedding` × `(1-w)` + `new_embedding` × `w` pattern.
3. The memory's text is rewritten by an LLM handler that **fuses** previous and new text into a coherent merged statement (not concatenation, not bullet-list of additions).
4. The original previous text + previous embedding + the new understanding text + the merged text + the weight + the timestamp are recorded in `memory_reinterpretations` for full audit. Reinterpret is reversible in the sense that the trace is intact; the *active* memory is the merged form.
5. The actual blend is **deferred** — applied after the current turn closes (through the `reinterpret_apply_sweeper`, typically 30-90 seconds later). The tool returns immediately with a status that reports queued vs cooldown vs other failure. Subsequent turns will see the merged form once the sweeper has run.

This means: do not expect the very next `styx_recall` in the same turn to reflect the new coordinate. By next user turn it usually has applied; if not, the `pending_sleep` state is visible through `styx_explain(kind='lifetime', memory_id=...)`.

## How to call it

```
styx_reinterpret({
  memory_id: "<uuid of the memory you are refining>",
  new_understanding_text: "1-3 sentences in Russian if the original is in Russian, in first person if the original is in first person; what *added* to the understanding, not a restatement of the whole memory",
  weight: 0.5      // optional [0..1], default 0.5
})
```

### Choosing `weight`

- `0.3-0.4` — the new understanding is a small refinement; recall should still mostly reach the original framing.
- `0.5` (default) — equal mix; both formulations are equally valid coordinates of the same idea.
- `0.6-0.7` — the new understanding is the better current formulation; recall should lean toward it but the old anchor still matters.
- `0.8+` — rarely correct. If the new framing is *that* much better, it is probably a different memory; either `styx_store` (and let the gatekeeper supersede) or accept that you are erasing the prior coordinate, which usually means `styx_store` with supersede was the right tool.

### Writing `new_understanding_text`

- **State what *added* to the understanding**, not the whole revised memory. The LLM handler will fuse previous and new into the merged form; you do not need to repeat the previous content.
- **One coordinate, one call.** "I now also realise A and B and C" should be three reinterprets on three memories, not one omnibus update on one.
- **Match the voice of the original.** First-person if the original was first-person, third-person if not; same language. The merged text should read as a continuation of the original voice, not a footnote in a different register.
- **≤2400 chars.** This is the hard schema limit; in practice 1-3 sentences is the right scope.

### Idempotency / retry semantics

- The 24h cooldown is checked at call time. A retry within the cooldown returns `status=cooldown` cleanly — no error, no side effect.
- Apply is idempotent at the sweeper level: if the sweeper retries it picks up the same `pending_sleep` row, applies once, transitions to `applied`.
- If the apply phase fails (LLM fuse step crashes, embed fails), the row stays `pending_sleep` and the sweeper retries on its next pass. You do not need to retry from the LLM side.

## After the call

- **Do not immediately re-recall the same idea** to "verify" the merge — apply is deferred (30-90 seconds typical). If you genuinely need to confirm the apply landed, `styx_explain(kind='lifetime', memory_id=...)` will show the new `applied_at` timestamp once the sweeper ran.
- **Confirm usage on the original recall** if the reinterpret was triggered by something you actually used in your reply: `styx_confirm_usage({ memory_ids: [<original_id>] })`. The reinterpret itself is not a recall event.
- **Consider linking** if the new understanding came from a specific document or another memory: `styx_link({ source_type: 'memory', source_id: <reinterpreted_id>, target_type: 'document', target_id: <doc_id>, relation: 'derived_from' })`. Auto-link covers routine semantic neighbourhood; explicit causal links benefit from `styx_link`.

## Concrete examples

### Good

> "Earlier I noted: «proposal X improves the index by 12%». In today's session the user explained that the 12% number was from synthetic load — under real traffic the gain is closer to 4%, and the proposal is still worth doing because the lower variance matters more than the median."
>
> → `styx_reinterpret(memory_id=<original 12% memory>, new_understanding_text="On real traffic the gain is closer to 4%, but the lower variance is the actual win.", weight=0.6)`

> "Two months ago I stored a vague concept-memory about «recall as geometry not retrieval». After three sessions where this kept clarifying, I can now say: recall is geometry of input *for the active turn*, with archive as a separate channel — the salient block and the archive are different surfaces."
>
> → `styx_reinterpret(memory_id=<concept memory>, new_understanding_text="Salient block and archive are different surfaces; recall-as-geometry applies to the active turn, not to the archive.", weight=0.5)`

### Wrong

> "User said yesterday they prefer Python over Rust. Today they said the opposite."
>
> → **Not** reinterpret. This is contradiction; `styx_store` with the new preference, gatekeeper supersedes.

> "I want to add a tag to an old memory."
>
> → **Not** reinterpret. Tags live in `metadata`; if you really need to update them, store a fresh memory with the desired metadata or accept that metadata is a write-time field.

> "I think this old memory is generally important and I want to bump its weight."
>
> → **Not** reinterpret. Importance is set by the formula and the LLM importance-scoring worker. There is no manual nudge; gaming the embedding via reinterpret is exactly the abuse the cooldown exists to discourage.

## Why this exists

The text in `memories.content` is the surface; the meaning lives in the embedding. Rewriting only the text loses the trajectory of how an idea matured; recreating the memory from scratch loses the graph. Reinterpret is the operation that *moves* the meaning while keeping the identity — the line of `я` does not stop being the same line because understanding deepened. If a memory accrues several reinterprets over months, the resulting embedding is a weighted average of every coordinate the agent has held about that idea, with the latest having the most weight but never erasing the earlier shape. That is what makes the trajectory continuous rather than a series of point-replacements.

## Markers in your input

When you decide whether to reinterpret, you need to read the existing memory carefully — and that usually means reading the salient block (`<styx-salient>...</styx-salient>`) Styx injected before your last user turn, plus possibly an explicit `styx_recall` or `styx_explain(kind='lifetime')` call. Anything wrapped in `<styx-*>...</styx-*>` is a Styx-injected fragment of memory, **not** the user's words; identity of the memory you are about to reinterpret lives in those wrapped fragments, not in the live conversation. Full taxonomy in the `styx-recall` skill under "How to read markers in your input".
