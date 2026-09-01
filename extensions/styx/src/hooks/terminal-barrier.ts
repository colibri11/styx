/**
 * Per-agent/session hand-off from a terminal `agent_end` to the next
 * `before_prompt_build`.
 *
 * OpenClaw may dispatch hooks without making the next model pass wait for
 * durable side effects from the previous turn.  Tracking is synchronous:
 * `track()` installs the promise before the work factory gets its first
 * microtask, so an immediately-started prompt build can see it.
 */

export type TerminalDrainResult = {
  completed: boolean;
  pending: number;
};

export function terminalScopeKey(
  openclawAgentId: string,
  sessionId?: string,
  sessionKey?: string,
): string {
  return `${openclawAgentId}::${sessionId || sessionKey || "-"}`;
}

export class TerminalTurnBarrier {
  private readonly pendingByScope = new Map<string, Set<Promise<void>>>();
  private readonly inFlightByTurn = new Map<string, Promise<void>>();
  private readonly tailByScope = new Map<string, Promise<void>>();

  /** Track one terminal run, single-flight for the same scope + identity. */
  track(
    scope: string,
    identity: string,
    work: () => Promise<void>,
  ): Promise<void> {
    const turnKey = `${scope}\0${identity}`;
    const existing = this.inFlightByTurn.get(turnKey);
    if (existing) return existing;

    const scopePending = this.pendingByScope.get(scope) ?? new Set<Promise<void>>();
    this.pendingByScope.set(scope, scopePending);

    let tracked!: Promise<void>;
    // The rejection branch makes the barrier fail-open even if a future work
    // implementation accidentally lets an exception escape.
    const previous = this.tailByScope.get(scope) ?? Promise.resolve();
    tracked = previous
      .then(work)
      .then(() => undefined, () => undefined)
      .finally(() => {
        scopePending.delete(tracked);
        if (scopePending.size === 0) this.pendingByScope.delete(scope);
        if (this.inFlightByTurn.get(turnKey) === tracked) {
          this.inFlightByTurn.delete(turnKey);
        }
        if (this.tailByScope.get(scope) === tracked) {
          this.tailByScope.delete(scope);
        }
      });

    this.inFlightByTurn.set(turnKey, tracked);
    this.tailByScope.set(scope, tracked);
    scopePending.add(tracked);
    return tracked;
  }

  /** Wait for work visible at drain start, but never beyond `timeoutMs`. */
  async drain(scope: string, timeoutMs: number): Promise<TerminalDrainResult> {
    const snapshot = Array.from(this.pendingByScope.get(scope) ?? []);
    if (snapshot.length === 0) return { completed: true, pending: 0 };

    let timer: ReturnType<typeof setTimeout> | undefined;
    const completed = await Promise.race([
      Promise.allSettled(snapshot).then(() => true),
      new Promise<false>((resolve) => {
        timer = setTimeout(() => resolve(false), Math.max(0, timeoutMs));
      }),
    ]);
    if (timer !== undefined) clearTimeout(timer);
    if (!completed) {
      // Quarantine timed-out work from later drains. It remains single-flight
      // by turn identity and cleans itself up if it eventually settles, but a
      // permanently stuck observer cannot charge every subsequent turn the
      // same timeout again.
      const current = this.pendingByScope.get(scope);
      for (const promise of snapshot) current?.delete(promise);
      if (current?.size === 0) this.pendingByScope.delete(scope);
    }
    return { completed, pending: snapshot.length };
  }
}
