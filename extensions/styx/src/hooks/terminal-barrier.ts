/**
 * Legacy per-agent/session hand-off retained for direct compatibility tests.
 * Current OpenClaw uses its durable accepted-turn outbox instead.
 *
 * OpenClaw may dispatch hooks without making the next model pass wait for
 * durable side effects from the previous turn.  Tracking is synchronous:
 * `track()` installs the promise before the work factory gets its first
 * microtask, so an immediately-started prompt build can see it.
 */

export type TerminalDrainResult = {
  completed: boolean;
  pending: number;
  pendingIdentities: string[];
};

const STATE_TTL_MS = 30 * 60_000;
const PRETURN_TTL_MS = 2 * 60_000;
const MAX_STATE_ENTRIES = 2_048;

type Timed<T> = { value: T; touchedAt: number };

export function terminalScopeKey(
  openclawAgentId: string,
  sessionId?: string,
  sessionKey?: string,
): string {
  return `${openclawAgentId}::${sessionId || sessionKey || "-"}`;
}

export class TerminalTurnBarrier {
  private readonly pendingByScope = new Map<string, Map<string, Promise<void>>>();
  private readonly inFlightByTurn = new Map<string, Promise<void>>();
  private readonly tailByScope = new Map<string, Promise<void>>();
  private readonly snapshotsByTurn = new Map<string, Timed<string | null>>();
  private readonly ancestry = new Map<string, Timed<{
    actKey: string;
    parentActKey: string | null;
    snapshotToken: string | null;
  }>>();
  private readonly lastActByScope = new Map<string, Timed<string>>();
  private readonly preturnByScope = new Map<string, Timed<{
    identity: string | null;
    promise: Promise<unknown>;
  }>>();

  private prune(now = Date.now()): void {
    for (const [key, entry] of this.snapshotsByTurn) {
      if (now - entry.touchedAt > PRETURN_TTL_MS) this.snapshotsByTurn.delete(key);
    }
    for (const [key, entry] of this.preturnByScope) {
      if (now - entry.touchedAt > PRETURN_TTL_MS) this.preturnByScope.delete(key);
    }
    for (const [key, entry] of this.ancestry) {
      if (now - entry.touchedAt > STATE_TTL_MS) this.ancestry.delete(key);
    }
    for (const [key, entry] of this.lastActByScope) {
      if (now - entry.touchedAt > STATE_TTL_MS) this.lastActByScope.delete(key);
    }
    for (const map of [this.snapshotsByTurn, this.preturnByScope,
      this.ancestry, this.lastActByScope]) {
      while (map.size > MAX_STATE_ENTRIES) {
        const oldest = map.keys().next().value as string | undefined;
        if (oldest === undefined) break;
        map.delete(oldest);
      }
    }
  }

  /** Single canonical core preturn per active scope; terminal declaration closes it. */
  getOrCreatePreturn<T>(
    scope: string,
    identity: string | null,
    work: () => Promise<T>,
  ): Promise<T> {
    const now = Date.now();
    this.prune(now);
    const existing = this.preturnByScope.get(scope);
    if (existing) {
      if ((identity === null && existing.value.identity === null) ||
          existing.value.identity === identity || identity === null) {
        existing.touchedAt = now;
        this.preturnByScope.delete(scope);
        this.preturnByScope.set(scope, existing);
        return existing.value.promise as Promise<T>;
      }
      // A newly identified run must not inherit an unkeyed or predecessor
      // response: core must see the same physical host key as agent_end.
      this.preturnByScope.delete(scope);
    }
    let pending!: Promise<T>;
    pending = work().catch((error) => {
      if (this.preturnByScope.get(scope)?.value.promise === pending) {
        this.preturnByScope.delete(scope);
      }
      throw error;
    });
    this.preturnByScope.set(scope, {
      value: { identity, promise: pending }, touchedAt: now,
    });
    this.prune(now);
    return pending;
  }

  rememberSnapshot(
    scope: string,
    snapshotToken: string | null,
    identity?: string | null,
  ): void {
    // Unkeyed FIFO snapshots can attach a cancelled/stale preturn to the next
    // physical act. If the host has no run/turn identity, omit the fence.
    if (!identity) return;
    const now = Date.now();
    this.prune(now);
    const key = `${scope}\0${identity}`;
    this.snapshotsByTurn.delete(key);
    this.snapshotsByTurn.set(key, { value: snapshotToken, touchedAt: now });
    this.prune(now);
  }

  /** Return the last declared physical act without mutating ancestry. */
  predecessorActKey(scope: string): string | null {
    const now = Date.now();
    this.prune(now);
    const entry = this.lastActByScope.get(scope);
    if (!entry) return null;
    entry.touchedAt = now;
    this.lastActByScope.delete(scope);
    this.lastActByScope.set(scope, entry);
    return entry.value;
  }

  /** Declare physical ancestry once; retries return the original coordinates. */
  declareAct(
    scope: string,
    identity: string,
    actKey: string,
  ): { actKey: string; parentActKey: string | null; snapshotToken: string | null } {
    const coordinate = `${scope}\0${identity}`;
    const now = Date.now();
    this.prune(now);
    const existing = this.ancestry.get(coordinate);
    if (existing) {
      existing.touchedAt = now;
      this.ancestry.delete(coordinate);
      this.ancestry.set(coordinate, existing);
      return existing.value;
    }
    const keyedSnapshot = this.snapshotsByTurn.get(coordinate);
    const parent = this.lastActByScope.get(scope);
    const declared = {
      actKey,
      parentActKey: parent?.value ?? null,
      snapshotToken: keyedSnapshot?.value ?? null,
    };
    this.snapshotsByTurn.delete(coordinate);
    this.preturnByScope.delete(scope);
    this.ancestry.set(coordinate, { value: declared, touchedAt: now });
    this.lastActByScope.set(scope, { value: actKey, touchedAt: now });
    this.prune(now);
    return declared;
  }

  /** Track one terminal run, single-flight for the same scope + identity. */
  track(
    scope: string,
    identity: string,
    work: () => Promise<void>,
  ): Promise<void> {
    const turnKey = `${scope}\0${identity}`;
    const existing = this.inFlightByTurn.get(turnKey);
    if (existing) return existing;

    const scopePending = this.pendingByScope.get(scope) ?? new Map<string, Promise<void>>();
    this.pendingByScope.set(scope, scopePending);

    let tracked!: Promise<void>;
    // The rejection branch makes the barrier fail-open even if a future work
    // implementation accidentally lets an exception escape.
    const previous = this.tailByScope.get(scope) ?? Promise.resolve();
    tracked = previous
      .then(work)
      .then(() => undefined, () => undefined)
      .finally(() => {
        scopePending.delete(identity);
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
    scopePending.set(identity, tracked);
    return tracked;
  }

  /** Wait for work visible at drain start, but never beyond `timeoutMs`. */
  async drain(scope: string, timeoutMs: number): Promise<TerminalDrainResult> {
    const entries = Array.from(this.pendingByScope.get(scope)?.entries() ?? []);
    const snapshot = entries.map(([, promise]) => promise);
    const pendingIdentities = entries.map(([identity]) => identity);
    if (snapshot.length === 0) {
      return { completed: true, pending: 0, pendingIdentities: [] };
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    const completed = await Promise.race([
      Promise.allSettled(snapshot).then(() => true),
      new Promise<false>((resolve) => {
        timer = setTimeout(() => resolve(false), Math.max(0, timeoutMs));
      }),
    ]);
    if (timer !== undefined) clearTimeout(timer);
    // Never delete a timed-out predecessor. The next prompt may fail open, but
    // it must report the pending host identities to the fenced preturn and all
    // later terminal commits remain ordered behind the same per-scope tail.
    return { completed, pending: snapshot.length, pendingIdentities };
  }
}
