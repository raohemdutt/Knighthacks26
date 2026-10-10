"use client";

/**
 * The day in words (contexts/day narrator): the latest `day.narrated` of each moment and the plain asks for
 * pending approval cards, kept in one place so the Now screen, the voice, the toasts and the Ask screen all
 * read the same sentences.
 *
 * It is filled once from GET /api/day/brief and then kept current by the `day.narrated` events on /ws
 * (AtlasProvider feeds it), so nothing polls. Until the first answer arrives `mode` is null: callers then
 * behave as they did before narration existed (raw event texts), and never both.
 */

import { useSyncExternalStore } from "react";
import { api, type AtlasEvent, type DayBrief, type DayMoment, type DayNarration } from "./api";

export interface BriefState {
  /** null until the box has answered; "off" when it has no narrator (or no Day loop). */
  mode: DayBrief["mode"] | null;
  engine: DayBrief["engine"] | null;
  latest: Partial<Record<DayMoment, DayNarration>>;
  /** action id -> how Atlas asks for it. */
  asks: Record<string, DayNarration>;
}

const EMPTY: BriefState = { mode: null, engine: null, latest: {}, asks: {} };

class BriefStore {
  private state: BriefState = EMPTY;
  private subs = new Set<() => void>();
  private loading: Promise<void> | null = null;

  subscribe = (fn: () => void) => {
    this.subs.add(fn);
    return () => {
      this.subs.delete(fn);
    };
  };

  snapshot = () => this.state;

  /** True once the box said it narrates: raw day texts are then never toasted or spoken as well. */
  on(): boolean {
    return this.state.mode !== null && this.state.mode !== "off";
  }

  askFor(actionId: string): DayNarration | undefined {
    return this.state.asks[actionId];
  }

  load(): Promise<void> {
    if (this.loading) return this.loading;
    this.loading = api
      .dayBrief()
      .then((b) => {
        // events that arrived while the request was in flight are newer than its answer
        const latest = { ...b.latest };
        for (const [moment, mine] of Object.entries(this.state.latest) as [DayMoment, DayNarration][]) {
          const theirs = latest[moment];
          if (!theirs || Date.parse(mine.ts) > Date.parse(theirs.ts)) latest[moment] = mine;
        }
        this.set({ mode: b.mode, engine: b.engine, latest, asks: { ...b.asks, ...this.state.asks } });
      })
      .catch(() => {
        // no Day loop on this box (503), or it is unreachable: no narration to prefer
        if (this.state.mode === null) this.set({ ...this.state, mode: "off", engine: "off" });
      })
      .finally(() => {
        this.loading = null;
      });
    return this.loading;
  }

  /** A `day.narrated` event from /ws (or the answer to a POST). */
  apply(e: AtlasEvent | DayNarration): void {
    if (e.type !== "day.narrated") return;
    const n = e as unknown as DayNarration;
    if (typeof n.moment !== "string" || typeof n.spoken !== "string") return;
    const s = this.state;
    const mode = s.mode === null || s.mode === "off" ? "template" : s.mode; // an event proves the narrator is on
    if (n.moment === "approval") {
      if (n.ref) this.set({ ...s, mode, asks: { ...s.asks, [n.ref]: n } });
      return;
    }
    const last = s.latest[n.moment];
    if (last && Date.parse(last.ts) > Date.parse(n.ts)) return;
    this.set({ ...s, mode, latest: { ...s.latest, [n.moment]: n } });
  }

  private set(next: BriefState): void {
    this.state = next;
    this.subs.forEach((fn) => fn());
  }
}

export const briefStore = new BriefStore();

export function useBrief(): BriefState {
  return useSyncExternalStore(briefStore.subscribe, briefStore.snapshot, () => EMPTY);
}

/**
 * The narration to give someone who asks "brief me" (`brief`: the rest of the day) or "how am I doing"
 * (`status`): asked of the box so it is current, falling back to the latest one it already wrote.
 */
export async function briefMe(what: "brief" | "status"): Promise<DayNarration | null> {
  const moment: DayMoment = what === "status" ? "now" : "morning";
  try {
    const made = await api.dayBriefMake({ moment, fast: true });
    briefStore.apply(made.narration);
    return made.narration;
  } catch {
    const s = briefStore.snapshot();
    return s.latest[moment] ?? s.latest[moment === "now" ? "morning" : "now"] ?? null;
  }
}
