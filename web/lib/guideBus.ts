/**
 * How a screen hands a question to the look guide. The guide itself (components/guide/GuideLayer.tsx) is mounted once
 * for the whole app; any screen calls one of these.
 *
 *   askGuide(question, route)    true when the question is about the app and the guide took it (no backend call)
 *   pointAt(run)                 point at a target that is already resolved
 *   offer(say, options)          "Did you mean ...?" with the closest targets as buttons
 */

import { classify, type GuideTarget } from "./guide";

export interface GuideRun {
  target: GuideTarget;
  /** From a link: draw the pointer at this fraction of its way (0..1) and hold it there. */
  t?: number | null;
  /** Said and shown on arrival instead of the target's own last sentence. */
  arrive?: string;
  /** A line under the caption that is shown but not said. May depend on what was found on the page. */
  note?: (el: Element) => string | null;
  /** One button in the caption ("Show me in the room"). */
  action?: { label: string; run: GuideRun };
}

export type GuideMessage = { kind: "run"; run: GuideRun } | { kind: "offer"; say: string; options: { label: string; run?: GuideRun; ask?: string }[] } | { kind: "calibrate" };

const listeners = new Set<(m: GuideMessage) => void>();
let waiting: GuideMessage | null = null;

export function onGuide(fn: (m: GuideMessage) => void): () => void {
  listeners.add(fn);
  if (waiting) {
    const m = waiting;
    waiting = null;
    fn(m);
  }
  return () => listeners.delete(fn);
}

function send(m: GuideMessage): void {
  if (listeners.size === 0) waiting = m;
  for (const fn of listeners) fn(m);
}

export const pointAt = (run: GuideRun): void => send({ kind: "run", run });
export const offer = (say: string, options: { label: string; run?: GuideRun; ask?: string }[]): void => send({ kind: "offer", say, options });
export const calibrateGaze = (): void => send({ kind: "calibrate" });

/** A where-question about the app itself: point there and say true. Anything else: false, and the caller carries on as before. */
export function askGuide(question: string, route: string | null): boolean {
  const asked = classify(question, route);
  if (asked?.kind !== "app") return false;
  pointAt({ target: asked.target });
  return true;
}
