/**
 * How a screen hands a question to the look guide. The guide itself (components/guide/GuideLayer.tsx) is mounted once
 * for the whole app; any screen calls one of these.
 *
 *   askGuide(question, route)    true when the question is about the app and the guide took it (no backend call)
 *   pointAt(run)                 point at a target that is already resolved
 *   offer(say, options)          "Did you mean ...?" with the closest targets as buttons
 */

import { classify, type GuideTarget, type Locator } from "./guide";
import { binIntent, binRun } from "./guideBin";

export interface GuideRun {
  target: GuideTarget;
  /** From a link: draw the pointer at this fraction of its way (0..1) and hold it there. */
  t?: number | null;
  /** Points the pointer passes through on its way, in order (a walking route's waypoints on a map). Those not on screen are skipped. */
  via?: Locator[];
  /** More sentences for the way there, after the target's own (the walking route in words). */
  extra?: string[];
  /** Said and shown on arrival instead of the target's own last sentence. */
  arrive?: string;
  /** A line under the caption that is shown but not said. May depend on what was found on the page. */
  note?: (el: Element) => string | null;
  /**
   * The spotlight at the end of the way: once the dot arrives it fades and the thing itself is marked. `label` is the
   * two or three words on it ("Here they are", "Throw it here"). `ring: false` when the thing lights up by itself
   * (a bin in the room twin) and a ring on top would only hide it.
   */
  spot?: { label?: string; ring?: boolean };
  /** Exact or about, and why (lib/guide.ts `precisionOf`). "About" draws a dashed, soft outline and is said as "about". */
  precision?: import("./guide").Precision;
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
  if (binIntent(question) !== null) {
    // "which bin for the cup?": the room lights the bin up; the rule sentence comes from the box's rules file
    void binRun(question).then((run) => run && pointAt(run), () => undefined);
    return true;
  }
  const asked = classify(question, route);
  if (asked?.kind !== "app") return false;
  pointAt({ target: asked.target });
  return true;
}
