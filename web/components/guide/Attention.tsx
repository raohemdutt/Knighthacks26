"use client";

import { useState, useSyncExternalStore, type ReactNode } from "react";
import ui from "@/components/ui.module.css";
import { attention, summary, type AttentionState, type Job, type Missing } from "@/lib/guideAttention";
import { pointAt } from "@/lib/guideBus";
import g from "./guide.module.css";

/**
 * The attention check's small parts (lib/guideAttention.ts holds the state and the rules):
 *  - <Seen id> a "seen" tick on something the operator's eyes have rested on;
 *  - <AttentionLine> "You've looked at 3 of 4 things that need you", or the one line inviting gaze when it is off;
 *  - useLookFirst() the prompt before an action: says what has not been looked at, sends the pointer there, and
 *    offers "Continue anyway". It never blocks;
 *  - <ScanPath> the order and dwell of what was looked at, drawn from the elements' rectangles (no screenshot, no camera picture).
 */
const OFF: AttentionState = { mode: "off", seen: {}, scan: [] };
export const useAttention = (): AttentionState => useSyncExternalStore(attention.subscribe, attention.get, () => OFF);

export function Seen({ id, also = [] }: { id: string; also?: string[] }) {
  const a = useAttention();
  if (a.mode === "off" || ![id, ...also].some((k) => a.seen[k])) return null;
  return (
    <span className={g.seen} data-seen={id} title="Your eyes rested here">
      ✓ seen
    </span>
  );
}

export function AttentionLine({ jobs }: { jobs: Job[] }) {
  const a = useAttention();
  if (a.mode === "off") return <p className={g.attention}>Turn on “Use my eyes” in Settings to keep track of what you have looked at.</p>;
  return (
    <p className={g.attention} data-attention={a.mode} aria-live="polite">
      <b>{summary(a.seen, jobs).text}</b> {a.mode === "sim" ? "Simulated gaze, not a camera." : "By gaze: looked at, not necessarily read."}
    </p>
  );
}

/** Send the pointer to something on this screen that has not been looked at. It is on the tab the person is on, so nothing is opened. */
function point(m: Missing): void {
  pointAt({ target: { id: m.target, route: null, label: "Look here first", at: [{ id: m.target }], steps: [{ say: m.say }] } });
}

/**
 * `ask(missing, go)`: when something that matters for the action has not been looked at, show `prompt` instead of acting,
 * with the pointer on its way there; "Continue anyway" runs `go`. With gaze off, or nothing missing, `go` runs at once.
 */
export function useLookFirst(): { check: (missing: () => Missing | null, go: () => void) => void; prompt: ReactNode } {
  const a = useAttention();
  const [held, setHeld] = useState<{ missing: Missing; go: () => void } | null>(null);
  const check = (missing: () => Missing | null, go: () => void) => {
    const m = a.mode === "off" ? null : missing();
    if (!m) return go();
    setHeld({ missing: m, go });
    point(m);
  };
  const prompt = held ? (
    <div className={g.lookFirst} role="group" aria-label="Look first" data-look-first={held.missing.target}>
      <p>
        <b>{held.missing.say}</b> I am pointing at it now.
      </p>
      <div className={`${ui.row} ${ui.wrap}`}>
        <button type="button" className={ui.btn} onClick={() => point(held.missing)}>
          Show me again
        </button>
        <button type="button" className={ui.btn} onClick={() => (setHeld(null), held.go())}>
          Continue anyway
        </button>
      </div>
    </div>
  ) : null;
  return { check, prompt };
}

/** The scan path: numbered boxes where the eyes rested, in order, joined by a line. Built from element rectangles only. */
export function ScanPath({ names = {} }: { names?: Record<string, string> }) {
  const a = useAttention();
  const stops = a.scan.filter((s) => a.seen[s.id]).map((s) => ({ ...s, box: a.seen[s.id]!.box }));
  if (a.mode === "off" || stops.length === 0) return null;
  const x0 = Math.min(...stops.map((s) => s.box.left));
  const y0 = Math.min(...stops.map((s) => s.box.top));
  const w = Math.max(...stops.map((s) => s.box.right)) - x0;
  const h = Math.max(...stops.map((s) => s.box.bottom)) - y0;
  const k = Math.max(w / 640, h / 300, 1e-6); // fit into 640 x 300
  const at = (s: (typeof stops)[number]) => ({ x: 20 + (s.box.left - x0) / k, y: 20 + (s.box.top - y0) / k, w: Math.max(8, (s.box.right - s.box.left) / k), h: Math.max(8, (s.box.bottom - s.box.top) / k) });
  const mid = (s: (typeof stops)[number]) => ({ x: at(s).x + at(s).w / 2, y: at(s).y + at(s).h / 2 });
  return (
    <figure className={g.scan} data-scan-path={stops.length}>
      <svg viewBox={`0 0 ${w / k + 40} ${h / k + 40}`} role="img" aria-label={`Where you looked, in order: ${stops.map((s, i) => `${i + 1} ${names[s.id] ?? s.id}`).join(", ")}`}>
        {stops.map((s, i) => (
          <rect key={`b${i}`} className={g.scanBox} x={at(s).x} y={at(s).y} width={at(s).w} height={at(s).h} rx="4" />
        ))}
        <polyline className={g.scanLine} points={stops.map((s) => `${mid(s).x},${mid(s).y}`).join(" ")} />
        {stops.map((s, i) => (
          <g key={`n${i}`} transform={`translate(${mid(s).x} ${mid(s).y})`}>
            <circle className={g.scanDot} r={Math.min(16, 8 + s.ms / 400)} />
            <text className={g.scanNum} y="4" textAnchor="middle">
              {i + 1}
            </text>
          </g>
        ))}
      </svg>
      <figcaption>
        Where your eyes rested, in order ({a.mode === "sim" ? "simulated" : "by gaze"}):{" "}
        {stops.map((s, i) => `${i + 1}. ${names[s.id] ?? s.id} (${(s.ms / 1000).toFixed(1)} s)`).join(" · ")}. Drawn from the positions of things on the screen, not from a picture.
      </figcaption>
    </figure>
  );
}
