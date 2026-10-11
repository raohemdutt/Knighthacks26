/**
 * The live power graph's pure rules (components/lab/LivePower.tsx), kept free of imports so scripts/check-lab.mjs can
 * run them in node. Readings come from GET /api/circle/shared-brain, one per poll, stamped with the browser's clock
 * when the answer arrived. Nothing here makes up a value: a pod that is offline or has no reading is a gap.
 */

/** The fields of one shared-brain member the graph reads (lib/sharedBrain.ts PairMember). */
export interface PowerMember {
  name: string;
  node: string;
  online: boolean;
  watts_now: number | null;
  watts_idle: number | null;
}

/** One poll: when it arrived (ms, browser clock) and each pod's watts by node, null where there is no reading. */
export interface PowerReading {
  t: number;
  w: Record<string, number | null>;
}

/** The pods the graph draws, in order. */
export const POWER_PODS: { node: string; name: string }[] = [
  { node: "pod-a", name: "pod C" },
  { node: "pod-b", name: "pod D" },
];

/** Two minutes on the x axis. */
export const POWER_SPAN_MS = 120_000;

/** Two readings further apart than this are not joined and not integrated: the meters missed polls in between. */
export const POWER_MAX_GAP_MS = 6_000;

/**
 * The reading one poll gives. `route` null means the poll failed; a route that is not configured, a member that is
 * offline or a member without watts_now gives null for that pod, so its line stops there.
 */
export function readingFrom(route: { configured: boolean; members: PowerMember[] } | null, t: number): PowerReading {
  const w: Record<string, number | null> = {};
  for (const p of POWER_PODS) {
    const m = route && route.configured ? route.members.find((x) => x.node === p.node) : undefined;
    w[p.node] = m && m.online && typeof m.watts_now === "number" && Number.isFinite(m.watts_now) ? m.watts_now : null;
  }
  return { t, w };
}

/** The buffer with `r` added and anything older than the span (plus one poll of margin) dropped. */
export function appendReading(buf: PowerReading[], r: PowerReading, spanMs = POWER_SPAN_MS): PowerReading[] {
  const keep = r.t - spanMs - 5_000;
  return [...buf.filter((x) => x.t >= keep && x.t < r.t), r];
}

/**
 * Joules between two readings, trapezoid, summed over the pods that have a reading at both ends. Pairs further apart
 * than `maxGapMs` add nothing: the area across a missed stretch would be a guess.
 */
export function trapezoidJoules(a: PowerReading, b: PowerReading, maxGapMs = POWER_MAX_GAP_MS): number {
  const dt = (b.t - a.t) / 1000;
  if (!(dt > 0) || b.t - a.t > maxGapMs) return 0;
  let j = 0;
  for (const p of POWER_PODS) {
    const wa = a.w[p.node];
    const wb = b.w[p.node];
    if (typeof wa === "number" && typeof wb === "number") j += ((wa + wb) / 2) * dt;
  }
  return j;
}

/** True when no pod has a reading: the poll failed, the route is not configured, or every member is offline. */
export function metersOffline(r: PowerReading | null): boolean {
  return !r || POWER_PODS.every((p) => r.w[p.node] === null || r.w[p.node] === undefined);
}

/**
 * One pod's line for the chart: [t, watts] pairs, with a break ("-") where its reading is missing or two readings are
 * further apart than `maxGapMs`, so the line stops instead of bridging the gap.
 */
export function podLine(buf: PowerReading[], node: string, maxGapMs = POWER_MAX_GAP_MS): [number, number | "-"][] {
  const out: [number, number | "-"][] = [];
  let prev: number | null = null;
  for (const r of buf) {
    const w = r.w[node];
    if (prev !== null && r.t - prev > maxGapMs) out.push([prev + 1, "-"]);
    out.push([r.t, typeof w === "number" ? w : "-"]);
    prev = r.t;
  }
  return out;
}

/** "0.42 kJ" for the running energy counter; whole kJ past 100. */
export function kjText(joules: number): string {
  const kj = joules / 1000;
  return `${kj >= 100 ? kj.toFixed(0) : kj >= 10 ? kj.toFixed(1) : kj.toFixed(2)} kJ`;
}
