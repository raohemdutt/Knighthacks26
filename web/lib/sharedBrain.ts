/**
 * The Circle's Shared brain card (G6): the shape of GET /api/circle/shared-brain (contexts/circle/brain_pair.py) and
 * the card's pure rules, kept free of imports so scripts/check-lab.mjs can run them in node.
 */

export interface PairMember {
  name: string; // "pod C"
  role: "head" | "half";
  node: string; // pod-a, pod-b
  layers: number;
  online: boolean;
  watts_now: number | null;
  watts_idle: number | null;
  vram_gb: number | null; // GiB that half of the model holds on the pod, measured at load
}

export interface SharedBrainPair {
  configured: boolean;
  model: { name: string; params_b: number; active_b: number; quant: string; served: string[] };
  members: PairMember[];
  link: { kind: string; rtt_ms: number; mbps: number };
  serving: { tok_s: number | null; ttft_s: number | null; requests_running: number | null };
  updated: string;
  reason: string | null;
}

/** What the stage shows: both friends bumping, one resting alone, or nobody ("no friends online"). */
export type PairState = "both" | "one" | "none";

export function pairState(b: SharedBrainPair | null): PairState {
  if (!b || !b.configured) return "none";
  const up = b.members.filter((m) => m.online).length;
  return up >= 2 ? "both" : up === 1 ? "one" : "none";
}

/** "Qwen3.5-122B-A10B" reads as "Qwen3.5-122B": the active-parameter suffix is said on the line below. */
export function shortModel(name: string): string {
  return name.replace(/-A\d+(\.\d+)?B$/i, "");
}

const COUNT = ["no", "one", "two", "three", "four"];

/** "Qwen3.5-122B, split 24 and 24 layers across two friends". */
export function modelLine(b: SharedBrainPair): string {
  const layers = b.members.map((m) => m.layers);
  const split = layers.length === 2 ? `${layers[0]} and ${layers[1]}` : layers.join(", ");
  return `${shortModel(b.model.name)}, split ${split} layers across ${COUNT[b.members.length] ?? b.members.length} friends`;
}

/** "0.36 ms over a tunnel". */
export function linkLine(b: SharedBrainPair): string {
  return `${b.link.rtt_ms} ms over a ${b.link.kind}`;
}

/** "10.0 tokens a second, first token in 0.29 s", or what is missing. */
export function servingLine(b: SharedBrainPair): string {
  const { tok_s, ttft_s, requests_running } = b.serving;
  if (tok_s === null && ttft_s === null) return "not serving right now";
  const parts = [];
  if (tok_s !== null) parts.push(`${tok_s.toFixed(1)} tokens a second`);
  if (ttft_s !== null) parts.push(`first token in ${ttft_s.toFixed(2)} s`);
  let line = parts.join(", ");
  if (requests_running) line += `, ${requests_running} answering now`;
  return line;
}

/** A member's watts now, whole watts: "61 W", or "--" when its sampler did not answer. */
export function wattsText(w: number | null): string {
  return w === null ? "--" : `${Math.round(w)} W`;
}
