/**
 * The live chat's pure rules (components/lab/LiveChat.tsx), kept free of imports so scripts/check-lab.mjs can run them
 * in node. The chat posts to POST /api/learning/lab/live-chat (contexts/learning/live_chat.py), which answers as
 * Server-Sent Events: `token` {text} as the model streams, `tool` {name, arguments} when it looks in the box's memory,
 * then one `done` or one `error`.
 */

export type ChatTarget = "pair" | "box";

export const CHAT_TARGETS: { id: ChatTarget; label: string }[] = [
  { id: "pair", label: "the pair, 122B on two pods" },
  { id: "box", label: "the box, 4B" },
];

export const WATER_BOTTLE = "Have you seen my water bottle?";

/** The energy a `done` event carries: both pods' samplers for the pair, the Jetson's for the box; null = meters offline. */
export interface ChatEnergy {
  joules: number;
  watts_avg?: number | null;
  idle_w?: number | null;
  seconds?: number;
  pods?: Record<string, { joules: number; watts_avg: number; idle_w: number | null; samples: number } | null> | null;
}

export interface ChatDone {
  answer: string;
  target: ChatTarget;
  serving: string | null;
  tokens: number;
  seconds: number;
  tok_s: number | null;
  ttft_s: number | null;
  calls: { name: string; arguments: Record<string, unknown> }[];
  energy: ChatEnergy | null;
}

export type ChatEvent =
  | { event: "token"; data: { text: string } }
  | { event: "tool"; data: { name: string; arguments: Record<string, unknown> } }
  | { event: "done"; data: ChatDone }
  | { event: "error"; data: { error: string; seconds?: number } };

/**
 * The complete events in `buffer` (blocks ended by a blank line) and what is left over for the next read. A block
 * without an event name or with data that is not JSON is skipped.
 */
export function parseSse(buffer: string): { events: ChatEvent[]; rest: string } {
  const text = buffer.replace(/\r\n/g, "\n");
  const blocks = text.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    let name = "";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (!name || !data.length) continue;
    try {
      events.push({ event: name, data: JSON.parse(data.join("\n")) } as ChatEvent);
    } catch {
      /* not JSON: not an event we can show */
    }
  }
  return { events, rest };
}

const num = (x: number, digits: number) => (Number.isFinite(x) ? x.toFixed(digits) : "?");

/** "N tokens in S s, X tok/s, about J joules", or "meters offline" in place of the joules when energy is null. */
/**
 * Tokens a second from the done event's own totals: tokens over the decode time, which is seconds minus ttft_s when
 * ttft_s is there and smaller than seconds, else seconds. The event's tok_s is not used (a near-zero window made it
 * read 28453 on the box). Null without tokens or time.
 */
export function rateOf(d: Pick<ChatDone, "tokens" | "seconds" | "ttft_s">): number | null {
  if (!(d.tokens > 0) || !(d.seconds > 0)) return null;
  const decode = typeof d.ttft_s === "number" && d.ttft_s >= 0 && d.ttft_s < d.seconds ? d.seconds - d.ttft_s : d.seconds;
  return decode > 0 ? d.tokens / decode : null;
}

export function footerText(d: ChatDone): string {
  const parts = [`${d.tokens} token${d.tokens === 1 ? "" : "s"} in ${num(d.seconds, 1)} s`];
  const rate = rateOf(d);
  if (rate !== null) parts.push(`${num(rate, 1)} tok/s`);
  parts.push(d.energy && typeof d.energy.joules === "number" ? `about ${Math.round(d.energy.joules)} joules` : "meters offline");
  return parts.join(", ");
}

/** The newest `keep` items, newest last. */
export function keepLast<T>(list: T[], keep = 3): T[] {
  return list.length > keep ? list.slice(list.length - keep) : list;
}

/** Who the chat asks: one side or both at once. */
export type ChatMode = "both" | ChatTarget;

export const CHAT_MODES: { id: ChatMode; label: string }[] = [
  { id: "box", label: "only the box" },
  { id: "pair", label: "only the pair" },
  { id: "both", label: "both" },
];

/** The heading over each side's transcript. */
export const SIDE_NAMES: Record<ChatTarget, string> = {
  box: "The box, 4B",
  pair: "The network brain, 122B on two pods",
};

export const targetsOf = (mode: ChatMode): ChatTarget[] => (mode === "both" ? ["box", "pair"] : [mode]);

const toolSet = (d: ChatDone) => Array.from(new Set(d.calls.map((c) => c.name))).sort();

function times(r: number): string {
  return `${r >= 10 ? Math.round(r) : Number(r.toFixed(1))}x`;
}

/**
 * One line comparing the two answers in words, from their done events only: which tools each used, and how their
 * joules compare. A side without a reading says so instead of a ratio.
 */
export function compareText(box: ChatDone, pair: ChatDone): string {
  const a = toolSet(box);
  const b = toolSet(pair);
  const tools =
    a.join(",") === b.join(",")
      ? a.length
        ? `same tool (${a.join(", ")})`
        : "neither looked in memory"
      : `different tools: the box used ${a.join(", ") || "none"}, the brain used ${b.join(", ") || "none"}`;
  const jb = box.energy && typeof box.energy.joules === "number" ? box.energy.joules : null;
  const jp = pair.energy && typeof pair.energy.joules === "number" ? pair.energy.joules : null;
  let energy: string;
  if (jb === null || jp === null) {
    const off = [jb === null ? "the box" : null, jp === null ? "the pods" : null].filter(Boolean).join(" and ");
    energy = `joules not compared, meters offline on ${off}`;
  } else if (jb <= 0 || jp <= 0) {
    energy = "joules not compared, a reading of zero";
  } else if (jp >= jb) {
    energy = `the brain used ${times(jp / jb)} the joules`;
  } else {
    energy = `the box used ${times(jb / jp)} the joules`;
  }
  return `${tools}, ${energy}`;
}
