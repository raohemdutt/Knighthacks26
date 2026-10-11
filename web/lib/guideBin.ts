/**
 * "Where does this go?" (Waste Watch / Sort It): the item, the bin, and the rule sentence from the box's own rules file.
 *
 * Thin on purpose: the bin and the sentence come from GET /api/impact/story (`rules.items`, or `rules.default_rule`
 * for anything not listed), and the way to the bin is the one the room twin already draws when Sort It previews an
 * item. This file only works out which item was meant, tells Sort It to preview it, and words the answer. The red
 * dot travels to the room and fades; the twin's own highlight ("Cup goes here", the route) is the spotlight.
 */

import { api } from "./api";
import { plainText, precisionOf, type GuideTarget } from "./guide";
import type { GuideRun } from "./guideBus";

const ASK =
  /^(?:(?:hey |ok |okay )?atlas )?(?:where (?:does|do|should|would) (?:this|that|it|the|my|a|an|these|those)? ?(.*?) ?go(?: in)?|which bin (?:is |does |do |should )?(?:for )?(?:this|that|it|the|my|a|an)? ?(.*?)(?: go(?: in| into)?| belong(?: in)?)?|where (?:do|should|can) (?:i|we) (?:throw|put|toss|bin|recycle|chuck|drop)(?: away| out)? ?(?:this|that|it|the|my|a|an)? ?(.*?)(?: away| out)?|(?:can|do) (?:i|we) recycle (?:this|that|it|the|my|a|an)? ?(.*?)|is (?:this|that|the|my|a|an)? ?(.*?) recyclable)$/;

/** "Which bin for the cup?" -> "cup"; "Where does this go?" -> "" (whatever is in hand); null when it is not that kind of question. */
export function binIntent(text: string): string | null {
  const m = ASK.exec(plainText(text));
  if (!m) return null;
  return (m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5] ?? "").replace(/^(?:this|that|it|one)$/, "").trim();
}

/** The rules file's name for what was said: "coffee cups" -> "cup", "soda can" -> "can". "" when nothing matches. */
export function binItem(said: string, labels: string[]): string {
  const words = said.split(" ").map((w) => (w.length > 3 && w.endsWith("s") ? w.slice(0, -1) : w));
  const whole = words.join(" ");
  return [...labels].sort((a, b) => b.length - a.length).find((l) => whole.includes(l) || words.includes(l.split(" ").pop()!)) ?? "";
}

// what Sort It is asked to preview; it reads this when it mounts and whenever it changes
let wanted: string | null = null;
const listeners = new Set<(item: string) => void>();
export const binPick = {
  get: (): string | null => wanted,
  subscribe: (fn: (item: string) => void): (() => void) => (listeners.add(fn), () => listeners.delete(fn)),
  set(item: string): void {
    wanted = item;
    listeners.forEach((fn) => fn(item));
  },
};

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** The pointer's run for a bin question, or null when it is not one. `holding` is the item Sort It is showing now. */
export async function binRun(question: string, t: number | null = null): Promise<GuideRun | null> {
  const said = binIntent(question);
  if (said === null) return null;
  const rules = (await api.impactStory()).rules;
  const item = said ? binItem(said, rules.items.map((i) => i.label)) || said : (wanted ?? rules.items[0]?.label ?? "can");
  const row = rules.items.find((i) => i.label === item);
  const bin = row?.bin ?? "trash"; // the box's own fallback: what is not listed stays out of recycling
  const rule = row?.rule ?? rules.default_rule ?? `I do not have a rule for ${item}, so it goes in ${bin}.`;
  binPick.set(item);
  const target: GuideTarget = {
    id: `bin:${item}`,
    route: "/play/",
    label: `${cap(item)} → ${cap(bin)}`,
    at: [{ id: "play:twin" }, { sel: "main svg[role=img]" }, { sel: "main h1" }],
    steps: [{ say: "Look at the room: the bin that lights up." }, { say: `${cap(bin)}. ${rule}` }],
  };
  return { target, t, precision: precisionOf("floor"), spot: { ring: false, label: "Throw it here" }, note: () => "The lit bin and the line on the floor are the room map's own: the way there is the one the box worked out." };
}
