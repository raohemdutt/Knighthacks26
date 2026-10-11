import type { CircleWho, ThreadState } from "@/lib/api";
import c from "./circle.module.css";

const COLOURS = [c.c0, c.c1, c.c2, c.c3, c.c4, c.c5].map((x) => x ?? "");

/** One colour per member's agent: their place in the Circle's member list, so it is the same on every tab. */
export function colourOf(members: CircleWho[], memberId: string | null): string {
  if (!memberId) return "";
  const at = members.findIndex((m) => m.member_id === memberId);
  if (at >= 0) return COLOURS[at % COLOURS.length] ?? "";
  let sum = 0;
  for (const ch of memberId) sum += ch.charCodeAt(0);
  return COLOURS[sum % COLOURS.length] ?? "";
}

export const STATE: Record<ThreadState, { label: string; className: string }> = {
  proposed: { label: "Needs your OK", className: c.proposed ?? "" },
  waiting: { label: "Waiting for answers", className: c.waiting ?? "" },
  countered: { label: "Another time suggested", className: c.countered ?? "" },
  agreed: { label: "Agreed", className: c.agreed ?? "" },
  declined: { label: "Someone is busy", className: c.declined ?? "" },
};

export type CircleTab = "people" | "plans" | "league" | "sharing";
export const TABS: { id: CircleTab; label: string }[] = [
  { id: "people", label: "People" },
  { id: "plans", label: "Plans" },
  { id: "league", label: "League" },
  { id: "sharing", label: "Sharing" },
];
