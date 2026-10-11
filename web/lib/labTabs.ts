/**
 * The Learning lab's three tabs. They are a grouping and nothing more: the five views the lab lane built (lib/labBench.ts
 * LAB_TABS: Training run, Alone vs with a friend, Speed and energy, Routing, Experiments) are each shown whole and
 * unchanged, two to a tab where they belong together.
 *
 *   The model          the loop in 3D, then the Training run view, then the Experiments view
 *   Speed and energy   the Speed and energy view, then the Routing view
 *   Together           the Alone vs with a friend view (Compare)
 *
 * Every `?tab=` value the screen ever took still works: it opens the tab that holds that view and scrolls to the view,
 * whose section carries the same word as its id (#run, #experiments, #speed, #routing, #compare).
 *
 * No imports at run time (the type is erased), so `node scripts/check-lab.mjs` can run this file as it is.
 */
import type { LabTab } from "./labBench";

export type LabGroup = "model" | "speed" | "together";

export const LAB_GROUPS: { key: LabGroup; title: string; views: LabTab[] }[] = [
  { key: "model", title: "The model", views: ["run", "experiments"] },
  { key: "speed", title: "Speed and energy", views: ["speed", "routing"] },
  { key: "together", title: "Together", views: ["compare"] },
];

/** The tab that holds a view. */
export function labGroupOf(view: LabTab): LabGroup {
  return LAB_GROUPS.find((g) => g.views.includes(view))?.key ?? "model";
}

/** The view a `?tab=` value (or a `#` anchor) asks for: the five old values as they were, and the two new tab names. */
export function labViewOf(asked: string | null | undefined): LabTab | null {
  if (asked === "run" || asked === "experiments" || asked === "speed" || asked === "routing" || asked === "compare") return asked;
  if (asked === "model") return "run";
  if (asked === "together") return "compare";
  return null;
}

/** What `?tab=` says for a view: nothing for the default, else the view's own old value, so old links stay as written. */
export function labTabParam(view: LabTab): string | null {
  return view === "run" ? null : view;
}
