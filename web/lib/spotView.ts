"use client";

import { useSyncExternalStore } from "react";

/**
 * Which part of Inspection (/spot/) is on show. Inspection is its own workspace inside Atlas: the sidebar lists these
 * views in place of Atlas's sections, and the page renders the one the address names. The address is the only state:
 * ?view=<name>, or the older ?tab=run / ?tab=analysis, so old links keep opening the same thing.
 */
export type SpotView = "shift" | "equipment" | "rounds" | "approvals" | "training" | "analysis" | "handoff";

/** The menu, in order. The labels are the menu's own; each one opens a part of Inspection that already exists. */
export const SPOT_MENU: { view: SpotView; label: string; href: string; keywords: string }[] = [
  { view: "shift", label: "My shift", href: "/spot", keywords: "inspection status floor plan do this first watch out" },
  { view: "equipment", label: "Equipment", href: "/spot?view=equipment", keywords: "inspection assets all equipment list" },
  { view: "rounds", label: "Rounds", href: "/spot?view=rounds", keywords: "inspection round mission steps replay" },
  { view: "approvals", label: "Approvals", href: "/spot?view=approvals", keywords: "inspection approve deny work order" },
  { view: "training", label: "Training", href: "/spot?tab=run", keywords: "inspection shift run plan a round game" },
  { view: "analysis", label: "Analysis", href: "/spot?tab=analysis", keywords: "inspection findings evidence numbers" },
  { view: "handoff", label: "Handoff", href: "/spot?view=handoff", keywords: "inspection shift handoff note" },
];

const VIEWS = new Set<string>(SPOT_MENU.map((m) => m.view));

export function viewOf(search: string): SpotView {
  const q = new URLSearchParams(search);
  const v = q.get("view")?.toLowerCase();
  if (v && VIEWS.has(v)) return v as SpotView;
  if (q.get("tab") === "run") return "training";
  if (q.get("tab") === "analysis") return "analysis";
  if (q.get("round")) return "rounds"; // ?round=M017&step=6: that round, at that step
  if (q.get("all")) return "equipment"; // ?all=1: all equipment
  return "shift";
}

const listeners = new Set<() => void>();
let patched = false;

/** The router moves between views with history.pushState / replaceState, which fire no event: tell the listeners. */
function patch() {
  if (patched || typeof window === "undefined") return;
  patched = true;
  const tell = () => window.setTimeout(() => listeners.forEach((fn) => fn()), 0);
  for (const name of ["pushState", "replaceState"] as const) {
    const real = window.history[name];
    window.history[name] = function (this: History, ...args: Parameters<History["pushState"]>) {
      const out = real.apply(this, args);
      tell();
      return out;
    };
  }
  window.addEventListener("popstate", tell);
}

function subscribe(fn: () => void): () => void {
  patch();
  listeners.add(fn);
  return () => void listeners.delete(fn);
}

const get = (): SpotView => viewOf(window.location.search);
const onServer = (): SpotView => "shift";

export function useSpotView(): SpotView {
  return useSyncExternalStore(subscribe, get, onServer);
}

/** Move to a view from inside the page (the in-page buttons): the address changes, nothing reloads. */
export function goSpotView(view: SpotView) {
  if (typeof window === "undefined" || get() === view) return;
  const item = SPOT_MENU.find((m) => m.view === view)!;
  const query = item.href.split("?")[1];
  window.history.pushState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
}
