"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * The theme: "dark" (Night) or "light" (Day). It lives on <html data-theme>, set before the first paint by the
 * script in app/layout.tsx and remembered under this key. The 3D room watches the attribute to relight itself.
 */

export type Theme = "dark" | "light";
export const THEME_KEY = "atlas.theme";

export function readTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Nowhere to remember it; the choice still holds for this visit.
  }
  // The browser's own chrome (the address bar on a phone) follows.
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta) meta.content = theme === "dark" ? "#070b14" : "#eef2f8";
}

/**
 * Flip the theme. With the View Transitions API the new theme spreads out from the point that was tapped
 * (globals.css animates ::view-transition-new(root) as a growing circle); elsewhere it simply switches.
 */
export function setTheme(theme: Theme, from?: { x: number; y: number }) {
  const doc = document as Document & { startViewTransition?: (update: () => void) => { finished: Promise<void> } };
  const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  if (!doc.startViewTransition || calm) {
    apply(theme);
    return;
  }
  const root = document.documentElement;
  if (from) {
    root.style.setProperty("--vt-x", `${from.x}px`);
    root.style.setProperty("--vt-y", `${from.y}px`);
  }
  doc.startViewTransition(() => apply(theme));
}

/** The current theme, kept in step with the attribute however it changes. */
export function useTheme(): [Theme, (next: Theme, from?: { x: number; y: number }) => void] {
  const [theme, setState] = useState<Theme>("dark");
  useEffect(() => {
    setState(readTheme());
    const watcher = new MutationObserver(() => setState(readTheme()));
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => watcher.disconnect();
  }, []);
  const set = useCallback((next: Theme, from?: { x: number; y: number }) => setTheme(next, from), []);
  return [theme, set];
}
