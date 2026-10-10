import type { CSSProperties } from "react";

/**
 * The video's design system: soft UI (neumorphism), minimal.
 *
 * One surface colour for everything; depth comes only from two shadows (light from the top left, shade to the
 * bottom right). A raised element sits on the surface; a pressed one sinks into it. One accent, used only for the
 * key a judge presses. Few words per screen, one idea per screen.
 */
export const theme = {
  surface: "#e4e8ee",
  light: "#ffffff",
  shade: "#b7c0cd",
  ink: "#2b313b",
  muted: "#6f7886",
  accent: "#d9774a",
  font: "Inter, 'Segoe UI', Arial, sans-serif",
  radius: 32,
};

/** Type scale: four sizes, emphasis by weight, never by a fifth size. */
export const type = { small: 28, body: 40, title: 64, display: 112 };

export const raised: CSSProperties = {
  background: theme.surface,
  borderRadius: theme.radius,
  boxShadow: `14px 14px 28px ${theme.shade}, -14px -14px 28px ${theme.light}`,
};

export const pressed: CSSProperties = {
  background: theme.surface,
  borderRadius: theme.radius,
  boxShadow: `inset 10px 10px 20px ${theme.shade}, inset -10px -10px 20px ${theme.light}`,
};

/** Between raised (0) and pressed (1): the key going down. */
export function press(t: number): CSSProperties {
  const out = 14 * (1 - t);
  const inn = 10 * t;
  return {
    background: theme.surface,
    borderRadius: theme.radius,
    boxShadow: [
      `${out}px ${out}px ${2 * out}px ${theme.shade}`,
      `${-out}px ${-out}px ${2 * out}px ${theme.light}`,
      `inset ${inn}px ${inn}px ${2 * inn}px ${theme.shade}`,
      `inset ${-inn}px ${-inn}px ${2 * inn}px ${theme.light}`,
    ].join(", "),
  };
}

export const page: CSSProperties = {
  backgroundColor: theme.surface,
  color: theme.ink,
  fontFamily: theme.font,
  padding: "0 160px",
  justifyContent: "center",
};

/** Frames the title card holds at the start of each slot: covers the number-key landing drift. */
export const TITLE_FRAMES = 45;
