import type { CSSProperties, ReactNode } from "react";
import { press, pressed, raised, theme, type } from "./theme";

/** A keycap: raised, or sunk by `down` (0 to 1). The one place the accent colour appears. */
export function Key({ label, size = 160, down = 0, style }: { label: string; size?: number; down?: number; style?: CSSProperties }) {
  return (
    <div
      style={{
        ...press(down),
        width: size,
        height: size,
        borderRadius: size * 0.22,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.5,
        fontWeight: 700,
        color: theme.accent,
        ...style,
      }}
    >
      {label}
    </div>
  );
}

/** A raised card. */
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ ...raised, padding: "40px 48px", ...style }}>{children}</div>;
}

/** A sunk well, for a screen or a number. */
export function Well({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ ...pressed, padding: "40px 48px", ...style }}>{children}</div>;
}

/** Muted label text. */
export function Muted({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ fontSize: type.small, color: theme.muted, ...style }}>{children}</div>;
}
