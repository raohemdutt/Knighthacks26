import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import type { Slot } from "./Atlas";
import { TITLE_FRAMES, theme } from "./theme";

/** 1.5 s at the start of a slot: where the judge landed, and which tracks this minute serves. */
export function TitleCard({ slot }: { slot: Slot }) {
  const frame = useCurrentFrame();
  const fade = interpolate(frame, [TITLE_FRAMES - 10, TITLE_FRAMES], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, color: theme.text, justifyContent: "center", padding: "0 160px", opacity: fade }}>
      <div style={{ fontSize: 40, color: theme.muted }}>
        {slot.stamp} · key {slot.key}
      </div>
      <div style={{ fontSize: 120, fontWeight: 700, margin: "24px 0" }}>{slot.title}</div>
      <div style={{ fontSize: 44, color: theme.warm }}>{slot.watchers.join(" · ")}</div>
    </AbsoluteFill>
  );
}
