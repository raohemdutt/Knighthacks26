import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import type { Slot } from "./Atlas";
import { Key, Muted } from "./Soft";
import { page, TITLE_FRAMES, type } from "./theme";

/** 1.5 s at the start of a slot: the key goes down, the slot's title, and which tracks this minute serves. */
export function TitleCard({ slot }: { slot: Slot }) {
  const frame = useCurrentFrame();
  const down = interpolate(frame, [0, 10], [0, 1], { extrapolateRight: "clamp" });
  const fade = interpolate(frame, [TITLE_FRAMES - 8, TITLE_FRAMES], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ ...page, opacity: fade }}>
      <div style={{ display: "flex", alignItems: "center", gap: 96 }}>
        <Key label={String(slot.key)} size={240} down={down} />
        <div>
          <Muted>{slot.stamp}</Muted>
          <div style={{ fontSize: type.display, fontWeight: 700, lineHeight: 1.05, margin: "12px 0 28px" }}>{slot.title}</div>
          <div style={{ fontSize: type.body, fontWeight: 500 }}>{slot.watchers.join("  ·  ")}</div>
        </div>
      </div>
    </AbsoluteFill>
  );
}
