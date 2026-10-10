import { AbsoluteFill } from "remotion";
import type { Slot } from "./Atlas";
import { theme } from "./theme";

/** Until a slot's recording exists at public/clips/slot-<key>.mp4: what it must show, so the cut is reviewable. */
export function Placeholder({ slot }: { slot: Slot }) {
  return (
    <AbsoluteFill style={{ backgroundColor: theme.panel, color: theme.text, justifyContent: "center", padding: "0 160px" }}>
      <div style={{ fontSize: 36, color: theme.cool }}>RECORDING NEEDED · public/clips/slot-{slot.key}.mp4</div>
      <div style={{ fontSize: 88, fontWeight: 700, margin: "24px 0" }}>{slot.title}</div>
      <div style={{ fontSize: 44, lineHeight: 1.35, color: theme.muted, maxWidth: 1500 }}>{slot.shows}</div>
    </AbsoluteFill>
  );
}
