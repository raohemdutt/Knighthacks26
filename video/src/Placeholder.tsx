import { AbsoluteFill } from "remotion";
import type { Slot } from "./Atlas";
import { Muted, Well } from "./Soft";
import { page, type } from "./theme";

/** Until a slot's recording exists at public/clips/slot-<key>.mp4: a sunk screen saying what it must show. */
export function Placeholder({ slot }: { slot: Slot }) {
  return (
    <AbsoluteFill style={page}>
      <Well style={{ padding: "96px 120px", display: "flex", flexDirection: "column", justifyContent: "center", minHeight: 560 }}>
        <Muted>recording needed · public/clips/slot-{slot.key}.mp4</Muted>
        <div style={{ fontSize: type.title, fontWeight: 700, margin: "20px 0 28px" }}>{slot.title}</div>
        <div style={{ fontSize: type.body, lineHeight: 1.35, maxWidth: 1400 }}>{slot.shows}</div>
      </Well>
    </AbsoluteFill>
  );
}
