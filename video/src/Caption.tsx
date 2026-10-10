import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card } from "./Soft";
import { theme, type } from "./theme";

export const CAPTION_FRAMES = 240; // 8 s: long enough to read the line, gone before the recording needs the room

/** The narrator's opening line for a slot, in one raised card low on the screen, over the recording. */
export function Caption({ text }: { text: string }) {
  const frame = useCurrentFrame();
  const a = interpolate(frame, [0, 12, CAPTION_FRAMES - 15, CAPTION_FRAMES], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 72, pointerEvents: "none" }}>
      <Card style={{ padding: "26px 44px", maxWidth: 1500, opacity: a, transform: `translateY(${(1 - a) * 12}px)`, fontFamily: theme.font }}>
        <div style={{ fontSize: type.body * 0.9, fontWeight: 500, color: theme.ink, lineHeight: 1.3, textAlign: "center" }}>{text}</div>
      </Card>
    </AbsoluteFill>
  );
}
