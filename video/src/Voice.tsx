import { AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Card } from "./Soft";
import { theme, type } from "./theme";
import voice from "./voice.json";

type Line = { n: number; text: string; start: number; duration: number; file: string };
type VoiceSlot = { title: string; lines: Line[]; speech_s: number; ends_at_s: number };

const SLOTS = voice.slots as Record<string, VoiceSlot>;

export function voiceFor(key: number): VoiceSlot | undefined {
  return SLOTS[String(key)];
}

/** One spoken line's caption: in with the voice, out shortly after it ends. */
function LineCaption({ text, frames }: { text: string; frames: number }) {
  const frame = useCurrentFrame();
  const a = interpolate(frame, [0, 8, frames - 10, frames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 64, pointerEvents: "none" }}>
      <Card style={{ padding: "22px 40px", maxWidth: 1560, opacity: a, transform: `translateY(${(1 - a) * 10}px)`, fontFamily: theme.font }}>
        <div style={{ fontSize: type.body * 0.8, fontWeight: 500, color: theme.ink, lineHeight: 1.3, textAlign: "center" }}>{text}</div>
      </Card>
    </AbsoluteFill>
  );
}

/**
 * The narrator for one slot: every line of script.yaml, spoken at its timed start (video/scripts/voice.py), with a
 * caption that follows the voice. Mount it at the slot's own frame 0; starts are seconds within the slot.
 */
export function Voice({ slot, captions = true }: { slot: number; captions?: boolean }) {
  const { fps } = useVideoConfig();
  const v = voiceFor(slot);
  if (!v) return null;
  return (
    <>
      {v.lines.map((l) => {
        const from = Math.round(l.start * fps);
        const frames = Math.round((l.duration + 0.5) * fps);
        return (
          <Sequence key={l.n} from={from} durationInFrames={frames} name={`voice ${slot}.${l.n}`}>
            <Audio src={staticFile(l.file)} />
            {captions ? <LineCaption text={l.text} frames={frames} /> : null}
          </Sequence>
        );
      })}
    </>
  );
}
