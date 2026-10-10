import { AbsoluteFill, OffthreadVideo, Sequence, staticFile } from "remotion";
import { IndexCard } from "./IndexCard";
import { Intro3D } from "./Intro3D";
import { Placeholder } from "./Placeholder";
import { TitleCard } from "./TitleCard";
import { TITLE_FRAMES, theme } from "./theme";
import timeline from "./timeline.json";

export type Slot = (typeof timeline.slots)[number];

const INDEX_FRAMES = 150; // the index card's 5 s at the start, and again in the intro's last 5 s

/** Slot k: a title card naming the tracks it serves, then its recording (or a placeholder until it exists). */
function SlotBody({ slot }: { slot: Slot }) {
  const body = timeline.slot_frames - TITLE_FRAMES;
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <Sequence durationInFrames={TITLE_FRAMES}>
        <TitleCard slot={slot} />
      </Sequence>
      <Sequence from={TITLE_FRAMES} durationInFrames={body}>
        {slot.clip ? <OffthreadVideo src={staticFile(slot.clip)} /> : <Placeholder slot={slot} />}
      </Sequence>
    </AbsoluteFill>
  );
}

/** Slot 0: index card, the 3D Atlas box, index card again so judges know which key to press. */
function Intro({ slot }: { slot: Slot }) {
  const middle = timeline.slot_frames - 2 * INDEX_FRAMES;
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <Sequence durationInFrames={INDEX_FRAMES}>
        <IndexCard text={timeline.index_card} />
      </Sequence>
      <Sequence from={INDEX_FRAMES} durationInFrames={middle}>
        {slot.clip ? <OffthreadVideo src={staticFile(slot.clip)} /> : <Intro3D />}
      </Sequence>
      <Sequence from={INDEX_FRAMES + middle} durationInFrames={INDEX_FRAMES}>
        <IndexCard text={timeline.index_card} />
      </Sequence>
    </AbsoluteFill>
  );
}

export const Atlas: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: theme.bg, fontFamily: theme.font }}>
    {timeline.slots.map((slot) => (
      <Sequence key={slot.key} from={slot.start_frame} durationInFrames={timeline.slot_frames} name={`${slot.key} ${slot.title}`}>
        {slot.key === 0 ? <Intro slot={slot} /> : <SlotBody slot={slot} />}
      </Sequence>
    ))}
  </AbsoluteFill>
);
