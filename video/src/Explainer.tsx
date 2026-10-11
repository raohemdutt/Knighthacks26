import { AbsoluteFill } from "remotion";
import type { Slot } from "./Atlas";
import { Diagram } from "./Diagram";
import { Placeholder } from "./Placeholder";
import { SCENES } from "./scenes";

/** A slot with no recording: its bird's-eye diagram, drawn in step with the narration; a placeholder if it has none. */
export function Explainer({ slot }: { slot: Slot }) {
  const scene = SCENES[slot.key];
  if (!scene) return <Placeholder slot={slot} />;
  return (
    <AbsoluteFill>
      <Diagram scene={scene} appear={10} step={7} />
    </AbsoluteFill>
  );
}
