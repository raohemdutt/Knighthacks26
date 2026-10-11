import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Key, Muted } from "./Soft";
import { page, type } from "./theme";
import timeline from "./timeline.json";

/**
 * The 5 s index: one sentence, then a row of keys. Each key carries its track underneath, so the judge reads one
 * number and nothing else. The general judges get one quiet line: they press nothing.
 */
export function IndexCard() {
  const frame = useCurrentFrame();
  const keyed = timeline.tracks.filter((t) => t.key !== null);
  const general = timeline.tracks.find((t) => t.key === null);
  return (
    <AbsoluteFill style={page}>
      <div style={{ fontSize: type.title, fontWeight: 600, lineHeight: 1.15, maxWidth: 1400 }}>
        Watch one minute. Press your key. Watch two.
      </div>
      <div style={{ display: "flex", gap: 44, marginTop: 96 }}>
        {keyed.map((t, i) => {
          const rise = interpolate(frame - i * 4, [0, 14], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          return (
            <div key={t.name} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 28, width: 190 }}>
              <Key label={String(t.key)} size={160} down={rise} />
              <div style={{ fontSize: type.small, fontWeight: 600, textAlign: "center", lineHeight: 1.2 }}>{t.name}</div>
            </div>
          );
        })}
      </div>
      {general ? <Muted style={{ marginTop: 80 }}>{general.name}: press nothing, keep watching.</Muted> : null}
    </AbsoluteFill>
  );
}
