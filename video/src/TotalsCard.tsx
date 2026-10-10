import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card, Muted } from "./Soft";
import { page, theme, type } from "./theme";
import totals from "./totals.json";

type Total = { name: string; value: string; source: string; events: number };

/** Measured rows only: a number the log does not hold is left out, never shown as zero or typed in. */
export const MEASURED: Total[] = (totals as Total[]).filter((t) => t.events > 0 && !t.value.startsWith("none"));

/**
 * "Running at our table, all weekend": the hackathon totals from the box's own event log
 * (`npm run totals` = `python -m contexts.demo totals --json`, D11), one raised tile each, source in small type.
 */
export function TotalsCard() {
  const frame = useCurrentFrame();
  const rows = MEASURED.slice(0, 6);
  return (
    <AbsoluteFill style={page}>
      <div style={{ fontSize: type.title, fontWeight: 600, marginBottom: 56 }}>Running at our table, all weekend</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 48 }}>
        {rows.map((t, i) => {
          const a = interpolate(frame - i * 5, [0, 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          return (
            <Card key={t.name} style={{ opacity: a, transform: `translateY(${(1 - a) * 16}px)` }}>
              <div style={{ fontSize: type.display * 0.8, fontWeight: 700, color: theme.accent, lineHeight: 1 }}>{t.value}</div>
              <div style={{ fontSize: type.body * 0.85, fontWeight: 500, margin: "18px 0 10px" }}>{t.name}</div>
              <Muted style={{ fontSize: 22 }}>{t.source.replace(/`/g, "")}</Muted>
            </Card>
          );
        })}
      </div>
    </AbsoluteFill>
  );
}
