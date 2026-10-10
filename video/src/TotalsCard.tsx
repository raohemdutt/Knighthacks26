import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { theme } from "./theme";
import totals from "./totals.json";

type Total = { name: string; value: string; source: string; events: number };

/** Measured rows only: a number the log does not hold is left out, never shown as zero or typed in. */
export const MEASURED: Total[] = (totals as Total[]).filter((t) => t.events > 0 && !t.value.startsWith("none"));

/**
 * "Running at our table since Saturday": the hackathon totals from the box's own event log
 * (`npm run totals` = `python -m contexts.demo totals --json`, D11), each with its source in small type.
 */
export function TotalsCard() {
  const frame = useCurrentFrame();
  const rows = MEASURED.slice(0, 6);
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, color: theme.text, justifyContent: "center", padding: "0 160px" }}>
      <div style={{ fontSize: 56, fontWeight: 700, marginBottom: 40 }}>Running at our table, all weekend</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "28px 80px" }}>
        {rows.map((t, i) => (
          <div key={t.name} style={{ opacity: interpolate(frame - i * 6, [0, 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) }}>
            <div style={{ fontSize: 72, fontWeight: 700, color: theme.warm }}>{t.value}</div>
            <div style={{ fontSize: 34 }}>{t.name}</div>
            <div style={{ fontSize: 22, color: theme.muted }}>{t.source.replace(/`/g, "")}</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
}
