import { AbsoluteFill } from "remotion";
import { theme } from "./theme";
import timeline from "./timeline.json";

/** The 5 s index: watch one minute, press your key, watch two. Built from the same track list as the chapters. */
export function IndexCard({ text }: { text: string }) {
  const [headline] = text.split("\n");
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, color: theme.text, justifyContent: "center", padding: "0 160px" }}>
      <div style={{ fontSize: 64, fontWeight: 700, marginBottom: 48 }}>{headline}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px 80px", fontSize: 44 }}>
        {timeline.tracks.map((t) => (
          <div
            key={t.name}
            style={{ display: "flex", justifyContent: "space-between", borderBottom: `2px solid ${theme.panel}`, gridColumn: t.key === null ? "1 / -1" : undefined }}
          >
            <span>{t.name}</span>
            <b style={{ color: theme.warm }}>{t.key === null ? "keep watching" : `press ${t.key}`}</b>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
}
