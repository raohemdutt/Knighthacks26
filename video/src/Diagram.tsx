import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { theme, type, raised, pressed } from "./theme";

/**
 * Bird's-eye architecture diagrams in the soft-UI system: nodes are raised cards (a sunk one marks the thing being
 * talked about), edges are soft lines that draw themselves, and a pulse travels along each edge to show flow.
 * One picture per idea, few words per node. Coordinates are in a 1920x1080 frame.
 */

export type Node = { id: string; label: string; sub?: string; x: number; y: number; w?: number; h?: number; sunk?: boolean; icon?: string };
export type Edge = { from: string; to: string; at?: number; label?: string; dashed?: boolean; labelBelow?: boolean };
export type Scene = { nodes: Node[]; edges: Edge[]; groups?: { label: string; x: number; y: number; w: number; h: number }[] };

const W = 300;
const H = 120;

function center(n: Node) {
  return { x: n.x + (n.w ?? W) / 2, y: n.y + (n.h ?? H) / 2 };
}

/** Where a straight line from a to b leaves a's box: the edge starts and ends at the card borders, not the centres. */
function border(a: Node, toward: { x: number; y: number }) {
  const c = center(a);
  const hw = (a.w ?? W) / 2;
  const hh = (a.h ?? H) / 2;
  const dx = toward.x - c.x;
  const dy = toward.y - c.y;
  const s = Math.min(hw / Math.max(Math.abs(dx), 1e-6), hh / Math.max(Math.abs(dy), 1e-6));
  return { x: c.x + dx * s, y: c.y + dy * s };
}

export function Diagram({ scene, appear = 0, step = 6 }: { scene: Scene; appear?: number; step?: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const byId = Object.fromEntries(scene.nodes.map((n) => [n.id, n]));
  return (
    <AbsoluteFill style={{ backgroundColor: theme.surface, fontFamily: theme.font }}>
    <div style={{ position: "absolute", left: 0, top: 30, width: 1920, height: 1080, transform: "scale(0.8)", transformOrigin: "960px 0px" }}>
      {scene.groups?.map((g, i) => {
        const a = spring({ frame: frame - appear - i * step, fps, config: { damping: 20 } });
        return (
          <div key={g.label} style={{ position: "absolute", left: g.x, top: g.y, width: g.w, height: g.h, ...pressed, borderRadius: 44, opacity: a }}>
            <div style={{ position: "absolute", left: 32, top: 22, fontSize: type.small, color: theme.muted, fontWeight: 600, letterSpacing: 1 }}>{g.label.toUpperCase()}</div>
          </div>
        );
      })}
      <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0 }}>
        {scene.edges.map((e, i) => {
          const a = byId[e.from];
          const b = byId[e.to];
          if (!a || !b) return null;
          const p = border(a, center(b));
          const q = border(b, center(a));
          const start = appear + scene.nodes.length * step + i * step * 2 + (e.at ?? 0);
          const draw = interpolate(frame - start, [0, 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const len = Math.hypot(q.x - p.x, q.y - p.y);
          const t = ((frame - start) % 90) / 90;
          const px = p.x + (q.x - p.x) * t;
          const py = p.y + (q.y - p.y) * t;
          const mx = (p.x + q.x) / 2;
          const my = (p.y + q.y) / 2;
          return (
            <g key={`${e.from}-${e.to}`}>
              {/* the groove: a light line over a shade line, offset like the cards' shadows */}
              <line x1={p.x + 2} y1={p.y + 2} x2={q.x + 2} y2={q.y + 2} stroke={theme.light} strokeWidth={6} strokeLinecap="round"
                    strokeDasharray={len} strokeDashoffset={len * (1 - draw)} />
              <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke={theme.shade} strokeWidth={6} strokeLinecap="round"
                    strokeDasharray={e.dashed ? `18 18` : len} strokeDashoffset={e.dashed ? 0 : len * (1 - draw)} opacity={e.dashed ? draw : 1} />
              {draw >= 1 ? <circle cx={px} cy={py} r={9} fill={theme.accent} opacity={0.9} /> : null}
              {e.label && draw > 0.5 ? (
                <text x={mx} y={e.labelBelow ? my + 36 : my - 16} textAnchor="middle" fontSize={24} fontFamily={theme.font} fill={theme.muted} fontWeight={600}>
                  {e.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      {scene.nodes.map((n, i) => {
        const a = spring({ frame: frame - appear - i * step, fps, config: { damping: 18 } });
        return (
          <div
            key={n.id}
            style={{
              position: "absolute",
              left: n.x,
              top: n.y,
              width: n.w ?? W,
              height: n.h ?? H,
              ...(n.sunk ? pressed : raised),
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              opacity: a,
              transform: `translateY(${(1 - a) * 18}px)`,
              color: theme.ink,
              padding: "0 20px",
              boxSizing: "border-box",
              textAlign: "center",
            }}
          >
            {n.icon ? <div style={{ fontSize: 34, lineHeight: 1 }}>{n.icon}</div> : null}
            <div style={{ fontSize: type.body * 0.75, fontWeight: 700, lineHeight: 1.1 }}>{n.label}</div>
            {n.sub ? <div style={{ fontSize: type.small * 0.8, color: theme.muted, lineHeight: 1.15 }}>{n.sub}</div> : null}
          </div>
        );
      })}
    </div>
    </AbsoluteFill>
  );
}
