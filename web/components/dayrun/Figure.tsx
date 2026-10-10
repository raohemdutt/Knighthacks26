import type { Doing } from "@/lib/dayplay";
import dr from "./dayrun.module.css";

/**
 * The little person of the playback, drawn in SVG user units with the feet at
 * (x, y). `phase` is seconds of real time: it drives the walking legs, the bob
 * and the small activity cue. With `still` nothing moves (reduced motion, or a
 * paused frame).
 */
export function Figure({
  x,
  y,
  dir = 1,
  doing = "idle",
  phase = 0,
  still = false,
  tone = "you",
  label,
  thinking = false,
  scale = 1,
}: {
  x: number;
  y: number;
  dir?: 1 | -1;
  doing?: Doing;
  phase?: number;
  still?: boolean;
  tone?: "you" | "friend" | "ghost";
  label?: string;
  thinking?: boolean;
  scale?: number;
}) {
  const p = still ? 0.13 : phase;
  const walking = doing === "walk" && !thinking;
  const swing = walking ? Math.sin(p * Math.PI * 2 * 2.4) : 0;
  const bob = walking ? Math.abs(Math.cos(p * Math.PI * 2 * 2.4)) * 1.6 : doing === "sleep" ? 0 : Math.sin(p * Math.PI * 2 * 0.5) * 0.5;
  const cls = tone === "friend" ? dr.figFriend : tone === "ghost" ? dr.figGhost : dr.figYou;
  const asleep = doing === "sleep";

  return (
    <g transform={`translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${scale})`} className={cls}>
      <ellipse className={dr.figShadow} cx={0} cy={0.5} rx={asleep ? 13 : 7} ry={2.4} />
      {asleep ? (
        <g>
          {/* lying down, head to the left */}
          <rect className={dr.figBody} x={-7} y={-8} width={18} height={7.5} rx={3.5} />
          <circle className={dr.figHead} cx={-10.5} cy={-5.5} r={5} />
          {[0, 1, 2].map((i) => {
            const f = (((still ? 0.35 : p * 0.45) + i / 3) % 1 + 1) % 1;
            return (
              <text key={i} className={dr.figZ} x={-6 + f * 12} y={-13 - f * 16} fontSize={7 + f * 5} opacity={still ? 0.9 : Math.sin(f * Math.PI)}>
                z
              </text>
            );
          })}
        </g>
      ) : (
        <g transform={`translate(0 ${(-bob).toFixed(2)}) scale(${dir} 1)`}>
          <line className={dr.figLimb} x1={-1.5} y1={-8} x2={-1.5 + swing * 4.5} y2={walking ? -Math.max(0, -swing) * 1.5 : 0} />
          <line className={dr.figLimb} x1={1.5} y1={-8} x2={1.5 - swing * 4.5} y2={walking ? -Math.max(0, swing) * 1.5 : 0} />
          <rect className={dr.figPack} x={-7} y={-18} width={4} height={8} rx={1.5} />
          <rect className={dr.figBody} x={-4.5} y={-20} width={9} height={13} rx={4} />
          <line className={dr.figArm} x1={0} y1={-17} x2={-swing * 4} y2={-10.5} />
          <circle className={dr.figHead} cx={0} cy={-25} r={5.4} />
          <circle className={dr.figEye} cx={2.4} cy={-25.6} r={0.9} />
        </g>
      )}
      {thinking ? <Cue kind="think" phase={p} /> : !asleep && doing !== "walk" && doing !== "idle" ? <Cue kind={doing} phase={p} /> : null}
      {label ? (
        <text className={dr.figLabel} x={0} y={12} textAnchor="middle">
          {label}
        </text>
      ) : null}
    </g>
  );
}

/** A small bubble over the head: what the figure is doing. */
function Cue({ kind, phase }: { kind: Exclude<Doing, "walk" | "idle" | "sleep"> | "think"; phase: number }) {
  const lift = Math.sin(phase * Math.PI * 2 * 1.1);
  return (
    <g transform="translate(13 -36)">
      <circle className={dr.cueBubble} r={9.5} />
      <circle className={dr.cueBubble} cx={-8.5} cy={9.5} r={2} />
      {kind === "eat" ? (
        <g className={dr.cueInk} transform={`translate(0 ${(lift * 1.2).toFixed(2)})`}>
          <path d="M-3.5 -5.5 V-1.5 M-5.2 -5.5 V-2.5 Q-5.2 -1 -3.5 -1 Q-1.8 -1 -1.8 -2.5 V-5.5 M-3.5 -1 V5.5" />
          <path d="M3.2 5.5 V-5.5 Q5.6 -3.5 5 0.5 H3.2" />
        </g>
      ) : kind === "study" ? (
        <g className={dr.cueInk}>
          <path d="M0 -3.6 Q-3 -5.4 -6 -4.2 V3.8 Q-3 2.6 0 4.4 Q3 2.6 6 3.8 V-4.2 Q3 -5.4 0 -3.6 V4.4" />
          <path d={`M2 ${(-1.6 + lift * 0.4).toFixed(2)} H4.4 M2 0.6 H4.4`} />
        </g>
      ) : kind === "rest" ? (
        <g className={dr.cueInk}>
          <path d="M-4.5 -1 H3 V2.4 Q3 5 0.4 5 H-1.9 Q-4.5 5 -4.5 2.4 Z M3 0 H4.6 Q5.8 0 5.8 1.3 Q5.8 2.6 4.6 2.6 H3" />
          <path d={`M-2.4 ${(-3 - lift).toFixed(2)} q1 -1.2 0 -2.4 M0.6 ${(-3 + lift).toFixed(2)} q1 -1.2 0 -2.4`} />
        </g>
      ) : kind === "event" ? (
        <path className={dr.cueFill} d="M0 -5.6 L1.6 -1.9 L5.6 -1.6 L2.5 1 L3.5 5 L0 2.8 L-3.5 5 L-2.5 1 L-5.6 -1.6 L-1.6 -1.9 Z" />
      ) : kind === "think" ? (
        <g className={dr.cueFill}>
          {[-3.6, 0, 3.6].map((cx, i) => (
            <circle key={cx} cx={cx} cy={Math.sin(phase * Math.PI * 2 * 1.4 - i * 0.9) * 1.4} r={1.3} />
          ))}
        </g>
      ) : (
        <g className={dr.cueInk}>
          {/* class, or any other fixed block: a board with a line of writing */}
          <path d="M-5.6 -4.6 H5.6 V2.4 H-5.6 Z M-3.4 -1.9 H2.6 M-3.4 0 H0.6 M0 2.4 V5.4 M-2.6 5.4 H2.6" />
        </g>
      )}
    </g>
  );
}
