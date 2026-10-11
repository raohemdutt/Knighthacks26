"use client";

import { useEffect, useId, useRef, useState } from "react";
import { type LabStory, isNum, pct } from "@/lib/lab";
import { lossWords } from "@/lib/labWords";
import lab from "./lab.module.css";
import story from "./story.module.css";

/** The rendered width of an element, so the SVGs draw at real pixels and their labels stay readable on a phone. */
function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function niceCeil(x: number, step: number): number {
  return Math.max(step, Math.ceil(x / step) * step);
}

/** The next round number at or above `v`: 0.127 -> 0.2, 4.75 -> 5, 2.08 -> 2.5. */
function niceTop(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v - 1e-12) return Number((m * p).toPrecision(3));
  return 10 * p;
}

const PAD = { left: 42, right: 14 };

/**
 * Loss over training time, drawn up to the playhead, and the card's power under it on the same x-axis.
 * The filled area under the power line is the energy used so far. `frac` is 0..1 through training.
 */
export function TrainingChart({ training, power, frac }: { training: LabStory["training"]; power: LabStory["power"]; frac: number }) {
  const [ref, width] = useWidth();
  const clip = useId();
  const pts = training.points.filter((p) => p.loss !== null) as { t: number; step: number | null; loss: number }[];
  const tMax = Math.max(isNum(training.wall_s) ? training.wall_s.value : 0, pts.length ? pts[pts.length - 1]!.t : 0, 1);
  const tNow = frac * tMax;
  const w = Math.max(width, 240);
  const inner = w - PAD.left - PAD.right;
  const x = (t: number) => PAD.left + (t / tMax) * inner;

  const lossH = 190;
  const lTop = 10;
  const lBot = lossH - 22;
  const yMax = niceTop(Math.max(...pts.map((p) => p.loss), 0.01)); // fitted to the run: a small loss is not drawn flat along the floor
  const y = (v: number) => lBot - (v / yMax) * (lBot - lTop);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");
  const head = [...pts].reverse().find((p) => p.t <= tNow + 1e-6);
  // a run of under two minutes is counted in seconds: "0 min" alone under a whole chart says nothing
  const inSeconds = tMax < 120;
  const minuteStep = tMax > 1500 ? 600 : tMax > 600 ? 300 : tMax > 240 ? 120 : inSeconds ? (tMax > 60 ? 30 : tMax > 20 ? 10 : 5) : 60;
  const ticks: number[] = [];
  for (let t = 0; t <= tMax + 1; t += minuteStep) ticks.push(t);
  const tick = (t: number) => (inSeconds ? (t === 0 ? "0 s" : `${t} s`) : t === 0 ? "0 min" : `${t / 60} min`);
  const firstLoss = pts[0]?.loss;
  const lastLoss = pts[pts.length - 1]?.loss;
  const lossTitle =
    firstLoss === undefined || lastLoss === undefined
      ? ""
      : lastLoss < firstLoss
        ? `Loss fell from ${lossWords(firstLoss)} to ${lossWords(lastLoss)} over ${pts.length} steps.`
        : `Loss did not fall: ${lossWords(firstLoss)} at the first step, ${lossWords(lastLoss)} at the last of ${pts.length}.`;

  const pw = power.points;
  const powerH = 120;
  const pTop = 8;
  const pBot = powerH - 22;
  const wMax = niceCeil(Math.max(...pw.map((p) => p.w), 1), 100);
  const py = (v: number) => pBot - (v / wMax) * (pBot - pTop);
  // each reading holds for its step (dt), or runs straight to the next sample
  let line = "";
  pw.forEach((p, i) => {
    if (p.dt !== undefined) line += `${i ? "L" : "M"}${x(p.t - p.dt).toFixed(1)},${py(p.w).toFixed(1)} L${x(p.t).toFixed(1)},${py(p.w).toFixed(1)} `;
    else line += `${i ? "L" : "M"}${x(p.t).toFixed(1)},${py(p.w).toFixed(1)} `;
  });
  const first = pw[0];
  const last = pw[pw.length - 1];
  const area = first && last ? `${line}L${x(last.t).toFixed(1)},${pBot} L${x(first.dt !== undefined ? first.t - first.dt : first.t).toFixed(1)},${pBot} Z` : "";

  return (
    <div ref={ref} className={lab.chart}>
      {pts.length === 0 ? (
        <p className={lab.gap}>
          <b>Nothing about how training went has reached the box for this run.</b> The curve is drawn step by step as the steps arrive.
        </p>
      ) : width > 0 ? (
        <>
          <div className={lab.chartTitle}>
            <span>
              <b>{lossTitle}</b> Loss is how wrong the model is on its training examples: lower is better.
            </span>
          </div>
          <svg width={w} height={lossH} role="img" aria-label={`${lossTitle} Time runs left to right, in minutes of training.`}>
            <defs>
              <clipPath id={clip}>
                <rect x={0} y={0} width={x(tNow) + 1} height={lossH + powerH} />
              </clipPath>
            </defs>
            {[0, yMax / 2, yMax].map((v) => (
              <g key={v}>
                <line className={lab.gridline} x1={PAD.left} x2={w - PAD.right} y1={y(v)} y2={y(v)} />
                <text className={lab.axis} x={PAD.left - 8} y={y(v) + 4} textAnchor="end">
                  {String(Number(v.toPrecision(2)))}
                </text>
              </g>
            ))}
            {ticks.map((t) => (
              <text key={t} className={lab.axis} x={x(t)} y={lossH - 4} textAnchor={t === 0 ? "start" : x(t) > w - 30 ? "end" : "middle"}>
                {tick(t)}
              </text>
            ))}
            <path className={lab.lossGhost} d={path} />
            <path className={lab.lossLine} d={path} clipPath={`url(#${clip})`} />
            <line className={lab.playhead} x1={x(tNow)} x2={x(tNow)} y1={lTop} y2={lBot} />
            {head ? (
              <>
                <circle className={lab.headDot} cx={x(head.t)} cy={y(head.loss)} r={6} />
                <text className={lab.headLabel} x={x(head.t) + (x(head.t) > w - 110 ? -10 : 10)} y={y(head.loss) - 10} textAnchor={x(head.t) > w - 110 ? "end" : "start"}>
                  loss {lossWords(head.loss)}
                </text>
              </>
            ) : null}
          </svg>
        </>
      ) : null}

      {pw.length === 0 ? (
        <p className={lab.gap}>
          <b>No power was measured during this run.</b> So its energy is not known, and none is shown.
        </p>
      ) : width > 0 ? (
        <>
          <div className={lab.chartTitle}>
            <span>
              <b>Power while it trained{power.node ? ` (${power.node})` : ""}, in watts.</b> The filled area is the energy used.
            </span>
          </div>
          <svg width={w} height={powerH} role="img" aria-label={`Power while training, in watts, from ${Math.round(Math.min(...pw.map((p) => p.w)))} to ${Math.round(Math.max(...pw.map((p) => p.w)))} W`}>
            <defs>
              <clipPath id={`${clip}p`}>
                <rect x={0} y={0} width={x(tNow) + 1} height={powerH} />
              </clipPath>
            </defs>
            {[0, wMax / 2, wMax].map((v) => (
              <g key={v}>
                <line className={lab.gridline} x1={PAD.left} x2={w - PAD.right} y1={py(v)} y2={py(v)} />
                <text className={lab.axis} x={PAD.left - 8} y={py(v) + 4} textAnchor="end">
                  {v === wMax ? `${v} W` : v}
                </text>
              </g>
            ))}
            {ticks.map((t) => (
              <text key={t} className={lab.axis} x={x(t)} y={powerH - 4} textAnchor={t === 0 ? "start" : x(t) > w - 30 ? "end" : "middle"}>
                {tick(t)}
              </text>
            ))}
            <g clipPath={`url(#${clip}p)`}>
              <path className={lab.powerArea} d={area} />
              <path className={lab.powerLine} d={line} />
            </g>
            <line className={lab.playhead} x1={x(tNow)} x2={x(tNow)} y1={pTop} y2={pBot} />
          </svg>
        </>
      ) : null}
    </div>
  );
}

/**
 * The gate, drawn: the score before and after on one 0 to 100% line with the run-to-run noise as a band around
 * "before", the teacher as a dashed tick, and under it one dot per question group at how much it changed.
 */
export function GateChart({ ev }: { ev: LabStory["eval"] }) {
  const [ref, width] = useWidth();
  const { current, candidate, teacher } = ev.scores;
  if (!isNum(current) || !isNum(candidate)) return null;
  const w = Math.max(width, 240);
  const left = 14;
  const right = 14;
  const x = (v: number) => left + v * (w - left - right);
  const yAxis = 76;
  const up = candidate.value >= current.value;
  const noise = isNum(ev.noise) ? ev.noise.value : null;
  // Two labels that would run into each other sit on two lines, "before" above "after"; a label near an edge is
  // anchored to that edge, so neither is ever cut off or printed over the other (100% and 100% used to collide).
  const close = Math.abs(x(candidate.value) - x(current.value)) < 110;
  const anchor = (v: number): "start" | "middle" | "end" => (x(v) < 60 ? "start" : x(v) > w - 60 ? "end" : "middle");
  const teacherNear = isNum(teacher) && Math.abs(x(teacher.value) - x(current.value)) < 130 && close;
  const groups = ev.groups ?? [];
  const many = groups.length > 16 || Math.max(0, ...countStacks(groups.map((g) => g.diff.toFixed(3)))) > 8; // too many to draw as dots
  const span = Math.max(0.25, Math.ceil(Math.max(...groups.map((g) => Math.abs(g.diff)), 0) * 4) / 4);
  const gx = (d: number) => left + ((d + span) / (2 * span)) * (w - left - right);
  const stacks = new Map<string, number>();
  const better = groups.filter((g) => g.diff > 0).length;
  const worse = groups.filter((g) => g.diff < 0).length;
  const gHeight = 40 + Math.max(1, ...countStacks(groups.map((g) => g.diff.toFixed(3)))) * 16 + 20;

  return (
    <div ref={ref} className={lab.chart}>
      {width > 0 ? (
        <>
          <div className={lab.chartTitle}>
            <span>
              <b>
                {candidate.value > current.value
                  ? `The score rose from ${pct(current.value)} to ${pct(candidate.value)} of the held-back questions.`
                  : candidate.value < current.value
                    ? `The score fell from ${pct(current.value)} to ${pct(candidate.value)} of the held-back questions.`
                    : `The score did not move: ${pct(current.value)} of the held-back questions right, before and after.`}
              </b>
              {isNum(teacher) ? ` The teacher, asked the same questions, got ${pct(teacher.value)}.` : ""}
            </span>
          </div>
          <svg width={w} height={116} role="img" aria-label={`Share of held-back questions right: before ${pct(current.value)}, after ${pct(candidate.value)}${isNum(teacher) ? `, teacher ${pct(teacher.value)}` : ""}`}>
            <line className={lab.scoreAxis} x1={x(0)} x2={x(1)} y1={yAxis} y2={yAxis} />
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line className={lab.gridline} x1={x(v)} x2={x(v)} y1={yAxis - 5} y2={yAxis + 5} />
                <text className={lab.axis} x={x(v)} y={112} textAnchor={v === 0 ? "start" : v === 1 ? "end" : "middle"}>
                  {v === 1 ? "100% right" : pct(v)}
                </text>
              </g>
            ))}
            {noise !== null ? <rect className={lab.noiseBand} x={x(Math.max(0, current.value - noise))} width={Math.max(3, x(Math.min(1, current.value + noise)) - x(Math.max(0, current.value - noise)))} y={yAxis - 13} height={26} rx={3} /> : null}
            <line className={`${lab.gainLine} ${up ? "" : lab.lossGain}`} x1={x(current.value)} x2={x(candidate.value)} y1={yAxis} y2={yAxis} />
            {isNum(teacher) ? (
              <>
                <line className={lab.teacherTick} x1={x(teacher.value)} x2={x(teacher.value)} y1={teacherNear ? yAxis - 12 : 18} y2={yAxis + 14} />
                {/* beside the two score labels there is no room for a third: the tick stays, and the title above says the teacher's score */}
                {teacherNear ? null : (
                  <text className={lab.axis} x={x(teacher.value)} y={12} textAnchor={anchor(teacher.value)}>
                    teacher {pct(teacher.value)}
                  </text>
                )}
              </>
            ) : null}
            <circle className={lab.dotBefore} cx={x(current.value)} cy={yAxis} r={7} />
            <circle className={`${lab.dotAfter} ${up ? "" : lab.dotAfterBad}`} cx={x(candidate.value)} cy={yAxis} r={8} />
            <text className={lab.gLabel} x={x(current.value)} y={yAxis - (close ? 36 : 18)} textAnchor={anchor(current.value)}>
              before {pct(current.value)}
            </text>
            <text className={`${lab.gLabel} ${up ? lab.gUp : lab.gDown}`} x={x(candidate.value)} y={yAxis - 18} textAnchor={anchor(candidate.value)}>
              after {pct(candidate.value)}
            </text>
          </svg>
          <p className={lab.note}>
            {noise !== null
              ? noise > 0
                ? `The grey band around "before" is the wobble between two runs of the same model: about ${pct(noise, 1)}. A gain has to be larger than that to count.`
                : "Scoring the same model twice gave the same score, so there is no wobble to draw around “before”."
              : "The wobble between two runs of the same model was not measured for this run, so no band is drawn."}
          </p>
          {groups.length && many ? (
            <>
              <div className={lab.chartTitle}>
                <span>
                  <b>
                    Of {groups.length} groups of questions, {better} got better, {groups.length - better - worse} did not move and {worse} got worse.
                  </b>{" "}
                  A group is one case, asked in one or more wordings.
                </span>
              </div>
              <div className={story.tri} role="img" aria-label={`${better} of ${groups.length} question groups got better, ${groups.length - better - worse} did not move, ${worse} got worse`}>
                {worse ? <span className={story.triWorse} style={{ flexGrow: worse }}>{worse} worse</span> : null}
                {groups.length - better - worse ? <span className={story.triSame} style={{ flexGrow: groups.length - better - worse }}>{groups.length - better - worse} same</span> : null}
                {better ? <span className={story.triBetter} style={{ flexGrow: better }}>{better} better</span> : null}
              </div>
              <p className={lab.note}>The test asks how often a pattern this lopsided would appear if the change were only chance.</p>
            </>
          ) : groups.length ? (
            <>
              <div className={lab.chartTitle}>
                <span>
                  <b>
                    Of {groups.length} groups of questions, {better} got better, {groups.length - better - worse} did not move and {worse} got worse.
                  </b>{" "}
                  Each dot is one group (one case, in several wordings), placed by how much its score moved.
                </span>
              </div>
              <svg width={w} height={gHeight} role="img" aria-label={`${better} of ${groups.length} question groups got better, ${worse} got worse`}>
                <line className={lab.gridline} x1={gx(0)} x2={gx(0)} y1={4} y2={gHeight - 18} />
                <line className={lab.scoreAxis} x1={gx(-span)} x2={gx(span)} y1={gHeight - 18} y2={gHeight - 18} />
                <text className={lab.axis} x={gx(-span)} y={gHeight - 3}>
                  worse −{pct(span)}
                </text>
                <text className={lab.axis} x={gx(0)} y={gHeight - 3} textAnchor="middle">
                  no change
                </text>
                <text className={lab.axis} x={gx(span)} y={gHeight - 3} textAnchor="end">
                  better +{pct(span)}
                </text>
                {groups.map((g) => {
                  const key = g.diff.toFixed(3);
                  const k = stacks.get(key) ?? 0;
                  stacks.set(key, k + 1);
                  return (
                    <circle key={g.group} className={g.diff > 0 ? lab.gUp : g.diff < 0 ? lab.gDown : lab.gFlat} cx={gx(g.diff)} cy={gHeight - 30 - k * 16} r={6.5}>
                      <title>{`${g.group}: ${g.before} of ${g.n} right before, ${g.after} after`}</title>
                    </circle>
                  );
                })}
              </svg>
              <p className={lab.note}>The test asks how often a pattern this lopsided would appear if the change were only chance.</p>
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function countStacks(keys: string[]): number[] {
  const m = new Map<string, number>();
  for (const k of keys) m.set(k, (m.get(k) ?? 0) + 1);
  return [...m.values()];
}
