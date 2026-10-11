"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { clamp01 } from "@/lib/lab";
import { type BenchPoint, type LabSim, type PolicyName, type SimKind, type SimParams, type SimRun, fmt, joules, shortRef } from "@/lib/labBench";
import { POLICY_WORDS, metricWords, routingTakeaway, said } from "@/lib/labWords";
import { LabelKey, LockIcon, PointDetail, Tag, useReduced, useWidth } from "./Labels";
import { Sheet } from "./LabRun";
import { ChartWords, Fold, SourceChip } from "./Story";
import b from "./bench.module.css";

/** The two rules by the names the screen uses; the code's own names ("capability", "energy") are in their meaning. */
const POLICY_TITLE: Record<PolicyName, string> = { capability: POLICY_WORDS.capability.title, energy: POLICY_WORDS.energy.title };

/**
 * Routing what-if: for a stream of requests, where each one is answered (the box or the lab's teacher) under each
 * routing policy, and what that costs. A simulation: POST /api/learning/lab/routing/simulate runs the real routing
 * hook and its two policies on numbers the lab recorded, and this screen says so wherever a modelled number appears.
 *
 *   /lab/?tab=routing&preset=quiet|study|burst    start from a preset
 *   /lab/?tab=routing&t=0.5                       the position in the replay (0 to 1); without it the replay plays once
 *   /lab/?tab=routing&policy=capability&basis=decode
 */

/** The four kinds of request by a short plain name; the code's own word ("chat", "eval") is in the fold of its reasons. */
const KIND_WORDS: Record<string, string> = { chat: "Questions", caption: "Camera descriptions", label: "Labelling examples", eval: "Practice scoring" };
const SHAPES = ["circle", "square", "triangle", "diamond"] as const;
const PLAY_SECONDS = 18;
const FLIGHT = 0.07; // how long a dot is on its way, as a share of the replay

function shape(i: number, x: number, y: number, r: number, className: string, key?: string | number) {
  const s = SHAPES[i % SHAPES.length];
  if (s === "square") return <rect key={key} className={className} x={x - r} y={y - r} width={2 * r} height={2 * r} rx={1} />;
  if (s === "diamond") return <path key={key} className={className} d={`M${x},${y - r - 1.5} L${x + r + 1.5},${y} L${x},${y + r + 1.5} L${x - r - 1.5},${y} Z`} />;
  if (s === "triangle") return <path key={key} className={className} d={`M${x},${y - r - 1} L${x + r + 1},${y + r} L${x - r - 1},${y + r} Z`} />;
  return <circle key={key} className={className} cx={x} cy={y} r={r} />;
}

function KindMark({ i, className }: { i: number; className: string }) {
  return (
    <svg className={b.kindMark} width={16} height={16} aria-hidden="true">
      {shape(i, 8, 8, 5, className)}
    </svg>
  );
}

/** One policy's side: requests leave "you" and arrive at the box or at the lab, and the energy meter fills. */
function Lane({ sim, run, name, u, on, scale }: { sim: LabSim; run: SimRun; name: PolicyName; u: number; on: boolean; scale: number }) {
  const [ref, width] = useWidth();
  const req = sim.requests!;
  const kinds = sim.kinds!;
  const w = Math.max(width, 250);
  const h = 176;
  const from = { x: 62, y: h / 2 };
  const box = { x: w - 122, y: 42 };
  const labNode = { x: w - 122, y: h - 42 };
  const curve = (to: { x: number; y: number }) => `M${from.x},${from.y} C${from.x + (to.x - from.x) * 0.5},${from.y} ${from.x + (to.x - from.x) * 0.5},${to.y} ${to.x},${to.y}`;
  const along = (to: { x: number; y: number }, p: number) => {
    const c1 = { x: from.x + (to.x - from.x) * 0.5, y: from.y };
    const c2 = { x: from.x + (to.x - from.x) * 0.5, y: to.y };
    const q = 1 - p;
    return { x: q ** 3 * from.x + 3 * q * q * p * c1.x + 3 * q * p * p * c2.x + p ** 3 * to.x, y: q ** 3 * from.y + 3 * q * q * p * c1.y + 3 * q * p * p * c2.y + p ** 3 * to.y };
  };
  // u runs past 1 by FLIGHT so the last dots land; a request counts once it has arrived
  const now = u * (1 + FLIGHT) * req.window_s;
  const flight = FLIGHT * req.window_s;
  let atBox = 0;
  let atLab = 0;
  let jBox = 0;
  let jLab = 0;
  const dots: React.ReactNode[] = [];
  for (let i = 0; i < req.n; i++) {
    const t = req.t[i]!;
    if (t > now) break;
    const k = req.kind[i]!;
    const target = run.targets[i]!;
    const row = run.by_kind[kinds[k]!.kind]!;
    const p = (now - t) / flight;
    if (p >= 1 || u >= 1) {
      if (target === "t") {
        atLab++;
        jLab += row.joules;
      } else {
        atBox++;
        jBox += row.joules;
      }
    } else {
      const pos = along(target === "t" ? labNode : box, p);
      dots.push(shape(k, pos.x, pos.y, 4.5, target === "p" ? b.reqLocked! : target === "t" ? b.reqLab! : b.reqOpen!, i));
    }
  }
  const total = jBox + jLab;
  return (
    <div className={`${b.lane} ${on ? b.laneOn : ""}`}>
      <div className={b.laneHead}>
        {POLICY_TITLE[name]}
        {on ? <span>selected</span> : null}
      </div>
      <div ref={ref} className={b.chart}>
        {width > 0 ? (
          <svg width={w} height={h} role="img" aria-label={`${POLICY_TITLE[name]}: ${run.to_student} requests answered on the box, ${run.to_teacher} sent to the lab's teacher`}>
            <path className={b.path} d={curve(box)} />
            <path className={`${b.path} ${b.pathLab}`} d={curve(labNode)} />
            <rect className={b.nodeBox} x={4} y={from.y - 20} width={58} height={40} rx={12} />
            <text className={b.nodeText} x={33} y={from.y + 5} textAnchor="middle">
              You
            </text>
            <rect className={b.nodeBox} x={box.x} y={box.y - 26} width={118} height={52} rx={12} />
            <text className={b.nodeText} x={box.x + 10} y={box.y - 5}>
              The box (Jetson)
            </text>
            <text className={b.nodeSub} x={box.x + 10} y={box.y + 13}>
              {atBox} answered
            </text>
            <rect className={b.nodeBox} x={labNode.x} y={labNode.y - 26} width={118} height={52} rx={12} />
            <text className={b.nodeText} x={labNode.x + 10} y={labNode.y - 5}>
              The lab (teacher)
            </text>
            <text className={b.nodeSub} x={labNode.x + 10} y={labNode.y + 13}>
              {atLab} answered
            </text>
            {dots}
          </svg>
        ) : null}
      </div>
      <div className={b.meter}>
        <div className={b.meterTrack} role="img" aria-label={`Energy so far: ${joules(total)}, of which ${joules(jLab)} at the lab`}>
          <div className={b.meterBox} style={{ width: `${(jBox / scale) * 100}%` }} />
          <div className={b.meterLab} style={{ width: `${(jLab / scale) * 100}%` }} />
        </div>
        <span className={b.meterNum}>{joules(total)}</span>
        <span className={b.note}>
          <span className={`${b.swatch} ${b.meterBox}`} /> box {joules(jBox)} · <span className={`${b.swatch} ${b.meterLab}`} /> lab {joules(jLab)}
        </span>
        <Tag kind="modelled" />
      </div>
      <p className={b.note}>
        {run.teacher_batch !== null ? `The teacher answers about ${fmt(run.teacher_batch, run.teacher_batch >= 10 ? 0 : 1)} at once here (${fmt(run.teacher_j_per_token ?? 0)} J per token).` : "Nothing is sent to the teacher."}
        {run.saturated ? ` That is the most at once the lab has measured (${Math.max(...(sim.inputs?.teacher.batches ?? [0]))}): more would have to wait in a queue, which this what-if does not count.` : ""}
      </p>
    </div>
  );
}

/** Joules per token against the teacher's batch size, the student as a flat line, and where the two cross. */
function BreakEven({ sim, policy }: { sim: LabSim; policy: PolicyName }) {
  const [ref, width] = useWidth();
  const be = sim.break_even!;
  const w = Math.max(width, 260);
  const h = 190;
  const pad = { left: 44, right: 16, top: 18, bot: 26 };
  const xs = be.curve.map((p) => p.batch);
  const ys = be.curve.map((p) => p.teacher_j_per_token).concat([be.student_j_per_token]);
  const x0 = Math.log2(Math.min(...xs));
  const x1 = Math.log2(Math.max(...xs));
  const y0 = Math.floor(Math.log10(Math.min(...ys)));
  const y1 = Math.ceil(Math.log10(Math.max(...ys)));
  const x = (v: number) => pad.left + ((Math.log2(v) - x0) / (x1 - x0 || 1)) * (w - pad.left - pad.right);
  const y = (v: number) => h - pad.bot - ((Math.log10(v) - y0) / (y1 - y0 || 1)) * (h - pad.top - pad.bot);
  const d = be.curve.map((p, i) => `${i ? "L" : "M"}${x(p.batch).toFixed(1)},${y(p.teacher_j_per_token).toFixed(1)}`).join(" ");
  const ticks: number[] = [];
  for (let e = y0; e <= y1; e++) ticks.push(10 ** e);
  const nowBatch = be.now[policy];
  const nowY = nowBatch !== null && nowBatch !== undefined ? (be.curve.reduce((a, p) => (Math.abs(p.batch - nowBatch) < Math.abs(a.batch - nowBatch) ? p : a), be.curve[0]!).teacher_j_per_token) : null;
  return (
    <div ref={ref} className={b.chart}>
      {width > 0 ? (
        <svg width={w} height={h} role="img" aria-label={be.batch !== null ? `Energy for one token against how many requests the teacher answers at once: the teacher costs less than the box above ${fmt(be.batch, 0)} at once` : (be.reason ?? "")}>
          <text className={b.axis} x={4} y={10}>
            J/token
          </text>
          {ticks.map((v) => (
            <g key={v}>
              <line className={b.gridline} x1={pad.left} x2={w - pad.right} y1={y(v)} y2={y(v)} />
              <text className={b.axis} x={pad.left - 8} y={y(v) + 4} textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          {be.curve
            .filter((p) => p.recorded)
            .map((p) => (
              <text key={p.batch} className={b.axis} x={x(p.batch)} y={h - 6} textAnchor={p.batch === xs[0] ? "start" : p.batch === xs[xs.length - 1] ? "end" : "middle"} dx={p.batch === xs[0] ? -6 : p.batch === xs[xs.length - 1] ? 8 : 0}>
                {p.batch === xs[0] || p.batch === xs[xs.length - 1] ? `${p.batch} at once` : p.batch}
              </text>
            ))}
          <line className={b.refLine} x1={pad.left} x2={w - pad.right} y1={y(be.student_j_per_token)} y2={y(be.student_j_per_token)} />
          <text className={b.refLabel} x={pad.left + 6} y={y(be.student_j_per_token) + 16}>
            the box: {fmt(be.student_j_per_token)} J/token
          </text>
          <path className={`${b.line} ${b.energy} ${b.lineLow}`} d={d} />
          {be.curve
            .filter((p) => p.recorded)
            .map((p) => (
              <path key={p.batch} className={`${b.dot} ${b.energy}`} d={`M${x(p.batch)},${y(p.teacher_j_per_token) - 6.5} l6.5,6.5 l-6.5,6.5 l-6.5,-6.5 Z`} />
            ))}
          <text className={b.valueLabel} x={x(xs[0]!) + 12} y={y(be.curve[0]!.teacher_j_per_token) + 4}>
            the teacher
          </text>
          {be.batch !== null ? (
            <>
              <line className={b.breakEven} x1={x(be.batch)} x2={x(be.batch)} y1={pad.top - 4} y2={h - pad.bot} />
              <text className={b.valueLabel} x={x(be.batch) - 8} y={pad.top + 8} textAnchor="end">
                equal at {fmt(be.batch, 0)} at once
              </text>
            </>
          ) : null}
          {nowBatch !== null && nowBatch !== undefined && nowY !== null ? (
            <>
              <circle className={b.nowMark} cx={x(nowBatch)} cy={y(nowY)} r={8} />
              <text className={b.valueLabel} x={x(nowBatch) + (x(nowBatch) > w - 80 ? -14 : 14)} y={y(nowY) - 12} textAnchor={x(nowBatch) > w - 80 ? "end" : "start"}>
                now
              </text>
            </>
          ) : null}
        </svg>
      ) : null}
    </div>
  );
}

function Where({ target, locked }: { target: "student" | "teacher"; locked: boolean }) {
  return (
    <span className={b.where}>
      {locked ? <LockIcon /> : null}
      {target === "teacher" ? "Lab" : "Box"}
    </span>
  );
}

function Assumptions({ sim }: { sim: LabSim }) {
  const inp = sim.inputs;
  const nums: [string, BenchPoint | null | undefined][] = inp
    ? [
        ["Student energy per token used", inp.student_j_per_token],
        ["Student power while answering", inp.student.watts],
        ["Student decode speed", inp.student.tok_s],
        ["Student time before the first token", inp.student.first_token_s],
        ...inp.teacher.j_per_token.map((p): [string, BenchPoint] => [`Teacher joules per token, batch ${p.x}`, p]),
        ...inp.teacher.tok_s.map((p): [string, BenchPoint] => [`Teacher tokens a second, batch ${p.x}`, p]),
        ...inp.teacher.ttft_s.map((p): [string, BenchPoint] => [`Teacher time to first token, batch ${p.x}`, p]),
      ]
    : [];
  return (
    <>
      <p className={ui.muted}>{sim.model_statement}. Every decision is made by the code the box runs, on these numbers.</p>
      <h3>What the model assumes</h3>
      <ol className={b.howto} style={{ fontSize: 15 }}>
        {sim.assumptions.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ol>
      <h3>The code it runs</h3>
      <p className={b.code}>{sim.calls.join(" · ")}</p>
      <h3>The recorded numbers it stands on</h3>
      <div className={b.list}>
        {nums.map(([label, p]) =>
          p ? (
            <div key={label} className={b.listRow}>
              <div className={b.listTop}>
                <span>
                  {label} <Tag kind={p.kind} />
                </span>
                <b>
                  {fmt(p.value)} {p.unit}
                </b>
              </div>
              <span className={`${b.note} ${b.code}`}>{p.kind === "derived" ? `${p.formula}; ${(p.inputs ?? []).map(shortRef).join(", ")}` : shortRef(p.source.ref)}</span>
              {p.low_confidence ? <span className={b.note}>Low confidence: {p.reason}</span> : null}
            </div>
          ) : null,
        )}
      </div>
      {inp ? <p className={b.note}>{inp.teacher.confidence_note}</p> : null}
    </>
  );
}

export function Routing({ initialPreset, initialT, initialPolicy, initialBasis, initialDrawer, forceReduced }: { initialPreset: string | null; initialT: number | null; initialPolicy: string | null; initialBasis: string | null; initialDrawer: string | null; forceReduced: boolean }) {
  const reduced = useReduced(forceReduced);
  const [params, setParams] = useState<Partial<SimParams>>(() => ({
    ...(initialPreset ? { preset: initialPreset } : {}),
    ...(initialPolicy === "capability" || initialPolicy === "energy" ? { policy: initialPolicy } : {}),
    ...(initialBasis === "decode" || initialBasis === "request" ? { student_basis: initialBasis } : {}),
  }));
  const [sim, setSim] = useState<LabSim | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [u, setU] = useState(initialT ?? 1);
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(initialT !== null);
  const [drawer, setDrawer] = useState(initialDrawer === "model");
  const [advanced, setAdvanced] = useState(false);
  const [num, setNum] = useState<BenchPoint | null>(null);
  const uRef = useRef(u);
  uRef.current = u;

  const key = JSON.stringify(params);
  useEffect(() => {
    const ctl = new AbortController();
    const id = window.setTimeout(() => {
      api
        .labRoutingSim(JSON.parse(key) as Partial<SimParams>, ctl.signal)
        .then((data) => (setSim(data), setError(null)))
        .catch((err: unknown) => {
          if (!ctl.signal.aborted) setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
        });
    }, 120);
    return () => (window.clearTimeout(id), ctl.abort());
  }, [key]);

  // the replay plays once when the tab opens, unless a position was asked for or motion is reduced
  useEffect(() => {
    if (sim?.available && !started) {
      setStarted(true);
      if (!reduced) {
        setU(0);
        setPlaying(true);
      }
    }
  }, [sim, started, reduced]);

  useEffect(() => {
    if (!playing || reduced) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const next = clamp01(uRef.current + (now - last) / 1000 / PLAY_SECONDS);
      last = now;
      setU(next);
      if (next >= 1) setPlaying(false);
      else raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [playing, reduced]);

  const closeDrawer = useCallback(() => setDrawer(false), []);
  /** Change one input: the full set of inputs is sent from then on, so the preset no longer applies. */
  const edit = useCallback(
    (change: Partial<SimParams>) => {
      if (!sim) return;
      const p = sim.params;
      setParams({ mix: p.mix, rpm: p.rpm, minutes: p.minutes, tokens: p.tokens, policy: p.policy, student_basis: p.student_basis, seed: p.seed, ...change });
    },
    [sim],
  );
  const scale = useMemo(() => (sim?.policies ? Math.max(sim.policies.capability.energy_j, sim.policies.energy.energy_j, 1) : 1), [sim]);

  if (error && !sim) return <ErrorBox what="the routing what-if" error={error} onRetry={() => setParams({ ...params })} />;
  if (!sim) return <Loading what="the routing what-if" />;
  const p = sim.params;
  const kinds: SimKind[] = sim.kinds ?? [];
  const cap = sim.policies?.capability;
  const eng = sim.policies?.energy;
  const mixTotal = Object.values(p.mix).reduce((a, v) => a + v, 0) || 1;
  const toggle = () => {
    if (reduced) {
      setU((v) => (v >= 1 ? 0.25 : Math.min(1, Math.round((v + 0.25) * 4) / 4)));
      return;
    }
    if (!playing && uRef.current >= 1) setU(0);
    setPlaying((x) => !x);
  };
  const take = routingTakeaway(sim);
  const seconds = sim.requests ? Math.min(1, u) * sim.requests.window_s : 0;
  const clockText = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className={b.page}>
      {/* opened from the routing card on the Speed and energy tab: the card above already says the result in one line */}
      <p className={b.note}>
        Routing is the rule that decides, for each request, whether the box&apos;s own small model answers or the request goes to the lab&apos;s big teacher. Change what is being
        asked and how often, and see where each request goes under two rules and what that costs.
      </p>
      {take ? <SourceChip sources={take.sources} about="The routing result" /> : null}
      <div className={b.cardHead} style={{ justifyContent: "flex-start" }}>
        <button type="button" className={ui.btn} onClick={() => setDrawer(true)}>
          How this is modelled
        </button>
      </div>
      <div className={b.key} role="note" aria-label="The two rules" data-guide="lab:rules">
        {(["capability", "energy"] as PolicyName[]).map((name) => (
          <span key={name}>
            <b>{POLICY_WORDS[name].title}.</b> {POLICY_WORDS[name].means}
          </span>
        ))}
      </div>

      {sim.measured ? (
        <div className={b.card} data-guide="lab:routing-measured">
          <div className={b.cardHead}>
            <h2>Measured once for real: the two rules side by side</h2>
            <Tag kind="measured" />
          </div>
          <div className={b.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>What was measured</th>
                  <th className={ui.num}>{POLICY_TITLE.capability}</th>
                  <th className={ui.num}>{POLICY_TITLE.energy}</th>
                  <th className={ui.num}>Change</th>
                </tr>
              </thead>
              <tbody>
                {sim.measured.rows.map((r) => {
                  const value = (v: number) => (r.unit === "fraction" ? `${said(v * 100)}%` : `${fmt(v)} ${r.unit ?? ""}`);
                  return (
                    <tr key={`${r.metric}|${r.node}`}>
                      <td>
                        {metricWords(r.metric)}
                        {r.low_confidence ? <span className={b.low}> low confidence</span> : null}
                      </td>
                      <td className={ui.num}>
                        <button type="button" className={b.numBtn} onClick={() => setNum(r.capability)}>
                          {value(r.capability.value)}
                        </button>
                      </td>
                      <td className={ui.num}>
                        <button type="button" className={b.numBtn} onClick={() => setNum(r.energy)}>
                          {value(r.energy.value)}
                        </button>
                      </td>
                      <td className={ui.num}>{r.change_pct !== null ? `${r.change_pct > 0 ? "+" : ""}${r.change_pct}%` : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className={b.note}>
            Each side ran over at least {sim.measured.min_requests} requests on {[...new Set(sim.measured.rows.map((r) => r.node))].join(", ")}. Tap a number for the row it came from. Everything below
            this card is a what-if around this measurement.
          </p>
          {num ? <PointDetail p={num} onClose={() => setNum(null)} /> : null}
        </div>
      ) : null}

      <p className={b.banner} role="note">
        <Tag kind="modelled">{sim.label === "what-if" ? "What-if" : "Simulation"}</Tag>
        <span>
          <b>{sim.model_statement}.</b> {sim.measured ? "" : `${sim.measured_reason![0]!.toUpperCase()}${sim.measured_reason!.slice(1)}.`}
        </span>
      </p>
      <LabelKey kinds={["measured", "derived", "modelled"]} />

      {!sim.available ? (
        <p className={`${b.banner} ${b.bannerWatch}`}>
          <b>Nothing to model from yet.</b> {sim.reason}
        </p>
      ) : (
        <div className={`${b.two} ${b.twoRouting}`}>
          <div className={b.col}>
            <div className={b.card}>
              <h2>What is being asked</h2>
              <div className={b.controls}>
                <div className={ui.segmented} role="group" aria-label="Presets">
                  {sim.presets.map((x) => (
                    <button key={x.id} type="button" className={`${ui.seg} ${p.preset === x.id ? ui.segOn : ""}`} aria-pressed={p.preset === x.id} onClick={() => setParams({ preset: x.id, policy: p.policy, student_basis: p.student_basis, tokens: p.tokens })}>
                      {x.title}
                    </button>
                  ))}
                </div>
                <label className={b.field}>
                  <span>Requests a minute</span>
                  <span className={b.fieldValue}>{p.rpm}</span>
                  <input type="range" min={1} max={120} step={1} value={p.rpm} onChange={(e) => edit({ rpm: Number(e.target.value) })} />
                </label>
                {kinds.map((k, i) => (
                  <label key={k.kind} className={`${b.field} ${k.locked ? b.fieldLocked : ""}`}>
                    <span>
                      <KindMark i={i} className={k.locked ? b.reqLocked! : b.reqOpen!} /> {k.title.replace(/for the gate$/, "for the test")}
                      {k.locked ? (
                        <span className={b.lock}>
                          {" "}
                          <LockIcon /> never leaves the box
                        </span>
                      ) : k.student_can ? null : (
                        <span className={b.note}> · the small model cannot do this</span>
                      )}
                    </span>
                    <span className={b.fieldValue}>{Math.round(((p.mix[k.kind] ?? 0) / mixTotal) * 100)}%</span>
                    <input type="range" min={0} max={100} step={5} value={p.mix[k.kind] ?? 0} aria-label={`Share of requests: ${k.title}`} onChange={(e) => edit({ mix: { ...p.mix, [k.kind]: Number(e.target.value) } })} />
                  </label>
                ))}
                <div>
                  <span className={b.stepLabel}>Rule to highlight</span>
                  <div className={ui.segmented} role="group" aria-label="Routing policy">
                    {(["capability", "energy"] as PolicyName[]).map((name) => (
                      <button key={name} type="button" className={`${ui.seg} ${p.policy === name ? ui.segOn : ""}`} aria-pressed={p.policy === name} onClick={() => edit({ policy: name })}>
                        {POLICY_TITLE[name]}
                      </button>
                    ))}
                  </div>
                  <p className={b.note}>{POLICY_WORDS[p.policy].means}</p>
                </div>
                <button type="button" className={`${ui.chip} ${advanced ? ui.chipOn : ""}`} aria-expanded={advanced} onClick={() => setAdvanced((v) => !v)} style={{ alignSelf: "flex-start" }}>
                  Assumptions you can change
                </button>
                {advanced ? (
                  <>
                    <div className={b.advanced}>
                      {kinds.map((k) => (
                        <label key={k.kind}>
                          <span>
                            Answer length in tokens, {(KIND_WORDS[k.kind] ?? k.kind).toLowerCase()} <Tag kind="modelled">assumed</Tag>
                          </span>
                          <input className={b.tokenInput} type="number" min={4} max={1024} step={8} value={p.tokens[k.kind] ?? k.tokens} aria-label={`Tokens per ${k.kind} answer`} onChange={(e) => edit({ tokens: { ...p.tokens, [k.kind]: Math.max(4, Math.min(1024, Math.round(Number(e.target.value) || 4))) } })} />
                        </label>
                      ))}
                    </div>
                    <div>
                      <span className={b.stepLabel}>The box&apos;s energy per token counts</span>
                      <div className={ui.segmented} role="group" aria-label="How the box's energy per token is counted">
                        {(["request", "decode"] as const).map((basis) => {
                          const v = sim.inputs?.student_bases[basis];
                          return (
                            <button key={basis} type="button" disabled={!v} className={`${ui.seg} ${p.student_basis === basis ? ui.segOn : ""}`} aria-pressed={p.student_basis === basis} onClick={() => edit({ student_basis: basis })}>
                              {basis === "request" ? "The whole request" : "Decoding only"}
                              {v ? ` (${fmt(v.value)} J)` : ""}
                            </button>
                          );
                        })}
                      </div>
                      <p className={b.note}>The teacher&apos;s figure covers the whole request, so that is the like-for-like choice. Both are derived, not measured.</p>
                    </div>
                  </>
                ) : null}
              </div>
            </div>
            <div className={b.lockRule}>
              <LockIcon />
              <span>
                <b>Locked rule: the privacy line.</b> {sim.privacy.rule}
                <span className={b.note} style={{ display: "block", marginTop: 4 }}>
                  Checked against the real hook with the teacher priced at almost nothing: {sim.privacy.locked.map((x) => x.kind).join(" and ")} still stay on the box.
                </span>
              </span>
            </div>
          </div>

          <div className={`${b.col} ${b.flowCol}`}>
            <div className={b.card}>
              <div className={b.cardHead}>
                <h2>Where each request goes</h2>
                <Tag kind="modelled" />
              </div>
              <div className={b.player}>
                <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={toggle}>
                  {reduced ? "Next step" : playing ? "Pause" : u >= 1 ? "Replay" : "Play"}
                </button>
                <input type="range" min={0} max={1000} value={Math.round(u * 1000)} aria-label="Position in the replay" onChange={(e) => (setPlaying(false), setU(Number(e.target.value) / 1000))} />
                <span className={b.fieldValue}>
                  {clockText(seconds)} of {clockText(sim.requests!.window_s)}
                </span>
              </div>
              <div className={b.lanes}>
                {cap ? <Lane sim={sim} run={cap} name="capability" u={u} on={p.policy === "capability"} scale={scale} /> : null}
                {eng ? <Lane sim={sim} run={eng} name="energy" u={u} on={p.policy === "energy"} scale={scale} /> : null}
              </div>
              <div className={b.legend}>
                {kinds.map((k, i) => (
                  <span key={k.kind}>
                    <KindMark i={i} className={k.locked ? b.reqLocked! : b.reqOpen!} />
                    {(KIND_WORDS[k.kind] ?? k.kind).toLowerCase()}
                    {k.locked ? " (stays on the box)" : ""}
                  </span>
                ))}
                <span>
                  <KindMark i={3} className={b.reqLab!} /> on its way to the lab
                </span>
              </div>
              {reduced ? <p className={b.note}>Reduced motion is on: nothing moves. Next step advances a quarter at a time; the counts and meters show the totals so far.</p> : null}
            </div>

            <ChartWords guide="lab:routing-words">
              <b>The picture in words.</b> Over {sim.params.minutes} minutes, {sim.requests!.n} requests. {POLICY_TITLE.capability}: {cap!.to_student} answered on the box, {cap!.to_teacher} by the
              teacher, {joules(cap!.energy_j)}. {POLICY_TITLE.energy}: {eng!.to_student} on the box, {eng!.to_teacher} by the teacher, {joules(eng!.energy_j)}. {eng!.locked} are about your own
              space and never leave the box under either rule.
              {eng!.cold_start_note ? ` ${eng!.cold_start_note}` : ""}
            </ChartWords>
          </div>

          <div className={b.card}>
            <div className={b.cardHead}>
              <h2>When does the teacher become the cheaper one?</h2>
              <Tag kind="measured">teacher: measured</Tag>
            </div>
            <BreakEven sim={sim} policy={p.policy} />
            <p className={b.note}>
              {sim.break_even!.batch !== null
                ? `Per token, the teacher beats the box once it answers about ${fmt(sim.break_even!.batch, 0)} requests at once, which takes about ${fmt(sim.break_even!.rpm_to_teacher ?? 0, 0)} requests a minute sent its way. Below that, sending work to the lab costs more energy.`
                : sim.break_even!.reason}{" "}
              Energy for one token (up the side) against how many requests the teacher answers at once (along the bottom); both scales are logarithmic. The box&apos;s line is worked out, not
              measured, and between measured points the teacher&apos;s curve is filled in.
            </p>
          </div>

          <div className={b.card}>
            <h2>By kind of request</h2>
            <div className={b.tableWrap}>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th>Kind of request</th>
                    <th className={ui.num}>How many</th>
                    <th>{POLICY_TITLE.capability}</th>
                    <th>{POLICY_TITLE.energy}</th>
                  </tr>
                </thead>
                <tbody>
                  {kinds.map((k, i) => {
                    const c = cap!.by_kind[k.kind]!;
                    const e = eng!.by_kind[k.kind]!;
                    return (
                      <tr key={k.kind}>
                        <td>
                          <KindMark i={i} className={k.locked ? b.reqLocked! : b.reqOpen!} /> {KIND_WORDS[k.kind] ?? k.kind}
                        </td>
                        <td className={ui.num}>{k.n}</td>
                        <td>
                          <Where target={c.target} locked={k.locked} />
                          <span className={b.note} style={{ display: "block", whiteSpace: "nowrap" }}>
                            {fmt(c.joules, 0)} J, {fmt(c.latency_s, 0)} s
                          </span>
                        </td>
                        <td>
                          <Where target={e.target} locked={k.locked} />
                          <span className={b.note} style={{ display: "block", whiteSpace: "nowrap" }}>
                            {fmt(e.joules, 0)} J, {fmt(e.latency_s, 0)} s
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className={b.note}>
              Where one request of each kind is answered, with the energy and the time for that one answer <Tag kind="modelled" />. A lock means it never leaves the box.
            </p>
            <Fold summary="The routing code's own reasons">
              <span className={b.note}>What the box&apos;s routing code said for each kind under {POLICY_TITLE[p.policy].toLowerCase()}, word for word:</span>
              {kinds.map((k) => (
                <span key={k.kind} className={b.code}>
                  {k.kind}: {sim.policies![p.policy].by_kind[k.kind]!.reason}
                </span>
              ))}
            </Fold>
          </div>
        </div>
      )}

      {drawer ? (
        <Sheet title="How this is modelled" onClose={closeDrawer}>
          <Assumptions sim={sim} />
        </Sheet>
      ) : null}
    </div>
  );
}
