"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { type BenchPoint, type BenchPowerNode, type BenchSweep, type LabBench, batchReadout, fmt, shortRef, when } from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";
import { LabelKey, PointDetail, Tag, useReduced, useWidth } from "./Labels";
import { Sheet } from "./LabRun";
import b from "./bench.module.css";

/**
 * Speed and energy: the lab's recorded benchmark rows (GET /api/learning/lab/bench), drawn as they are.
 *
 * The batching story is three small charts on one batch-size axis (tokens a second per person, tokens a second in
 * total, joules per token), with the teacher's watts on a strip under them. One selector moves a cursor through all of
 * them and rewrites the sentence above from the rows at that batch size. A low-confidence point is hollow and its
 * line dashed; tapping any point shows the row it came from.
 *
 *   /lab/?tab=speed&batch=16      open at a batch size
 *   /lab/?tab=speed&ttft=1        with the time-to-first-token chart
 *   /lab/?tab=speed&point=joules_per_token   with that point's source open
 */

type Shape = "circle" | "square" | "diamond" | "triangle";
type Tone = "speed" | "energy" | "neutral";
const PAD = { left: 46, right: 20 };
const SOLID: Record<Tone, string> = { speed: b.dotSolidSpeed!, energy: b.dotSolidEnergy!, neutral: b.dotSolidNeutral! };

function mark(shape: Shape, x: number, y: number, r: number, className: string, key?: string) {
  if (shape === "square") return <rect key={key} className={className} x={x - r} y={y - r} width={2 * r} height={2 * r} rx={1.5} />;
  if (shape === "diamond") return <path key={key} className={className} d={`M${x},${y - r - 1.5} L${x + r + 1.5},${y} L${x},${y + r + 1.5} L${x - r - 1.5},${y} Z`} />;
  if (shape === "triangle") return <path key={key} className={className} d={`M${x},${y - r - 1} L${x + r + 1},${y + r} L${x - r - 1},${y + r} Z`} />;
  return <circle key={key} className={className} cx={x} cy={y} r={r} />;
}

function niceMax(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v || 1));
  for (const m of [1, 1.5, 2, 3, 4, 5, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

interface Ref {
  lo: number;
  hi: number;
  label: string;
}

/** One measure against batch size. `cursor` is the selector's position as a fraction along the batch list. */
function Panel({
  title,
  hint,
  unit,
  series,
  batches,
  cursor,
  selected,
  log,
  height = 132,
  refBand,
  edgeNote,
  picked,
  onPick,
  width,
}: {
  title: string;
  hint: string;
  unit: string;
  series: { name: string; label: string; points: BenchPoint[]; tone: Tone; shape: Shape; lineless?: boolean }[];
  batches: number[];
  cursor: number;
  selected: number;
  log?: boolean;
  height?: number;
  refBand?: Ref | null;
  edgeNote?: string | null;
  picked: string | null;
  onPick: (key: string) => void;
  width: number;
}) {
  const w = Math.max(width, 260);
  const inner = w - PAD.left - PAD.right;
  const lx = (v: number) => Math.log2(v);
  const x0 = lx(batches[0]!);
  const x1 = lx(batches[batches.length - 1]!);
  const x = (batch: number) => PAD.left + ((lx(batch) - x0) / (x1 - x0 || 1)) * inner;
  const top = 14;
  const bot = height - 22;
  const values = series.flatMap((s) => s.points.map((p) => p.value)).concat(refBand ? [refBand.lo, refBand.hi] : []);
  const vMax = Math.max(...values, 0.001);
  const vMin = Math.min(...values);
  let y: (v: number) => number;
  let ticks: number[];
  if (log) {
    const lo = Math.floor(Math.log10(vMin));
    const hi = Math.ceil(Math.log10(vMax));
    y = (v) => bot - ((Math.log10(v) - lo) / (hi - lo || 1)) * (bot - top);
    ticks = [];
    for (let e = lo; e <= hi; e++) ticks.push(10 ** e);
    if (hi - lo <= 2) for (let e = lo; e < hi; e++) ticks.push(3 * 10 ** e);
  } else {
    const max = niceMax(vMax);
    y = (v) => bot - (v / max) * (bot - top);
    ticks = [0, max / 2, max];
  }
  // the cursor sits between two batch sizes while it moves
  const i = Math.min(batches.length - 1, Math.max(0, Math.floor(cursor)));
  const j = Math.min(batches.length - 1, i + 1);
  const cx = x(batches[i]!) + (x(batches[j]!) - x(batches[i]!)) * (cursor - i);
  const now = series[0]?.points.find((p) => p.x === selected);

  return (
    <>
      <div className={b.panelTitle}>
        <span>
          <b>{title}</b> {hint}
        </span>
        {now ? (
          <span className={b.panelNow}>
            {fmt(now.value)} {unit}
          </span>
        ) : null}
      </div>
      <svg width={w} height={height} role="group" aria-label={`${title} by batch size`}>
        {ticks.map((v) => (
          <g key={v}>
            <line className={b.gridline} x1={PAD.left} x2={w - PAD.right} y1={y(v)} y2={y(v)} />
            <text className={b.axis} x={PAD.left - 8} y={y(v) + 4} textAnchor="end">
              {v >= 10 ? v.toFixed(0) : String(Number(v.toPrecision(2)))}
            </text>
          </g>
        ))}
        {batches.map((batch) => (
          <text key={batch} className={b.axis} x={x(batch)} y={height - 5} textAnchor="middle">
            {batch === batches[0] ? `batch ${batch}` : batch}
          </text>
        ))}
        <rect className={b.cursorBand} x={cx - 13} y={top - 8} width={26} height={bot - top + 12} rx={8} />
        <line className={b.cursor} x1={cx} x2={cx} y1={top - 6} y2={bot + 2} />
        {refBand ? (
          <g>
            <rect className={b.refBand} x={PAD.left} width={inner} y={y(refBand.hi)} height={Math.max(3, y(refBand.lo) - y(refBand.hi))} />
            <line className={b.refLine} x1={PAD.left} x2={w - PAD.right} y1={y(refBand.hi)} y2={y(refBand.hi)} />
            <line className={b.refLine} x1={PAD.left} x2={w - PAD.right} y1={y(refBand.lo)} y2={y(refBand.lo)} />
            <text className={b.refLabel} x={PAD.left + 6} y={y(refBand.hi) - 6}>
              {refBand.label}
            </text>
          </g>
        ) : null}
        {edgeNote ? (
          <text className={b.refLabel} x={w - PAD.right} y={top - 2} textAnchor="end">
            {edgeNote}
          </text>
        ) : null}
        {series.map((s) =>
          s.lineless
            ? null
            : s.points.slice(1).map((p, k) => {
                const a = s.points[k]!;
                return <line key={`${s.name}-${p.x}`} className={`${b.line} ${b[s.tone]} ${a.low_confidence || p.low_confidence ? b.lineLow : ""}`} x1={x(a.x!)} y1={y(a.value)} x2={x(p.x!)} y2={y(p.value)} />;
              }),
        )}
        {series.map((s) =>
          s.points.map((p) => {
            const key = `${s.name}@${p.x}`;
            const px = x(p.x!);
            const py = y(p.value);
            const on = p.x === selected;
            return (
              <g key={key}>
                {mark(s.shape, px, py, on ? 6.5 : 5, p.low_confidence ? `${b.dot} ${b[s.tone]}` : `${b.dot} ${SOLID[s.tone]}`)}
                {picked === key ? <circle className={b.picked} cx={px} cy={py} r={12} /> : null}
                {on && !s.lineless ? (
                  <text className={b.valueLabel} x={px + (px > w - 90 ? -12 : 12)} y={py + (py < top + 16 ? 18 : -10)} textAnchor={px > w - 90 ? "end" : "start"}>
                    {fmt(p.value)}
                  </text>
                ) : null}
                <circle
                  className={b.hit}
                  cx={px}
                  cy={py}
                  r={17}
                  tabIndex={0}
                  role="button"
                  aria-label={`${s.label} at batch ${p.x}: ${fmt(p.value)} ${unit}${p.low_confidence ? ", low confidence" : ""}. Show its source.`}
                  onClick={() => onPick(key)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onPick(key);
                    }
                  }}
                />
              </g>
            );
          }),
        )}
      </svg>
    </>
  );
}

function LegendMark({ shape, tone, hollow }: { shape: Shape; tone: Tone; hollow?: boolean }) {
  return (
    <svg width={30} height={14} aria-hidden="true">
      <line className={`${b.line} ${b[tone]} ${hollow ? b.lineLow : ""}`} x1={1} x2={29} y1={7} y2={7} />
      {mark(shape, 15, 7, 4.5, hollow ? `${b.dot} ${b[tone]}` : `${b.dot} ${SOLID[tone]}`)}
    </svg>
  );
}

function Batching({ sweep, bench, initialBatch, initialTtft, initialPoint, reduced }: { sweep: BenchSweep; bench: LabBench; initialBatch: number | null; initialTtft: boolean; initialPoint: string | null; reduced: boolean }) {
  const batches = sweep.batches;
  const start = Math.max(0, initialBatch !== null ? batches.indexOf(initialBatch) : batches.length > 3 ? 3 : batches.length - 1);
  const [sel, setSel] = useState(start);
  const [cursor, setCursor] = useState(start);
  const [ttft, setTtft] = useState(initialTtft);
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<string | null>(initialPoint ? `${initialPoint}@${batches[start]}` : null);
  const [ref, width] = useWidth();
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;
  const batch = batches[sel]!;

  // the cursor glides to the chosen batch size; with reduced motion it is simply there
  useEffect(() => {
    if (reduced) {
      setCursor(sel);
      return;
    }
    const from = cursorRef.current;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const u = Math.min(1, (now - t0) / 420);
      setCursor(from + (sel - from) * (1 - (1 - u) ** 3));
      if (u < 1) raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [sel, reduced]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => {
      if (sel >= batches.length - 1) setPlaying(false);
      else setSel(sel + 1);
    }, 1500);
    return () => window.clearTimeout(id);
  }, [playing, sel, batches.length]);

  const m = sweep.metrics;
  const student = bench.student;
  const jLo = student?.j_per_token?.value;
  const jHi = student?.j_per_token_request?.value;
  const band: Ref | null = jLo !== undefined ? { lo: Math.min(jLo, jHi ?? jLo), hi: Math.max(jLo, jHi ?? jLo), label: `Jetson student ${fmt(jLo)}${jHi !== undefined ? ` to ${fmt(jHi)}` : ""} J/token (derived)` } : null;
  const pick = useCallback((key: string) => setPicked((p) => (p === key ? null : key)), []);
  const pickedPoint = useMemo(() => {
    if (!picked) return null;
    const [name, at] = picked.split("@");
    return m[name ?? ""]?.find((p) => String(p.x) === at) ?? null;
  }, [picked, m]);
  const readout = batchReadout(sweep, batch);
  const shared = { batches, cursor, selected: batch, picked, onPick: pick, width };
  const allLow = sweep.low_confidence === sweep.points;

  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h2>One teacher, many people at once</h2>
        <Tag kind="measured" />
      </div>
      {sweep.other_chip ? (
        <p className={`${b.banner} ${b.bannerWatch}`}>
          <b>Other chip (R9700).</b> These rows were measured on the lab&apos;s earlier cards, not on the Strix Halo pods it runs now.
        </p>
      ) : null}
      <p className={b.note}>
        {sweep.model ?? "The teacher"} on {sweep.node}
        {sweep.chip ? ` (${sweep.chip})` : ""}, measured {when(sweep.ts)}. Batch size is how many requests it answers at the same time.
      </p>

      <div className={b.stepper} role="group" aria-label="Batch size">
        <button type="button" className={b.stepBtn} disabled={sel === 0} aria-label="Smaller batch" onClick={() => (setPlaying(false), setSel(sel - 1))}>
          ‹
        </button>
        {batches.map((v, k) => (
          <button key={v} type="button" className={`${b.stepBtn} ${k === sel ? b.stepOn : ""}`} aria-pressed={k === sel} onClick={() => (setPlaying(false), setSel(k))}>
            {v}
          </button>
        ))}
        <button type="button" className={b.stepBtn} disabled={sel === batches.length - 1} aria-label="Larger batch" onClick={() => (setPlaying(false), setSel(sel + 1))}>
          ›
        </button>
        {reduced ? null : (
          <button type="button" className={b.stepBtn} onClick={() => (playing ? setPlaying(false) : (sel >= batches.length - 1 && setSel(0), setPlaying(true)))}>
            {playing ? "Pause" : "Play 1 to " + batches[batches.length - 1]}
          </button>
        )}
      </div>

      <input
        className={b.slider}
        type="range"
        min={0}
        max={batches.length - 1}
        step={1}
        value={sel}
        aria-label="Batch size"
        aria-valuetext={`batch ${batch}`}
        onChange={(e) => (setPlaying(false), setSel(Number(e.target.value)))}
      />

      {readout ? (
        <p className={b.readout} role="status">
          {readout}
          <span className={b.readoutSub}>
            Computed from {["decode_tok_s", "joules_per_token"].map((n) => shortRef(m[n]?.find((p) => p.x === batch)?.source.ref)).join(" and ")}.
          </span>
        </p>
      ) : null}

      <div ref={ref} className={b.chart}>
        {width > 0 ? (
          <>
            {m.decode_tok_s ? (
              <Panel
                {...shared}
                title="Speed for each person"
                hint="tokens a second per request: falls as more share the card"
                unit="tok/s"
                series={[{ name: "decode_tok_s", label: "Tokens a second per request", points: m.decode_tok_s, tone: "speed", shape: "circle" }]}
                edgeNote={student?.tok_s ? `Jetson student: ${fmt(student.tok_s.value)} tok/s, above this scale` : null}
              />
            ) : null}
            {m.decode_tok_s_total ? (
              <Panel
                {...shared}
                title="Speed in total"
                hint="tokens a second across all requests: rises"
                unit="tok/s"
                series={[{ name: "decode_tok_s_total", label: "Tokens a second in total", points: m.decode_tok_s_total, tone: "speed", shape: "square" }]}
              />
            ) : null}
            {m.joules_per_token ? (
              <Panel
                {...shared}
                title="Energy for each token"
                hint="joules per token, log scale: falls steeply"
                unit="J/token"
                log
                height={170}
                refBand={band}
                series={[{ name: "joules_per_token", label: "Joules per token", points: m.joules_per_token, tone: "energy", shape: "diamond" }]}
              />
            ) : null}
            {ttft && m.ttft_s ? (
              <Panel
                {...shared}
                title="Wait before the first word"
                hint="seconds to the first token: the price of sharing"
                unit="s"
                series={[{ name: "ttft_s", label: "Time to first token", points: m.ttft_s, tone: "neutral", shape: "triangle" }]}
              />
            ) : null}
            {m.watts_avg ? (
              <Panel
                {...shared}
                title="Power drawn"
                hint="watts, average (line) and peak (squares): flat, so more answers per watt"
                unit="W"
                height={96}
                series={[
                  { name: "watts_avg", label: "Average watts", points: m.watts_avg, tone: "energy", shape: "circle" },
                  ...(m.watts_peak ? [{ name: "watts_peak", label: "Peak watts", points: m.watts_peak, tone: "neutral" as Tone, shape: "square" as Shape, lineless: true }] : []),
                ]}
              />
            ) : null}
          </>
        ) : null}
      </div>

      <div className={b.legend}>
        <span>
          <LegendMark shape="circle" tone="speed" hollow /> hollow and dashed: low confidence
        </span>
        <span>
          <LegendMark shape="circle" tone="speed" /> solid: confident
        </span>
        <span>
          <svg width={30} height={14} aria-hidden="true">
            <line className={b.refLine} x1={1} x2={29} y1={7} y2={7} />
          </svg>
          Jetson student, for reference
        </span>
      </div>
      <p className={`${b.banner} ${allLow ? b.bannerWatch : ""}`}>
        <span>
          <b>{allLow ? "Read these as a first look, not a result." : `${sweep.low_confidence} of ${sweep.points} points are low confidence.`}</b> {sweep.confidence_note} Tap a point to see its row.
        </span>
      </p>
      {pickedPoint ? <PointDetail p={pickedPoint} onClose={() => setPicked(null)} /> : null}
      {m.ttft_s ? (
        <button type="button" className={`${ui.chip} ${ttft ? ui.chipOn : ""}`} aria-pressed={ttft} onClick={() => setTtft((v) => !v)} style={{ alignSelf: "flex-start" }}>
          {ttft ? "Hide" : "Show"} time to first token
        </button>
      ) : null}
    </div>
  );
}

function StudentCard({ bench }: { bench: LabBench }) {
  const s = bench.student;
  const [open, setOpen] = useState<string | null>(null);
  if (!s)
    return (
      <div className={b.card}>
        <h2>The box&apos;s own model</h2>
        <p className={b.note}>The metrics file has no row for the student on the Jetson yet.</p>
      </div>
    );
  const cells: [string, string, BenchPoint | null][] = [
    ["watts", "power while answering", s.watts],
    ["tok", "decode speed", s.tok_s],
    ["jpt", "per token while decoding", s.j_per_token],
    ["jptr", "per token, whole request", s.j_per_token_request],
  ];
  const shown = cells.find(([k]) => k === open)?.[2] ?? null;
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h2>The box&apos;s own model, for comparison</h2>
      </div>
      <p className={b.note}>The small student model on the Jetson answers one request at a time. Its joules per token is not measured directly: it is worked out from its watts and its speed.</p>
      <div className={b.stats}>
        {cells.map(([key, label, p]) =>
          p ? (
            <button key={key} type="button" className={`${b.stat} ${open === key ? b.statOn : ""}`} aria-expanded={open === key} onClick={() => setOpen(open === key ? null : key)}>
              <span className={b.statNum}>
                {fmt(p.value)} {p.unit}
              </span>
              <span className={b.statLabel}>{label}</span>
              <span>
                <Tag kind={p.kind} />
                {p.low_confidence ? <span className={b.low}> low confidence</span> : null}
              </span>
              {p.kind === "derived" && p.formula ? <span className={b.note}>= {p.formula}</span> : null}
            </button>
          ) : null,
        )}
      </div>
      {shown ? <PointDetail p={shown} onClose={() => setOpen(null)} /> : <p className={b.note}>Tap a number to see the row it came from.</p>}
    </div>
  );
}

function PowerCard({ bench }: { bench: LabBench }) {
  const [open, setOpen] = useState<string | null>(null);
  const nodes = bench.power.nodes;
  const max = Math.max(1, ...nodes.flatMap((n) => [n.idle?.value ?? 0, n.serving?.value ?? 0]));
  const find = (key: string): BenchPoint | null => {
    const [node, which] = key.split("|");
    const n = nodes.find((x) => x.node === node);
    return (n?.[which as "idle" | "serving"] as BenchPoint | null | undefined) ?? null;
  };
  const row = (n: BenchPowerNode, which: "idle" | "serving") => {
    const p = n[which];
    const key = `${n.node}|${which}`;
    return (
      <div key={key} style={{ display: "contents" }}>
        <span className={b.note}>{which === "idle" ? "On, idle" : "Answering"}</span>
        {p ? (
          <button type="button" className={b.barTrack} aria-expanded={open === key} aria-label={`${n.node} ${which}: ${fmt(p.value)} watts${p.low_confidence ? ", low confidence" : ""}. Show its source.`} onClick={() => setOpen(open === key ? null : key)}>
            <span className={`${b.barFill} ${which === "idle" ? b.barIdle : ""} ${p.low_confidence ? b.barLow : ""}`} style={{ width: `${Math.max(1, (p.value / max) * 72)}%` }} />
            <span className={b.barValue}>{fmt(p.value)} W</span>
            {p.low_confidence ? <span className={b.low}>low confidence</span> : null}
          </button>
        ) : (
          <span className={b.note}>no row yet</span>
        )}
      </div>
    );
  };
  if (nodes.length === 0) return null;
  const shown = open ? find(open) : null;
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h2>What it costs to be on</h2>
        <Tag kind="measured" />
      </div>
      <p className={b.note}>The newest idle and answering power row for each machine. A striped bar is a low-confidence row.</p>
      <div className={b.bars}>
        {nodes.map((n) => (
          <div key={n.node} style={{ display: "contents" }}>
            <span className={b.barNode}>
              {n.node === "jetson" ? "Jetson (the box)" : n.node}
              {n.serving?.state && n.node !== "jetson" ? <span className={b.note}> · answering at {n.serving.state}</span> : null}
            </span>
            {row(n, "idle")}
            {row(n, "serving")}
          </div>
        ))}
      </div>
      {bench.power.other_chip.length ? (
        <p className={b.note}>
          Not drawn: {bench.power.other_chip.length} power row{bench.power.other_chip.length === 1 ? "" : "s"} from the other chip (R9700) or untagged from before the lab moved (
          <span className={b.code}>metrics.jsonl:{bench.power.other_chip.map((p) => p.line).join(", ")}</span>). They are in &ldquo;All rows&rdquo;.
        </p>
      ) : null}
      {bench.power.absent.length ? <p className={b.note}>No power row yet for {bench.power.absent.join(", ")}. Nothing is drawn for a machine the file does not cover.</p> : null}
      {shown ? <PointDetail p={shown} onClose={() => setOpen(null)} /> : null}
    </div>
  );
}

function What({ bench, sweep }: { bench: LabBench; sweep: BenchSweep | undefined }) {
  const ref = (name: string) => {
    const pts = sweep?.metrics[name] ?? [];
    return pts.length ? `${shortRef(pts[0]!.source.ref)} to ${pts[pts.length - 1]!.line}` : "no row";
  };
  const lines: [string, string][] = [
    ["Batch size is how many requests the big teacher model answers at the same moment.", "conditions.batch_size of each row"],
    ["Each person gets fewer tokens a second as the batch grows; together they get more.", `${ref("decode_tok_s")}; ${ref("decode_tok_s_total")}`],
    ["The card draws about the same power either way, so each token costs far less energy.", `${ref("joules_per_token")}; ${ref("watts_avg")}`],
    ["The green band is the small model on the box: derived from its watts and its speed, not measured.", bench.student?.j_per_token?.source.ref.replace(/lab\/refine\//g, "") ?? "no row"],
    ["Hollow points and dashed lines are low confidence. Tap any point for its row and the reason.", `${bench.low_confidence_rows} of ${bench.rows} rows in ${shortRef(bench.file)}`],
  ];
  return (
    <ol className={b.howto}>
      {lines.map(([text, from]) => (
        <li key={text}>
          {text}
          <span className={b.note} style={{ display: "block" }}>
            From: <span className={b.code}>{from}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

function AllRows({ bench }: { bench: LabBench }) {
  return (
    <>
      <p className={ui.muted}>
        {bench.rows} rows in {bench.file}; {bench.charted_rows} are drawn on this screen or on the Training tab. The rest are listed here so none is dropped silently. {bench.other_chip_rows} rows
        are from the other chip (R9700) or untagged from before the lab moved to the Strix Halo pods; they are never drawn on the current lab&apos;s charts.
      </p>
      <h3>Present but not charted</h3>
      <div className={b.list}>
        {bench.not_charted.map((x) => (
          <div key={`${x.metric}|${x.node}`} className={b.listRow}>
            <div className={b.listTop}>
              <span>
                {x.metric} <span className={b.note}>on {x.node}</span>
              </span>
              <b>
                {fmt(x.latest.value)} {x.unit}
              </b>
            </div>
            <span className={b.note}>
              {x.rows} row{x.rows === 1 ? "" : "s"}: <span className={b.code}>metrics.jsonl:{x.lines.join(", ")}</span>
              {x.latest.low_confidence ? " · newest is low confidence" : ""}
              {x.other_chip ? ` · ${x.other_chip === x.rows ? "all" : x.other_chip} from the other chip (R9700), not the current lab` : ""}
            </span>
          </div>
        ))}
      </div>
      <h3>Metric families</h3>
      <div className={b.list}>
        {bench.families.map((f) => (
          <div key={f.family} className={b.listRow}>
            <div className={b.listTop}>
              <span>{f.family}</span>
              <b>
                {f.charted} of {f.rows} drawn
              </b>
            </div>
            <span className={b.note}>
              {f.metrics.length} metric{f.metrics.length === 1 ? "" : "s"} on {f.nodes.join(", ")}; {f.low_confidence} low confidence
            </span>
          </div>
        ))}
      </div>
      {bench.skipped.length ? <p className={b.note}>Lines that are not rows: {bench.skipped.map((s) => `${s.line} (${s.problem})`).join(", ")}.</p> : null}
    </>
  );
}

export function SpeedEnergy({ initialBatch, initialTtft, initialPoint, initialDrawer, forceReduced }: { initialBatch: number | null; initialTtft: boolean; initialPoint: string | null; initialDrawer: string | null; forceReduced: boolean }) {
  const bench = useFetch((signal) => api.labBench(signal), "bench");
  const reduced = useReduced(forceReduced);
  const [sweepId, setSweepId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<"what" | "rows" | null>(initialDrawer === "what" || initialDrawer === "rows" ? initialDrawer : null);
  const close = useCallback(() => setDrawer(null), []);
  if (bench.loading && !bench.data) return <Loading what="the lab's measurements" />;
  if (bench.error && !bench.data) return <ErrorBox what="the lab's measurements" error={bench.error} onRetry={bench.reload} />;
  const data = bench.data!;
  const sweep = data.sweeps.find((s) => s.id === sweepId) ?? data.sweeps[0];
  return (
    <div className={b.page}>
      <div className={ui.pageHead}>
        <h1>Speed and energy</h1>
        <p className={b.sub}>What the lab measured on its own hardware: how fast the big teacher model answers, and what each token costs in energy.</p>
        <div className={b.cardHead} style={{ justifyContent: "flex-start", marginTop: 8 }}>
          <button type="button" className={`${ui.btn}`} onClick={() => setDrawer("what")}>
            What am I looking at?
          </button>
          <button type="button" className={`${ui.btn}`} onClick={() => setDrawer("rows")}>
            All {data.rows} rows
          </button>
        </div>
      </div>
      <LabelKey kinds={["measured", "derived"]} />
      {data.rows === 0 ? <p className={b.banner}>{data.empty_reason}</p> : null}
      {data.sweeps.length > 1 ? (
        <div className={ui.chips} role="group" aria-label="Which measurement run">
          {data.sweeps.map((s) => (
            <button key={s.id} type="button" className={`${ui.chip} ${s.id === sweep?.id ? ui.chipOn : ""}`} aria-pressed={s.id === sweep?.id} onClick={() => setSweepId(s.id)}>
              {s.family.replace(/_/g, " ")} · {s.node} · {when(s.ts)}
              {s.other_chip ? " · other chip (R9700)" : ""}
            </button>
          ))}
        </div>
      ) : null}
      <div className={b.two}>
        {sweep ? (
          <Batching key={sweep.id} sweep={sweep} bench={data} initialBatch={initialBatch} initialTtft={initialTtft} initialPoint={initialPoint} reduced={reduced} />
        ) : (
          <div className={b.card}>
            <h2>No batching measurement yet</h2>
            <p className={b.note}>The metrics file has no metric measured at two or more batch sizes. The chart appears as soon as the lab records one.</p>
          </div>
        )}
        <div className={b.col}>
          <StudentCard bench={data} />
          <PowerCard bench={data} />
        </div>
      </div>
      {drawer === "what" ? (
        <Sheet title="What am I looking at?" onClose={close}>
          <What bench={data} sweep={sweep} />
        </Sheet>
      ) : null}
      {drawer === "rows" ? (
        <Sheet title="Every row" onClose={close}>
          <AllRows bench={data} />
        </Sheet>
      ) : null}
    </div>
  );
}
