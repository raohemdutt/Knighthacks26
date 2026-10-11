"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { type LabChip, chipLabel, chipRole, chipShort } from "@/lib/lab";
import { type BenchChipRows, type BenchPoint, type BenchPowerNode, type BenchSweep, type LabBench, batchReadout, fmt, shortRef, when } from "@/lib/labBench";
import { batchTakeaway, engineName, headlineSweep, plainConfidence, routingLine, said, speedHero, sweepName } from "@/lib/labWords";
import { useFetch } from "@/lib/useFetch";
import { LabelKey, PointDetail, Tag, useReduced, useWidth } from "./Labels";
import { Sheet } from "./LabRun";
import { Routing } from "./Routing";
import { ChartWords, Fold, Hero, SourceChip, Takeaway, Term, Words } from "./Story";
import b from "./bench.module.css";

/**
 * Speed and energy: the lab's recorded benchmark rows (GET /api/learning/lab/bench), drawn as they are.
 *
 * The batching story is three small charts on one batch-size axis (tokens a second per person, tokens a second in
 * total, joules per token), with the teacher's watts on a strip under them. One selector moves a cursor through all of
 * them and rewrites the sentence above from the rows at that batch size. A low-confidence point is hollow and its
 * line dashed; tapping any point shows the row it came from.
 *
 * Every row carries the chip it was measured on (lib/lab.ts LabChip). The main chart is the current lab's sweep. A
 * sweep from any other chip is a section of its own under that chip's name ("Reference: NVIDIA A100 (Colab)",
 * "Earlier lab cards: Radeon R9700") and is never joined to the lab's curve. Where another chip recorded the same
 * metrics at the same batch sizes, "Compare with" lays its rows beside the lab's as a grey series with its own marker.
 *
 *   /lab/?tab=speed&batch=16      open at a batch size
 *   /lab/?tab=speed&ttft=1        with the time-to-first-token chart
 *   /lab/?tab=speed&point=joules_per_token   with that point's source open
 *   /lab/?tab=speed&compare=1     with the first comparison on (or compare=<chip key>, such as nvidia-a100)
 *   /lab/?tab=speed&sweep=<chip key>         open another chip's own section
 */

type Shape = "circle" | "square" | "diamond" | "triangle";
type Tone = "speed" | "energy" | "neutral";
const PAD = { left: 46, right: 20 };
const SOLID: Record<Tone, string> = { speed: b.dotSolidSpeed!, energy: b.dotSolidEnergy!, neutral: b.dotSolidNeutral! };
const CMP = "cmp:"; // a comparison series' name: the other chip's rows of the same metric

/** "metrics.jsonl:7, 8, 17" as it is; a long list as "metrics.jsonl:1 to 22", or "between metrics.jsonl:102 and 129" when lines between them are not in it. */
function lineList(lines: number[]): string {
  if (lines.length <= 6) return `metrics.jsonl:${lines.join(", ")}`;
  const first = lines[0]!;
  const last = lines[lines.length - 1]!;
  return last - first + 1 === lines.length ? `metrics.jsonl:${first} to ${last}` : `between metrics.jsonl:${first} and ${last}`;
}

/** "3 from Radeon R9700 (the lab's earlier cards, …); 11 from NVIDIA A100 (Colab) (a reference, …)". */
function chipCounts(groups: BenchChipRows[], withLines = false): string {
  return groups.map((g) => `${g.rows} from ${chipShort(g)} (${chipRole(g)}${withLines ? `; ${lineList(g.lines)}` : ""})`).join("; ");
}

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
  height = 144,
  refBand,
  edgeNote,
  picked,
  onPick,
  width,
}: {
  title: string;
  hint: string;
  unit: string;
  series: { name: string; label: string; short?: string; points: BenchPoint[]; tone: Tone; shape: Shape; lineless?: boolean }[];
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
  const top = 26; // room above the plot for the unit, so the axis says what it counts
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
  const beside = series.find((s) => s.name.startsWith(CMP));
  const besideNow = beside?.points.find((p) => p.x === selected);

  return (
    <>
      <div className={b.panelTitle}>
        <span>
          <b>{title}</b> {hint}
        </span>
        {now ? (
          <span className={b.panelNow}>
            {fmt(now.value)} {unit} at {selected === 1 ? "1 request" : `${selected} at once`}
            {beside && besideNow ? (
              <span className={b.note}>
                {" "}
                · {beside.short}: {fmt(besideNow.value)}
              </span>
            ) : null}
          </span>
        ) : null}
      </div>
      <svg width={w} height={height} role="group" aria-label={`${title} ${hint}`}>
        <text className={b.axis} x={4} y={10}>
          {unit}
        </text>
        {ticks.map((v) => (
          <g key={v}>
            <line className={b.gridline} x1={PAD.left} x2={w - PAD.right} y1={y(v)} y2={y(v)} />
            <text className={b.axis} x={PAD.left - 8} y={y(v) + 4} textAnchor="end">
              {v >= 10 ? v.toFixed(0) : String(Number(v.toPrecision(2)))}
            </text>
          </g>
        ))}
        {batches.map((batch) => (
          <text key={batch} className={b.axis} x={x(batch)} y={height - 5} textAnchor={batch === batches[0] ? "start" : batch === batches[batches.length - 1] ? "end" : "middle"} dx={batch === batches[0] ? -6 : batch === batches[batches.length - 1] ? 8 : 0}>
            {batch === batches[0] ? `${batch} at once` : batch === batches[batches.length - 1] ? `${batch} at once` : batch}
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

/**
 * A chart's title, as the finding it shows: "Each token gets cheaper: 8.8 J down to 1.9 J". From the curve's own first
 * and last point; a change of under 8% either way is "barely moves".
 */
function finding(points: BenchPoint[] | undefined, words: { down: string; up: string; flat: string }, unit: string): string {
  const pts = points ?? [];
  const a = pts[0];
  const z = pts[pts.length - 1];
  if (!a || !z) return words.flat;
  const u = unit === "tokens a second" ? "" : ` ${unit}`;
  const tail = unit === "tokens a second" ? " tokens a second" : "";
  const change = (z.value - a.value) / (a.value || 1);
  if (Math.abs(change) < 0.08) return `${words.flat}: about ${said((a.value + z.value) / 2)}${u}${tail}.`;
  return change < 0 ? `${words.down}: ${said(a.value)}${u} down to ${said(z.value)}${u}${tail}.` : `${words.up}: ${said(a.value)}${u} up to ${said(z.value)}${u}${tail}.`;
}

function LegendMark({ shape, tone, hollow }: { shape: Shape; tone: Tone; hollow?: boolean }) {
  return (
    <svg width={30} height={14} aria-hidden="true">
      <line className={`${b.line} ${b[tone]} ${hollow ? b.lineLow : ""}`} x1={1} x2={29} y1={7} y2={7} />
      {mark(shape, 15, 7, 4.5, hollow ? `${b.dot} ${b[tone]}` : `${b.dot} ${SOLID[tone]}`)}
    </svg>
  );
}

function Batching({
  sweep,
  bench,
  initialBatch,
  initialTtft,
  initialPoint,
  initialCompare,
  reduced,
}: {
  sweep: BenchSweep;
  bench: LabBench;
  initialBatch: number | null;
  initialTtft: boolean;
  initialPoint: string | null;
  initialCompare: string | null;
  reduced: boolean;
}) {
  const batches = sweep.batches;
  // other chips' sweeps that can be read beside this one; only the current lab's sweep offers any
  const offers = useMemo(() => sweep.comparisons.flatMap((c) => bench.sweeps.filter((s) => s.id === c.id).map((other) => ({ ...c, other }))), [sweep, bench]);
  const [compareId, setCompareId] = useState<string | null>(() => (initialCompare ? (sweep.comparisons.find((c) => c.chip.key === initialCompare) ?? (initialCompare === "1" ? sweep.comparisons[0] : undefined))?.id ?? null : null));
  const cmp = offers.find((c) => c.id === compareId) ?? null;
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
  const band: Ref | null = jLo !== undefined ? { lo: Math.min(jLo, jHi ?? jLo), hi: Math.max(jLo, jHi ?? jLo), label: `the box's model: ${fmt(jLo)}${jHi !== undefined ? ` to ${fmt(jHi)}` : ""} J` } : null;
  const pick = useCallback((key: string) => setPicked((p) => (p === key ? null : key)), []);
  const pickedPoint = useMemo(() => {
    if (!picked) return null;
    const [name = "", at] = picked.split("@");
    const from = name.startsWith(CMP) ? cmp?.other.metrics[name.slice(CMP.length)] : m[name];
    return from?.find((p) => String(p.x) === at) ?? null;
  }, [picked, m, cmp]);
  const readout = batchReadout(sweep, batch);
  const shared = { batches, cursor, selected: batch, picked, onPick: pick, width };
  const allLow = sweep.low_confidence === sweep.points;
  // the other chip's rows of one metric, at the batch sizes both ran, as a grey series with a marker of its own
  const cmpShort = cmp ? chipShort(cmp.chip) : "";
  const beside = (name: string, label: string) => {
    const pts = cmp && cmp.metrics.includes(name) ? (cmp.other.metrics[name] ?? []).filter((p) => p.x !== null && p.x !== undefined && cmp.batches.includes(p.x)) : [];
    return pts.length ? [{ name: CMP + name, label: `${cmpShort}: ${label}`, short: cmpShort, points: pts, tone: "neutral" as Tone, shape: "triangle" as Shape }] : [];
  };
  const mineAt = (name: string) => m[name]?.find((p) => p.x === batch);
  const theirsAt = (name: string) => (cmp && cmp.metrics.includes(name) ? cmp.other.metrics[name]?.find((p) => p.x === batch) : undefined);
  const wMine = mineAt("watts_avg");
  const wTheirs = theirsAt("watts_avg");
  const scale = cmp ? ", log scale while comparing" : "";

  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h2>One teacher, many people at once</h2>
        <Tag kind="measured" />
      </div>
      {sweep.other_chip ? (
        <p className={`${b.banner} ${b.bannerWatch}`}>
          <span>
            <b>{chipLabel(sweep.chip)}.</b>{" "}
            {sweep.chip.role === "earlier"
              ? "These rows were measured on the lab's earlier cards, not on the Strix Halo pods it runs now."
              : sweep.chip.role === "reference"
                ? `These rows were measured on a different chip${sweep.chip.where ? `, on ${sweep.chip.where}` : ""}, not on the lab's Strix Halo pods.${sweep.chip_note ? ` The rows say: ${sweep.chip_note}.` : ""}`
                : "These rows mark themselves as measured on another chip and do not name it. They are not the lab's Strix Halo pods."}
            {sweep.chip.reason ? <span className={b.note}> How we know: {sweep.chip.reason}.</span> : null}
          </span>
        </p>
      ) : null}
      <p className={b.note}>
        <Term k="teacher">The teacher</Term>
        {engineName(sweep) ? ` (running on ${engineName(sweep)})` : ""} on {sweep.node}, measured {when(sweep.ts)}. Pick how many requests it answers <Term k="batch">at once</Term>:
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
            {playing ? "Pause" : `Play ${batches[0]} to ${batches[batches.length - 1]}`}
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
        <p className={b.readout} role="status" data-guide="lab:speed-readout">
          {readout}
          {cmp && (theirsAt("decode_tok_s") || theirsAt("joules_per_token")) ? (
            <span className={b.readoutSub}>
              For comparison, {cmpShort} with {batch === 1 ? "1 request" : `${batch} at once`}:{" "}
              {[
                [theirsAt("decode_tok_s"), "tokens a second for each request"],
                [theirsAt("joules_per_token"), "J for each token"],
              ]
                .filter((x): x is [BenchPoint, string] => !!x[0])
                .map(([p, words]) => `${fmt(p.value)} ${words}`)
                .join(", ")}
              . A different chip, not the lab&apos;s.
            </span>
          ) : null}
        </p>
      ) : null}
      {readout ? (
        <SourceChip
          about="The numbers in this sentence"
          sources={[
            ...(["decode_tok_s", "joules_per_token", "watts_avg"] as const).flatMap((n) => {
              const p = mineAt(n);
              return p ? [{ label: `${p.label}, ${batch === 1 ? "1 request" : `${batch} at once`}`, value: `${fmt(p.value)} ${p.unit ?? ""}`, from: shortRef(p.source.ref) }] : [];
            }),
            ...(["decode_tok_s", "joules_per_token"] as const).flatMap((n) => {
              const p = theirsAt(n);
              return p ? [{ label: `${cmpShort}: ${p.label}`, value: `${fmt(p.value)} ${p.unit ?? ""}`, from: shortRef(p.source.ref) }] : [];
            }),
          ]}
        />
      ) : null}

      {offers.length ? (
        <div className={ui.chips} role="group" aria-label="Compare with another chip">
          {offers.map((c) => (
            <button key={c.id} type="button" className={`${ui.chip} ${c.id === compareId ? ui.chipOn : ""}`} aria-pressed={c.id === compareId} onClick={() => (setPicked(null), setCompareId(c.id === compareId ? null : c.id))}>
              Compare with: {chipShort(c.chip)}
            </button>
          ))}
        </div>
      ) : null}
      {cmp ? (
        <p className={b.banner}>
          <span>
            <b>
              {chipLabel(cmp.chip)}, in grey. Not the lab&apos;s hardware.
            </b>{" "}
            A different{wMine && wTheirs && wTheirs.value >= 2 * wMine.value ? ", much larger" : ""} chip{cmp.other.gpu ? ` (${cmp.other.gpu})` : ""}
            {cmp.chip.where ? `, measured on ${cmp.chip.where}` : ", measured elsewhere"} {when(cmp.other.ts)}
            {cmp.other.chip_note ? `. The rows say: ${cmp.other.chip_note}` : ""}.{" "}
            {wMine && wTheirs
              ? `With ${batch === 1 ? "1 request" : `${batch} at once`} it drew ${fmt(wTheirs.value)} W against ${fmt(wMine.value)} W on ${sweep.node}${cmp.other.power_source ? `, and its power is read another way (${cmp.other.power_source})` : ""}. `
              : ""}
            {cmp.same_model ? "The same model" : `A different model (${cmp.other.model ?? "not named"})`};{" "}
            {cmp.batches.length === batches.length ? "the same loads" : `only the loads ${cmp.batches.join(", ")}`}. {plainConfidence(cmp.other.confidence_note)} Tap a grey point to see its row.
          </span>
        </p>
      ) : null}

      <div ref={ref} className={b.chart}>
        {width > 0 ? (
          <>
            {m.decode_tok_s ? (
              <Panel
                {...shared}
                title={finding(m.decode_tok_s, { down: "Each person gets slower", up: "Each person gets faster", flat: "Each person's speed barely moves" }, "tokens a second")}
                hint={`Tokens a second for one request, by how many share the teacher${scale}.`}
                unit="tok/s"
                log={!!cmp}
                series={[{ name: "decode_tok_s", label: "Tokens a second per request", points: m.decode_tok_s, tone: "speed", shape: "circle" }, ...beside("decode_tok_s", "tokens a second per request")]}
                edgeNote={student?.tok_s && !cmp && student.tok_s.value > niceMax(Math.max(...m.decode_tok_s.map((p) => p.value))) ? `the box's own model: ${fmt(student.tok_s.value)} tok/s, above this scale` : null}
              />
            ) : null}
            {m.decode_tok_s_total ? (
              <Panel
                {...shared}
                title={finding(m.decode_tok_s_total, { down: "Together they get less", up: "Together they get more", flat: "The total barely moves" }, "tokens a second")}
                hint={`Tokens a second across all requests together${scale}.`}
                unit="tok/s"
                log={!!cmp}
                series={[{ name: "decode_tok_s_total", label: "Tokens a second in total", points: m.decode_tok_s_total, tone: "speed", shape: "square" }, ...beside("decode_tok_s_total", "tokens a second in total")]}
              />
            ) : null}
            {m.joules_per_token ? (
              <Panel
                {...shared}
                title={finding(m.joules_per_token, { down: "Each token gets cheaper", up: "Each token gets dearer", flat: "A token costs about the same" }, "J")}
                hint="Energy for one token, in joules. Log scale: each step up the side is ten times more."
                unit="J/token"
                log
                height={182}
                refBand={band}
                series={[{ name: "joules_per_token", label: "Joules per token", points: m.joules_per_token, tone: "energy", shape: "diamond" }, ...beside("joules_per_token", "joules per token")]}
              />
            ) : null}
            {ttft && m.ttft_s ? (
              <Panel
                {...shared}
                title={finding(m.ttft_s, { down: "The first word comes sooner", up: "The first word takes longer", flat: "The wait for the first word barely moves" }, "s")}
                hint={`Seconds before the first token arrives: the price of sharing${scale}.`}
                unit="s"
                log={!!cmp}
                series={[{ name: "ttft_s", label: "Time to first token", points: m.ttft_s, tone: cmp ? "speed" : "neutral", shape: cmp ? "circle" : "triangle" }, ...beside("ttft_s", "time to first token")]}
              />
            ) : null}
            {m.watts_avg ? (
              <Panel
                {...shared}
                title={finding(m.watts_avg, { down: "Power drawn falls", up: "Power drawn rises", flat: "Power drawn barely moves" }, "W")}
                hint={cmp ? "Watts: the average (line) and the lab's peak (squares)." : "Watts: the average (line) and the peak (squares)."}
                unit="W"
                height={cmp ? 144 : 108}
                series={[
                  { name: "watts_avg", label: "Average watts", points: m.watts_avg, tone: "energy", shape: "circle" },
                  ...(m.watts_peak ? [{ name: "watts_peak", label: "Peak watts", points: m.watts_peak, tone: (cmp ? "energy" : "neutral") as Tone, shape: "square" as Shape, lineless: true }] : []),
                  ...beside("watts_avg", "average watts"),
                ]}
              />
            ) : null}
          </>
        ) : null}
      </div>

      <Fold summary="The charts in words" guide="lab:speed-words">
      <ChartWords>
        {(
          [
            ["joules_per_token", "Energy for a token", "J"],
            ["decode_tok_s_total", "Tokens a second in total", ""],
            ["decode_tok_s", "Tokens a second for each person", ""],
            ["watts_avg", "Power", "W"],
          ] as const
        )
          .filter(([n]) => m[n]?.length)
          .map(([n, words, unit]) => `${words}: ${m[n]!.map((p) => `${said(p.value)}${unit ? ` ${unit}` : ""} with ${p.x}`).join(", ")}.`)
          .join(" ")}{" "}
        Each number is for that many requests at once.
      </ChartWords>
      </Fold>
      <div className={b.legend}>
        <span>
          <LegendMark shape="circle" tone="speed" hollow /> hollow and dashed: <Term k="low">low confidence</Term>
        </span>
        <span>
          <LegendMark shape="circle" tone="speed" /> solid: measured often enough
        </span>
        <span>
          <svg width={30} height={14} aria-hidden="true">
            <line className={b.refLine} x1={1} x2={29} y1={7} y2={7} />
          </svg>
          the box&apos;s own model, for reference
        </span>
        {cmp ? (
          <span>
            <LegendMark shape="triangle" tone="neutral" hollow={cmp.other.low_confidence > 0} /> grey: {cmpShort}, a different chip, for comparison
          </span>
        ) : null}
      </div>
      <p className={`${b.banner} ${allLow ? b.bannerWatch : ""}`}>
        <span>
          <b>{allLow ? "A first look, not a result." : `${sweep.low_confidence} of ${sweep.points} points are low confidence.`}</b> Tap a point for its row.
        </span>
      </p>
      <Fold summary="How sure these numbers are">
        <span className={b.note}>{plainConfidence(sweep.confidence_note)}</span>
      </Fold>
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
        <p className={b.note}>The lab has not measured the box&apos;s own model yet, so there is nothing to compare the teacher with.</p>
      </div>
    );
  const cells: [string, string, BenchPoint | null][] = [
    ["watts", "power while answering", s.watts],
    ["tok", "speed while writing", s.tok_s],
    ["jpt", "energy per token, writing only", s.j_per_token],
    ["jptr", "energy per token, whole request", s.j_per_token_request],
  ];
  const shown = cells.find(([k]) => k === open)?.[2] ?? null;
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h2>The box&apos;s own model, for comparison</h2>
      </div>
      <p className={b.note}>
        <Term k="student">The small model on the box</Term> answers one request at a time. Its <Term k="joules">energy per token</Term> is not measured directly: it is{" "}
        <Term k="derived">worked out</Term> from its watts and its speed.
      </p>
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
          <span className={b.note}>not measured yet</span>
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
      <p className={b.note}>The newest measurement of each machine, in watts: switched on and idle, and while answering. A striped bar was measured too few times to be sure of. Tap a bar for its row.</p>
      <div className={b.bars}>
        {nodes.map((n) => (
          <div key={n.node} style={{ display: "contents" }}>
            <span className={b.barNode}>
              {n.node === "jetson" ? "The box (Jetson)" : n.node}
              {n.serving?.state && n.node !== "jetson" ? <span className={b.note}> · answering {n.serving.state.replace(/^batch (\d+)$/, (_m, k: string) => (k === "1" ? "1 request" : `${k} at once`))}</span> : null}
            </span>
            {row(n, "idle")}
            {row(n, "serving")}
          </div>
        ))}
      </div>
      {bench.power.absent.length ? <p className={b.note}>Not measured yet: {bench.power.absent.join(", ")}. Nothing is drawn for a machine with no measurement.</p> : null}
      {bench.power.other_chip.length ? (
        <Fold summary={`Not drawn: ${bench.power.other_chip.length} measurement${bench.power.other_chip.length === 1 ? "" : "s"} from other hardware`}>
          <span className={b.note}>These were not measured on the lab&apos;s Strix Halo pods, so they are kept off this chart: {chipCounts(bench.power.other_chips, true)}. They are listed under Every row.</span>
        </Fold>
      ) : null}
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
    ...(sweep?.comparisons ?? []).map((c): [string, string] => [
      `"Compare with: ${chipShort(c.chip)}" adds grey triangles: the same measurement on a different chip that is not the lab's. They are never joined to the lab's line.`,
      `${c.chip.reason ?? "the rows' own chip"}; sweep ${c.id}`,
    ]),
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
        were not measured on the lab&apos;s Strix Halo pods: {chipCounts(bench.chips.filter((g) => !g.current), true)}. They are never drawn as the current lab&apos;s; a chip with a sweep of its own
        has its own named section.
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
              {x.other_chip ? ` · not the current lab: ${x.other_chips.map((g) => `${g.rows === x.rows ? "all" : g.rows} from ${chipShort(g)}`).join(", ")}` : ""}
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

/**
 * Routing, as one card: which preset of requests, and one sentence of result worked out by the box's own routing code
 * (POST /api/learning/lab/routing/simulate). Everything else the old Routing tab showed is behind "How this is worked
 * out", and is only loaded when that is opened.
 */
function RoutingCard({ forceReduced, initialPreset, initialOpen }: { forceReduced: boolean; initialPreset: string | null; initialOpen: boolean }) {
  const [preset, setPreset] = useState<string | null>(initialPreset);
  const [open, setOpen] = useState(initialOpen);
  const sim = useFetch((signal) => api.labRoutingSim(preset ? { preset } : {}, signal), `sim-${preset ?? ""}`);
  const data = sim.data;
  const result = data ? routingLine(data) : null;
  return (
    <div className={b.card} data-guide="lab:routing">
      <div className={b.cardHead}>
        <h2>Box, or teacher?</h2>
        <Tag kind="modelled">What-if</Tag>
      </div>
      {data ? (
        <div className={ui.segmented} role="group" aria-label="What is being asked">
          {data.presets.map((x) => (
            <button key={x.id} type="button" className={`${ui.seg} ${data.params.preset === x.id ? ui.segOn : ""}`} aria-pressed={data.params.preset === x.id} onClick={() => setPreset(x.id)}>
              {x.title}
            </button>
          ))}
        </div>
      ) : null}
      {result ? (
        <>
          <p className={b.readout} role="status">
            {result.line}
          </p>
          <SourceChip sources={result.sources} about="The routing result" />
        </>
      ) : data ? (
        <p className={b.note}>Nothing to work this out from yet: {data.reason}</p>
      ) : sim.error ? (
        <p className={b.note}>The box could not work this out ({sim.error.message}).</p>
      ) : (
        <p className={b.note}>Working it out…</p>
      )}
      <details open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)} data-guide="lab:routing-how">
        <summary className={ui.chip} style={{ display: "inline-flex", cursor: "pointer", listStyle: "none" }}>
          How this is worked out
        </summary>
        {open ? (
          <div style={{ marginTop: 14 }}>
            <Routing key={data?.params.preset ?? "custom"} initialPreset={data?.params.preset ?? preset} initialT={1} initialPolicy={null} initialBasis={null} initialDrawer={null} forceReduced={forceReduced} />
          </div>
        ) : null}
      </details>
    </div>
  );
}

export function SpeedEnergy({
  initialBatch,
  initialTtft,
  initialPoint,
  initialDrawer,
  initialCompare = null,
  initialSweep = null,
  initialPreset = null,
  initialRouting = false,
  forceReduced,
}: {
  /** The routing card's preset, and whether its "How this is worked out" starts open (the old Routing tab's links). */
  initialPreset?: string | null;
  initialRouting?: boolean;
  initialBatch: number | null;
  initialTtft: boolean;
  initialPoint: string | null;
  initialDrawer: string | null;
  initialCompare?: string | null;
  initialSweep?: string | null;
  forceReduced: boolean;
}) {
  const bench = useFetch((signal) => api.labBench(signal), "bench");
  const reduced = useReduced(forceReduced);
  const [sweepId, setSweepId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<"what" | "rows" | null>(initialDrawer === "what" || initialDrawer === "rows" ? initialDrawer : null);
  const close = useCallback(() => setDrawer(null), []);
  if (bench.loading && !bench.data) return <Loading what="the lab's measurements" />;
  if (bench.error && !bench.data) return <ErrorBox what="the lab's measurements" error={bench.error} onRetry={bench.reload} />;
  const data = bench.data!;
  // The tab opens on the lab's newest full curve (three load levels or more), not on whatever was measured last: a
  // two-point run is not a curve. Every other run stays one tap away under "Other measurements".
  const lead = headlineSweep(data);
  const sweep = data.sweeps.find((s) => s.id === sweepId) ?? (initialSweep ? data.sweeps.find((s) => s.chip.key === initialSweep || s.id === initialSweep) : undefined) ?? lead;
  const take = batchTakeaway(lead, when);
  const hero = speedHero(data, when);
  // the runs, one group per chip: the current lab first, then each other chip under its own name
  const groups: { chip: LabChip; sweeps: BenchSweep[] }[] = [];
  for (const s of data.sweeps) {
    const mine = groups.find((g) => (g.chip.current && s.chip.current) || (g.chip.key === s.chip.key && (g.chip.role !== "reference" || g.chip.where === s.chip.where)));
    if (mine) mine.sweeps.push(s);
    else groups.push({ chip: s.chip, sweeps: [s] });
  }
  return (
    <div className={b.page}>
      <div className={ui.pageHead}>
        <h1>Speed and energy</h1>
      </div>
      {hero ? (
        <Hero label="What does an answer cost?" hero={hero} guide="lab:cost" />
      ) : take ? (
        <Takeaway label="What does an answer cost?" take={take} guide="lab:cost" />
      ) : data.rows === 0 ? (
        <p className={b.banner}>{data.empty_reason}</p>
      ) : null}
      <div className={b.two}>
        {sweep ? (
          <Batching key={sweep.id} sweep={sweep} bench={data} initialBatch={initialBatch} initialTtft={initialTtft} initialPoint={initialPoint} initialCompare={initialCompare} reduced={reduced} />
        ) : (
          <div className={b.card}>
            <h2>No batching measurement yet</h2>
            <p className={b.note}>The lab has not yet measured the teacher at two or more loads. The chart appears as soon as it does.</p>
          </div>
        )}
        <div className={b.col}>
          <RoutingCard forceReduced={forceReduced} initialPreset={initialPreset} initialOpen={initialRouting} />
          <StudentCard bench={data} />
          <PowerCard bench={data} />
        </div>
      </div>
      <div className={b.cardHead} style={{ justifyContent: "flex-start" }}>
        <button type="button" className={`${ui.btn}`} onClick={() => setDrawer("what")}>
          What am I looking at?
        </button>
        <button type="button" className={`${ui.btn}`} onClick={() => setDrawer("rows")}>
          Every row ({data.rows.toLocaleString()})
        </button>
      </div>
      {data.sweeps.length > 1 ? (
        <details className={b.col} style={{ gap: 12 }} open={!!initialSweep || (!!sweep && sweep.id !== lead?.id)} data-guide="lab:sweeps">
          <summary className={`${ui.chip}`} style={{ alignSelf: "flex-start", cursor: "pointer", listStyle: "none" }}>
            Other measurements ({data.sweeps.length - 1}){sweep && sweep.id !== lead?.id ? ": showing an older one" : ""}
          </summary>
          <p className={b.note} style={{ marginTop: 10 }}>
            Each is <Term k="sweep">a run of the same test at different loads</Term>. The one the sentence above is read from is the newest with three load levels or more on the lab&apos;s own
            chip.
          </p>
          {groups.map((g) => {
            const title = g.chip.current ? `The lab's own chip: ${g.chip.name}` : chipLabel(g.chip);
            return (
              <div key={`${g.chip.key}|${g.chip.where ?? ""}`} className={b.col} style={{ gap: 6, marginTop: 10 }}>
                <span className={b.note}>
                  {title}
                  {g.chip.current ? "" : ` · ${chipRole(g.chip)}`}
                </span>
                <div className={ui.chips} role="group" aria-label={`Measurement runs, ${title}`}>
                  {g.sweeps.map((s) => (
                    <button key={s.id} type="button" className={`${ui.chip} ${s.id === sweep?.id ? ui.chipOn : ""}`} aria-pressed={s.id === sweep?.id} onClick={() => setSweepId(s.id)}>
                      {sweepName(s, when)}
                      {s.id === lead?.id ? " · the one above" : ""}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </details>
      ) : null}
      <Words keys={["teacher", "student", "batch", "tokens", "joules", "low", "sweep", "measured", "derived"]} />
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
