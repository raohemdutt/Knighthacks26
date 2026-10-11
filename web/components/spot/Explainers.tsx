"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import ui from "@/components/ui.module.css";
import x from "./explain.module.css";
import { geometry } from "./FloorMap";
import type { Bar, Finding, Metric, Replay, Shift, Target } from "./types";

/**
 * The analysis, shown instead of tabled: one short animated explainer per finding. Each has one idea, a caption that
 * changes as it plays, a Replay button and "Show the numbers", which opens the exact rows, intervals and caveats that
 * were always there. Every count and height comes from the API (the plain shift view, the findings, the replay marks),
 * so a different kit draws a different picture.
 *
 * Motion follows the app: 420 to 700 ms, cubic-bezier(0.5, 0, 0.2, 1), nothing loops except one soft pulse. With
 * "reduce motion" on, each explainer is drawn at its last step and nothing moves.
 */

export type ExplainId = "hidden" | "alarms" | "battery" | "version" | "weak" | "missing";
export type Force = "end" | "mid" | null;

/** Which explainer backs which finding. Findings not listed open their table directly. */
export const EXPLAINS: Record<string, ExplainId> = {
  hidden_failures: "hidden",
  anomalies_hidden: "alarms",
  battery: "battery",
  version_comparison: "version",
  over_time: "version",
  reliability_by_location: "weak",
  reliability_by_type: "weak",
  sequence: "weak",
  validation: "missing",
};

interface Stage {
  caption: string;
  ms: number;
}

function useStages(stages: Stage[], force: Force) {
  const last = stages.length - 1;
  const [stage, setStage] = useState(force === "end" ? last : force === "mid" ? Math.ceil(last / 2) : 0);
  const [run, setRun] = useState(0); // 0: waiting to be seen
  const ref = useRef<HTMLDivElement>(null);
  const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (force || run > 0) return;
    if (still) {
      setStage(last);
      return;
    }
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      setRun(1);
      return;
    }
    const io = new IntersectionObserver((entries) => entries.some((e) => e.isIntersecting) && setRun(1), { threshold: 0.5 });
    io.observe(el);
    return () => io.disconnect();
  }, [force, run, still, last]);

  useEffect(() => {
    if (force || run === 0 || still || stage >= last) return;
    const t = window.setTimeout(() => setStage((s) => Math.min(last, s + 1)), stages[stage]?.ms ?? 1200);
    return () => window.clearTimeout(t);
  }, [force, run, still, stage, last, stages]);

  const replay = () => {
    setStage(0);
    setRun((r) => r + 1);
  };
  return { stage, ref, replay };
}

function Shell({
  title,
  stages,
  force,
  onNumbers,
  numbers = "Show the numbers",
  children,
}: {
  title: string;
  stages: Stage[];
  force: Force;
  onNumbers?: () => void;
  numbers?: string;
  children: (stage: number) => ReactNode;
}) {
  const { stage, ref, replay } = useStages(stages, force);
  return (
    <article className={x.card} ref={ref}>
      <h3 className={x.title}>{title}</h3>
      <div className={x.stageBox}>{children(stage)}</div>
      <p className={x.caption} aria-live="polite">
        {stages[stage]?.caption}
      </p>
      <div className={x.foot}>
        <span className={x.dots} aria-hidden="true">
          {stages.map((_, i) => (
            <i key={i} className={i <= stage ? x.dotOn : undefined} />
          ))}
        </span>
        <button type="button" className={x.btn} onClick={replay}>
          ↻ Replay
        </button>
        {onNumbers ? (
          <button type="button" className={`${x.btn} ${x.btnMain}`} onClick={onNumbers}>
            {numbers}
          </button>
        ) : null}
      </div>
    </article>
  );
}

const pctText = (k: number, n: number) => (n ? `${Math.round((100 * k) / n)}%` : "n/a");

// ---------- 1. "completed" hides failures ----------

function Hidden({ shift, force, onNumbers }: { shift: Shift; force: Force; onNumbers: () => void }) {
  const rounds = shift.rounds.filter((r) => !r.result.startsWith("did not finish"));
  const n = rounds.length;
  const k = rounds.filter((r) => r.failed > 0).length;
  const per = 9;
  const size = 52;
  const gap = 9;
  const rows = Math.ceil(n / per);
  const stages: Stage[] = [
    { caption: `${n} rounds are marked "completed". A tick each: looks fine.`, ms: 1500 },
    { caption: "Open them up. Every round is about ten checks.", ms: 1900 },
    { caption: "The checks that failed turn red.", ms: 1900 },
    { caption: `${k} of ${n} completed rounds had a failed check inside. That is why this screen shows checks, not rounds.`, ms: 0 },
  ];
  return (
    <Shell title={'"Completed" hides failures'} stages={stages} force={force} onNumbers={onNumbers}>
      {(stage) => (
        <svg viewBox={`0 0 560 ${rows * (size + gap) + 46}`} role="img" aria-label={`${k} of ${n} completed rounds had a failed check inside`}>
          {rounds.map((r, i) => {
            const cx = 6 + (i % per) * (size + gap);
            const cy = 4 + Math.floor(i / per) * (size + gap);
            const bad = r.failed > 0;
            const delay = `${i * 45}ms`;
            const checks = Math.max(1, Math.min(r.checks, 10));
            return (
              <g key={r.mission_id} transform={`translate(${cx} ${cy})`}>
                <rect className={`${x.t} ${x.tile} ${stage >= 3 && bad ? x.tileBad : ""}`} style={{ transitionDelay: delay }} width={size} height={size} rx="12" />
                <path className={`${x.t} ${x.tick}`} style={{ opacity: stage === 0 ? 1 : 0, transitionDelay: delay }} d={`M ${size * 0.3} ${size * 0.52} l ${size * 0.14} ${size * 0.14} l ${size * 0.27} ${-size * 0.3}`} />
                {Array.from({ length: checks }, (_, j) => (
                  <circle
                    key={j}
                    className={`${x.t} ${stage >= 2 && j >= checks - r.failed ? x.fillBad : x.fillOk}`}
                    style={{ opacity: stage >= 1 ? 1 : 0, transitionDelay: `${i * 45 + (stage >= 2 ? 0 : j * 12)}ms` }}
                    cx={9 + (j % 5) * 8.6}
                    cy={size / 2 - 5 + Math.floor(j / 5) * 10}
                    r="3.1"
                  />
                ))}
              </g>
            );
          })}
          <text className={`${x.t} ${x.big}`} style={{ opacity: stage >= 3 ? 1 : 0 }} x="6" y={rows * (size + gap) + 34}>
            {k} of {n}
            <tspan className={x.bigSub} dx="10">
              completed rounds had a failed check inside
            </tspan>
          </text>
        </svg>
      )}
    </Shell>
  );
}

// ---------- 2. alarms you can't trust yet ----------

function Alarms({ shift, force, onNumbers }: { shift: Shift; force: Force; onNumbers: () => void }) {
  const f = shift.findings.find((v) => v.id === "anomalies_hidden");
  const n = f?.facts.n ?? 0;
  const unsure = f?.facts.unsure ?? 0;
  const stands = f?.facts.stands ?? n - unsure;
  const per = 20;
  const stages: Stage[] = [
    { caption: `SPOT raised ${n} alarms. Which of them can you act on?`, ms: 1500 },
    { caption: "An alarm raised by a check that itself failed goes left: not confirmed.", ms: 2100 },
    { caption: "An alarm raised by a check that worked goes right: it stands.", ms: 1900 },
    { caption: `${unsure} of ${n} alarms are not confirmed, so send SPOT back first. ${stands} stand, so send a person.`, ms: 0 },
  ];
  const chip = (i: number, stage: number) => {
    const left = i < unsure;
    if ((left && stage >= 1) || (!left && stage >= 2)) {
      const j = left ? i : i - unsure;
      const cols = left ? 12 : 4;
      return { x: (left ? 14 : 400) + (j % cols) * 26, y: 74 + Math.floor(j / cols) * 22 };
    }
    return { x: 22 + (i % per) * 26, y: 10 + Math.floor(i / per) * 22 };
  };
  return (
    <Shell title="Alarms you can't trust yet" stages={stages} force={force} onNumbers={onNumbers}>
      {(stage) => (
        <svg viewBox="0 0 560 196" role="img" aria-label={`${unsure} of ${n} alarms are not confirmed; ${stands} stand`}>
          <rect className={x.bin} x="6" y="64" width="330" height="92" rx="14" />
          <rect className={x.bin} x="392" y="64" width="122" height="92" rx="14" />
          {Array.from({ length: n }, (_, i) => {
            const p = chip(i, stage);
            const left = i < unsure;
            const moved = (left && stage >= 1) || (!left && stage >= 2);
            return (
              <rect
                key={i}
                className={`${x.t} ${x.slow} ${moved ? (left ? x.fillWatch : x.fillBad) : x.fillMuted}`}
                style={{ transform: `translate(${p.x}px, ${p.y}px)`, transitionDelay: `${(left ? i : i - unsure) * 28}ms` }}
                width="20"
                height="15"
                rx="5"
              />
            );
          })}
          <text className={x.label} x="10" y="180">
            <tspan className={x.num}>{stage >= 1 ? unsure : 0}</tspan> not confirmed: the check that raised them failed
          </text>
          <text className={x.label} x="392" y="180">
            <tspan className={x.num}>{stage >= 2 ? stands : 0}</tspan> stand
          </text>
        </svg>
      )}
    </Shell>
  );
}

// ---------- 3. battery ----------

function Battery({ f, force, onNumbers }: { f: Finding; force: Force; onNumbers: () => void }) {
  const cut = Number(f.extra.threshold ?? 40);
  const bins = [...f.rows].sort((a, b) => Number(b.key.split("-")[0]) - Number(a.key.split("-")[0])); // fullest first
  const top = Math.max(0.01, ...bins.map((b) => b.rate ?? 0));
  const low = bins.filter((b) => Number(b.key.split("-")[1]) <= cut);
  const high = bins.filter((b) => !low.includes(b));
  const sum = (rows: Bar[]) => ({ k: rows.reduce((t, b) => t + b.k, 0), n: rows.reduce((t, b) => t + b.n, 0) });
  const lo = sum(low);
  const hi = sum(high);
  const levels = [100, ...bins.map((b) => Number(b.key.split("-")[0]))]; // battery left at each step
  const last = bins.length + 1;
  const stages: Stage[] = [
    { caption: "SPOT sets off with a full battery. Watch how many checks fail as it drains.", ms: 1500 },
    ...bins.map((b) => {
      const from = Number(b.key.split("-")[0]);
      const tone = from < 20 ? "Most of them fail." : from < cut ? "Failures climb." : "";
      return { caption: `Battery ${b.label}: ${b.k} of ${b.n} checks failed. ${tone}`.trim(), ms: 1500 };
    }),
    { caption: `Under ${cut}%: ${pctText(lo.k, lo.n)} of checks failed (${lo.k} of ${lo.n}). At ${cut}% or more: ${pctText(hi.k, hi.n)} (${hi.k} of ${hi.n}). Bring SPOT back to charge earlier.`, ms: 0 },
  ];
  const W = 560;
  const col = (W - 40) / bins.length;
  const base = 196;
  const H = 120;
  return (
    <Shell title="Low battery, more failed checks" stages={stages} force={force} onNumbers={onNumbers}>
      {(stage) => {
        const end = stage >= last;
        const left = end ? 0 : levels[Math.min(stage, levels.length - 1)] ?? 0;
        return (
          <svg viewBox={`0 0 ${W} 240`} role="img" aria-label={`Under ${cut}% battery ${lo.k} of ${lo.n} checks failed; above, ${hi.k} of ${hi.n}`}>
            {/* the battery, draining */}
            <g className={x.t} style={{ opacity: end ? 0 : 1 }}>
              <rect className={x.battShell} x="20" y="8" width="220" height="34" rx="9" />
              <rect className={x.battCap} x="242" y="18" width="7" height="14" rx="3" />
              <rect className={`${x.t} ${x.slow} ${left < 20 ? x.fillBad : left < cut ? x.fillWatch : x.fillOk}`} x="25" y="13" height="24" rx="5" style={{ width: `${Math.max(2, 2.1 * left)}px` }} />
              <text className={x.num} x="262" y="32">
                {left}%
              </text>
              <text className={x.label} x="320" y="32">
                battery left
              </text>
            </g>
            {/* one bar per stretch of battery: how many checks failed there */}
            {bins.map((b, i) => {
              const from = Number(b.key.split("-")[0]);
              const shown = stage > i && !end;
              const h = shown ? Math.max(3, (H * (b.rate ?? 0)) / top) : 0;
              return (
                <g key={b.key} className={x.t} style={{ opacity: end ? 0 : 1 }}>
                  <rect className={`${x.t} ${x.slow} ${from < 20 ? x.fillBad : from < cut ? x.fillWatch : x.fillMuted}`} x={24 + i * col} width={col - 12} rx="7" style={{ y: `${base - h}px`, height: `${h}px` } as CSSProperties} />
                  <text className={`${x.t} ${x.num}`} style={{ opacity: shown ? 1 : 0 }} x={24 + i * col + (col - 12) / 2} y={base - h - 7} textAnchor="middle">
                    {pctText(b.k, b.n)}
                  </text>
                  <text className={x.label} x={24 + i * col + (col - 12) / 2} y={base + 18} textAnchor="middle">
                    {b.label}
                  </text>
                </g>
              );
            })}
            <text className={`${x.t} ${x.label}`} style={{ opacity: end ? 0 : 1 }} x="24" y={base + 36}>
              share of checks that failed, by battery left
            </text>
            {/* the two-bar ending */}
            {[
              { label: `Battery under ${cut}%`, v: lo, cls: x.fillWatch },
              { label: `${cut}% or more`, v: hi, cls: x.fillMuted },
            ].map((r, i) => {
              const w = end ? Math.max(6, (330 * (r.v.n ? r.v.k / r.v.n : 0)) / Math.max(0.01, lo.n ? lo.k / lo.n : 0, hi.n ? hi.k / hi.n : 0)) : 0;
              return (
                <g key={r.label} className={x.t} style={{ opacity: end ? 1 : 0 }}>
                  <text className={x.label} x="20" y={76 + i * 78}>
                    {r.label}
                  </text>
                  <rect className={`${x.t} ${x.slow} ${r.cls}`} x="20" y={86 + i * 78} height="30" rx="9" style={{ width: `${w}px` }} />
                  <text className={`${x.t} ${x.big}`} x={34 + w} y={110 + i * 78}>
                    {pctText(r.v.k, r.v.n)}
                    <tspan className={x.bigSub} dx="8">
                      {r.v.k} of {r.v.n} failed
                    </tspan>
                  </text>
                </g>
              );
            })}
          </svg>
        );
      }}
    </Shell>
  );
}

// ---------- 4. is the new setup really better? ----------

function Version({ f, force, onNumbers }: { f: Finding; force: Force; onNumbers: () => void }) {
  const metrics = (f.extra.metrics as Metric[] | undefined) ?? [];
  const all = metrics.find((m) => m.key === "action_success");
  const same = metrics.find((m) => m.key === "action_success_matched_battery");
  const band = (f.extra.battery_band as [number, number] | undefined) ?? null;
  const below = f.extra.below_band as { k: number; n: number } | undefined;
  if (!all) return null;
  const p = (v: number | null) => `${Math.round(100 * (v ?? 0))}%`;
  const gapAll = Math.round(100 * ((all.b.value ?? 0) - (all.a.value ?? 0)));
  const gapSame = same ? Math.round(100 * ((same.b.value ?? 0) - (same.a.value ?? 0))) : null;
  const unclear = same ? same.verdict !== "changed" : false;
  const stages: Stage[] = [
    { caption: "SPOT's setup was updated part-way through. Did checks get more reliable?", ms: 1500 },
    { caption: `Looks like it: ${p(all.b.value)} of checks worked on the newer setup, ${p(all.a.value)} on the older. A ${gapAll}-point gap.`, ms: 2400 },
    {
      caption: band
        ? `But the newer rounds never ran below ${band[0]}% battery. The older ones did, for ${below?.n ?? "many"} checks, and low battery is where checks fail.`
        : "But the two setups did not run in the same conditions.",
      ms: 2800,
    },
    {
      caption: same
        ? `Compare only the battery range both ran in: ${p(same.b.value)} against ${p(same.a.value)}. The gap shrinks to ${gapSame} points${unclear ? ", too small to be sure" : ""}. We do not say the update is the reason.`
        : "There is too little shared ground to compare them fairly.",
      ms: 0,
    },
  ];
  const H = 124;
  const base = 168;
  return (
    <Shell title="Is the new setup really better?" stages={stages} force={force} onNumbers={onNumbers}>
      {(stage) => {
        const use = stage >= 3 && same ? same : all;
        const bars = [
          { label: "Older setup", side: use.a, cls: x.fillMuted },
          { label: "Newer setup", side: use.b, cls: x.fillOk },
        ];
        const ax = (v: number) => 316 + 2.2 * v; // the battery axis on the right
        return (
          <svg viewBox="0 0 560 250" role="img" aria-label={stages[stages.length - 1]!.caption}>
            {bars.map((b, i) => {
              const h = stage >= 1 ? H * (b.side.value ?? 0) : 0;
              return (
                <g key={b.label}>
                  <rect className={`${x.t} ${x.slow} ${b.cls}`} x={20 + i * 136} width="84" rx="10" style={{ y: `${base - h}px`, height: `${h}px` } as CSSProperties} />
                  <text className={`${x.t} ${x.big}`} style={{ opacity: stage >= 1 ? 1 : 0, transform: `translateY(${-h}px)` }} x={62 + i * 136} y={base - 8} textAnchor="middle">
                    {p(b.side.value)}
                  </text>
                  <text className={x.label} x={62 + i * 136} y={base + 20} textAnchor="middle">
                    {b.label}
                  </text>
                  <text className={x.label} x={62 + i * 136} y={base + 38} textAnchor="middle">
                    {b.side.k} of {b.side.n} worked
                  </text>
                </g>
              );
            })}
            <text className={`${x.t} ${x.label} ${x.strong}`} style={{ opacity: stage >= 3 ? 1 : 0 }} x="20" y="240">
              same battery level only
            </text>
            {band ? (
              <g className={x.t} style={{ opacity: stage >= 2 ? 1 : 0 }}>
                <text className={`${x.label} ${x.strong}`} x="316" y="24">
                  Battery while checking
                </text>
                <line className={x.axis} x1={ax(0)} x2={ax(100)} y1="146" y2="146" />
                {[0, 50, 100].map((v) => (
                  <text key={v} className={x.label} x={ax(v)} y="166" textAnchor="middle">
                    {v}%
                  </text>
                ))}
                {/* the older setup ran low; the newer one never did */}
                <rect className={`${x.t} ${x.slow} ${x.fillMuted}`} x={ax(8)} y="62" height="16" rx="8" style={{ width: stage >= 2 ? `${ax(band[1]) - ax(8)}px` : "0px" }} />
                <text className={x.label} x={ax(0)} y="54">
                  older setup
                </text>
                <rect className={`${x.t} ${x.slow} ${x.fillOk}`} x={ax(band[0])} y="112" height="16" rx="8" style={{ width: stage >= 2 ? `${ax(100) - ax(band[0])}px` : "0px" }} />
                <text className={x.label} x={ax(0)} y="125">
                  newer setup
                </text>
                <rect className={`${x.t} ${x.bracket}`} style={{ opacity: stage >= 3 ? 1 : 0 }} x={ax(band[0])} y="40" width={ax(band[1]) - ax(band[0])} height="100" rx="8" />
                <text className={`${x.t} ${x.label} ${x.strong}`} style={{ opacity: stage >= 3 ? 1 : 0 }} x={(ax(band[0]) + ax(band[1])) / 2} y="190" textAnchor="middle">
                  both ran here: {band[0]}% to {band[1]}%
                </text>
              </g>
            ) : null}
            <text className={`${x.t} ${x.big}`} style={{ opacity: stage >= 1 ? 1 : 0 }} x="316" y="238">
              {stage >= 3 && gapSame !== null ? `+${gapSame}` : `+${gapAll}`}
              <tspan className={x.bigSub} dx="8">
                {stage >= 3 && same ? (unclear ? "points: not clear" : "points") : "points"}
              </tspan>
            </text>
          </svg>
        );
      }}
    </Shell>
  );
}

// ---------- 5. weak spots on the map ----------

function Weak({ shift, where, kinds, order, force, onNumbers }: { shift: Shift; where?: Finding; kinds?: Finding; order?: Finding; force: Force; onNumbers: (id: string) => void }) {
  const geo = useMemo(() => (shift.map ? geometry(shift.map) : null), [shift.map]);
  if (!geo || !where) return null;
  const rows = where.rows.filter((b) => geo.pos[b.key]);
  const worst = rows.find((b) => b.key === where.extra.worst) ?? rows[0];
  const top = Math.max(0.01, ...rows.map((b) => b.rate ?? 0));
  const name = (id: string) => shift.map?.places.find((p) => p.location_id === id)?.name ?? id;
  const types = kinds?.rows ?? [];
  const typeTop = Math.max(0.01, ...types.map((b) => b.rate ?? 0));
  const worstType = types.find((b) => b.key === kinds?.extra.worst);
  const steps = order?.rows ?? [];
  const stepTop = Math.max(0.01, ...steps.map((b) => b.rate ?? 0));
  const tail = steps.slice(-2);
  const KIND: Record<string, string> = { ACOUSTIC_INSPECTION: "Listening", GAUGE_READING: "Gauge", NAVIGATION: "Walking", THERMAL_INSPECTION: "Heat", VISUAL_INSPECTION: "Camera" };
  const stages: Stage[] = [
    { caption: "Where does SPOT struggle? Each circle is a stop: the bigger it is, the more checks failed there.", ms: 2000 },
    { caption: worst ? `${name(worst.key)} is the weak spot: ${worst.k} of ${worst.n} checks failed there.` : "No stop stands out.", ms: 2400 },
    { caption: worstType ? `What it is doing matters too. ${KIND[worstType.key] ?? "One kind of"} checks fail most: ${worstType.k} of ${worstType.n}.` : "Every kind of check fails about as often.", ms: 2400 },
    {
      caption: tail.length ? `And when: the last two stops of a round fail most (${tail.reduce((t, b) => t + b.k, 0)} of ${tail.reduce((t, b) => t + b.n, 0)}), when the battery is lowest.` : "",
      ms: 0,
    },
  ];
  const scale = 560 / geo.w;
  const mapH = geo.h * scale;
  return (
    <Shell title="Weak spots: where, what and when" stages={stages} force={force} onNumbers={() => onNumbers(where.id)}>
      {(stage) => (
        <svg viewBox={`0 0 560 ${mapH + 118}`} role="img" aria-label={stages[1]!.caption}>
          <g transform={`scale(${scale})`}>
            {geo.image ? (
              <>
                <image className={x.plan} href={geo.image} x="0" y="0" width={geo.w} height={geo.h} preserveAspectRatio="none" />
                <rect x="0" y="0" width={geo.w} height={geo.h} fill="var(--surface)" opacity="0.72" />
              </>
            ) : (
              <rect x="0" y="0" width={geo.w} height={geo.h} rx="14" fill="var(--surface-2)" />
            )}
            {rows.map((b, i) => {
              const p = geo.pos[b.key]!;
              const isWorst = worst?.key === b.key;
              const r = 8 + 34 * Math.sqrt((b.rate ?? 0) / top);
              return (
                <g key={b.key} transform={`translate(${p.x} ${p.y})`}>
                  {isWorst && stage >= 1 ? <circle className={x.pulse} r={r} /> : null}
                  <circle className={`${x.t} ${x.slow} ${isWorst && stage >= 1 ? x.heatBad : x.heat}`} r={r} style={{ animationDelay: `${i * 40}ms` }} />
                  {isWorst && stage >= 1 ? (
                    <text className={x.mapLabel} y={-r - 8} textAnchor="middle" style={{ fontSize: 26 / scale }}>
                      {name(b.key)}: {b.k} of {b.n}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
          {/* what: by kind of check */}
          <g className={x.t} style={{ opacity: stage >= 2 ? 1 : 0.18 }}>
            <text className={`${x.label} ${x.strong}`} x="6" y={mapH + 22}>
              By kind of check
            </text>
            {types.map((b, i) => {
              const w = stage >= 2 ? (120 * (b.rate ?? 0)) / typeTop : 0;
              return (
                <g key={b.key} transform={`translate(6 ${mapH + 32 + i * 16})`}>
                  <text className={x.small} x="0" y="10">
                    {KIND[b.key] ?? b.label}
                  </text>
                  <rect className={`${x.t} ${x.slow} ${b.key === worstType?.key ? x.fillWatch : x.fillMuted}`} x="78" y="1" height="10" rx="5" style={{ width: `${w}px` }} />
                  <text className={x.small} x={84 + w} y="10">
                    {b.k} of {b.n}
                  </text>
                </g>
              );
            })}
          </g>
          {/* when: by position in the round */}
          <g className={x.t} style={{ opacity: stage >= 3 ? 1 : 0.18 }}>
            <text className={`${x.label} ${x.strong}`} x="300" y={mapH + 22}>
              By stop in the round
            </text>
            {steps.map((b, i) => {
              const h = stage >= 3 ? (58 * (b.rate ?? 0)) / stepTop : 0;
              const late = i >= steps.length - 2;
              return (
                <g key={b.key}>
                  <rect className={`${x.t} ${x.slow} ${late ? x.fillBad : x.fillMuted}`} x={300 + i * 25} width="18" rx="5" style={{ y: `${mapH + 96 - h}px`, height: `${h}px`, transitionDelay: `${i * 60}ms` } as CSSProperties} />
                  <text className={x.small} x={309 + i * 25} y={mapH + 110} textAnchor="middle">
                    {b.key}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      )}
    </Shell>
  );
}

// ---------- 6. what the data itself is missing ----------

function Missing({ live, shift, force, onNumbers }: { live: Replay; shift: Shift; force: Force; onNumbers: () => void }) {
  const marks = live.marks ?? [];
  const total = Math.max(1, live.total);
  const stop = live.rebuilt_from ?? null;
  const ghost = marks.filter((m) => m.source === "tables");
  const whole = ghost.filter((m) => stop !== null && m.start >= stop);
  const cut = ghost.find((m) => stop !== null && m.start < stop);
  const num = (id: string) => shift.rounds.find((r) => r.mission_id === id)?.number ?? id;
  const X = (i: number) => 10 + (540 * i) / total;
  const stages: Stage[] =
    stop === null
      ? [{ caption: `The event stream covers all ${marks.length} rounds: ${total} events, nothing missing.`, ms: 0 }]
      : [
          { caption: "The kit comes with a recording of events to replay, one block per round.", ms: 1600 },
          { caption: `Play it through and it stops at event ${stop}${cut ? `, in the middle of round ${num(cut.mission_id)}` : ""}.`, ms: 2600 },
          { caption: `${whole.length} round${whole.length === 1 ? "" : "s"} never appear${whole.length === 1 ? "s" : ""} in it at all.`, ms: 2000 },
          { caption: `We rebuilt the missing ${total - stop} events from the kit's tables and mark them as rebuilt, so the replay runs to the end and says which part is ours.`, ms: 0 },
        ];
  return (
    <Shell title="What the data itself is missing" stages={stages} force={force} onNumbers={onNumbers} numbers="Show the data check">
      {(stage) => {
        const head = stop === null ? total : stage === 0 ? 0 : stage >= 1 ? stop : 0;
        return (
          <svg viewBox="0 0 560 132" role="img" aria-label={stages[stages.length - 1]!.caption}>
            {marks.map((m) => {
              const rebuilt = m.source === "tables";
              const from = rebuilt && stop !== null ? Math.max(m.start, stop) : m.start;
              return (
                <g key={m.mission_id}>
                  {rebuilt && stop !== null && m.start < stop ? <rect className={x.fillMuted} x={X(m.start) + 1} y="40" width={Math.max(1, X(stop) - X(m.start) - 1)} height="34" rx="4" /> : null}
                  <rect
                    className={`${x.t} ${rebuilt ? (stage >= 3 ? x.ghostOn : x.ghost) : x.fillMuted}`}
                    style={{ opacity: rebuilt ? (stage >= 2 ? 1 : 0) : 1 }}
                    x={X(from) + 1}
                    y="40"
                    width={Math.max(1, X(m.end + 1) - X(from) - 2)}
                    height="34"
                    rx="4"
                  />
                </g>
              );
            })}
            <g className={`${x.t} ${x.sweep}`} style={{ transform: `translateX(${X(head)}px)` }}>
              <line className={x.head} x1="0" x2="0" y1="26" y2="88" />
              <text className={x.num} x="0" y="20" textAnchor={head > total * 0.8 ? "end" : "middle"}>
                {stage >= 1 && stop !== null ? `event ${stop}` : "start"}
              </text>
            </g>
            <text className={x.label} x="10" y="108">
              round {marks[0] ? num(marks[0].mission_id) : ""}
            </text>
            <text className={x.label} x="550" y="108" textAnchor="end">
              round {marks.length ? num(marks[marks.length - 1]!.mission_id) : ""}
            </text>
            <text className={`${x.t} ${x.label} ${x.strong}`} style={{ opacity: stage >= 3 ? 1 : 0 }} x={stop !== null ? Math.min(X(stop), 380) : 10} y="126">
              rebuilt from the tables
            </text>
          </svg>
        );
      }}
    </Shell>
  );
}

// ---------- one explainer by id, and all of them ----------

export function Explainer({
  id,
  shift,
  findings,
  live,
  force,
  onNumbers,
}: {
  id: ExplainId;
  shift: Shift;
  findings: Finding[];
  live: Replay | null;
  force: Force;
  onNumbers: (t: Target) => void;
}) {
  const f = (fid: string) => findings.find((v) => v.id === fid);
  const open = (fid: string) => () => onNumbers({ kind: "finding", id: fid });
  if (id === "hidden") return <Hidden shift={shift} force={force} onNumbers={open("hidden_failures")} />;
  if (id === "alarms") return <Alarms shift={shift} force={force} onNumbers={open("anomalies_hidden")} />;
  if (id === "battery") return f("battery") ? <Battery f={f("battery")!} force={force} onNumbers={open("battery")} /> : null;
  if (id === "version") return f("version_comparison") ? <Version f={f("version_comparison")!} force={force} onNumbers={open("version_comparison")} /> : null;
  if (id === "weak")
    return <Weak shift={shift} where={f("reliability_by_location")} kinds={f("reliability_by_type")} order={f("sequence")} force={force} onNumbers={(fid) => onNumbers({ kind: "finding", id: fid })} />;
  if (id === "missing") return live?.configured ? <Missing live={live} shift={shift} force={force} onNumbers={() => onNumbers({ kind: "validation", id: "validation" })} /> : null;
  return null;
}

export const EXPLAIN_ORDER: ExplainId[] = ["hidden", "alarms", "battery", "version", "weak", "missing"];

export function ExplainerGrid(props: Omit<Parameters<typeof Explainer>[0], "id">) {
  return (
    <div className={x.grid}>
      {EXPLAIN_ORDER.map((id) => (
        <Explainer key={id} id={id} {...props} />
      ))}
    </div>
  );
}

/** One explainer over the page, for a "Why?". */
export function ExplainSheet({ onClose, ...props }: Parameters<typeof Explainer>[0] & { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className={ui.overlay} onClick={onClose}>
      <div className={`${ui.sheet} ${x.sheet}`} role="dialog" aria-modal="true" aria-label="Why" onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <span className={x.eyebrow}>Why?</span>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        <Explainer {...props} />
      </div>
    </div>
  );
}
