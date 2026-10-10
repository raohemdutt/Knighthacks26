"use client";

import { Pill, RateBars, Track, pValue, pct, useOpen } from "./bits";
import s from "./spot.module.css";
import type { Bar, Finding, Metric } from "./types";

const VERDICT_TONE = { changed: "ok", "no clear change": "muted", "not enough data": "watch" } as const;

function Card({ f, children, note }: { f: Finding; children?: React.ReactNode; note?: string }) {
  const { open } = useOpen();
  return (
    <section className={s.card}>
      <div className={s.cardHead}>
        <h2>{f.title}</h2>
        <button type="button" className={s.linkBtn} onClick={() => open({ kind: "finding", id: f.id })}>
          Rows behind it
        </button>
      </div>
      <p className={s.sentence}>{f.enough ? f.sentence : `Not enough data. ${f.sentence}`}</p>
      {children}
      {note ? <p className={`${s.small} ${s.muted}`}>{note}</p> : null}
      {f.caveat ? <p className={s.caveat}>{f.caveat}</p> : null}
    </section>
  );
}

/** V1 against V2: each rate as a pair of bars with 95% intervals; means as a small table; a verdict on every line. */
export function VersionCard({ f }: { f: Finding }) {
  const { open } = useOpen();
  const metrics = (f.extra.metrics as Metric[] | undefined) ?? [];
  const a = f.extra.a as { version: string; label: string; missions: number; actions: number } | undefined;
  const b = f.extra.b as { version: string; label: string; missions: number; actions: number } | undefined;
  if (!a || !b) return <Card f={f} />;
  const rates = metrics.filter((m) => m.kind === "rate");
  const means = metrics.filter((m) => m.kind === "mean");
  const fmt = (m: Metric, v: number | null) => (v === null ? "n/a" : `${v.toFixed(v >= 10 ? 0 : 1)}${m.unit ? ` ${m.unit}` : ""}`);
  return (
    <Card f={f} note={`${a.label}: ${a.missions} missions, ${a.actions} actions. ${b.label}: ${b.missions} missions, ${b.actions} actions. Whiskers are 95% intervals. Tap a line for the rows.`}>
      <div>
        {rates.map((m) => (
          <button type="button" key={m.key} className={s.pair} onClick={() => open({ kind: "metric", id: m.key })}>
            <span className={s.pairHead}>
              <span>{m.label}</span>
              <Pill tone={VERDICT_TONE[m.verdict]}>
                {m.verdict}
                {m.diff !== null && m.verdict !== "not enough data" ? ` ${m.diff > 0 ? "+" : ""}${(100 * m.diff).toFixed(1)} pts` : ""}
              </Pill>
            </span>
            {([
              [a.version, m.a, "var(--v1)"],
              [b.version, m.b, "var(--v2)"],
            ] as const).map(([name, side, color]) => (
              <span key={name} className={s.pairLine}>
                <span>{name}</span>
                <Track rate={side.value} lo={side.lo} hi={side.hi} color={color} />
                <span>
                  <b>{pct(side.value, 1)}</b> ({side.k}/{side.n})
                </span>
              </span>
            ))}
          </button>
        ))}
      </div>
      <table className={s.means}>
        <thead>
          <tr>
            <th>Mean (min to max)</th>
            <th>{a.version}</th>
            <th>{b.version}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {means.map((m) => (
            <tr key={m.key}>
              <td>{m.label}</td>
              <td title={`n=${m.a.n}; ${fmt(m, m.a.lo)} to ${fmt(m, m.a.hi)}`}>
                {fmt(m, m.a.value)} <span className={s.muted}>n={m.a.n}</span>
              </td>
              <td title={`n=${m.b.n}; ${fmt(m, m.b.lo)} to ${fmt(m, m.b.hi)}`}>
                {fmt(m, m.b.value)} <span className={s.muted}>n={m.b.n}</span>
              </td>
              <td title={pValue(m.p_value)}>
                <Pill tone={VERDICT_TONE[m.verdict]}>{m.verdict}</Pill>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

/** Any finding whose rows are failure rates: bars with intervals and k/n. */
export function RateCard({ f, overall, limit, onRow }: { f: Finding; overall: number; limit?: number; onRow?: (b: Bar) => void }) {
  return (
    <Card f={f} note={limit && f.rows.length > limit ? `Top ${limit} of ${f.rows.length}; the rest are behind "Rows behind it".` : undefined}>
      <RateBars rows={f.rows} overall={overall} limit={limit} onRow={onRow} />
    </Card>
  );
}

/** Battery bins, split by version so the reader sees that only V1 ever ran low. */
export function BatteryCard({ f, overall }: { f: Finding; overall: number }) {
  return (
    <Card f={f}>
      <RateBars rows={f.rows} overall={overall} />
      <table className={s.means}>
        <thead>
          <tr>
            <th>Battery</th>
            {Object.keys((f.rows[0]?.extra.by_version as Record<string, unknown> | undefined) ?? {}).map((v) => (
              <th key={v}>{v} failed/n</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {f.rows.map((b) => (
            <tr key={b.key}>
              <td>{b.label}</td>
              {Object.entries((b.extra.by_version as Record<string, { k: number; n: number }> | undefined) ?? {}).map(([v, x]) => (
                <td key={v}>{x.n ? `${x.k}/${x.n}` : <span className={s.muted}>never ran here</span>}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

interface Point {
  mission_id: string;
  version: string;
  k: number;
  n: number;
  status: string;
  anomalies: number;
  end_battery: number;
}

/** Failed actions per mission in time order: the answer to "improving or degrading?" as a picture. */
export function OverTimeCard({ f }: { f: Finding }) {
  const { open } = useOpen();
  const points = (f.extra.points as Point[] | undefined) ?? [];
  const W = 520;
  const H = 150;
  const left = 26;
  const base = H - 26;
  const maxK = Math.max(3, ...points.map((p) => p.k));
  const step = (W - left - 6) / Math.max(points.length, 1);
  const versions = [...new Set(points.map((p) => p.version))];
  const change = points.findIndex((p) => p.version !== points[0]?.version);
  return (
    <Card f={f} note="One column per mission, oldest first. Height: failed actions out of 10. A hollow column is a mission marked FAILED; every filled column above zero was marked COMPLETED. Tap a column for the mission.">
      <svg className={s.chart} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Failed actions per mission over time">
        {[0, 1, 2, 3].filter((v) => v <= maxK).map((v) => (
          <g key={v}>
            <line x1={left} x2={W} y1={base - (v / maxK) * (base - 14)} y2={base - (v / maxK) * (base - 14)} stroke="var(--line)" strokeWidth="1" />
            <text x={left - 6} y={base - (v / maxK) * (base - 14) + 4} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {change > 0 ? (
          <g>
            <line x1={left + change * step - 1} x2={left + change * step - 1} y1="4" y2={base} stroke="var(--muted)" strokeDasharray="3 3" />
            <text x={left + change * step + 4} y="12">
              {versions[1]} starts
            </text>
            <text x={left + 2} y="12">
              {versions[0]}
            </text>
          </g>
        ) : null}
        {points.map((p, i) => {
          const h = Math.max(p.k === 0 ? 2 : 0, (p.k / maxK) * (base - 14));
          const color = p.version === versions[0] ? "var(--v1)" : "var(--v2)";
          const failed = p.status !== "COMPLETED";
          return (
            <g key={p.mission_id} onClick={() => open({ kind: "mission", id: p.mission_id })} style={{ cursor: "pointer" }}>
              <title>{`${p.mission_id} (${p.version}): ${p.k} of ${p.n} actions failed, marked ${p.status}, ${p.anomalies} anomalies, ended at ${p.end_battery}% battery`}</title>
              <rect x={left + i * step} y="4" width={step} height={base} fill="transparent" />
              <rect x={left + i * step + 2} y={base - h} width={step - 4} height={h} rx="2" fill={failed ? "none" : color} stroke={failed ? "var(--bad)" : "none"} strokeWidth="2" />
              {i % 3 === 0 ? (
                <text x={left + i * step + step / 2} y={H - 8} textAnchor="middle">
                  {p.mission_id.replace("M0", "M")}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </Card>
  );
}
