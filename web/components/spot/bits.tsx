"use client";

import { createContext, useContext, type ReactNode } from "react";
import s from "./spot.module.css";
import type { Bar, Evidence, Target } from "./types";

/** Opens the evidence panel for a mission, asset, waypoint or finding. Provided by the page. */
export interface Opener {
  open: (t: Target) => void;
  /** Which panel an id from the tables belongs to (an action opens its mission, an anomaly its asset). */
  resolve: (id: string) => Target | null;
}

export const OpenCtx = createContext<Opener>({ open: () => undefined, resolve: () => null });
export const useOpen = () => useContext(OpenCtx);

export const words = (v: string | null | undefined) => (v ?? "").replace(/_/g, " ").toLowerCase();
export const pct = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(digits)}%`);
export const range = (lo: number | null, hi: number | null) => (lo === null || hi === null ? "" : `${(100 * lo).toFixed(0)} to ${(100 * hi).toFixed(0)}%`);
export const pValue = (p: number | null | undefined) => (p === null || p === undefined ? "" : p < 0.001 ? "p<0.001" : `p=${p.toFixed(p < 0.1 ? 3 : 2)}`);

/** Kit timestamps have no zone: show them as written. */
export function stamp(iso: string | null | undefined, withDay = true): string {
  if (!iso) return "";
  const [day = "", time = ""] = iso.split("T");
  const [, m, d] = day.split("-");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const label = `${months[Number(m) - 1] ?? m} ${Number(d)}`;
  return withDay ? `${label}, ${time.slice(0, 5)}` : time.slice(0, 5);
}

export function Sev({ level }: { level: string | null | undefined }) {
  if (!level) return null;
  const cls = { CRITICAL: s.sevCRITICAL, HIGH: s.sevHIGH, MEDIUM: s.sevMEDIUM, LOW: s.sevLOW }[level] ?? s.toneMuted;
  return <span className={`${s.pill} ${cls}`}>{level}</span>;
}

export function Pill({ tone, children }: { tone: "ok" | "bad" | "watch" | "muted"; children: ReactNode }) {
  const cls = { ok: s.toneOk, bad: s.toneBad, watch: s.toneWatch, muted: s.toneMuted }[tone];
  return <span className={`${s.pill} ${cls}`}>{children}</span>;
}

/** One id from the tables; a button when there is a panel behind it. */
export function Id({ id }: { id: string }) {
  const { open, resolve } = useOpen();
  const target = resolve(id);
  if (!target) return <span className={s.id}>{id}</span>;
  return (
    <button type="button" className={s.id} onClick={(e) => (e.stopPropagation(), open(target))} title={`Open ${target.kind} ${target.id}`}>
      {id}
    </button>
  );
}

export function Ids({ ids, max = 14, label }: { ids: string[]; max?: number; label?: string }) {
  if (ids.length === 0) return null;
  return (
    <span className={s.ids}>
      {label ? <span>{label}</span> : null}
      {ids.slice(0, max).map((id) => (
        <Id key={id} id={id} />
      ))}
      {ids.length > max ? <span>and {ids.length - max} more</span> : null}
    </span>
  );
}

export function evidenceIds(e: Evidence | undefined | null): string[] {
  if (!e) return [];
  return [...e.assets, ...e.locations, ...e.missions, ...e.anomalies, ...e.actions, ...e.readings];
}

export function EvidenceLine({ evidence, max = 12 }: { evidence: Evidence | undefined | null; max?: number }) {
  const ids = evidenceIds(evidence);
  if (ids.length === 0) return null;
  return <Ids ids={ids} max={max} label={`Evidence (${ids.length} rows):`} />;
}

/** A horizontal bar with its 95% interval as a whisker. Scale is 0 to `max` (a share, 0..1). */
export function Track({ rate, lo, hi, max = 1, color, faded }: { rate: number | null; lo: number | null; hi: number | null; max?: number; color: string; faded?: boolean }) {
  const x = (v: number) => Math.max(0, Math.min(100, (100 * v) / max));
  return (
    <svg className={s.barTrack} viewBox="0 0 100 16" preserveAspectRatio="none" aria-hidden="true">
      <rect x="0" y="3" width="100" height="10" rx="2" fill="var(--bg)" />
      {rate !== null ? <rect x="0" y="3" width={x(rate)} height="10" rx="2" fill={color} opacity={faded ? 0.35 : 1} /> : null}
      {lo !== null && hi !== null ? (
        <g stroke="var(--text)" strokeWidth="1.4" vectorEffect="non-scaling-stroke" opacity={faded ? 0.5 : 0.9}>
          <line x1={x(lo)} x2={x(hi)} y1="8" y2="8" vectorEffect="non-scaling-stroke" />
          <line x1={x(lo)} x2={x(lo)} y1="4.5" y2="11.5" vectorEffect="non-scaling-stroke" />
          <line x1={x(hi)} x2={x(hi)} y1="4.5" y2="11.5" vectorEffect="non-scaling-stroke" />
        </g>
      ) : null}
    </svg>
  );
}

export function rateColor(rate: number | null, overall: number): string {
  if (rate === null) return "var(--unknown)";
  return rate > overall * 1.5 ? "var(--bad)" : rate > overall ? "var(--watch)" : "var(--ok)";
}

/** Failure-rate bars with k/n and intervals. Rows under the minimum n are faded and say so. */
export function RateBars({ rows, overall, max, onRow, limit }: { rows: Bar[]; overall: number; max?: number; onRow?: (b: Bar) => void; limit?: number }) {
  const shown = limit ? rows.slice(0, limit) : rows;
  const top = max ?? Math.min(1, Math.max(0.4, ...shown.map((b) => b.hi ?? b.rate ?? 0)));
  if (shown.length === 0) return null;
  return (
    <div className={s.rows}>
      {shown.map((b) => {
        const inner = (
          <>
            <span className={s.barLabel} title={b.label}>
              {b.label}
            </span>
            <Track rate={b.rate} lo={b.lo} hi={b.hi} max={top} color={b.enough ? rateColor(b.rate, overall) : "var(--unknown)"} faded={!b.enough} />
            <span className={s.barVal}>
              <b>{b.enough ? pct(b.rate) : "n/a"}</b> {b.k}/{b.n}
            </span>
          </>
        );
        return onRow ? (
          <button type="button" key={b.key} className={s.barRow} onClick={() => onRow(b)} title={b.note ?? `${b.k} of ${b.n} failed; 95% interval ${range(b.lo, b.hi)}`}>
            {inner}
          </button>
        ) : (
          <div key={b.key} className={s.barRow} title={b.note ?? `95% interval ${range(b.lo, b.hi)}`}>
            {inner}
          </div>
        );
      })}
      <div className={s.axis}>
        <div>failure rate, with 95% interval</div>
        <div>
          <span>0%</span>
          <span>{pct(top)}</span>
        </div>
        <div>failed/n</div>
      </div>
    </div>
  );
}
