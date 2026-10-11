"use client";

import { useEffect, useRef, useState } from "react";
import { chipShort } from "@/lib/lab";
import { type BenchPoint, fmt, shortRef, when } from "@/lib/labBench";
import b from "./bench.module.css";

export type NumberKind = "measured" | "derived" | "modelled" | "sample";
const TAG: Record<NumberKind, [string, string]> = {
  measured: ["Measured", b.tagMeasured!],
  derived: ["Derived", b.tagDerived!],
  modelled: ["Modelled", b.tagModelled!],
  sample: ["Sample", b.tagSample!],
};

/** Which of the four kinds a number is. The word is the label; the shape of the tag differs too, not only its tone. */
export function Tag({ kind, children }: { kind: NumberKind; children?: React.ReactNode }) {
  const [word, cls] = TAG[kind];
  return <span className={`${b.tag} ${cls}`}>{children ?? word}</span>;
}

/** The four labels side by side, so a reader learns them once. */
export function LabelKey({ kinds }: { kinds: NumberKind[] }) {
  const meaning: Record<NumberKind, string> = {
    measured: "a row the lab recorded",
    derived: "worked out from rows, formula shown",
    modelled: "what a model of those rows predicts",
    sample: "made-up example data",
  };
  return (
    <div className={b.key} aria-label="What the labels mean">
      {kinds.map((k) => (
        <span key={k}>
          <Tag kind={k} /> {meaning[k]}
        </span>
      ))}
    </div>
  );
}

/** Everything the lab recorded about one number: the row, when, how it was read and why it may not be trusted yet. */
export function PointDetail({ p, onClose }: { p: BenchPoint; onClose?: () => void }) {
  const spread = p.spread && p.spread.min !== undefined && p.spread.max !== undefined ? `${fmt(p.spread.min)} to ${fmt(p.spread.max)} ${p.unit ?? ""}` : null;
  const sampler = p.sampler ? [p.sampler.sampler, p.sampler.tool, p.sampler.period_ms ? `every ${p.sampler.period_ms} ms` : null, p.sampler.samples ? `${p.sampler.samples} samples` : null].filter(Boolean).join(", ") : null;
  return (
    <div className={b.detail} role="status">
      <div className={b.detailTop}>
        <span>
          {p.label}
          {p.x ? ` at batch ${p.x}` : ""} <Tag kind={p.kind} />
          {p.low_confidence ? <span className={b.low}> low confidence</span> : null}
        </span>
        <b>
          {fmt(p.value)} {p.unit}
        </b>
      </div>
      <dl>
        {p.low_confidence && p.reason ? (
          <>
            <dt>Why low confidence</dt>
            <dd>{p.reason}</dd>
          </>
        ) : null}
        {p.formula ? (
          <>
            <dt>Formula</dt>
            <dd>{p.formula}</dd>
          </>
        ) : null}
        <dt>Source</dt>
        <dd className={b.code}>{p.kind === "derived" ? (p.inputs ?? []).map(shortRef).join(", ") || shortRef(p.source.ref) : shortRef(p.source.ref)}</dd>
        {p.recorded_by ? (
          <>
            <dt>Recorded by</dt>
            <dd>{p.recorded_by}</dd>
          </>
        ) : null}
        {p.runs ? (
          <>
            <dt>Runs</dt>
            <dd>
              {p.statistic ?? `${p.runs} runs`}
              {spread ? `, from ${spread}` : ""}
            </dd>
          </>
        ) : null}
        {sampler ? (
          <>
            <dt>Sampler</dt>
            <dd>{sampler}</dd>
          </>
        ) : null}
        {p.sampler?.error ? (
          <>
            <dt>Error</dt>
            <dd>{p.sampler.error}</dd>
          </>
        ) : null}
        {p.state && !p.x ? (
          <>
            <dt>While</dt>
            <dd>{p.state}</dd>
          </>
        ) : null}
        {p.chip && p.chip.key !== "jetson" ? (
          <>
            <dt>Chip</dt>
            <dd>
              {chipShort(p.chip)}
              {p.chip.current ? "" : ", not the current lab"}
              {p.chip.reason ? `: ${p.chip.reason}` : ""}
            </dd>
          </>
        ) : null}
        {p.ts ? (
          <>
            <dt>Measured</dt>
            <dd>
              {when(p.ts)}
              {p.node ? ` on ${p.node}` : ""}
            </dd>
          </>
        ) : null}
      </dl>
      {onClose ? (
        <button type="button" className={b.numBtn} onClick={onClose}>
          Hide
        </button>
      ) : null}
    </div>
  );
}

/** The rendered width of an element, so SVGs draw at real pixels and their labels stay readable on a phone. */
export function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
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

/** True when the system, or `?motion=reduce`, asks for no animation. */
export function useReduced(force: boolean): boolean {
  const [reduced, setReduced] = useState(force);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(force || mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [force]);
  return reduced;
}

export function LockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </svg>
  );
}
