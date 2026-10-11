"use client";

import { useEffect, useId, useState } from "react";
import { GLOSSARY, type HeroData, type Source, type Takeaway as TakeawayData, type Tone } from "@/lib/labWords";
import s from "./story.module.css";

/**
 * The Learning lab's shared grammar, the one Today's Plan and the Calendar screen use: a tab opens on one sentence
 * that answers its question, every number it stands on is one tap away behind a small source chip, a term of the
 * lab's is said in plain words with the precise term on tap, and raw ids and file lines sit in a fold.
 *
 * The chip is the idea of the one on Today's Plan (components/dayrun/PlanDetails.tsx SourceChip), drawn here with the
 * same tokens; nothing is imported from that screen.
 */

const TONE: Record<Tone, string> = { good: s.good!, bad: s.bad!, watch: s.watch!, plain: "" };

/** The numbers a sentence stands on, each with where it was read. Shown when the chip is open. */
export function SourceList({ sources }: { sources: Source[] }) {
  return (
    <dl className={s.sources}>
      {sources.map((x, i) => (
        <div key={`${x.label}-${i}`} className={s.sourceRow}>
          <dt>{x.label}</dt>
          <dd>
            {x.value ? <b>{x.value}</b> : null}
            <span className={s.from}>{x.from}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** "Where this comes from": a small chip that opens the sources of the sentence beside it. */
export function SourceChip({ sources, about, label = "Where this comes from" }: { sources: Source[]; about: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  if (sources.length === 0) return null;
  return (
    <>
      <button type="button" className={s.src} aria-expanded={open} aria-controls={id} aria-label={`${about}: ${open ? "hide" : "show"} where it comes from`} onClick={() => setOpen((v) => !v)}>
        {label}
      </button>
      {open ? (
        <div id={id} className={s.srcBody}>
          <SourceList sources={sources} />
        </div>
      ) : null}
    </>
  );
}

/**
 * The top of a tab: a small label that says what question this answers, the sentence that answers it, at most one
 * more sentence, and the chip. `guide` is the look guide's hook (lib/guide.ts), so Atlas can point at it.
 */
export function Takeaway({ label, take, guide, children }: { label: string; take: TakeawayData; guide: string; children?: React.ReactNode }) {
  return (
    <section className={`${s.take} ${TONE[take.tone]}`} data-guide={guide} aria-label={label}>
      <span className={s.takeLabel}>{label}</span>
      <p className={s.takeLine}>{take.line}</p>
      {take.sub ? <p className={s.takeSub}>{take.sub}</p> : null}
      {children}
      <div className={s.takeFoot}>
        <SourceChip sources={take.sources} about={label} />
      </div>
    </section>
  );
}

/** A number that counts up to its value once, when it first shows. With reduced motion it is simply there. */
export function CountUp({ to, digits = 0 }: { to: number; digits?: number }) {
  const [v, setV] = useState(to); // the static page and a reader with motion off see the final number
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(window.location.search).get("motion") === "reduce") return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const u = Math.min(1, (now - t0) / 900);
      setV(to * (1 - (1 - u) ** 3));
      if (u < 1) raf = window.requestAnimationFrame(tick);
    };
    setV(0);
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [to]);
  return <>{v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}</>;
}

/** One piece of a hero's big line: a number that counts up, or plain text between numbers ("→", "of 228", "×"). */
export type Big = { n: number; digits?: number } | string;

/**
 * The top of a tab, as one slide: a small label, one big line with its number counting in, a statement of at most a
 * line, two or three tiles, and the chip that says where every one of those numbers was read. Longer words live
 * behind a tap, never here.
 */
export function Hero({ label, hero, guide, children }: { label: string; hero: HeroData; guide: string; children?: React.ReactNode }) {
  return (
    <section className={`${s.take} ${s.hero} ${TONE[hero.tone]}`} data-guide={guide} aria-label={label}>
      <span className={s.takeLabel}>{label}</span>
      {hero.big.length ? (
        <p className={s.heroBig} aria-label={hero.big.map((b) => (typeof b === "string" ? b : b.n.toLocaleString(undefined, { maximumFractionDigits: b.digits ?? 0 }))).join(" ")}>
          {hero.big.map((b, i) => (typeof b === "string" ? <span key={i} className={s.heroWord}>{b}</span> : <CountUp key={i} to={b.n} digits={b.digits} />))}
        </p>
      ) : null}
      <p className={s.heroLine}>{hero.line}</p>
      {hero.tiles.length ? (
        <div className={s.tiles}>
          {hero.tiles.map((t) => (
            <div key={t.label} className={s.tile}>
              <b>{t.value}</b>
              <span>{t.label}</span>
            </div>
          ))}
        </div>
      ) : null}
      {children}
      <div className={s.takeFoot}>
        <SourceChip sources={hero.sources} about={label} />
      </div>
    </section>
  );
}

/** One sentence about a state (running, passed, refused, older than the box's model, ...), with its tone as an edge. */
export function StateLine({ tone = "plain", children, guide }: { tone?: Tone; children: React.ReactNode; guide?: string }) {
  return (
    <p className={`${s.state} ${TONE[tone]}`} role="status" data-guide={guide}>
      {children}
    </p>
  );
}

/**
 * A word of the lab's, said plainly, with the precise term and what it means on tap. `k` is a key of the glossary
 * (lib/labWords.ts GLOSSARY); `children` overrides the words shown when a sentence needs another form.
 */
export function Term({ k, children }: { k: keyof typeof GLOSSARY | string; children?: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const w = GLOSSARY[k];
  if (!w) return <>{children}</>;
  return (
    <span className={s.termWrap}>
      <button type="button" className={s.term} aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)}>
        {children ?? w.plain}
      </button>
      {open ? (
        <span id={id} className={s.termBody} role="note">
          <b>{w.term}.</b> {w.means}
        </span>
      ) : null}
    </span>
  );
}

/** Raw ids, file paths and line references: kept, but folded away under a plain summary. */
export function Fold({ summary = "Where this comes from", children, guide }: { summary?: string; children: React.ReactNode; guide?: string }) {
  return (
    <details className={s.fold} data-guide={guide}>
      <summary>{summary}</summary>
      <div className={s.foldBody}>{children}</div>
    </details>
  );
}

/** What a chart shows, in words: under every chart, for a reader who cannot see it and for the look guide to read. */
export function ChartWords({ children, guide }: { children: React.ReactNode; guide?: string }) {
  return (
    <p className={s.chartWords} data-guide={guide}>
      {children}
    </p>
  );
}

/** The words this tab uses, each with the lab's term: one fold at the foot of a tab. */
export function Words({ keys }: { keys: (keyof typeof GLOSSARY | string)[] }) {
  return (
    <Fold summary="Words used here" guide="lab:words">
      <dl className={s.words}>
        {keys.map((k) => {
          const w = GLOSSARY[k];
          return w ? (
            <div key={k}>
              <dt>
                {w.plain}
                {w.term !== w.plain ? <span className={s.from}> ({w.term})</span> : null}
              </dt>
              <dd>{w.means}</dd>
            </div>
          ) : null;
        })}
      </dl>
    </Fold>
  );
}
