"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import type { LeagueView } from "@/lib/api";
import { humanize, percent } from "@/lib/format";
import type { ImpactFlowItem, ImpactStory, Range } from "@/lib/impact";
import im from "./impact.module.css";

/**
 * The week as one short story, played from the numbers in GET /api/impact/story (and GET /api/league for the
 * last part). Four small panels fill in turn: where each item went, the diversion ring against the goal, what
 * that kept out as one estimated range, and this table against the friends' tables. Nothing here is a constant
 * count: an empty week draws empty bins.
 *
 * One number drives everything: `u`, 0 to 1. At rest it is 1 and every panel shows its end state, which is
 * also what reduced motion shows (the button then steps panel by panel instead of animating).
 */

export const STORY_SECONDS = 15;
/** Where each part starts and ends on `u`. */
const PARTS = [
  { key: "bins", from: 0, to: 0.5 },
  { key: "ring", from: 0.5, to: 0.66 },
  { key: "kept", from: 0.66, to: 0.82 },
  { key: "bet", from: 0.82, to: 1 },
] as const;
type PartKey = (typeof PARTS)[number]["key"];

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));
const within = (u: number, key: PartKey) => {
  const p = PARTS.find((x) => x.key === key)!;
  return clamp01((u - p.from) / (p.to - p.from));
};
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Two significant figures: a range is not a place for false precision. */
function tidy(n: number): string {
  if (n === 0) return "0";
  const p = Math.pow(10, Math.floor(Math.log10(Math.abs(n))) - 1);
  const r = Math.round(n / p) * p;
  return r >= 100 ? String(Math.round(r)) : String(Math.round(r * 100) / 100);
}

export function rangeText(r: Range): string {
  const a = tidy(r.low);
  const b = tidy(r.high);
  return a === b ? a : `${a}–${b}`;
}

/** The friends' tables this week, from the League's team waste block. Null when there is nobody to compare with. */
export function bet(league: LeagueView | undefined) {
  const t = league?.team_waste;
  if (!league || !t) return null;
  const rows = t.members.filter((m) => m.diversion_rate !== null && m.recyclable > 0);
  const me = rows.find((m) => m.member_id === league.member_id);
  const others = rows.filter((m) => m.member_id !== league.member_id).sort((a, b) => (b.diversion_rate ?? 0) - (a.diversion_rate ?? 0));
  if (!me || others.length === 0) return null;
  const top = others[0]!;
  const name = (m: { member_id: string; display_name: string }) => (m.display_name === m.member_id ? humanize(m.member_id) : m.display_name);
  const mine = me.diversion_rate ?? 0;
  const theirs = top.diversion_rate ?? 0;
  const lead = mine > theirs ? "you're ahead on sorting this week" : mine < theirs ? `${name(top)} is ahead on sorting this week` : "level on sorting this week";
  const counts = league.targets.some((x) => x.source === "impact");
  return {
    me: { name: "Your table", rate: mine, recycled: me.recycled, recyclable: me.recyclable },
    others: others.map((m) => ({ name: `${name(m)}'s`, rate: m.diversion_rate ?? 0, recycled: m.recycled, recyclable: m.recyclable })),
    target: t.target,
    line: `Your table ${percent(mine)} · ${name(top)}'s ${percent(theirs)}: ${lead}.`,
    /** The stake is the League's own text for the week; recycling is named as part of it only when it is a League target. */
    stake: league.stake ? `${counts ? "Recycling is one of the things the League's stake is scored on" : "The League's stake"}: ${league.stake.text_for_week}` : null,
  };
}

/* ---------- part 1: the bins ---------- */

/** A right item takes one beat; a wrong one takes five, so its rule can be read. One and a half beats close the part. */
const WRONG_BEATS = 5;
const DROP_BEATS = 1.6;

/** When each item starts falling, in beats, and how many beats the part lasts. */
function schedule(items: ImpactFlowItem[]): { starts: number[]; total: number } {
  const starts: number[] = [];
  let at = 0;
  for (const it of items) {
    starts.push(at);
    at += it.correct ? 1 : WRONG_BEATS;
  }
  return { starts, total: at + 1.5 };
}

/** The item on stage at `p`: the last one that has started. Undefined once the part is closing. */
function onStage(items: ImpactFlowItem[], p: number): ImpactFlowItem | undefined {
  const { starts, total } = schedule(items);
  const beat = p * total;
  if (beat >= total - 1.5) return undefined;
  let k = -1;
  starts.forEach((st, i) => {
    if (st <= beat) k = i;
  });
  return k >= 0 ? items[k] : undefined;
}

const BIN_W = 62;
const BIN_TOP = 44;
const BIN_H = 62;

function Bins({ items, bins, p }: { items: ImpactFlowItem[]; bins: string[]; p: number }) {
  const two = bins.slice(0, 2);
  const per = Math.max(1, ...two.map((b) => items.filter((i) => i.bin === b).length));
  const cols = per > 24 ? 6 : per > 12 ? 5 : 4;
  const r = per > 24 ? 3.4 : per > 12 ? 4.2 : 5.4;
  const step = BIN_W / (cols + 0.6);
  const n = items.length;
  const { starts, total } = schedule(items);
  const beat = p * total;
  const landed = (k: number) => beat >= starts[k]! + DROP_BEATS * 0.6;
  const placed: Record<string, number> = {};
  return (
    <svg viewBox="0 0 160 132" className={im.stage} role="img" aria-label={`${plural(n, "thing")} went into the bins this week`}>
      {two.map((b, bi) => {
        const x = 12 + bi * (BIN_W + 12);
        const right = items.filter((i, k) => i.bin === b && i.correct && landed(k)).length;
        const wrong = items.filter((i, k) => i.bin === b && !i.correct && landed(k)).length;
        return (
          <g key={b}>
            <path d={`M${x} ${BIN_TOP} v${BIN_H - 10} a10 10 0 0 0 10 10 h${BIN_W - 20} a10 10 0 0 0 10 -10 v-${BIN_H - 10}`} className={im.bin} />
            <text x={x + BIN_W / 2} y={BIN_TOP + BIN_H + 13} textAnchor="middle" className={im.binLabel}>
              {humanize(b)}
            </text>
            <text x={x + BIN_W / 2} y={BIN_TOP + BIN_H + 24} textAnchor="middle" className={im.binCount}>
              {right} right{wrong ? ` · ${wrong} wrong` : ""}
            </text>
          </g>
        );
      })}
      {items.map((it, k) => {
        const bi = two.indexOf(it.bin);
        if (bi < 0) return null;
        const seat = placed[it.bin] ?? 0;
        placed[it.bin] = seat + 1;
        const t = clamp01((beat - starts[k]!) / DROP_BEATS);
        if (t <= 0) return null;
        const x = 12 + bi * (BIN_W + 12) + step * (0.8 + (seat % cols));
        const rest = BIN_TOP + BIN_H - 8 - r - Math.floor(seat / cols) * (r * 2 + 2);
        // a right one drops and settles; a wrong one hits, bounces and comes to rest
        const fall = clamp01(t / 0.6);
        const bounce = it.correct ? 0 : Math.sin(clamp01((t - 0.6) / 0.4) * Math.PI) * 16;
        const y = 8 + (rest - 8) * (fall * fall) - bounce;
        return it.correct ? (
          <circle key={it.item_id} cx={x} cy={y} r={r} className={im.dotRight} />
        ) : (
          <g key={it.item_id}>
            <circle cx={x} cy={y} r={r} className={im.dotWrong} />
            <path d={`M${x - r * 0.45} ${y - r * 0.45} l${r * 0.9} ${r * 0.9} M${x + r * 0.45} ${y - r * 0.45} l-${r * 0.9} ${r * 0.9}`} className={im.dotCross} />
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- part 2: the ring ---------- */

function Ring({ value, target, p }: { value: number | null; target: number; p: number }) {
  const R = 44;
  const C = 2 * Math.PI * R;
  const shown = (value ?? 0) * easeOut(p);
  const met = value !== null && value >= target;
  const a = target * 2 * Math.PI - Math.PI / 2;
  return (
    <svg
      viewBox="0 0 160 132"
      className={im.stage}
      role="img"
      aria-label={value === null ? "No recyclables seen yet this week" : `${percent(value)} of recyclables reached recycling; the goal is ${percent(target)}`}
    >
      <circle cx="80" cy="62" r={R} className={im.ringTrack} />
      {value !== null && shown > 0.004 ? (
        <circle cx="80" cy="62" r={R} className={`${im.ringFill} ${met ? im.ringMet : ""}`} strokeDasharray={`${C * shown} ${C}`} transform="rotate(-90 80 62)" />
      ) : null}
      <line x1={80 + Math.cos(a) * (R - 9)} y1={62 + Math.sin(a) * (R - 9)} x2={80 + Math.cos(a) * (R + 9)} y2={62 + Math.sin(a) * (R + 9)} className={im.ringGoal} />
      <text x={80 + Math.cos(a) * (R + 20)} y={62 + Math.sin(a) * (R + 20) + 3} textAnchor="middle" className={im.binCount}>
        goal
      </text>
      <text x="80" y="69" textAnchor="middle" className={im.ringNum}>
        {value === null ? "–" : percent(shown)}
      </text>
      <text x="80" y="126" textAnchor="middle" className={im.binLabel}>
        goal {percent(target)}
      </text>
    </svg>
  );
}

/* ---------- part 3: what that kept out, as a range ---------- */

function Kept({ charges, p }: { charges: Range; p: number }) {
  // the axis ends a little past the high end; the band is the range, never a single mark
  const max = Math.max(1, charges.high * 1.25);
  const x = (n: number) => 14 + (n / max) * 132;
  const grow = easeOut(p);
  const lo = x(charges.low);
  const hi = lo + (x(charges.high) - lo) * grow;
  return (
    <svg viewBox="0 0 160 132" className={im.stage} role="img" aria-label={`An estimate: the energy of ${rangeText(charges)} phone charges`}>
      <text x="80" y="52" textAnchor="middle" className={im.keptNum} opacity={clamp01(p * 3)}>
        {rangeText(charges)}
      </text>
      <text x="80" y="68" textAnchor="middle" className={im.binLabel} opacity={clamp01(p * 3)}>
        phone charges of energy
      </text>
      <line x1="14" y1="98" x2="146" y2="98" className={im.axis} />
      <rect x={lo} y="86" width={Math.max(0, hi - lo)} height="24" rx="6" className={im.band} />
      <text x="14" y="124" className={im.binCount}>
        0
      </text>
      {p > 0.5 ? (
        <>
          <text x={lo} y="124" textAnchor="middle" className={im.binCount}>
            low
          </text>
          <text x={x(charges.high)} y="124" textAnchor="middle" className={im.binCount}>
            high
          </text>
        </>
      ) : null}
    </svg>
  );
}

/* ---------- part 4: the friends' bet, or the goal line when there is no one to compare with ---------- */

function Bet({ b, p }: { b: NonNullable<ReturnType<typeof bet>>; p: number }) {
  const rows = [b.me, ...b.others].slice(0, 3);
  const grow = easeOut(p);
  const h = rows.length > 2 ? 30 : 40;
  const top = rows.length > 2 ? 14 : 20;
  const gx = 14 + b.target * 132;
  return (
    <svg viewBox="0 0 160 132" className={im.stage} role="img" aria-label={b.line}>
      {rows.map((r, i) => {
        const y = top + i * h;
        return (
          <g key={r.name}>
            <text x="14" y={y + 9} className={i === 0 ? im.betMe : im.binLabel}>
              {r.name} {percent(r.rate * grow)}
            </text>
            <rect x="14" y={y + 14} width="132" height="10" rx="5" className={im.betTrack} />
            <rect x="14" y={y + 14} width={132 * r.rate * grow} height="10" rx="5" className={i === 0 ? im.betFillMe : im.betFill} />
          </g>
        );
      })}
      <line x1={gx} y1={top + 11} x2={gx} y2="113" className={im.ringGoal} />
      <text x={gx} y="126" textAnchor="middle" className={im.binLabel}>
        goal {percent(b.target)}
      </text>
    </svg>
  );
}

function GoalLine({ toGo, target, met, p }: { toGo: number | null; target: number; met: boolean; p: number }) {
  const n = Math.min(toGo ?? 0, 10);
  const shown = Math.round(n * easeOut(p));
  return (
    <svg viewBox="0 0 160 132" className={im.stage} role="img" aria-label={met ? "The goal is met" : toGo === null ? "No recyclables yet" : `${toGo} more in the right bin reaches the goal`}>
      <text x="80" y="52" textAnchor="middle" className={im.keptNum}>
        {met ? "met" : toGo === null ? "–" : toGo}
      </text>
      <text x="80" y="68" textAnchor="middle" className={im.binLabel}>
        {met ? `goal ${percent(target)}` : "more right in a row"}
      </text>
      {Array.from({ length: n }, (_, i) => (
        <circle key={i} cx={80 - (n - 1) * 7 + i * 14} cy="96" r="5" className={i < shown ? im.dotNext : im.dotGhost} />
      ))}
      <text x="80" y="126" textAnchor="middle" className={im.binLabel}>
        reaches {percent(target)}
      </text>
    </svg>
  );
}

/* ---------- the story ---------- */

export function WeekStory({ v, league, initial, forceReduced, onWhy, onHow }: { v: ImpactStory; league: LeagueView | undefined; initial: number | null; forceReduced: boolean; onWhy: () => void; onHow: () => void }) {
  const [u, setU] = useState(initial ?? 1);
  const [playing, setPlaying] = useState(false);
  const [reduced, setReduced] = useState(forceReduced);
  const uRef = useRef(u);
  uRef.current = u;

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(forceReduced || mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [forceReduced]);

  // The clock: one animation frame at a time. With reduced motion nothing animates; the button steps part by part.
  useEffect(() => {
    if (!playing || reduced) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const next = clamp01(uRef.current + (now - last) / 1000 / STORY_SECONDS);
      last = now;
      setU(next);
      if (next >= 1) setPlaying(false);
      else raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [playing, reduced]);

  const box = useRef<HTMLDivElement>(null);
  const play = useCallback(() => {
    box.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    if (reduced) {
      setU((x) => (x >= 1 ? PARTS[0].to : (PARTS.find((p) => p.to > x + 1e-6) ?? PARTS[PARTS.length - 1]!).to));
      return;
    }
    setPlaying((on) => {
      if (!on && uRef.current >= 1) setU(0);
      return !on;
    });
  }, [reduced]);

  const flow = v.flow;
  const items = flow?.items ?? [];
  const g = v.goal.diversion;
  const eq = v.equivalents && v.equivalents.items > 0 ? v.equivalents : null;
  const b = bet(league);
  const part = PARTS.find((p) => u < p.to) ?? PARTS[PARTS.length - 1]!;
  const moving = playing || (u > 0 && u < 1);

  // One caption, from the same numbers the panels draw. While the bins fill it follows the item that is landing.
  const right = items.filter((i) => i.correct).length;
  const binsDone = items.length === 0 ? "Nothing has gone into the bins yet this week." : `${plural(items.length, "thing")} went into the bins: ${right} right, ${items.length - right} wrong.`;
  const landing = onStage(items, within(u, "bins"));
  const missed = [flow?.recyclable_in_trash ? `${flow.recyclable_in_trash} went in the trash` : "", flow?.not_binned ? `${flow.not_binned} never reached a bin` : ""].filter(Boolean).join("; ");
  const ringLine =
    g.value === null
      ? "No recyclables seen yet this week, so there is no rate to show."
      : `${g.recycled} of ${g.recyclable} recyclables reached recycling: ${percent(g.value)}, goal ${percent(g.target)}.${missed ? ` ${missed[0]!.toUpperCase()}${missed.slice(1)}.` : ""}`;
  const keptLine = eq
    ? `Those ${plural(eq.items, "item")} weigh about ${rangeText(eq.kept_out_g)} g; recycling them saves the energy of ${rangeText(eq.phone_charges)} phone charges. An estimate from EPA factors, shown as a range.`
    : "Nothing recycled yet this week, so there is nothing to estimate.";
  const goalLine = g.met ? "The goal is met this week." : g.to_go ? `${plural(g.to_go, "more item")} in the right bin, with none missed, reaches ${percent(g.target)}.` : "The first recyclable in the right bin starts the week.";
  const endLine = b ? b.line : goalLine;
  const caption =
    part.key === "bins" && u < PARTS[0].to
      ? landing
        ? landing.correct
          ? `${humanize(landing.item)} → ${landing.bin}. Right bin.`
          : `${humanize(landing.item)} → ${landing.bin}. Wrong bin: ${landing.rule}`
        : binsDone
      : part.key === "ring"
        ? ringLine
        : part.key === "kept"
          ? keptLine
          : endLine;

  const cls = (key: PartKey) => `${im.panel} ${moving && part.key === key ? im.panelNow : ""} ${moving && u < PARTS.find((p) => p.key === key)!.from ? im.panelWaiting : ""}`;

  return (
    <div className={im.story} ref={box}>
      <div className={im.storyActions}>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={play}>
          {reduced ? "Step through" : playing ? "Pause" : u >= 1 ? "See how" : "Resume"}
        </button>
        <button type="button" className={im.linkBtn} onClick={onWhy}>
          Why it matters
        </button>
        {eq ? (
          <button type="button" className={im.linkBtn} onClick={onHow}>
            How we got this number
          </button>
        ) : null}
      </div>
      <div className={im.panels}>
        <figure className={cls("bins")}>
          <figcaption>
            Where it went <span className={im.measured}>measured</span>
          </figcaption>
          <Bins items={items} bins={v.rules.bins} p={within(u, "bins")} />
        </figure>
        <figure className={cls("ring")}>
          <figcaption>
            Recycled <span className={im.measured}>measured</span>
          </figcaption>
          <Ring value={g.value} target={g.target} p={within(u, "ring")} />
        </figure>
        <figure className={`${cls("kept")} ${im.panelEstimate}`}>
          <figcaption>
            Kept out <span className={im.estimate}>estimate</span>
          </figcaption>
          {eq ? <Kept charges={eq.phone_charges} p={within(u, "kept")} /> : <p className={`${im.panelEmpty} ${ui.muted}`}>Nothing to estimate yet.</p>}
        </figure>
        <figure className={cls("bet")}>
          <figcaption>
            {b ? "The friends' bet" : "To the goal"} <span className={im.measured}>measured</span>
          </figcaption>
          {b ? <Bet b={b} p={within(u, "bet")} /> : <GoalLine toGo={g.to_go} target={g.target} met={g.met} p={within(u, "bet")} />}
        </figure>
      </div>

      <p className={im.caption} aria-live="polite">
        {caption}
      </p>
      {u >= 1 && b ? (
        <p className={`${ui.small} ${ui.muted}`}>
          {b.stake ? `${b.stake} ` : ""}
          {goalLine} <Link href="/circle/?tab=league">Open the League</Link>
        </p>
      ) : null}

    </div>
  );
}
