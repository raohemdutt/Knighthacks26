"use client";

import { Fragment, useState, type ReactNode } from "react";
import type { DayRunInitial, Explained, NumberSource, PlanBoard, SimResult, StripDay } from "@/lib/api";
import { ms } from "@/lib/dayrun";
import { dateWords, dur, hoursShort, stars, t12 } from "@/lib/plan";
import pl from "./plan.module.css";

/**
 * Where the numbers come from. Every figure a person might question carries a small chip naming its source
 * (Canvas, OpenStreetMap, your watch, the desk camera, your estimate, Atlas's guess): the designer's decision,
 * "source chips, not the Why? well". The label is the box's own (`explained.sources`, written in
 * contexts/play/dayrun/words.py from the same field the sentence is written from), never chosen here. Tapping a
 * chip opens the plain sentence the old "Why?" opened, unchanged: nothing was lost, it only moved behind the chip.
 * A box too old to send sources keeps the "Why?" button, so the sentence is always reachable.
 */

/** One piece of a fact's line, with the source of its number when the box named one. */
export interface Part {
  text: string;
  source?: NumberSource | null;
}

/** The chip. `open` and `onToggle` belong to the explanation it opens; several chips of one fact share them. */
export function SourceChip({ source, about, open, onToggle }: { source: NumberSource; about: string; open: boolean; onToggle: () => void }) {
  return (
    <button type="button" className={pl.src} data-src={source.id} aria-expanded={open} aria-label={`${about}: from ${source.label}. ${open ? "Hide" : "Show"} how it is worked out`} onClick={onToggle}>
      {source.label}
    </button>
  );
}

function Fact({ name, parts, side, why }: { name: string; parts: Part[]; side?: ReactNode; why?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const chips = parts.some((p) => p.source);
  return (
    <div className={pl.fact}>
      <div className={pl.factTop}>
        <p className={pl.factName}>
          {name}
          <span>
            {parts.map((p, i) => (
              <Fragment key={p.text}>
                {i ? " " : ""}
                <b className={pl.srcPart}>
                  {i ? "· " : ""}
                  {p.text}
                  {why && p.source ? <SourceChip source={p.source} about={`${name}, ${p.text}`} open={open} onToggle={() => setOpen(!open)} /> : null}
                </b>
              </Fragment>
            ))}
          </span>
        </p>
        <div className={pl.factSide}>
          {side}
          {why && !chips ? (
            <button type="button" className={pl.whyBtn} aria-expanded={open} onClick={() => setOpen(!open)}>
              Why?
            </button>
          ) : null}
        </div>
      </div>
      {open ? <div className={pl.srcNotes}>{why}</div> : null}
    </div>
  );
}

const STATUS: Record<string, { text: string; cls: string }> = {
  done: { text: "Done", cls: pl.tagGood ?? "" },
  on_track: { text: "On track", cls: pl.tagGood ?? "" },
  at_risk: { text: "At risk", cls: pl.tagWarn ?? "" },
  missed: { text: "Missed", cls: pl.tagWarn ?? "" },
};
const WORK_STEP = 30;

/** What is due: each deadline, how much work is left (yours to correct) and whether this plan keeps it safe. */
export function DueSoon({ board, explained, workLeft, onWorkLeft, readOnly }: { board: PlanBoard; explained: Explained | undefined; workLeft: Record<string, number>; onWorkLeft: (id: string, minutes: number) => void; readOnly: boolean }) {
  if (board.deadlines.length === 0) return null;
  return (
    <section className={pl.pane} aria-label="Due soon" data-guide="plan:due">
      <h2 className={pl.paneTitle}>Due soon</h2>
      <div className={pl.facts}>
        {board.deadlines.map((d) => {
          const e = explained?.deadlines[d.id];
          const left = workLeft[d.id] ?? d.work_left;
          const status = e?.status ? STATUS[e.status] : undefined;
          const from = explained?.sources?.deadlines[d.id];
          return (
            <Fact
              key={d.id}
              name={e?.title ?? d.title}
              parts={[
                { text: `Due ${e?.due ?? t12(d.due)}`, source: from?.due },
                e?.status === "done" ? { text: "this plan finishes it" } : { text: `about ${dur(left)} of work`, source: from?.work },
                ...(e?.status === "done" && from ? [{ text: `${dur(left)} of work`, source: from.work }] : []),
              ]}
              side={status ? <span className={`${pl.tagPill} ${status.cls}`}>{status.text}</span> : null}
              why={
                e ? (
                  <>
                    <p className={pl.srcNote}>{e.work}</p>
                    {e.risk ? <p className={pl.srcNote}>{e.risk}</p> : null}
                    {readOnly ? null : (
                      <div className={pl.edit}>
                        <span className={pl.stepper} role="group" aria-label="Work left">
                          <button type="button" aria-label="30 minutes less work" disabled={left <= WORK_STEP} onClick={() => onWorkLeft(d.id, Math.max(WORK_STEP, left - WORK_STEP))}>
                            −
                          </button>
                          <span>{dur(left)}</span>
                          <button type="button" aria-label="30 minutes more work" onClick={() => onWorkLeft(d.id, left + WORK_STEP)}>
                            +
                          </button>
                        </span>
                      </div>
                    )}
                  </>
                ) : undefined
              }
            />
          );
        })}
      </div>
    </section>
  );
}

/** The fold at the bottom: sleep, meals, walking and the score, each number with its source chip, and one small chart. */
export function PlanDetails({ result, explained, initial, t, tomorrow, open }: { result: SimResult; explained: Explained | undefined; /** Where the day starts (last night's sleep), for the number the sleep chip is about. */ initial?: DayRunInitial; t: number | null; tomorrow?: string | null; open?: boolean }) {
  const tot = result.totals;
  const t0 = ms(result.start);
  const span = Math.max(1, ms(result.end) - t0);
  const x = (iso: string) => (((ms(iso) - t0) / span) * 100).toFixed(2);
  const path = (pick: (k: SimResult["timeline"][number]) => number) => result.timeline.map((k, i) => `${i ? "L" : "M"}${x(k.t)} ${(100 - pick(k)).toFixed(1)}`).join("");
  const longest = [...result.routes].sort((a, b) => b.minutes - a.minutes)[0];
  const from = explained?.sources;
  const before = tot.meals_before ?? 0;
  return (
    <details className={pl.fold} open={open} data-guide="plan:details">
      <summary>Details</summary>
      <div className={pl.foldBody}>
        {tomorrow ? <p className={pl.well}>Next time: {tomorrow}</p> : null}
        <div className={pl.facts}>
          <Fact
            name="Sleep"
            parts={
              from && initial
                ? [{ text: `${hoursShort(tot.sleep_hours)} in this plan` }, { text: `${hoursShort(initial.sleep_hours)} last night`, source: from.sleep }]
                : [{ text: `${hoursShort(tot.sleep_hours)} in this plan`, source: from?.sleep }]
            }
            why={explained ? <p className={pl.srcNote}>{explained.sleep}</p> : undefined}
          />
          <Fact
            name="Meals"
            parts={before > 0 ? [{ text: `${tot.meals} today` }, { text: `${before} before this plan`, source: from?.meals }] : [{ text: `${tot.meals} today`, source: from?.meals }]}
            why={explained ? <p className={pl.srcNote}>{explained.meals}</p> : undefined}
          />
          <Fact
            name="Walking"
            parts={[
              {
                text: result.routes.length === 0 ? "No walks in this plan" : `${dur(tot.walk_minutes)} over ${result.routes.length} ${result.routes.length === 1 ? "walk" : "walks"}`,
                source: longest ? from?.walks[String(longest.index)] : undefined,
              },
            ]}
            why={explained && longest && explained.walks[String(longest.index)] ? <p className={pl.srcNote}>The longest: {explained.walks[String(longest.index)]}</p> : undefined}
          />
          <Fact
            name="Score"
            parts={[{ text: `${tot.points} of 100`, source: from?.points }]}
            side={
              <span className={pl.stars} aria-label={`${result.stars} of 3 stars`}>
                {stars(result.stars)}
              </span>
            }
            why={
              explained ? (
                <>
                  <p className={pl.srcNote}>{explained.points}</p>
                  <p className={pl.srcNote}>{explained.stars}</p>
                </>
              ) : undefined
            }
          />
        </div>
        <div>
          <div className={pl.chart} role="img" aria-label={`Energy and focus through the day: energy ends at ${Math.round(tot.energy_end)} of 100, lowest ${Math.round(tot.energy_min)}.`}>
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {t !== null ? <line className={pl.chartNow} x1={((t - t0) / span) * 100} x2={((t - t0) / span) * 100} y1={0} y2={100} /> : null}
              <path className={pl.lineFocus} d={path((k) => k.focus)} />
              <path className={pl.lineEnergy} d={path((k) => k.energy)} />
            </svg>
          </div>
          <p className={pl.chartKey}>
            <span>
              <i /> Energy
            </span>
            <span>
              <i className={pl.dash} /> Focus
            </span>
            <span>
              {t12(result.start)} to {t12(result.end)}
            </span>
          </p>
        </div>
      </div>
    </details>
  );
}

/** The next few days, from what the box already has for them. Nothing here is predicted. */
export function NextDays({ days, selected, onSelect }: { days: StripDay[]; selected: string; onSelect: (day: string) => void }) {
  const ahead = days.filter((d) => d.when === "future");
  if (ahead.length === 0) return null;
  return (
    <details className={pl.fold} open={ahead.some((d) => d.day === selected)} data-guide="plan:ahead">
      <summary>The next few days</summary>
      <div className={pl.foldBody}>
        <div className={pl.ahead}>
          {ahead.map((d) => {
            const due = d.due?.[0];
            const pressure = Math.max(0, Math.min(1, d.pressure ?? 0));
            return (
              <button key={d.day} type="button" className={pl.aheadRow} disabled={!d.has_data} aria-pressed={d.day === selected} onClick={() => onSelect(d.day)}>
                <span className={pl.aheadDay}>{dateWords(d.day).split(",")[0]} {d.day.slice(8).replace(/^0/, "")}</span>
                <span className={pl.aheadWhat}>
                  <b>{due ? `${due.title} due` : d.fixed ? `${d.fixed} on your calendar` : "Nothing fixed"}</b>
                  {dur(d.free_minutes ?? 0)} free{d.optional ? ` · ${d.optional} near you` : ""}
                </span>
                <span className={pl.press} role="img" aria-label={`Deadline pressure ${Math.round(pressure * 100)} percent`}>
                  <i style={{ width: `${Math.round(pressure * 100)}%` }} />
                </span>
              </button>
            );
          })}
        </div>
        <p className={pl.legendLine}>The bar is deadline pressure: how much of your work time the nearest deadline would take if you did nothing before that day.</p>
      </div>
    </details>
  );
}
