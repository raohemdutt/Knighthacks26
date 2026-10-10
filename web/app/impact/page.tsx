"use client";

import Link from "next/link";
import { useState } from "react";
import im from "@/components/impact/impact.module.css";
import { Keyframe } from "@/components/Keyframe";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel, humanize, percent, timeOrDay } from "@/lib/format";
import type { Confidence, ImpactDisposal, ImpactStory, Range } from "@/lib/impact";
import { useFetch } from "@/lib/useFetch";

/**
 * Impact: what Waste Watch's counts mean, and what the table can do about them this week.
 *
 * Counts (items, right or wrong bin, diversion rate) are measured by the camera. Everything in grams,
 * kWh, phone charges or CO2e is an ESTIMATE from config/impact.yaml: it is shown as a range, tagged
 * "estimate", and the factors with their sources and confidence are one tap away.
 */

/** Two significant figures: 81 -> 81, 227 -> 230, 2725 -> 2700. A range is not a place for false precision. */
function sig2(n: number): number {
  if (n === 0) return 0;
  const p = Math.pow(10, Math.floor(Math.log10(Math.abs(n))) - 1);
  return Math.round(n / p) * p;
}

function tidy(n: number): string {
  const r = sig2(n);
  return r >= 100 ? String(Math.round(r)) : String(Math.round(r * 100) / 100);
}

function range(r: Range): string {
  const a = tidy(r.low);
  const b = tidy(r.high);
  return a === b ? a : `${a}–${b}`;
}

/** Grams under a kilogram, kilograms above: [number, unit]. */
function mass(g: Range): [string, string] {
  return g.high >= 1000 ? [range({ low: g.low / 1000, high: g.high / 1000 }), "kg"] : [range(g), "g"];
}

const CONFIDENCE: Record<Confidence, string> = { high: "high confidence", medium: "medium confidence", low: "low confidence" };
const WEEKDAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function ImpactPage() {
  const [member, setMember] = useState<string | undefined>(undefined);
  const story = useFetch((s) => api.impactStory(member, s), member ?? "");
  const people = useFetch((s) => api.members(s)); // names for the picker; 404 on a box without recognition
  useAtlasEvents(["impact."], () => story.reload());

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Impact</h1>
        {story.data ? (
          <p className={ui.muted}>
            Week of {dayLabel(`${story.data.week.start}T12:00:00`)} · {story.data.rules.name}
          </p>
        ) : null}
      </div>
      {story.error ? (
        story.error.notConfigured ? (
          <Empty title="Waste Watch is not set up on this box">
            Once the box has a sorting rules file and a camera on the bins, this screen shows what the table threw away, whether it went in the right
            bin, and what the week adds up to.
          </Empty>
        ) : (
          <ErrorBox what="the Impact screen" error={story.error} onRetry={story.reload} />
        )
      ) : !story.data ? (
        <Loading what="this week's impact" />
      ) : (
        <Story
          v={story.data}
          member={member}
          setMember={setMember}
          nameOf={(id) => people.data?.find((m) => m.member_id === id)?.name ?? id}
          reload={story.reload}
        />
      )}
    </>
  );
}

function Story({
  v,
  member,
  setMember,
  nameOf,
  reload,
}: {
  v: ImpactStory;
  member: string | undefined;
  setMember: (m: string | undefined) => void;
  nameOf: (id: string) => string;
  reload: () => void;
}) {
  const eq = v.equivalents;
  const g = v.goal.diversion;
  const [num, unit] = eq ? mass(eq.kept_out_g) : ["", ""];
  const ba = v.before_after;

  return (
    <>
      {v.empty ? <Empty title="Nothing counted yet">{v.empty_text}</Empty> : null}

      {/* What it means, in real units */}
      <section className={ui.section}>
        <div className={im.hero}>
          <p className={im.heroLabel}>Kept out of landfill this week</p>
          {eq && eq.items > 0 ? (
            <>
              <p className={im.heroNum}>
                {num} <span className={im.heroUnit}>{unit}</span> <span className={im.estimate}>estimate</span>
              </p>
              <p className={ui.muted}>
                {g.recycled} of {g.recyclable} recyclables the camera saw reached recycling (that count is measured).
              </p>
            </>
          ) : (
            <>
              <p className={im.heroNum}>
                {g.recycled} <span className={im.heroUnit}>items</span>
              </p>
              <p className={ui.muted}>
                {eq ? "The first can, bottle or sheet of paper in the recycling bin starts this number." : "No factors file on this box, so counts are not turned into grams or energy."}
              </p>
            </>
          )}
          {v.milestones.latest ? (
            <div className={im.milestone}>
              <p>
                {/* The count above is live; a milestone is a threshold passed earlier, so it is named as one unless the two agree. */}
                <strong>
                  {v.milestones.latest.kind === "kept_out" && v.milestones.latest.threshold !== g.recycled
                    ? `Last milestone: ${v.milestones.latest.threshold} ${v.milestones.latest.threshold === 1 ? "item" : "items"} kept out of landfill (${g.recycled} so far this week).`
                    : v.milestones.latest.text}
                </strong>
              </p>
              {v.milestones.next ? <p className={`${ui.muted} ${ui.small}`}>{v.milestones.next.text}</p> : null}
            </div>
          ) : v.milestones.next && !v.empty ? (
            <p className={`${ui.muted} ${ui.small}`}>{v.milestones.next.text}</p>
          ) : null}
        </div>

        {eq && eq.items > 0 ? (
          <>
            <div className={im.tiles}>
              <div className={im.tile}>
                <p className={im.tileNum}>{range(eq.phone_charges)}</p>
                <p className={im.tileLabel}>
                  phone charges&apos; worth of energy saved <span className={im.estimate}>estimate</span>
                </p>
              </div>
              <div className={im.tile}>
                <p className={im.tileNum}>{range(eq.energy_kwh)} kWh</p>
                <p className={im.tileLabel}>
                  not spent making new material <span className={im.estimate}>estimate</span>
                </p>
              </div>
              <div className={im.tile}>
                <p className={im.tileNum}>{range(eq.co2e_kg)} kg</p>
                <p className={im.tileLabel}>
                  CO2e avoided <span className={im.estimate}>estimate</span>
                </p>
              </div>
              <div className={im.tile}>
                <p className={im.tileNum}>{v.contamination.wrong_in_recycling}</p>
                <p className={im.tileLabel}>wrong {v.contamination.wrong_in_recycling === 1 ? "item" : "items"} in recycling (measured)</p>
              </div>
            </div>
            <details className={im.details}>
              <summary>How these are estimated</summary>
              <div className={im.detailsBody}>
                <p>{eq.disclaimer}</p>
                <p className={ui.muted}>
                  Each figure is the measured count multiplied by a low and a high factor from <code>{eq.file}</code>. They are illustrative
                  factors, to be replaced with a campus&apos;s own.
                </p>
                {eq.by_item.map((r) => (
                  <div key={r.item} className={im.factor}>
                    <p>
                      <strong>
                        {r.count} × {r.item}
                      </strong>
                      : {r.each_g.low}–{r.each_g.high} g each ({CONFIDENCE[r.mass_confidence]})
                    </p>
                    <p className={`${ui.muted} ${ui.small}`}>{r.mass_source}</p>
                    {r.material_source ? (
                      <p className={`${ui.muted} ${ui.small}`}>
                        {humanize(r.material ?? "material")} ({r.material_confidence ? CONFIDENCE[r.material_confidence] : "no confidence given"}):{" "}
                        {r.material_source}
                      </p>
                    ) : (
                      <p className={`${ui.muted} ${ui.small}`}>No energy or CO2e is claimed for this material.</p>
                    )}
                  </div>
                ))}
                {eq.unweighed.length > 0 ? (
                  <p className={im.factor}>
                    Counted but given no weight (no entry in the factors file): {eq.unweighed.map((u) => `${u.count} × ${u.item}`).join(", ")}.
                  </p>
                ) : null}
                <p className={`${im.factor} ${ui.muted} ${ui.small}`}>
                  Phone charge ({CONFIDENCE[eq.phone_charge_confidence]}): {eq.phone_charge_source}
                </p>
              </div>
            </details>
          </>
        ) : null}
      </section>

      {/* Goals and momentum */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>This week&apos;s goal</h2>
        <div className={ui.card}>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <h3>{g.value === null ? "No recyclables yet" : `${percent(g.value)} of recyclables recycled`}</h3>
            <span className={`${ui.tag} ${g.met ? ui.tagAccent : ""}`}>{g.met ? "goal met" : `goal ${percent(g.target)}`}</span>
          </div>
          <div className={im.goalBar} role="img" aria-label={`${g.value === null ? "nothing" : percent(g.value)} against a goal of ${percent(g.target)}`}>
            <div className={`${im.goalFill} ${g.met ? im.goalFillMet : ""}`} style={{ width: `${Math.round((g.value ?? 0) * 100)}%` }} />
            <span className={im.goalMark} style={{ left: `${Math.round(g.target * 100)}%` }} aria-hidden="true" />
            <span className={im.goalMarkLabel} style={{ left: `${Math.round(g.target * 100)}%` }} aria-hidden="true">
              goal {percent(g.target)}
            </span>
          </div>
          <p>{g.text}</p>
          <p className={v.goal.contamination.met ? `${ui.small} ${ui.muted}` : ui.small}>
            Wrong items in recycling: {v.goal.contamination.value} of at most {v.goal.contamination.ceiling} this week
            {v.goal.contamination.met ? (v.goal.contamination.left > 0 ? ` (${v.goal.contamination.left} to spare).` : " (none to spare).") : ": over the ceiling."}
          </p>
          <div className={im.week} role="list" aria-label="Each day this week">
            {v.days.map((d, i) => {
              const future = d.day > v.week.today;
              const cls = [im.day, d.clean ? im.dayClean : "", d.day === v.week.today ? im.dayToday : "", future ? im.dayFuture : ""].join(" ");
              const label = d.disposals === 0 ? "nothing thrown away" : `${d.right} of ${d.disposals} in the right bin`;
              return (
                <div key={d.day} role="listitem" className={cls} title={`${WEEKDAY[i]}: ${label}`} aria-label={`${WEEKDAY[i]}: ${label}`}>
                  <span>{WEEKDAY[i]}</span>
                  <span className={im.dayMark}>{d.disposals === 0 ? "·" : `${d.right}/${d.disposals}`}</span>
                </div>
              );
            })}
          </div>
          <p className={`${ui.small} ${ui.muted}`}>Right bin / thrown away, per day. A highlighted day had nothing in the wrong bin.</p>
        </div>
        <div className={im.pair}>
          <div className={im.tile}>
            <p className={im.tileNum}>
              {v.clean_streak} {v.clean_streak === 1 ? "day" : "days"}
            </p>
            <p className={im.tileLabel}>{v.clean_streak > 0 ? "clean in a row" : "clean streak: the next day with nothing in the wrong bin starts one"}</p>
          </div>
          <div className={im.tile}>
            <p className={im.tileNum}>{v.best_day ? dayLabel(`${v.best_day.day}T12:00:00`) : "–"}</p>
            <p className={im.tileLabel}>
              {v.best_day ? `best day: ${v.best_day.right} of ${v.best_day.disposals} in the right bin` : "best day: needs a day with two or more items"}
            </p>
          </div>
        </div>
        <p className={`${ui.small} ${ui.muted}`}>{v.projection.text}</p>
      </section>

      {/* The moment of disposal */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>What went where</h2>
        {v.scope.per_person && v.scope.members.length > 0 ? (
          <div className={ui.chips} role="group" aria-label="Whose disposals">
            <button type="button" className={`${ui.chip} ${member === undefined ? ui.chipOn : ""}`} aria-pressed={member === undefined} onClick={() => setMember(undefined)}>
              The table
            </button>
            {v.scope.members.map((id) => (
              <button key={id} type="button" className={`${ui.chip} ${member === id ? ui.chipOn : ""}`} aria-pressed={member === id} onClick={() => setMember(id)}>
                {nameOf(id)}
              </button>
            ))}
          </div>
        ) : null}
        <p className={`${ui.small} ${ui.muted}`}>{v.scope.note}</p>
        {v.feed.length === 0 ? (
          <Empty title="No disposals yet">
            Each thing the camera sees go into a bin appears here with its picture, the bin, and whether that was right. A wrong one comes with the
            rule and a way to practise it.
          </Empty>
        ) : (
          <ul className={ui.list}>
            {v.feed.map((d) => (
              <Disposal key={d.item_id} d={d} member={member} who={d.member_id ? nameOf(d.member_id) : null} reload={reload} />
            ))}
          </ul>
        )}
      </section>

      {/* The practice set */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Practise in Sort It</h2>
        {v.practice.size === 0 ? (
          <Empty title="Nothing to practise">Items that land in the wrong bin this week, and any you tap &quot;Teach me&quot; on, are listed here for a round of Sort It.</Empty>
        ) : (
          <div className={ui.card}>
            <ul className={ui.list}>
              {v.practice.items.map((p) => (
                <li key={p.item}>
                  <strong>{humanize(p.item)}</strong> goes in {p.belongs_in}
                  <span className={ui.muted}>
                    {" "}
                    · {p.mistakes > 0 ? `${p.mistakes} ${p.mistakes === 1 ? "miss" : "misses"} this week` : "you asked to learn it"}
                  </span>
                </li>
              ))}
            </ul>
            <Link href="/play/" className={`${ui.btn} ${ui.btnPrimary}`}>
              Play a round of Sort It
            </Link>
          </div>
        )}
      </section>

      {/* Before and after nudges */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Before and after nudges</h2>
        <div className={ui.card}>
          {ba.available && ba.enough && ba.before && ba.after ? (
            <>
              <div className={im.bars} role="img" aria-label={`Recyclables recycled: ${percent(ba.before.diversion_rate ?? 0)} before nudges, ${percent(ba.after.diversion_rate ?? 0)} after`}>
                <div className={im.barCol} title={`Before: ${ba.before.recycled} of ${ba.before.n} recycled`}>
                  <span className={im.barValue}>{percent(ba.before.diversion_rate ?? 0)}</span>
                  <div className={im.bar} style={{ height: `${Math.round((ba.before.diversion_rate ?? 0) * 75)}%` }} />
                </div>
                <div className={im.barCol} title={`After: ${ba.after.recycled} of ${ba.after.n} recycled`}>
                  <span className={im.barValue}>{percent(ba.after.diversion_rate ?? 0)}</span>
                  <div className={`${im.bar} ${im.barAfter}`} style={{ height: `${Math.round((ba.after.diversion_rate ?? 0) * 75)}%` }} />
                </div>
              </div>
              <div className={im.barLabels}>
                <span>
                  Before · n = {ba.before.n}
                </span>
                <span>
                  After · n = {ba.after.n}
                </span>
              </div>
              <p className={`${ui.small} ${ui.muted}`}>
                Share of recyclables that reached recycling, before and after the first sorting nudge ({timeOrDay(ba.nudges_on_at)}). n is the
                recyclables seen on each side. Measured counts from this table; small numbers, so read it as a direction, not a result.
              </p>
            </>
          ) : (
            <>
              <h3>Not enough data yet</h3>
              <p className={ui.muted}>{ba.text}</p>
            </>
          )}
        </div>
      </section>

      {/* The cost of contamination */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Why a wrong item matters</h2>
        <div className={ui.card}>
          <p>
            {v.contamination.text} <span className={im.estimate}>rule of thumb</span>
          </p>
          <p className={`${ui.small} ${ui.muted}`}>
            {v.contamination.source} ({CONFIDENCE[v.contamination.confidence]}.)
          </p>
          {v.contamination.wrong_in_recycling > 0 ? (
            <p className={ui.small}>
              This week: {v.contamination.wrong_in_recycling} wrong {v.contamination.wrong_in_recycling === 1 ? "item" : "items"} in recycling, on days when{" "}
              {v.contamination.good_items_sharing_the_bin} good {v.contamination.good_items_sharing_the_bin === 1 ? "item" : "items"} went in the same
              bin.
            </p>
          ) : null}
        </div>
      </section>

      {/* Which rules file is active */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>The rules in use</h2>
        <div className={ui.card}>
          <h3>{v.rules.name}</h3>
          <p className={`${ui.small} ${ui.muted} ${ui.breakWords}`}>
            <code>{v.rules.file}</code>
            {v.rules.overridden && v.profile ? ` (the ${v.profile} profile's file replaces config/sorting.yaml)` : ""}
          </p>
          {v.rules.note ? <p className={ui.small}>{v.rules.note}</p> : null}
          <div className={im.rulesGrid}>
            {v.rules.bins.map((b) => (
              <div key={b}>
                <strong>{humanize(b)}</strong>
                <ul>
                  {v.rules.items
                    .filter((i) => i.bin === b)
                    .map((i) => (
                      <li key={i.label} title={i.rule}>
                        {i.label}
                      </li>
                    ))}
                </ul>
              </div>
            ))}
          </div>
          <p className={`${ui.small} ${ui.muted}`}>{v.rules.swap}</p>
        </div>
      </section>
    </>
  );
}

function Disposal({ d, member, who, reload }: { d: ImpactDisposal; member: string | undefined; who: string | null; reload: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [added, setAdded] = useState(d.in_practice);
  const [error, setError] = useState<ApiError | undefined>(undefined);

  const teachMe = async () => {
    setOpen(true);
    if (added || busy) return;
    setBusy(true);
    setError(undefined);
    try {
      await api.addPractice(d.item, d.item_id, member ?? d.member_id ?? undefined);
      setAdded(true);
      reload();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className={ui.card}>
      <div className={im.disposal}>
        {d.frame_ref && d.frame_stored ? (
          <div className={im.thumb}>
            <Keyframe frameRef={d.frame_ref} compact />
          </div>
        ) : (
          <div className={im.noPicture}>no picture stored</div>
        )}
        <div>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <h3>
              {humanize(d.item)} → {d.bin}
            </h3>
            <span className={`${im.verdict} ${d.correct ? im.right : im.wrong}`}>{d.correct ? "Right bin" : "Wrong bin"}</span>
          </div>
          <p className={`${ui.small} ${ui.muted}`}>
            {timeOrDay(d.ts)}
            {who ? ` · ${who}` : ""}
          </p>
          {!d.correct ? (
            <>
              <p className={im.rule}>{d.rule}</p>
              {open ? (
                <p className={ui.small}>
                  It belongs in <strong>{d.belongs_in}</strong>. {added ? `${humanize(d.item)} is in your Sort It practice set.` : ""}
                </p>
              ) : null}
              <div className={`${ui.row} ${ui.wrap}`} style={{ marginTop: 8 }}>
                {!open || !added ? (
                  <button type="button" className={ui.btn} onClick={teachMe} disabled={busy}>
                    {busy ? "Adding…" : "Teach me"}
                  </button>
                ) : (
                  <Link href="/play/" className={ui.btn}>
                    Practise it in Sort It
                  </Link>
                )}
              </div>
              {error ? <WriteError error={error} /> : null}
            </>
          ) : null}
        </div>
      </div>
    </li>
  );
}
