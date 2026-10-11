"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import im from "@/components/impact/impact.module.css";
import { WeekStory, rangeText } from "@/components/impact/WeekStory";
import { Keyframe } from "@/components/Keyframe";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, type LeagueView, api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel, humanize, percent, safeUrl, timeOrDay } from "@/lib/format";
import type { Confidence, ImpactDisposal, ImpactStory, Range } from "@/lib/impact";
import { useFetch } from "@/lib/useFetch";

/**
 * Impact: one table's recycling this week, as a short story and the facts under it.
 *
 * Two kinds of number, always labelled. MEASURED: counts from this table's camera (items, right or wrong bin,
 * the diversion rate). ESTIMATE: those counts multiplied by published factors in config/impact.yaml, always a
 * range. Nothing on this screen is a projection.
 *
 *   /impact/?t=0.4            open with the story paused part-way (0 to 1); without it the finished story is shown
 *   /impact/?explain=why      open the "Why it matters" panel (also: how)
 *   /impact/?motion=reduce    as if the system asked for reduced motion
 */

const CONFIDENCE: Record<Confidence, string> = { high: "high confidence", medium: "medium confidence", low: "low confidence" };
const WEEKDAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
type Panel = "why" | "how" | null;

/** Grams under a kilogram, kilograms above. */
function mass(g: Range): string {
  return g.high >= 1000 ? `${rangeText({ low: g.low / 1000, high: g.high / 1000 })} kg` : `${rangeText(g)} g`;
}

/** A citation as text, with its last URL made into a link. */
function Cite({ text }: { text: string | null | undefined }) {
  if (!text) return null;
  const parts = text.split(/(https:\/\/\S+)/);
  return (
    <span className={ui.breakWords}>
      {parts.map((p, i) => {
        const url = p.startsWith("https://") ? safeUrl(p) : null;
        return url ? (
          <a key={i} href={url} target="_blank" rel="noreferrer">
            {p}
          </a>
        ) : (
          p
        );
      })}
    </span>
  );
}

export default function ImpactPage() {
  const [member, setMember] = useState<string | undefined>(undefined);
  const [params, setParams] = useState<{ t: number | null; reduce: boolean; panel: Panel } | null>(null);
  const story = useFetch((s) => api.impactStory(member, s), member ?? "");
  const people = useFetch((s) => api.members(s)); // names for the picker; 404 on a box without recognition
  const league = useFetch((s) => api.league(s)); // the friends' tables, for the end of the story
  const reloadLeague = league.reload;
  useAtlasEvents(["impact.", "play.round.scored"], () => story.reload());
  useAtlasEvents(["play.league.", "impact.count.updated"], () => reloadLeague());

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.get("t");
    const e = q.get("explain");
    setParams({ t: t !== null && !Number.isNaN(Number(t)) ? Math.min(1, Math.max(0, Number(t))) : null, reduce: q.get("motion") === "reduce", panel: e === "why" || e === "how" ? e : null });
  }, []);

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
      ) : !story.data || !params ? (
        <Loading what="this week's impact" />
      ) : (
        <Story
          v={story.data}
          league={league.data}
          params={params}
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
  league,
  params,
  member,
  setMember,
  nameOf,
  reload,
}: {
  v: ImpactStory;
  league: LeagueView | undefined;
  params: { t: number | null; reduce: boolean; panel: Panel };
  member: string | undefined;
  setMember: (m: string | undefined) => void;
  nameOf: (id: string) => string;
  reload: () => void;
}) {
  const g = v.goal.diversion;
  const ba = v.before_after;
  const [panel, setPanel] = useState<Panel>(params.panel);
  const close = useCallback(() => setPanel(null), []);

  return (
    <>
      {v.empty ? <Empty title="Nothing counted yet">{v.empty_text}</Empty> : null}

      {/* The week as a short story */}
      <section className={ui.section}>
        <h2 className={im.headline}>{v.headline ?? `${g.recycled} of ${g.recyclable} recyclables reached recycling this week.`}</h2>
        <WeekStory v={v} league={league} initial={params.t} forceReduced={params.reduce} onWhy={() => setPanel("why")} onHow={() => setPanel("how")} />
      </section>

      {/* The goal */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>This week&apos;s goal</h2>
        <div className={ui.card}>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <h3>{g.value === null ? "No recyclables yet" : `${percent(g.value)} of recyclables recycled`}</h3>
            <span className={`${ui.tag} ${g.met ? ui.tagAccent : ""}`}>{g.met ? "goal met" : `goal ${percent(g.target)}`}</span>
          </div>
          <p>{g.text}</p>
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
          <p className={`${ui.small} ${ui.muted}`}>
            Right bin / thrown away, per day; a highlighted day had nothing in the wrong bin. {v.clean_streak > 0 ? `${v.clean_streak} clean ${v.clean_streak === 1 ? "day" : "days"} in a row. ` : ""}
            Wrong items in recycling: {v.goal.contamination.value} of at most {v.goal.contamination.ceiling}
            {v.goal.contamination.met ? "." : ": over the ceiling."}
            {v.milestones.next ? ` ${v.milestones.next.text}` : ""}
          </p>
          {ba.available && ba.enough && ba.before && ba.after ? (
            <p className={ui.small}>
              Before the first sorting nudge ({timeOrDay(ba.nudges_on_at)}): {percent(ba.before.diversion_rate ?? 0)} of {ba.before.n} recyclables recycled. After:{" "}
              {percent(ba.after.diversion_rate ?? 0)} of {ba.after.n}. <span className={im.measured}>measured</span> Small numbers: a direction, not a result.
            </p>
          ) : ba.available ? (
            <p className={`${ui.small} ${ui.muted}`}>Before and after nudges: {ba.text.replace(/^Not enough data yet: /, "not enough data yet, ")}</p>
          ) : null}
        </div>
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

      {/* The practice set, and the loop it closes */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Practise in Sort It</h2>
        {v.loop ? (
          <ol className={im.loop} aria-label="From a mistake to fewer of them">
            <li>
              <strong>{v.loop.seen}</strong>
              <span>{v.loop.seen === 1 ? "mistake" : "mistakes"} seen</span>
            </li>
            <li>
              <strong>{v.loop.taught}</strong>
              <span>&quot;Teach me&quot; taps</span>
            </li>
            <li>
              <strong>{v.loop.practised.rounds}</strong>
              <span>
                Sort It {v.loop.practised.rounds === 1 ? "round" : "rounds"} on them{v.loop.practised.rounds ? `, ${v.loop.practised.right} right` : ""}
              </span>
            </li>
            <li className={v.loop.trend.available ? "" : im.loopWaiting}>
              <strong>{v.loop.trend.available ? `${v.loop.trend.previous} → ${v.loop.trend.current}` : "–"}</strong>
              <span>{v.loop.trend.available ? "wrong, last week to this" : "next week: needs a second week to show a trend"}</span>
            </li>
          </ol>
        ) : null}
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

      {/* Which rules file is active, and where it comes from */}
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>The rules in use</h2>
        <div className={ui.card}>
          <h3>{v.rules.name}</h3>
          <p className={`${ui.small} ${ui.muted} ${ui.breakWords}`}>
            {v.rules.source ? (
              <>
                From <a href={safeUrl(v.rules.source.url) ?? undefined} target="_blank" rel="noreferrer">{`“${v.rules.source.title}”, ${v.rules.source.publisher}`}</a>, read {v.rules.source.read}.{" "}
              </>
            ) : v.rules.note ? (
              `${v.rules.note} `
            ) : null}
            <code>{v.rules.file}</code>
            {v.rules.overridden && v.profile ? ` (the ${v.profile} profile's file replaces config/sorting.yaml)` : ""}
          </p>
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
                        {i.basis === "assumed" ? <span className={im.assumed}> *</span> : null}
                      </li>
                    ))}
                </ul>
              </div>
            ))}
          </div>
          {v.rules.items.some((i) => i.basis === "assumed") ? (
            <p className={`${ui.small} ${ui.muted}`}>
              <span className={im.assumed}>*</span> Not stated by {v.rules.place ?? "the published list"}: our assumption.
            </p>
          ) : null}
          {v.rules.compare && v.rules.compare.length > 0 ? (
            <details className={im.inlineDetails}>
              <summary>Same item, different rule nearby</summary>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th scope="col">Item</th>
                    {Object.keys(v.rules.compare[0]!)
                      .filter((k) => k !== "item")
                      .map((k) => (
                        <th key={k} scope="col">
                          {k === "ucf" ? "UCF" : humanize(k)}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {v.rules.compare.map((c) => (
                    <tr key={c.item}>
                      <td>{c.item}</td>
                      {Object.keys(c)
                        .filter((k) => k !== "item")
                        .map((k) => (
                          <td key={k}>{c[k]}</td>
                        ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {(v.rules.compare_sources ?? []).map((s) => (
                <p key={s} className={`${ui.small} ${ui.muted}`}>
                  <Cite text={s} />
                </p>
              ))}
            </details>
          ) : null}
          <p className={`${ui.small} ${ui.muted}`}>{v.rules.swap}</p>
        </div>
      </section>

      {panel === "why" ? <Why v={v} onClose={close} /> : null}
      {panel === "how" ? <How v={v} onClose={close} /> : null}
    </>
  );
}

function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>{title}</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

/** What one wrong item does: the sourced statement, this week's measured count, and the problem here. */
function Why({ v, onClose }: { v: ImpactStory; onClose: () => void }) {
  const c = v.contamination;
  return (
    <Sheet title="Why it matters" onClose={onClose}>
      {c.quote ? (
        <blockquote className={im.quote}>
          <p>“{c.quote}”</p>
          <footer className={`${ui.small} ${ui.muted}`}>
            {c.url ? (
              <a href={safeUrl(c.url) ?? undefined} target="_blank" rel="noreferrer">
                {c.attribution}
              </a>
            ) : (
              c.attribution
            )}
          </footer>
        </blockquote>
      ) : (
        <p>
          {c.text} <span className={im.estimate}>rule of thumb</span>
        </p>
      )}
      <p className={ui.small}>
        <span className={im.measured}>measured</span> This week at this table: {c.wrong_in_recycling} wrong {c.wrong_in_recycling === 1 ? "item" : "items"} in recycling
        {c.wrong_in_recycling > 0 ? `, on days when ${c.good_items_sharing_the_bin} good ${c.good_items_sharing_the_bin === 1 ? "item" : "items"} went in the same bin` : ""}.
      </p>
      <p className={`${ui.small} ${ui.muted}`}>
        EPA says &quot;can&quot;, not &quot;will&quot;, and gives no number. We found no published count of wrong items that spoils a bag, so this screen does not claim one.
      </p>
      {v.problem ? (
        <>
          <h3>The problem here</h3>
          <p>{v.problem.text}</p>
          <ul className={im.sources}>
            {v.problem.sources.map((s) => (
              <li key={s.claim} className={ui.small}>
                <span className={ui.tag}>{s.scope}</span> {s.claim} <span className={ui.muted}><Cite text={s.source} /></span>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </Sheet>
  );
}

/** How the estimate was made: count x mass x factor for each item, with the document each number came from. */
function How({ v, onClose }: { v: ImpactStory; onClose: () => void }) {
  const eq = v.equivalents;
  if (!eq) return null;
  return (
    <Sheet title="How we got this number" onClose={onClose}>
      <p>
        <span className={im.estimate}>estimate</span> {eq.disclaimer}
      </p>
      <div className={im.sum}>
        <div>
          <strong>{mass(eq.kept_out_g)}</strong>
          <span>kept out of landfill</span>
        </div>
        <div>
          <strong>{rangeText(eq.energy_kwh)} kWh</strong>
          <span>energy not spent</span>
        </div>
        <div>
          <strong>{rangeText(eq.co2e_kg)} kg</strong>
          <span>CO2e avoided</span>
        </div>
        <div>
          <strong>{rangeText(eq.phone_charges)}</strong>
          <span>phone charges</span>
        </div>
      </div>
      {eq.by_item.map((r) => (
        <div key={r.item} className={im.factor}>
          <p>
            <strong>
              {r.count} × {r.item}
            </strong>{" "}
            × {r.each_g.low}–{r.each_g.high} g{r.energy_kwh_per_kg ? ` × ${r.energy_kwh_per_kg.low}–${r.energy_kwh_per_kg.high} kWh/kg` : ""}
            {r.energy_kwh ? ` = ${rangeText(r.energy_kwh)} kWh` : ` = ${rangeText(r.mass_g)} g`}
            {r.co2e_kg && r.co2e_kg_per_kg ? `; × ${r.co2e_kg_per_kg.low}–${r.co2e_kg_per_kg.high} kg CO2e/kg = ${rangeText(r.co2e_kg)} kg CO2e` : ""}
          </p>
          <p className={`${ui.muted} ${ui.small}`}>
            Mass ({CONFIDENCE[r.mass_confidence]}): <Cite text={r.mass_source} />
          </p>
          {r.material_source ? (
            <p className={`${ui.muted} ${ui.small}`}>
              {humanize(r.material ?? "material")} ({r.material_confidence ? CONFIDENCE[r.material_confidence] : "no confidence given"}): <Cite text={r.material_source} />
            </p>
          ) : (
            <p className={`${ui.muted} ${ui.small}`}>No energy or CO2e is claimed for this material.</p>
          )}
        </div>
      ))}
      {eq.unweighed.length > 0 ? (
        <p className={im.factor}>
          Counted, but given no weight (no published mass was found): {eq.unweighed.map((u) => `${u.count} × ${u.item}`).join(", ")}.
        </p>
      ) : null}
      <div className={im.factor}>
        <p>
          <strong>Phone charges</strong> = energy ÷ {eq.phone_charge_wh ? `${eq.phone_charge_wh.low}–${eq.phone_charge_wh.high} Wh` : "one charge"} a charge
        </p>
        <p className={`${ui.muted} ${ui.small}`}>
          ({CONFIDENCE[eq.phone_charge_confidence]}) <Cite text={eq.phone_charge_source} />
        </p>
        {eq.energy_note ? <p className={ui.small}>{eq.energy_note}</p> : null}
      </div>
      <p className={`${ui.muted} ${ui.small}`}>
        Every factor is in <code>{eq.file}</code> with its source and the unit conversion. They are US averages, not UCF&apos;s own figures.
      </p>
    </Sheet>
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
