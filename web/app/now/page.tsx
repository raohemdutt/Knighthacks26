"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import {
  ApiError,
  api,
  keyframeUrl,
  type CoachCalibration,
  type CoachEvidence,
  type CoachSession,
  type DayCoachNote,
  type DayNarration,
  type DayNow,
  type DayPattern,
  type DayPlan,
  type DayReplan,
  type DayReview,
  type DayStatus,
  type PlanMove,
  type PlanSource,
} from "@/lib/api";
import { useAtlas, useAtlasEvents } from "@/lib/atlas";
import { briefMe, useBrief, type BriefState } from "@/lib/brief";
import { cameraName, dayLabel, humanize, relative, time, timeOrDay } from "@/lib/format";
import { useWords } from "@/lib/profile";
import { useFetch } from "@/lib/useFetch";
import { useVoice } from "@/lib/voice";
import n from "./now.module.css";

/**
 * Now: the day Atlas runs unasked (contexts/day), told top to bottom the way a day goes.
 *
 *  1. What Atlas would say right now (the narrator's paragraph, `day.narrated`), with a button that says it.
 *  2. One thing to do about it: approve the plan, leave, re-plan from here, or change it in Day Run.
 *  3. The day so far in four numbers.
 *  4. The ribbon: what is behind you (dimmed, with what the box measured), where you are, what is next and when
 *     to leave for it, tonight's sleep. Blocks from your calendar are solid; Atlas's suggestions are dashed, and
 *     tapping one says why it is there. The coach and the evening review sit in the ribbon where they happened.
 *  5. Why this plan: the sources, what was missing, the patterns, what the cameras last saw, friends.
 *
 * Live from the day.* events on /ws. Without a narrator on the box the first paragraph is put together here
 * from the same status and plan.
 */

const RISK_TEXT: Record<DayStatus["risk"], string> = {
  ok: "On time",
  leave_now: "Leave now",
  late: "Late",
  infeasible: "You will miss it",
};

const ORIGIN_TEXT: Record<string, string> = {
  canvas: "Canvas",
  calendar: "Outlook calendar",
  campus: "campus events",
  groupme: "GroupMe",
  "desk camera": "desk camera",
  watch: "your watch",
  circle: "Circle",
  review: "yesterday's review",
  coach: "the coach",
};

const EVIDENCE_TEXT: Record<CoachEvidence["kind"], string> = {
  chat: "Group chat",
  phone: "Phone in view",
  door: "At the door",
  food: "Drink or food",
};

const NIGHT_MIN = 180; // a sleep block shorter than this is a nap inside the day
const BREAK_MAX = 30; // a rest this short between two blocks of the main activity belongs to them

/**
 * The day's blocks as the ribbon shows them: a run of Atlas's own blocks of the main activity, with the short rests
 * between them, is one row ("3 study blocks"), the way the brief says it. The block you are in, anything from the
 * calendar, meals and long rests stay rows of their own, and a run never straddles now.
 */
function grouped(today: PlanMove[], current: PlanMove | undefined, nowMs: number, keep: (m: PlanMove) => boolean): RibbonItem[] {
  const out: RibbonItem[] = [];
  let run: PlanMove[] = [];
  const flush = () => {
    const tail: PlanMove[] = []; // a rest after the last block is a row of its own
    while (run.length > 0 && run[run.length - 1]?.kind === "rest") tail.unshift(run.pop() as PlanMove);
    if (run.filter((m) => m.kind === "study").length >= 2) out.push({ kind: "group", at: ms(run[0]?.start), moves: run });
    else for (const m of run) out.push({ kind: "block", at: ms(m.start), move: m });
    for (const m of tail) out.push({ kind: "block", at: ms(m.start), move: m });
    run = [];
  };
  for (const m of today) {
    const first = run[0];
    const sameSide = !first || ms(first.end) <= nowMs === ms(m.end) <= nowMs;
    const joins = !m.fixed && m !== current && !keep(m) && sameSide && (m.kind === "study" || (m.kind === "rest" && m.minutes <= BREAK_MAX && run.length > 0));
    if (joins) {
      run.push(m);
      continue;
    }
    flush();
    if (!m.fixed && m !== current && !keep(m) && m.kind === "study") run.push(m);
    else out.push({ kind: "block", at: ms(m.start), move: m });
  }
  flush();
  return out;
}
const ICON = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));
const ms = (iso: string | null | undefined) => (iso ? Date.parse(iso) : NaN);
const sentence = (text: string) => (text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : text);

export default function NowPage() {
  const now = useFetch((s) => api.dayNow(s));
  const patterns = useFetch((s) => api.dayPatterns(s));
  const brief = useBrief();
  useAtlasEvents(["day.", "workspace.action.completed", "workspace.approval.requested", "experience.approval.decided"], () => now.reload());
  useAtlasEvents(["habits.day.closed"], () => patterns.reload());

  if (now.error) {
    return (
      <>
        <div className={ui.pageHead}>
          <h1>Now</h1>
        </div>
        {now.error.notConfigured ? (
          <Empty title="The Day loop is not on this box">It needs the place packs (config/places) so it can plan where to be and how long the walks are.</Empty>
        ) : (
          <ErrorBox what="the day" error={now.error} onRetry={now.reload} />
        )}
      </>
    );
  }
  if (!now.data) {
    return (
      <>
        <div className={ui.pageHead}>
          <h1>Now</h1>
        </div>
        <Loading what="the day" />
      </>
    );
  }
  return <Day d={now.data} brief={brief} patterns={patterns.data?.patterns ?? []} reload={now.reload} />;
}

// ---------- the story ----------

interface Said {
  text: string;
  /** The narration behind it, when the box wrote one. */
  narration?: DayNarration;
  label?: string;
}

/** What leads the screen, and the quieter paragraph under it: the narrator's words when the box has them. */
function story(d: DayNow, brief: BriefState): { lead: Said; sub?: Said; what: "brief" | "status" } {
  const L = brief.latest;
  const today = (x: DayNarration | undefined) => (x && x.day === d.day ? x : undefined);
  const nowSaid = today(L.now);
  const morning = today(L.morning);
  const said = (x: DayNarration, label?: string): Said => ({ text: x.display, narration: x, label });
  if (d.replan && L.replan && L.replan.plan_id === d.replan.plan_id) return { lead: said(L.replan), sub: nowSaid ? said(nowSaid, "Right now") : undefined, what: "status" };
  if (d.review && today(L.evening)) return { lead: said(L.evening as DayNarration), sub: undefined, what: "brief" };
  if (nowSaid) return { lead: said(nowSaid), sub: morning ? said(morning, `The brief from ${time(morning.ts)}`) : undefined, what: "status" };
  if (morning) return { lead: said(morning), what: "brief" };
  return { lead: { text: fallbackLead(d) }, what: d.status ? "status" : "brief" };
}

/** The first paragraph on a box with no narrator: the same facts, plainly. */
function fallbackLead(d: DayNow): string {
  const s = d.status;
  if (d.replan) return d.replan.why;
  if (!d.plan) return "No plan for today yet. Atlas plans the day at wake time, or as soon as it starts after; you can also make one now.";
  if (!s) return d.plan.type === "day.plan.proposed" ? d.plan.text : d.plan.why;
  const doing = s.block_title ? `${s.block_title}${s.block_end ? ` until ${time(s.block_end)}` : ""}` : s.next_block ? `Free until ${s.next_block.title} at ${time(s.next_block.start)}` : "Nothing planned right now";
  const drift = s.drift_minutes > 0 ? `, ${s.drift_minutes} min behind` : s.drift_minutes < 0 ? `, ${-s.drift_minutes} min ahead` : "";
  const e = s.next_event;
  const next = e ? ` ${RISK_TEXT[s.risk]}: ${e.title} at ${time(e.start)}${e.place_name ? ` at ${e.place_name}` : ""}${e.leave_by && e.walk_minutes > 0 ? `, leave by ${time(e.leave_by)}` : ""}.` : "";
  return `${doing}${drift}.${next}`;
}

function useGreeting(): string {
  const [hello, setHello] = useState("Hello");
  useEffect(() => {
    const h = new Date().getHours();
    setHello(h < 5 ? "Still up" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening");
  }, []);
  return hello;
}

function Day({ d, brief, patterns, reload }: { d: DayNow; brief: BriefState; patterns: DayPattern[]; reload: () => void }) {
  const s = d.status;
  const voice = useVoice();
  const hello = useGreeting();
  const { lead, sub, what } = story(d, brief);
  const warn = s?.risk === "late" || s?.risk === "infeasible" || s?.risk === "leave_now" || Boolean(d.replan);
  const [speaking, setSpeaking] = useState(false);
  const nextRef = useRef<HTMLDivElement>(null);

  /** Say where things stand: asked of the box so it is current; the words on screen follow it. */
  const speak = async (kind: "brief" | "status" = what) => {
    if (speaking) {
      voice.stop();
      return;
    }
    setSpeaking(true);
    const fresh = d.plan ? await briefMe(kind) : null;
    const text = fresh?.spoken ?? lead.narration?.spoken ?? lead.text;
    voice.say(text, { tag: "answer", replace: true, raw: Boolean(fresh ?? lead.narration), onDone: () => setSpeaking(false) });
  };

  return (
    <>
      <section className={`${n.hero} ${warn ? n.heroWarn : ""}`} aria-live="polite">
        <h1 className={n.greeting}>
          <strong>{hello}</strong>
          <span>
            {dayLabel(`${d.day}T12:00:00`)}, as of {timeOrDay(d.now)}
          </span>
        </h1>
        <p className={n.lead}>{lead.text}</p>
        {sub ? (
          <p className={n.sub}>
            <span className={n.subLabel}>{sub.label}</span>
            {sub.text}
          </p>
        ) : null}
        <div className={n.heroRow}>
          <button
            type="button"
            className={`${n.speak} ${speaking ? n.speakOn : ""}`}
            disabled={voice.ready && !voice.canSpeak}
            onClick={() => void speak()}
            aria-pressed={speaking}
            title={voice.ready && !voice.canSpeak ? "This browser cannot speak" : undefined}
          >
            <svg viewBox="0 0 24 24" {...ICON} aria-hidden="true">
              <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" />
              <path d="M15.5 9a4.2 4.2 0 010 6M18 6.5a7.8 7.8 0 010 11" />
            </svg>
            {speaking ? "Stop" : "Brief me"}
          </button>
          <span className={n.engine}>
            {lead.narration ? (lead.narration.engine === "model" ? "In Atlas's own words" : "Atlas, from the plan and the cameras") : "From the plan and the cameras"}
          </span>
        </div>
      </section>

      <Action d={d} reload={reload} onLeave={() => (void speak("status"), nextRef.current?.scrollIntoView({ block: "center", behavior: "smooth" }))} />

      {d.replan ? <ReplanFold r={d.replan} /> : null}

      {s ? <Strip s={s} /> : null}

      {d.plan ? (
        <section className={ui.section} aria-label="Your day">
          <h2 className={ui.sectionTitle}>{d.plan.type === "day.replan.proposed" ? "Your day (re-planned)" : "Your day"}</h2>
          <Ribbon d={d} brief={brief} nextRef={nextRef} />
          <p className={n.legend}>
            <span>
              <i className={n.sq} /> on your calendar
            </span>
            <span>
              <i /> suggested by Atlas (tap for why)
            </span>
          </p>
        </section>
      ) : null}

      <Why d={d} patterns={patterns} />
    </>
  );
}

// ---------- one primary action at a time ----------

function Action({ d, reload, onLeave }: { d: DayNow; reload: () => void; onLeave: () => void }) {
  const { approvals } = useAtlas();
  const [busy, setBusy] = useState<"plan" | "here" | null>(null);
  const [error, setError] = useState<ApiError | undefined>();
  const s = d.status;
  const run = (body: { from_here?: boolean }, which: "plan" | "here") => {
    setBusy(which);
    setError(undefined);
    api
      .dayMakePlan(body)
      .then(() => reload())
      .catch((err: unknown) => setError(asApiError(err)))
      .finally(() => setBusy(null));
  };
  const waiting = (id: string | null | undefined) => Boolean(id) && (approvals.loading || approvals.items.some((a) => a.id === id));
  const planAgain = (primary: boolean, label = "Plan again") => (
    <button key="plan" type="button" className={`${ui.btn} ${primary ? `${ui.btnPrimary} ${n.primary}` : ""}`} disabled={busy !== null} onClick={() => run({}, "plan")}>
      {busy === "plan" ? "Planning…" : label}
    </button>
  );
  const fromHere = (primary: boolean) => (
    <button key="here" type="button" className={`${ui.btn} ${primary ? `${ui.btnPrimary} ${n.primary}` : ""}`} disabled={busy !== null} onClick={() => run({ from_here: true }, "here")}>
      {busy === "here" ? "Planning…" : "Re-plan from here"}
    </button>
  );
  const dayRun = (primary: boolean) => (
    <Link key="dayrun" href="/dayrun/?plan=day" className={`${ui.btn} ${primary ? `${ui.btnPrimary} ${n.primary}` : ""}`}>
      Change it in Day Run
    </Link>
  );
  const approve = (primary: boolean, label: string) => (
    <Link key="approve" href="/approvals/" className={`${ui.btn} ${primary ? `${ui.btnPrimary} ${n.primary}` : ""}`}>
      {label}
    </Link>
  );

  let primary: ReactNode;
  let hint: string;
  let rest: ReactNode[];
  const e = s?.next_event;
  if (!d.plan) {
    primary = planAgain(true, "Plan my day now");
    hint = "A starting point from what Atlas already knows. Nothing is written until you approve its card.";
    rest = [];
  } else if (d.replan && waiting(d.replan.action_id)) {
    primary = approve(true, "Review the new plan");
    hint = "Approve or deny it in Approvals. Until then the day stays as it was.";
    rest = [fromHere(false), dayRun(false), planAgain(false)];
  } else if (s?.risk === "leave_now" && e) {
    primary = (
      <button key="leave" type="button" className={`${ui.btn} ${ui.btnPrimary} ${n.primary}`} onClick={onLeave}>
        Leave now
      </button>
    );
    hint = `${e.title} at ${time(e.start)}${e.place_name ? `, ${e.place_name}` : ""}${e.walk_minutes ? `: a ${e.walk_minutes} min walk` : ""}.`;
    rest = [fromHere(false), dayRun(false), planAgain(false)];
  } else if (s && (s.risk === "late" || s.risk === "infeasible")) {
    primary = fromHere(true);
    hint = e ? `${RISK_TEXT[s.risk]}: ${e.title} at ${time(e.start)}. A new plan from where you are, with one card to approve.` : "A new plan from where you are, with one card to approve.";
    rest = [dayRun(false), planAgain(false)];
  } else if (!d.committed && waiting(d.plan.action_id)) {
    primary = approve(true, "Approve the plan");
    hint = "One card in Approvals puts the blocks on your calendar. Nothing is written until then.";
    rest = [fromHere(false), dayRun(false), planAgain(false)];
  } else if (s && s.drift_minutes >= 20) {
    primary = fromHere(true);
    hint = `You are about ${s.drift_minutes} minutes behind. A new plan from where you are, with one card to approve.`;
    rest = [dayRun(false), planAgain(false)];
  } else {
    primary = dayRun(true);
    hint = d.committed ? "Approved and on your calendar. It is still yours to change." : d.plan.action_id ? "Its card has expired or was answered; plan again for a new one, or edit it yourself." : "The plan is a starting point: edit it yourself.";
    rest = [fromHere(false), planAgain(false)];
  }

  return (
    <section className={n.action} aria-label="What to do">
      {primary}
      <p className={n.hint}>{hint}</p>
      {error ? <WriteError error={error} /> : null}
      {rest.length > 0 ? (
        <details className={n.more}>
          <summary>More</summary>
          <div className={n.moreBody}>{rest}</div>
        </details>
      ) : null}
    </section>
  );
}

function ReplanFold({ r }: { r: DayReplan }) {
  return (
    <details className={n.why}>
      <summary>The new plan, block by block ({r.moves.filter((m) => !m.fixed).length} to add)</summary>
      <div className={n.whyBody}>
        <p className={ui.breakWords}>{r.why}</p>
        <p className={`${ui.muted} ${ui.small}`}>
          {r.asked ? "You asked for it." : "Drift triggered it."} {r.outcome.summary} {r.outcome.points} points.
        </p>
        <Blocks moves={r.moves} />
      </div>
    </details>
  );
}

// ---------- the day so far, as a strip ----------

function Strip({ s }: { s: DayStatus }) {
  const m = s.meters;
  const cells: [string, string][] = [
    [String(m.meals), m.meals === 1 ? "meal seen" : "meals seen"],
    [String(m.desk_minutes), "min at the desk"],
    [String(m.recycled), "recycled"],
    [m.sleep_hours === null ? "–" : `${m.sleep_hours.toFixed(1)} h`, m.sleep_hours === null ? "sleep not measured" : m.sleep_source === "watch" ? "slept (watch)" : "slept (desk camera)"],
  ];
  return (
    <section className={n.strip} aria-label="Today so far">
      {cells.map(([num, label]) => (
        <div key={label} className={n.cell}>
          <p className={n.cellNum}>{num}</p>
          <p className={n.cellLabel}>{label}</p>
        </div>
      ))}
    </section>
  );
}

// ---------- the ribbon ----------

type RibbonItem =
  | { kind: "block"; at: number; move: PlanMove }
  | { kind: "group"; at: number; moves: PlanMove[] }
  | { kind: "here"; at: number }
  | { kind: "review"; at: number; review: DayReview };

/** Why a block is in the plan, from the inputs the plan names. */
function reasonFor(m: PlanMove, plan: DayPlan | DayReplan): string {
  const sources: PlanSource[] = plan.type === "day.plan.proposed" ? plan.sources : [];
  const find = (test: (s: PlanSource) => boolean) => sources.find(test);
  const friend = m.with_friend ? ` With ${humanize(m.with_friend)}, who shared a place.` : "";
  if (m.fixed) {
    const src = find((s) => s.id === m.id);
    return `It is on your calendar${src?.origin ? ` (${ORIGIN_TEXT[src.origin] ?? src.origin})` : ""}: Atlas plans around it and never moves it.`;
  }
  if (m.kind === "study") {
    const due = m.deadline ? find((s) => s.id === m.deadline) : undefined;
    const coach = find((s) => s.origin === "coach");
    return `${due ? `Because ${due.text}.` : "Time for your main work, where the plan found room for it."}${coach ? " Its length follows the coach's advice from yesterday." : ""}${friend}`;
  }
  if (m.kind === "eat") {
    const meals = find((s) => s.kind === "habit" && s.id === "meals");
    return `A meal here keeps your energy from running out before the next block${meals ? ` (${meals.text})` : ""}.${friend}`;
  }
  if (m.kind === "rest") {
    const sleep = find((s) => s.kind === "habit" && s.id === "sleep");
    return `${sleep ? `Because you ${sleep.text}: ` : ""}${sleep ? "a" : "A"} rest brings your energy back before what comes next.`;
  }
  if (m.kind === "sleep") return `${m.minutes < NIGHT_MIN ? "A nap to get you through the day" : `The plan protects ${plan.outcome.sleep_hours} h of sleep`}; tomorrow starts from tonight.`;
  return `Suggested by Atlas where the day had room for it.${friend}`;
}

function Ribbon({ d, brief, nextRef }: { d: DayNow; brief: BriefState; nextRef: RefObject<HTMLDivElement | null> }) {
  const plan = d.plan as DayPlan | DayReplan;
  const s = d.status;
  const nowMs = ms(d.now);
  const [open, setOpen] = useState<string | null>(null);
  const w = useWords();
  const moves = [...plan.moves].sort((a, b) => ms(a.start) - ms(b.start));
  const nights = moves.filter((m) => m.kind === "sleep" && m.minutes >= NIGHT_MIN);
  const night = nights.length > 0 ? nights.reduce((a, b) => (b.minutes > a.minutes ? b : a)) : undefined;
  const today = moves.filter((m) => !night || ms(m.start) <= ms(night.start));
  const tomorrow = moves.filter((m) => night && ms(m.start) >= ms(night.end));
  const current = today.find((m) => ms(m.start) <= nowMs && nowMs < ms(m.end));

  const notes = new Map<string, DayCoachNote>();
  for (const note of [d.coach, ...(d.coach_earlier ?? [])]) if (note && note.plan_id === plan.plan_id && !notes.has(note.block_id)) notes.set(note.block_id, note);
  const sessions = new Map<string, CoachSession>();
  for (const x of d.review?.sessions ?? []) sessions.set(x.block_id, x);

  // a block the coach spoke about that is not in today's ribbon (yesterday's plan, or one a re-plan replaced) leads it
  const shown = new Set([...today, ...tomorrow].map((m) => m.id));
  const earlier = [d.coach, ...(d.coach_earlier ?? [])].filter((x): x is DayCoachNote => Boolean(x) && !(x!.plan_id === plan.plan_id && shown.has(x!.block_id)));

  const items: RibbonItem[] = grouped(today, current, nowMs, (m) => s?.next_event?.id === m.id);
  if (!current) items.push({ kind: "here", at: nowMs });
  if (d.review) items.push({ kind: "review", at: ms(d.review.ts), review: d.review });
  items.sort((a, b) => a.at - b.at);

  const hereLine = s?.seen_at_name ? `Seen at ${s.seen_at_name} (${s.seen_how}, ${time(s.seen_ts)})` : "Not seen yet today";
  const lastSeen = d.last_seen
    ? `Last on camera: ${d.last_seen.caption ?? `${d.last_seen.labels.join(", ") || "activity"} on the ${cameraName(d.last_seen.camera_id).toLowerCase()}`} (${cameraName(d.last_seen.camera_id)}, ${timeOrDay(d.last_seen.start)}${d.last_seen.end ? ` to ${time(d.last_seen.end)}` : ", still going"})`
    : null;
  const drift = s && s.drift_minutes > 0 ? <span className={`${n.pill} ${n.pillWarn}`}>{s.drift_minutes} min behind</span> : s && s.drift_minutes < 0 ? <span className={`${n.pill} ${n.pillGood}`}>{-s.drift_minutes} min ahead</span> : s ? <span className={`${n.pill} ${n.pillGood}`}>on plan</span> : null;

  const block = (m: PlanMove, tomorrowBlock = false) => {
    const past = !tomorrowBlock && ms(m.end) <= nowMs;
    const isNow = !tomorrowBlock && m === current;
    const isNext = !tomorrowBlock && s?.next_event?.id === m.id && !isNow;
    const e = isNext || (isNow && s?.next_event?.id === m.id) ? s?.next_event : undefined;
    const risky = Boolean(e) && s !== null && s.risk !== "ok";
    const note = notes.get(m.id);
    const session = note?.session ?? sessions.get(m.id);
    const followed = d.review ? (d.review.followed.includes(m.title) ? "happened" : d.review.slipped.includes(m.title) ? "slipped" : null) : null;
    const pct = isNow ? Math.min(100, Math.max(0, ((nowMs - ms(m.start)) / Math.max(1, ms(m.end) - ms(m.start))) * 100)) : 0;
    const said = note && brief.latest.coach && brief.latest.coach.ref === note.event_id ? brief.latest.coach.display : note?.text;
    const key = `${tomorrowBlock ? "t" : ""}${m.id}:${m.start}`;
    return (
      <div key={key} className={`${n.item} ${past ? n.past : ""}`} ref={isNext || (e && isNow) ? nextRef : undefined}>
        <span className={n.when}>{time(m.start)}</span>
        <span className={n.rail}>
          <span className={`${n.dot} ${m.fixed ? n.dotFixed : ""} ${isNow ? n.dotNow : ""}`} />
        </span>
        <div>
          <button
            type="button"
            className={`${n.body} ${m.fixed ? n.bodyFixed : ""} ${isNow ? n.bodyNow : ""} ${isNext ? n.bodyNext : ""} ${risky ? n.bodyWarn : ""}`}
            aria-expanded={open === key}
            onClick={() => setOpen(open === key ? null : key)}
          >
            <span className={n.titleRow}>
              <span className={n.title}>
                {m.title}
                {m.with_friend ? ` with ${humanize(m.with_friend)}` : ""}
              </span>
              <span className={n.tags}>
                {isNow ? <span className={`${n.pill} ${n.pillHere}`}>you are here</span> : null}
                {isNow ? drift : null}
                {e && risky ? <span className={`${n.pill} ${n.pillWarn}`}>{RISK_TEXT[s!.risk]}</span> : null}
                {followed === "happened" ? <span className={`${n.pill} ${n.pillGood}`}>happened</span> : followed === "slipped" ? <span className={n.pill}>slipped</span> : null}
              </span>
            </span>
            <span className={n.meta}>
              {time(m.start)} to {time(m.end)}
              {m.place_name && !m.title.includes(m.place_name) ? ` · ${m.place_name}` : ""}
              {isNow && s?.should_be_name ? ` · the plan has you at ${s.should_be_name}` : ""}
            </span>
            {isNow ? (
              <>
                <span className={n.progress} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} aria-label="How far through this block">
                  <span className={n.progressFill} style={{ width: `${pct}%`, display: "block" }} />
                </span>
                <span className={n.meta}>{hereLine}</span>
              </>
            ) : null}
            {e ? (
              <span className={n.leave}>
                {e.walk_minutes > 0 && e.leave_by ? `Leave by ${time(e.leave_by)} · ${e.walk_minutes} min walk` : "You are already there"}
              </span>
            ) : null}
            {past && session ? (
              <span className={n.meta}>
                {session.measured ? `${session.desk_minutes ?? 0} of ${session.planned_minutes} min at the desk, ${session.breaks.length} ${session.breaks.length === 1 ? "break" : "breaks"}` : `Not measured (${session.not_measured ?? "the box could not see it"})`}
              </span>
            ) : null}
            {open === key ? <span className={n.reason}>{reasonFor(m, plan)}</span> : null}
          </button>
          {note ? <CoachNote note={note} said={said ?? note.text} /> : null}
        </div>
      </div>
    );
  };

  const group = (run: PlanMove[]) => {
    const first = run[0] as PlanMove;
    const last = run[run.length - 1] as PlanMove;
    const blocks = run.filter((m) => m.kind === "study");
    const past = ms(last.end) <= nowMs;
    const key = `g:${first.id}:${first.start}`;
    const measured = blocks.map((m) => notes.get(m.id)?.session ?? sessions.get(m.id)).filter((x): x is CoachSession => Boolean(x?.measured));
    const each = Math.round(blocks.reduce((sum, m) => sum + m.minutes, 0) / blocks.length);
    return (
      <div key={key} className={`${n.item} ${past ? n.past : ""}`}>
        <span className={n.when}>{time(first.start)}</span>
        <span className={n.rail}>
          <span className={n.dot} />
        </span>
        <div>
          <button type="button" className={n.body} aria-expanded={open === key} onClick={() => setOpen(open === key ? null : key)}>
            <span className={n.titleRow}>
              <span className={n.title}>
                {blocks.length} {w.activity} blocks{first.place_name ? ` at ${first.place_name}` : ""}
              </span>
              <span className={n.tags}>
                <span className={n.pill}>{each} min each</span>
              </span>
            </span>
            <span className={n.meta}>
              {time(first.start)} to {time(last.end)}, short breaks between
            </span>
            {past && measured.length > 0 ? (
              <span className={n.meta}>
                {measured.reduce((sum, x) => sum + (x.desk_minutes ?? 0), 0)} of {measured.reduce((sum, x) => sum + x.planned_minutes, 0)} min at the {w.workplace}
              </span>
            ) : null}
            {open === key ? <span className={n.reason}>{reasonFor(first, plan)}</span> : null}
          </button>
          {open === key ? <div className={n.ribbon}>{run.map((m) => block(m))}</div> : blocks.map((m) => notes.get(m.id)).map((note) => (note ? <CoachNote key={note.event_id} note={note} said={note.text} about={note.session.title} /> : null))}
        </div>
      </div>
    );
  };

  return (
    <div className={n.ribbon}>
      {[...earlier].reverse().map((note) => (
        <div key={note.event_id} className={n.item}>
          <span className={n.when}>{time(note.session.end)}</span>
          <span className={n.rail}>
            <span className={`${n.dot} ${n.dotNote}`} />
          </span>
          <CoachNote
            note={note}
            said={brief.latest.coach && brief.latest.coach.ref === note.event_id ? brief.latest.coach.display : note.text}
            about={`${note.session.title}, ${timeOrDay(note.session.start)} to ${time(note.session.end)}`}
          />
        </div>
      ))}
      {items.map((it) => {
        if (it.kind === "block") return block(it.move);
        if (it.kind === "group") return group(it.moves);
        if (it.kind === "here")
          return (
            <div key="here" className={n.item}>
              <span className={n.when}>{time(d.now)}</span>
              <span className={n.rail}>
                <span className={`${n.dot} ${n.dotNow}`} />
              </span>
              <div className={n.here}>
                <span className={n.titleRow}>
                  <span className={n.title}>{s?.next_block ? `Free until ${time(s.next_block.start)}` : "Nothing planned right now"}</span>
                  <span className={n.tags}>
                    <span className={`${n.pill} ${n.pillHere}`}>you are here</span>
                    {drift}
                  </span>
                </span>
                <span className={n.meta}>
                  {hereLine}
                  {s?.next_block?.place_name ? ` · next at ${s.next_block.place_name}` : ""}
                </span>
                {lastSeen ? <span className={n.meta}>{lastSeen}</span> : null}
              </div>
            </div>
          );
        return (
          <div key="review" className={n.item}>
            <span className={n.when}>{time(it.review.ts)}</span>
            <span className={n.rail}>
              <span className={`${n.dot} ${n.dotNote}`} />
            </span>
            <ReviewNote r={it.review} said={brief.latest.evening && brief.latest.evening.ref === it.review.event_id ? brief.latest.evening.display : null} label="Evening review" />
          </div>
        );
      })}
      {tomorrow.length > 0 ? (
        <div className={n.item}>
          <span className={n.when}>{dayLabel(tomorrow[0]?.start)}</span>
          <span className={n.rail}>
            <span className={`${n.dot} ${n.dotNote}`} />
          </span>
          <details className={n.fold}>
            <summary>
              The morning after: {tomorrow.length} {tomorrow.length === 1 ? "block" : "blocks"} from {time(tomorrow[0]?.start)}
            </summary>
            <div className={n.ribbon}>{tomorrow.map((m) => block(m, true))}</div>
          </details>
        </div>
      ) : null}
    </div>
  );
}

/** The coach, where the block ended: what it said, and behind a fold the numbers and the evidence. */
function CoachNote({ note, said, about }: { note: DayCoachNote; said: string; about?: string }) {
  return (
    <div className={n.note}>
      <span className={n.noteLabel}>Coach · {relative(note.ts)}</span>
      {about ? <span className={n.meta}>{about}</span> : null}
      <p className={n.noteText}>{said}</p>
      <details className={n.fold}>
        <summary>The numbers and the evidence</summary>
        <div className={n.foldBody}>
          <SessionRows s={note.session} />
          <Advice suggestions={note.suggestions} calibration={note.calibration} />
          <p className={`${ui.muted} ${ui.small}`}>The camera sees the desk, not the screen, and a trigger is only what happened in the two minutes before a break.</p>
        </div>
      </details>
    </div>
  );
}

function ReviewNote({ r, said, label }: { r: DayReview; said: string | null; label: string }) {
  return (
    <div className={n.note}>
      <span className={n.titleRow}>
        <span className={n.noteLabel}>{label}</span>
        <span className={`${n.pill} ${r.stars > 0 ? n.pillGood : ""}`}>
          {r.score === null ? "not measured" : `${r.score}% followed`} · {"★".repeat(r.stars) || "no stars"} · {r.points} points
        </span>
      </span>
      <p className={n.noteText}>{said ?? `${r.score === null ? "Nothing could be measured." : `You followed ${r.score}%.`} Tomorrow: ${r.change_tomorrow}`}</p>
      <details className={n.fold}>
        <summary>The whole review</summary>
        <div className={n.foldBody}>
          <p className={`${ui.small} ${ui.breakWords}`}>{r.text}</p>
          {r.followed.length > 0 ? <p className={ui.small}>Happened: {r.followed.join(", ")}.</p> : null}
          {r.slipped.length > 0 ? <p className={ui.small}>Slipped: {r.slipped.join(", ")}.</p> : null}
          <p className={ui.nudge}>Tomorrow: {r.change_tomorrow}</p>
          {r.sessions && r.sessions.length > 0 ? (
            <>
              <h3>Coach</h3>
              {r.coach_text ? <p className={`${ui.small} ${ui.breakWords}`}>{r.coach_text}</p> : null}
              {r.sessions.map((x) => (
                <SessionRows key={`${x.block_id}:${x.start}`} s={x} />
              ))}
              <Advice suggestions={r.suggestions ?? []} calibration={r.calibration ?? null} />
            </>
          ) : null}
          <p className={`${ui.muted} ${ui.small}`}>
            Streak {r.streak} {r.streak === 1 ? "day" : "days"} with a star. The plan would have earned {r.plan_stars}.
          </p>
        </div>
      </details>
    </div>
  );
}

// ---------- why this plan ----------

function Why({ d, patterns }: { d: DayNow; patterns: DayPattern[] }) {
  const p = d.plan;
  const s = d.status;
  return (
    <details className={n.why}>
      <summary>Why this plan</summary>
      <div className={n.whyBody}>
        {p ? (
          <>
            <p className={`${ui.small} ${ui.breakWords}`}>
              {p.outcome.summary} {p.outcome.points} points. {p.type === "day.plan.proposed" && p.by === "person" ? "Your own moves" : "Chosen by Atlas"}
              {d.committed ? ", approved and on your calendar." : p.action_id ? ", not approved yet." : "."}
            </p>
            {p.type === "day.replan.proposed" ? <p className={`${ui.small} ${ui.breakWords}`}>{p.why}</p> : null}
            {p.type === "day.plan.proposed" ? (
              <>
                <h3>What it was built from</h3>
                <Sources sources={p.sources} missing={p.missing} />
                <details className={n.fold}>
                  <summary>The plan as the box wrote it</summary>
                  <p className={`${ui.small} ${ui.muted} ${ui.breakWords}`}>{p.text}</p>
                </details>
              </>
            ) : null}
            <details className={n.fold}>
              <summary>Every block ({p.moves.length})</summary>
              <Blocks moves={p.moves} />
            </details>
          </>
        ) : (
          <p className={`${ui.muted} ${ui.small}`}>Atlas plans at wake time from what it already knows. It is a starting point: remake it, re-plan from where you are, or edit it in Day Run.</p>
        )}

        {s ? (
          <>
            <h3>Right now, in detail</h3>
            <dl className={ui.kv}>
              <dt>Should be</dt>
              <dd>{s.should_be_name ?? (s.block_id ? "wherever you are (online)" : s.next_block?.place_name ? `${s.next_block.place_name} by ${time(s.next_block.start)}` : "anywhere")}</dd>
              <dt>Seen</dt>
              <dd>{s.seen_at_name ? `${s.seen_at_name} (${s.seen_how}, ${time(s.seen_ts)})` : "not seen yet today"}</dd>
              {s.block_end ? (
                <>
                  <dt>Until</dt>
                  <dd>{time(s.block_end)}</dd>
                </>
              ) : null}
              {s.next_block ? (
                <>
                  <dt>Then</dt>
                  <dd>
                    {s.next_block.title} at {time(s.next_block.start)}
                  </dd>
                </>
              ) : null}
              {s.next_event ? (
                <>
                  <dt>Next fixed</dt>
                  <dd>
                    {RISK_TEXT[s.risk]}: {s.next_event.title} at {time(s.next_event.start)}
                    {s.next_event.place_name ? ` at ${s.next_event.place_name}` : ""}
                    {s.next_event.walk_minutes > 0 && s.next_event.leave_by ? ` (${s.next_event.walk_minutes} min walk, leave by ${time(s.next_event.leave_by)})` : ""}
                  </dd>
                </>
              ) : null}
            </dl>
            <p className={`${ui.muted} ${ui.small}`}>
              {s.reason ? `${sentence(s.reason)}, ` : ""}
              {relative(s.now)}. Deadline risk {Math.round(s.deadline_risk * 100)}%.
            </p>
          </>
        ) : (
          <p className={`${ui.muted} ${ui.small}`}>{p ? "The status updates as the cameras and connectors report." : "Meals, desk time, recycling and last night's sleep appear once the day has a status."}</p>
        )}

        <h3>What Atlas sees</h3>
        {d.last_seen ? (
          <p className={`${ui.small} ${ui.breakWords}`}>
            {d.last_seen.caption ?? `${d.last_seen.labels.join(", ") || "Activity"} on the ${cameraName(d.last_seen.camera_id).toLowerCase()}`}{" "}
            <span className={ui.muted}>
              ({cameraName(d.last_seen.camera_id)}, {timeOrDay(d.last_seen.start)}
              {d.last_seen.end ? ` to ${time(d.last_seen.end)}` : ", still going"})
            </span>
          </p>
        ) : (
          <p className={`${ui.muted} ${ui.small}`}>No camera activity yet. The last thing a camera saw appears here.</p>
        )}

        <h3>Friends</h3>
        {d.friends.length === 0 ? (
          <p className={`${ui.muted} ${ui.small}`}>Nobody in your Circle is sharing a place right now.</p>
        ) : (
          d.friends.map((f) => (
            <p key={f.member_id} className={ui.small}>
              <strong>{humanize(f.member_id)}</strong> is at {f.place}{" "}
              <span className={ui.muted}>
                (shared {timeOrDay(f.ts)}, until {time(f.expires_at)})
              </span>
            </p>
          ))
        )}

        {patterns.length > 0 ? (
          <>
            <h3>Patterns</h3>
            <p className={`${ui.muted} ${ui.small}`}>Counted from the habit history on this box, nothing fitted.</p>
            {patterns.map((x) => (
              <p key={x.text} className={ui.small}>
                {x.text}
              </p>
            ))}
          </>
        ) : null}

        {!d.review && d.yesterday ? (
          <>
            <h3>Yesterday</h3>
            <ReviewNote r={d.yesterday} said={null} label="Yesterday's review" />
          </>
        ) : null}
      </div>
    </details>
  );
}

// ---------- the pieces the folds are made of ----------

function Blocks({ moves }: { moves: PlanMove[] }) {
  return (
    <ul className={ui.list}>
      {moves.map((m) => (
        <li key={`${m.id}:${m.start}`} className={`${ui.row} ${ui.between} ${ui.wrap} ${ui.small}`}>
          <span>
            {m.fixed ? <span className={ui.tag}>on the calendar</span> : null} {m.title}
            {m.with_friend ? ` with ${m.with_friend}` : ""}
          </span>
          <span className={ui.muted}>
            {timeOrDay(m.start)} to {time(m.end)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Sources({ sources, missing }: { sources: PlanSource[]; missing: string[] }) {
  const named = sources.filter((s) => s.kind !== "config" && s.kind !== "place");
  const rest = sources.filter((s) => s.kind === "config" || s.kind === "place");
  return (
    <div className={ui.sources}>
      {named.map((s) => (
        <div key={`${s.kind}:${s.id}`} className={ui.source}>
          <span>{s.text}</span>
          <span className={`${ui.muted} ${ui.small}`}>
            {s.origin ? ORIGIN_TEXT[s.origin] ?? s.origin : s.kind} · {s.id}
          </span>
        </div>
      ))}
      {rest.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>{rest.map((s) => sentence(s.text)).join(". ")}.</p> : null}
      {missing.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>Not available: {missing.join("; ")}.</p> : null}
    </div>
  );
}

/** One piece of evidence: what it was and when; a camera moment opens its picture when the box stored one. */
function EvidenceChip({ e }: { e: CoachEvidence }) {
  const [missing, setMissing] = useState(false);
  const words = `${EVIDENCE_TEXT[e.kind]}, ${time(e.at)}${e.title ? `: ${e.title}` : ""}`;
  if (!e.frame_ref || missing) {
    return (
      <li className={ui.source}>
        <span className={ui.breakWords}>{words}</span>
        <span className={`${ui.muted} ${ui.small}`}>
          {e.frame_ref ? "picture not stored on this box · " : ""}
          {e.id}
        </span>
      </li>
    );
  }
  return (
    <li className={ui.source}>
      <a href={keyframeUrl(e.frame_ref)} target="_blank" rel="noreferrer noopener" className={ui.breakWords}>
        {words}
      </a>
      {/* probes for the picture: without one the row falls back to plain words */}
      <img src={keyframeUrl(e.frame_ref)} alt="" hidden onError={() => setMissing(true)} />
      <span className={`${ui.muted} ${ui.small}`}>tap for the picture · {e.id}</span>
    </li>
  );
}

function SessionRows({ s }: { s: CoachSession }) {
  if (!s.measured) {
    return (
      <p className={`${ui.muted} ${ui.small}`}>
        {s.title}, {time(s.start)} to {time(s.end)}: not measured ({s.not_measured ?? "the box could not see it"}).
      </p>
    );
  }
  const evidence = s.breaks.flatMap((b) => b.evidence);
  const cells: [string, string][] = [
    [`${s.desk_minutes ?? 0}/${s.planned_minutes}`, "min done / planned"],
    [String(s.breaks.length), s.breaks.length === 1 ? "break" : "breaks"],
    [String(s.longest_stretch_minutes ?? 0), "longest stretch"],
    [String(s.late_start_minutes ?? 0), "min to sit down"],
  ];
  return (
    <>
      <div className={n.strip}>
        {cells.map(([num, label]) => (
          <div key={label} className={n.cell}>
            <p className={n.cellNum}>{num}</p>
            <p className={n.cellLabel}>{label}</p>
          </div>
        ))}
      </div>
      <p className={`${ui.small} ${ui.breakWords}`}>{s.text}</p>
      {evidence.length > 0 ? (
        <ul className={ui.sources}>
          {evidence.map((e) => (
            <EvidenceChip key={`${e.kind}:${e.id}:${e.at}`} e={e} />
          ))}
        </ul>
      ) : null}
    </>
  );
}

function Advice({ suggestions, calibration }: { suggestions: string[]; calibration?: CoachCalibration | null }) {
  return (
    <>
      {suggestions.map((t) => (
        <p key={t} className={ui.nudge}>
          {t}
        </p>
      ))}
      {calibration ? <p className={`${ui.muted} ${ui.small}`}>{calibration.text}</p> : null}
    </>
  );
}
