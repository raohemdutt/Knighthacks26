"use client";

import Link from "next/link";
import { useState } from "react";
import { Keyframe } from "@/components/Keyframe";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CheckinCard as Card, type CheckinEvidence, type CheckinKind, type CheckinSignal } from "@/lib/api";
import { cameraName, dayLabel, relative, time } from "@/lib/format";
import c from "./checkin.module.css";

/**
 * The check-in (contexts/day/application/checkin.py): Atlas speaking up unasked, as one card at the top of the day.
 *
 * The sentence, then one chip per signal. A chip opens what is behind it, looked up on the box: the night against
 * your usual one, the resting heart rate as a line against your own baseline (its shape only; the values are
 * fetched and shown only after "Show numbers", on this device), how the looks read, the drink's last frame, the
 * earlier days. When the look memory has an earlier look like this one, one more chip ("Looked tired · like Tue") opens
 * that remembered moment in one or two plain lines: what was going on, and what helped. "Done" puts it away; "Move the block" asks the box for a re-plan, which changes nothing until its
 * card is approved.
 *
 * `ahead` is one quiet line under the chips ("You usually start to look strained around now. A short break tends to
 * help."), from what Atlas has learned about this person's looks. It is on this card only: never said aloud, never pushed.
 */

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));
const hm = (minutes: number) => `${Math.floor(minutes / 60)} h ${Math.round(minutes % 60)} min`;

/**
 * With no check-in on screen, the same quiet line on its own: the person has been at the desk about as long as it
 * usually takes for a look to start. Shown, never said. Nothing renders when there is nothing to say.
 */
export function LookAhead({ line }: { line?: string }) {
  if (!line) return null;
  return (
    <p className={`${ui.muted} ${ui.small}`} data-guide="checkin:look-ahead" aria-label="A pattern Atlas has noticed">
      {line}{" "}
      <Link href="/people/" className={c.link}>
        How Atlas knows
      </Link>
    </p>
  );
}

export function CheckinCard({ card, said, onChanged }: { card: Card; said?: string; onChanged: () => void }) {
  const [open, setOpen] = useState<CheckinKind | "callback" | null>(null);
  const [busy, setBusy] = useState<"done" | "move" | null>(null);
  const [gone, setGone] = useState(false);
  const [error, setError] = useState<ApiError | undefined>();
  const [numbers, setNumbers] = useState<CheckinEvidence["resting_hr"] | null>(null);

  if (gone || card.answered === "done") return null;
  const moved = card.answered === "move";
  const shown = card.signals.find((s) => s.kind === open);

  const answer = (which: "done" | "move") => {
    setBusy(which);
    setError(undefined);
    (which === "done" ? api.dayCheckinDone(card.event_id) : api.dayCheckinMove(card.event_id))
      .then(() => {
        if (which === "done") setGone(true);
        onChanged();
      })
      .catch((err: unknown) => setError(asApiError(err)))
      .finally(() => setBusy(null));
  };
  const showNumbers = () => {
    api
      .dayCheckin(true)
      .then((v) => setNumbers(v.latest?.event_id === card.event_id ? (v.latest.evidence.resting_hr ?? null) : null))
      .catch((err: unknown) => setError(asApiError(err)));
  };

  return (
    <section className={c.card} aria-label="Check-in from Atlas">
      <p className={c.label}>
        <span>Check-in · {relative(card.ts)}</span>
        <span className={c.unasked}>Atlas spoke up</span>
      </p>
      <p className={c.said}>{said ?? card.display}</p>

      <div className={`${ui.chips} ${c.chips}`}>
        {card.signals.map((s) => (
          <button
            key={s.kind}
            type="button"
            className={`${ui.chip} ${c.chip} ${open === s.kind ? ui.chipOn : ""}`}
            aria-expanded={open === s.kind}
            onClick={() => setOpen(open === s.kind ? null : s.kind)}
          >
            {s.summary}
          </button>
        ))}
        {card.callback ? (
          <button
            type="button"
            className={`${ui.chip} ${c.chip} ${open === "callback" ? ui.chipOn : ""}`}
            aria-expanded={open === "callback"}
            data-guide="checkin:look-back"
            onClick={() => setOpen(open === "callback" ? null : "callback")}
          >
            {card.callback.chip}
          </button>
        ) : null}
      </div>

      {open === "callback" && card.callback ? (
        <div className={c.evidence} data-look-back={card.callback.day}>
          {card.callback.lines.map((line) => (
            <p key={line} className={c.detail}>
              {line}
            </p>
          ))}
          <p className={c.fine}>
            A look Atlas remembers, kept on this box as words and numbers, never a picture.{" "}
            <Link href="/people/" className={c.link}>
              See or forget them on People
            </Link>
          </p>
        </div>
      ) : shown ? (
        <div className={c.evidence}>
          <p className={c.detail}>{shown.detail}</p>
          <Behind signal={shown} evidence={card.evidence} numbers={numbers} onNumbers={showNumbers} />
        </div>
      ) : (
        <p className={c.hint}>Tap one to see what is behind it.</p>
      )}

      {card.ahead ? (
        <p className={c.fine} data-guide="checkin:look-ahead">
          {card.ahead}{" "}
          <Link href="/people/" className={c.link}>
            How Atlas knows
          </Link>
        </p>
      ) : null}

      {moved ? (
        <p className={c.moved}>
          A new plan is waiting for your yes. Nothing changes until you approve it.{" "}
          <Link href="/approvals/" className={c.link}>
            Review it
          </Link>
        </p>
      ) : (
        <div className={c.buttons}>
          <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => answer("done")}>
            {busy === "done" ? "Putting it away…" : "Done"}
          </button>
          {card.proposes_replan ? (
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy !== null} onClick={() => answer("move")}>
              {busy === "move" ? "Planning…" : "Move the block"}
            </button>
          ) : null}
        </div>
      )}
      {error ? <WriteError error={error} /> : null}
      <p className={c.foot}>Read from what this box recorded and kept on it. Not medical advice.</p>
    </section>
  );
}

function Behind({
  signal,
  evidence,
  numbers,
  onNumbers,
}: {
  signal: CheckinSignal;
  evidence: CheckinEvidence;
  numbers: CheckinEvidence["resting_hr"] | null;
  onNumbers: () => void;
}) {
  if (signal.kind === "sleep" && evidence.sleep) return <Night e={evidence.sleep} />;
  if (signal.kind === "resting_hr" && evidence.resting_hr) return <Resting e={evidence.resting_hr} numbers={numbers} onNumbers={onNumbers} />;
  if (signal.kind === "expression" && evidence.expression) return <Looks e={evidence.expression} />;
  if (signal.kind === "sitting" && evidence.sitting?.since) return <p className={c.fine}>Sat down at {time(evidence.sitting.since)}.</p>;
  if (signal.kind === "drink" && evidence.drink) return <Drink e={evidence.drink} />;
  if (signal.kind === "history" && evidence.history) {
    return (
      <ul className={c.days}>
        {evidence.history.days.map((d) => (
          <li key={d.day}>
            <strong>{dayLabel(`${d.day}T12:00:00`)}</strong>
            <span>{d.text}</span>
          </li>
        ))}
      </ul>
    );
  }
  return null;
}

const STAGES: [string, string][] = [
  ["deep", "Deep"],
  ["core", "Core"],
  ["rem", "REM"],
  ["awake", "Awake"],
];

/** Last night against the person's own usual night, on one scale, and the night's stages as one bar. */
function Night({ e }: { e: NonNullable<CheckinEvidence["sleep"]> }) {
  const asleep = e.asleep_min ?? 0;
  const usual = e.usual_min ?? 0;
  const scale = Math.max(asleep, usual, 1);
  const stages = STAGES.filter(([k]) => (e.stages[k] ?? 0) > 0);
  const total = stages.reduce((sum, [k]) => sum + (e.stages[k] ?? 0), 0);
  return (
    <div className={c.night}>
      <Bar label="Last night" value={hm(asleep)} share={asleep / scale} strong />
      {usual > 0 ? <Bar label={`Your usual (${e.nights} nights)`} value={hm(usual)} share={usual / scale} /> : null}
      {total > 0 ? (
        <>
          <div className={c.stages} role="img" aria-label={`Stages: ${stages.map(([k, name]) => `${name} ${hm(e.stages[k] ?? 0)}`).join(", ")}`}>
            {stages.map(([k]) => (
              <i key={k} className={c[`stage_${k}`]} style={{ flexGrow: e.stages[k] }} />
            ))}
          </div>
          <p className={c.legend}>
            {stages.map(([k, name]) => (
              <span key={k}>
                <i className={c[`stage_${k}`]} />
                {name} {hm(e.stages[k] ?? 0)}
              </span>
            ))}
          </p>
        </>
      ) : null}
    </div>
  );
}

function Bar({ label, value, share, strong = false }: { label: string; value: string; share: number; strong?: boolean }) {
  return (
    <div className={c.bar}>
      <p className={c.barHead}>
        <span>{label}</span>
        <strong>{value}</strong>
      </p>
      <div className={c.track}>
        <div className={strong ? c.fillStrong : c.fill} style={{ width: `${Math.max(4, Math.round(share * 100))}%` }} />
      </div>
    </div>
  );
}

/**
 * Resting heart rate against the person's own baseline: the shape of the last two weeks and nothing else. No axis, no
 * values and no tooltip on purpose. "Show numbers" fetches the values for this device only.
 */
function Resting({ e, numbers, onNumbers }: { e: NonNullable<CheckinEvidence["resting_hr"]>; numbers: CheckinEvidence["resting_hr"] | null; onNumbers: () => void }) {
  const W = 280;
  const H = 72;
  const pad = 8;
  const rel = e.series.map((p) => p.rel);
  const lo = Math.min(1, ...rel);
  const hi = Math.max(1, ...rel);
  const span = hi - lo || 1;
  const x = (i: number) => pad + (rel.length > 1 ? (i * (W - 2 * pad)) / (rel.length - 1) : (W - 2 * pad) / 2);
  const y = (v: number) => pad + (1 - (v - lo) / span) * (H - 2 * pad);
  const last = rel.length - 1;
  return (
    <div className={c.resting}>
      <svg className={c.spark} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Resting heart rate over ${e.series.length} days against your own baseline: today is above it`}>
        <line className={c.base} x1={pad} x2={W - pad} y1={y(1)} y2={y(1)} />
        <polyline className={c.line} points={rel.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
        {last >= 0 ? <circle className={c.today} cx={x(last)} cy={y(rel[last] ?? 1)} r={5} /> : null}
      </svg>
      <p className={c.legend}>
        <span>
          <i className={c.keyLine} />
          each day
        </span>
        <span>
          <i className={c.keyBase} />
          your baseline ({e.days} days)
        </span>
        <span>
          <i className={c.keyToday} />
          today
        </span>
      </p>
      {numbers?.today !== undefined && numbers.baseline !== undefined ? (
        <p className={c.fine}>
          Today {Math.round(numbers.today)} bpm; your baseline {Math.round(numbers.baseline)} bpm. Shown on this device only.
        </p>
      ) : (
        <button type="button" className={`${ui.chip} ${c.numbers}`} onClick={onNumbers}>
          Show numbers
        </button>
      )}
    </div>
  );
}

/** How the looks read, window by window: a taller bar is a larger share that read as strained. */
function Looks({ e }: { e: NonNullable<CheckinEvidence["expression"]> }) {
  const first = e.windows[0];
  const last = e.windows[e.windows.length - 1];
  return (
    <div>
      <div className={c.looks} role="img" aria-label={`${e.windows.length} stretches of looks; the share that read as strained in each`}>
        {e.windows.map((w) => (
          <i key={w.start} className={w.strained >= 0.5 ? c.lookHigh : c.look} style={{ height: `${Math.max(8, Math.round(w.strained * 100))}%` }} />
        ))}
      </div>
      {first && last ? (
        <p className={c.fine}>
          {time(first.start)} to {time(last.end)}. A taller bar: more of the looks read as strained. A noisy reading of a face&rsquo;s shape, not an emotion.
        </p>
      ) : null}
    </div>
  );
}

/** The frame behind the drink clause, blurred until it is tapped: the pick-up when one was seen, else where the container was last seen. */
function Drink({ e }: { e: NonNullable<CheckinEvidence["drink"]> }) {
  const [clear, setClear] = useState(false);
  const frame = e.frame_ref ?? e.last_seen?.frame_ref ?? null;
  const seen = e.last_seen;
  return (
    <div className={c.drink}>
      {frame ? (
        <button type="button" className={`${c.frame} ${clear ? "" : c.blurred}`} onClick={() => setClear(!clear)} aria-pressed={clear} aria-label={clear ? "Blur the frame" : "Show the frame"}>
          <Keyframe frameRef={frame} compact />
          {clear ? null : <span className={c.reveal}>Tap to show</span>}
        </button>
      ) : null}
      <p className={c.fine}>
        {e.picked_up && e.since ? `Picked up at ${time(e.since)}. ` : "Not seen picked up this sitting. "}
        {seen ? `The ${seen.label} was last seen on the ${cameraName(seen.frame_ref?.split(":")[1] ?? "cam-desk").toLowerCase()} at ${time(seen.ts)}.` : ""} It cannot tell water from coffee, or
        see a drink taken out of view.
      </p>
    </div>
  );
}
