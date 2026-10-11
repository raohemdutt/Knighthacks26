"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type OptionalAnswer, type OptionalChange, type OptionalEvent, type OptionalView } from "@/lib/api";
import { useAtlas, useAtlasEvents } from "@/lib/atlas";
import { time } from "@/lib/format";
import s from "./optional.module.css";

/**
 * Atlas suggesting an optional event ("Near you"), as one small card over whatever screen is open.
 *
 * It appears when the box says a suggestion (`day.narrated`, moment "suggest": on its own, or as the last clause of
 * the line it was merged into) and shows what the box's own list says about that event (GET /api/day/optional):
 * who is going, when and where, the walk, and how it fits today's plan. Two answers:
 *
 *  - "Add it" asks the box (POST /api/day/optional/{id}). That never changes the plan: the box makes a new plan
 *    with the event in it and one approval card. The card then shows what would move and one way on: "Review and
 *    approve", which opens the Approvals screen. Nothing here approves anything.
 *  - "Not today" is recorded on the box and the event is not suggested again.
 *
 * The same two answers can be spoken in the half minute after Atlas says the suggestion (lib/voice.tsx); the card
 * follows along from the box's events. Mounted once, by AtlasProvider, in the place the toasts use: under the header
 * and below any toast that is up, so it never sits on the tab bar or on the Ask screen's microphone.
 */

const WATCHED = ["day.narrated", "day.optional.answered", "day.replan.proposed", "day.plan.committed", "workspace.action.completed"];
const DISMISSED_KEY = "atlas.optional.dismissed";
/** A card that only says "it is in your plan now" puts itself away after this long. */
const ADDED_MS = 12_000;

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

/** "Riley", "Riley and Sam", "Riley, Sam and Alex" */
function people(names: string[]): string {
  return names.length <= 1 ? (names[0] ?? "") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** "17:15" (the box's local time, as the answer carries it) as "5:15 PM". */
function clock(hm: string): string {
  const [h, m] = hm.split(":").map(Number);
  if (h === undefined || m === undefined || Number.isNaN(h) || Number.isNaN(m)) return hm;
  return `${h % 12 || 12}:${String(m).padStart(2, "0")} ${h >= 12 ? "PM" : "AM"}`;
}

function who(e: OptionalEvent): string {
  const going = e.going.filter((n) => n !== e.host);
  if (e.host) return going.length ? `Hosted by ${e.host}, with ${people(going)}` : `Hosted by ${e.host}`;
  return going.length ? `With ${people(going)}` : "";
}

function fitLine(e: OptionalEvent): string {
  const f = e.fit;
  if (f.fits === "no") return `Does not fit today: ${f.reason}.`;
  if (f.fits === "nearly" && f.displaces.length) {
    const moved = f.displaces.map((d) => d.title);
    return `Nearly fits: ${people(moved)} would move.`;
  }
  if (f.fits === "nearly") return `Nearly fits: you would leave ${f.leave_early_minutes} min early for what comes next.`;
  return "Fits your plan.";
}

function walkLine(e: OptionalEvent): string {
  const f = e.fit;
  if (!f.leave_by || f.walk_minutes < 3) return "";
  return `${f.walk_minutes} min walk, leave by ${time(f.leave_by)}.`;
}

function changeLine(c: OptionalChange): string {
  const was = `${c.title}, ${clock(c.start)}`;
  if (c.change === "dropped") return `${was}: taken out`;
  if (c.change === "moved") return `${was}: moves to ${clock(c.new_start ?? c.start)}`;
  return `${was}: ${c.change}, now until ${clock(c.new_end ?? c.end)}`;
}

/** The event the card is about: the one Atlas brought up most recently that is still to come and still open. */
function pick(view: OptionalView, keep: string | null): OptionalEvent | null {
  const open = view.events.filter((e) => e.suggested_at !== null && Date.parse(e.start) > Date.parse(view.now));
  const kept = keep ? view.events.find((e) => e.id === keep && e.state === "accepted" && Date.parse(e.end) > Date.parse(view.now)) : undefined;
  const waiting = open.filter((e) => e.state === "suggested" && e.answer !== "no");
  waiting.sort((a, b) => Date.parse(b.suggested_at ?? "") - Date.parse(a.suggested_at ?? ""));
  return waiting[0] ?? kept ?? null;
}

export function OptionalSuggestion() {
  const path = usePathname() || "/";
  const { toasts } = useAtlas();
  const [below, setBelow] = useState(0); // the height of the toasts above the card, so neither covers the other
  const [event, setEvent] = useState<OptionalEvent | null>(null);
  const [answer, setAnswer] = useState<OptionalAnswer | null>(null); // what the box said to this phone's "Add it"
  const [busy, setBusy] = useState<"yes" | "no" | null>(null);
  const [error, setError] = useState<ApiError | undefined>();
  const [dismissed, setDismissed] = useState<string | null>(null);
  const shown = useRef<string | null>(null); // the event on the card, so an approval elsewhere can be told on it
  const fresh = useRef(0);

  useEffect(() => {
    try {
      setDismissed(window.sessionStorage.getItem(DISMISSED_KEY));
    } catch {
      /* private mode: the card just cannot remember it was put away */
    }
  }, []);

  const load = useCallback(() => {
    const turn = ++fresh.current;
    api
      .dayOptional()
      .then((view) => {
        if (turn !== fresh.current) return; // a newer answer is on its way
        const next = pick(view, shown.current);
        shown.current = next ? next.id : null;
        setEvent(next);
      })
      .catch(() => undefined); // no Day loop on this box, or the box is away: there is nothing to suggest
  }, []);

  useEffect(load, [load]);

  useEffect(() => {
    // Shell.tsx draws the toasts in the one assertive live region; its layout height does not count their entry animation.
    const stack = document.querySelector<HTMLElement>('[aria-live="assertive"]');
    setBelow(stack && stack.childElementCount > 0 ? stack.offsetHeight + 8 : 0);
  }, [toasts]);
  useAtlasEvents(WATCHED, (e) => {
    if (e.type !== "day.narrated" || e.moment === "suggest") load();
  });

  // What the box answered belongs to one event: a different one on the card starts clean.
  const id = event?.id ?? null;
  useEffect(() => {
    setAnswer((a) => (a && a.id !== id ? null : a));
    setError(undefined);
  }, [id]);

  const added = event?.state === "accepted";
  useEffect(() => {
    if (!added || !id) return;
    const timer = window.setTimeout(() => {
      shown.current = null;
      setEvent(null);
    }, ADDED_MS);
    return () => window.clearTimeout(timer);
  }, [added, id]);

  const dismiss = () => {
    if (!event) return;
    try {
      window.sessionStorage.setItem(DISMISSED_KEY, event.id);
    } catch {
      /* as above */
    }
    setDismissed(event.id);
  };

  const say = (which: "yes" | "no") => {
    if (!event) return;
    setBusy(which);
    setError(undefined);
    api
      .dayOptionalAnswer(event.id, which)
      .then((r) => {
        setAnswer(which === "yes" ? r : null);
        load();
      })
      .catch((err: unknown) => setError(asApiError(err)))
      .finally(() => setBusy(null));
  };

  // On the Approvals screen the approval card itself is on screen; the operator console is not the owner's phone.
  if (!event || event.id === dismissed || path.startsWith("/approvals") || path.startsWith("/operator")) return null;

  const waiting = event.state === "suggested" && event.awaiting_approval;
  const noted = answer && answer.id === event.id && !answer.proposed ? answer : null;
  const ranOut = event.state === "suggested" && event.answer === "yes" && !event.awaiting_approval && !noted;
  const moves = answer && answer.id === event.id ? (answer.moves ?? []) : [];
  const adds = answer && answer.id === event.id ? (answer.adds ?? []) : [];
  const names = who(event);
  const walk = walkLine(event);

  return (
    <aside className={s.card} style={{ "--below": `${below}px` } as CSSProperties} aria-label="A suggestion from Atlas" aria-live="polite">
      <div className={s.head}>
        <p className={s.label}>
          <span>Near you</span>
          <span className={s.unasked}>{added ? "In your plan" : waiting ? "Waiting for your yes" : "Atlas suggests"}</span>
        </p>
        <button type="button" className={s.close} aria-label="Put this suggestion away" onClick={dismiss}>
          ×
        </button>
      </div>
      <p className={s.title}>{event.title}</p>
      {names ? <p className={s.who}>{names}</p> : null}
      <p className={s.when}>
        {time(event.start)} to {time(event.end)}
        {event.place_name ? ` · ${event.place_name}` : ""}
        {event.source ? ` · via ${event.source}` : ""}
      </p>

      {added ? (
        <p className={s.line}>It is a block of your day now.{walk ? ` ${walk}` : ""}</p>
      ) : waiting ? (
        <>
          <p className={s.line}>Atlas made a new plan with it. Nothing changes until you approve it.</p>
          {moves.length || adds.length ? (
            <ul className={s.changes} aria-label="What would move">
              {moves.map((c) => (
                <li key={`${c.id}-${c.start}`}>{changeLine(c)}</li>
              ))}
              {adds.map((a) => (
                <li key={`new-${a.id}-${a.start}`}>
                  New: {a.title}, {clock(a.start)}
                </li>
              ))}
            </ul>
          ) : null}
          <div className={s.buttons}>
            <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary} ${s.go}`}>
              Review and approve
            </Link>
          </div>
        </>
      ) : noted ? (
        <p className={s.line}>{noted.reason ?? "Noted."}</p>
      ) : (
        <>
          <p className={s.line}>
            {fitLine(event)}
            {walk ? ` ${walk}` : ""}
          </p>
          {ranOut ? <p className={s.fine}>The approval card ran out, so nothing changed. You can add it again.</p> : null}
          <div className={s.buttons}>
            <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => say("no")}>
              {busy === "no" ? "Putting it away…" : "Not today"}
            </button>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy !== null || event.fit.fits === "no"} onClick={() => say("yes")}>
              {busy === "yes" ? "Planning…" : "Add it"}
            </button>
          </div>
        </>
      )}
      {error ? <WriteError error={error} /> : null}
    </aside>
  );
}
