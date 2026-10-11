"use client";

import Link from "next/link";
import type { Answer } from "@/lib/api";
import ui from "@/components/ui.module.css";

/** What each Ask Atlas tool is called on screen. */
const TOOL_NAMES: Record<string, string> = {
  "calendar.list": "Read your calendar",
  "calendar.add": "Proposed a calendar event",
  "calendar.find": "Searched your calendar",
  "reminders.add": "Proposed a reminder",
  "memory.last_seen": "Looked up where it was last seen",
  "memory.search": "Searched what the cameras saw",
  "memory.recap": "Recapped what the cameras saw",
  "room.map": "Opened the room map",
  "people.who_is_here": "Checked who is here",
  "camera.status": "Checked the cameras",
  "camera.pause": "Proposed a camera change",
  "box.model": "Checked the box's model",
  "box.energy_last_answer": "Read the energy of the last answer",
  "learning.status": "Read the Learning lab",
  "friends.shared_brain": "Checked the shared brain",
  "navigate.screen": "Opened a screen",
};

/** Screens this app has, for the "Open ..." link under an answer. */
const SCREEN_NAMES: Record<string, string> = {
  "/now/": "Calendar",
  "/approvals/": "Approvals",
  "/map/": "Room map",
  "/timeline/": "Timeline",
  "/live/": "Live",
  "/lab/": "Learning lab",
  "/stats/": "Box stats",
  "/circle/": "Circle",
  "/privacy/": "Privacy",
  "/people/": "People",
};

function whenWords(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
}

/** The tool an answer used, what it found in plain words, and where to go next (the approval, the screen). */
export function AskTool({ answer }: { answer: Answer }) {
  const tool = answer.tool;
  if (!tool) return null;
  const r = answer.tool_result;
  const label = TOOL_NAMES[tool] ?? tool;
  const path = answer.link?.split("?")[0] ?? "";
  const screen = SCREEN_NAMES[path];
  const events = Array.isArray(r?.events) ? r.events.slice(0, 6) : [];
  const proposed = r?.event;
  return (
    <div className={ui.evidence} data-guide="ask-tool">
      <p className={ui.small}>
        <span className={ui.tag}>{tool}</span> {label}
        {r && r.ok === false ? " (could not do it)" : null}
      </p>
      {proposed ? (
        <p className={ui.small}>
          <strong>{proposed.title}</strong>, {whenWords(proposed.start)}
          {proposed.attendees && proposed.attendees.length ? ` with ${proposed.attendees.join(", ")}` : null}
          {answer.action_id ? (
            <>
              {" "}
              <Link href={`/approvals/?focus=${answer.action_id}`}>Approve it</Link>
            </>
          ) : null}
        </p>
      ) : null}
      {events.length ? (
        <ul className={ui.small}>
          {events.map((e, i) => (
            <li key={`${e.title}-${i}`}>
              {e.url ? (
                <a href={e.url} target="_blank" rel="noreferrer">
                  {e.title}
                </a>
              ) : (
                e.title
              )}
              {e.when ? `, ${whenWords(e.when)}` : null}
            </li>
          ))}
        </ul>
      ) : null}
      {answer.action_id && !proposed ? (
        <p className={ui.small}>
          <Link href={`/approvals/?focus=${answer.action_id}`}>Approve it on the Approvals screen</Link>
        </p>
      ) : null}
      {screen && !answer.action_id ? (
        <p className={ui.small}>
          <Link href={answer.link ?? path}>Open {screen}</Link>
        </p>
      ) : null}
    </div>
  );
}

/** "the box" / "the network brain" / "the box's rules": the small line under an answer. */
export function answeredBy(answer: Answer): string | null {
  return answer.answered_by ?? null;
}

/** Tools whose answer is a screen: the Ask tab opens it. */
export const OPENS_SCREEN = new Set(["navigate.screen", "room.map"]);
