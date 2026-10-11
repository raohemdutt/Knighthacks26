"use client";

import { useSyncExternalStore } from "react";
import type { Item, PlacePack } from "./api";

/**
 * "Near you": optional events a person can say yes to.
 *
 * They are ordinary connector items (kind calendar_event, external_id `nearby:<slug>:<start>`), so the box needs no
 * new endpoint. The facts that have no field of their own travel in the summary, the same " · " shape as the
 * `outlook:` and `campus:` items:
 *
 *   until 8:30 PM · John C. Hitt Library · hosted by Andria · with Maya, Priya · optional · via GroupMe
 *
 * Tags the screen understands: `hosted by <Name>`, `with <Name>, <Name>`, `free`, `optional`, `via <Source>`.
 * Anything else is kept as plain text. Saying yes is kept in this browser for now (see useGoing).
 */

export const NEARBY_PREFIX = "nearby:";
const DEFAULT_MINUTES = 60;

export interface Nearby {
  id: string; // external_id
  title: string;
  start: number; // ms
  end: number; // ms
  until: string; // the clock time exactly as the summary says it
  place: string | null;
  /** The pack place id ("ucf.library") when the name matches one, so a walk can be worked out. */
  placeId: string | null;
  host: string | null;
  friends: string[];
  free: boolean;
  optional: boolean;
  source: string | null;
  extra: string[];
  /** The Canvas assignment this event looks like help with, by course code. */
  helpsWith: Item | null;
  /** What the host said in the group chat about it, when a message from them is on the box. */
  said: { text: string; where: string } | null;
}

export const isNearby = (i: Item): boolean => i.kind === "calendar_event" && i.external_id.startsWith(NEARBY_PREFIX);

/** "until 8:30 PM" -> the moment on the start's day (the next day when it is earlier than the start). */
function endOf(start: number, until: string): number {
  const m = /^until\s+(\d{1,2}):(\d{2})\s*(AM|PM)$/i.exec(until.trim());
  if (!m) return start + DEFAULT_MINUTES * 60_000;
  let h = Number(m[1]) % 12;
  if (m[3]?.toUpperCase() === "PM") h += 12;
  const d = new Date(start);
  d.setHours(h, Number(m[2]), 0, 0);
  if (d.getTime() <= start) d.setDate(d.getDate() + 1);
  return d.getTime();
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

/** The campus pack's place for a name as people write it ("Library", "Student Union"). */
export function placeIdFor(name: string | null, packs: PlacePack[] | undefined): string | null {
  if (!name || !packs) return null;
  const want = norm(name);
  for (const pack of packs) {
    for (const [id, p] of Object.entries(pack.places)) {
      if ([p.name, ...p.aliases, id].some((x) => norm(x) === want)) return `${pack.name}.${id}`;
    }
  }
  return null;
}

const GENERIC = new Set(["with", "friends", "free", "night", "optional", "that", "this", "from", "have", "there"]);
/** The words of four letters or more, for telling whether a chat message is about an event. */
const words4 = (s: string): Set<string> => new Set(norm(s).split(" ").filter((w) => w.length >= 4 && !GENERIC.has(w)));

/** The course code in a title: "COP4533". */
const course = (s: string) => /\b([A-Z]{3}\s?\d{4}[A-Z]?)\b/.exec(s)?.[1]?.replace(/\s/g, "") ?? null;

export function parseNearby(items: Item[], packs?: PlacePack[]): Nearby[] {
  const assignments = items.filter((i) => i.kind === "assignment" && i.when);
  const messages = items.filter((i) => i.kind === "message");
  const out: Nearby[] = [];
  for (const i of items) {
    if (!isNearby(i) || !i.when) continue;
    const start = Date.parse(i.when);
    if (Number.isNaN(start)) continue;
    const [first = "", ...rest] = (i.summary ?? "").split(" · ").map((s) => s.trim()).filter(Boolean);
    const hasUntil = /^until\s/i.test(first);
    const until = hasUntil ? first : "";
    const parts = hasUntil ? rest : [first, ...rest].filter(Boolean);
    let place: string | null = null;
    let host: string | null = null;
    let friends: string[] = [];
    let free = false;
    let optional = false;
    let source: string | null = null;
    const extra: string[] = [];
    for (const p of parts) {
      const t = p.toLowerCase();
      if (t.startsWith("hosted by ")) host = p.slice(10).trim();
      else if (t.startsWith("with ")) friends = p.slice(5).split(/,|\band\b/).map((n) => n.trim()).filter(Boolean);
      else if (t === "free") free = true;
      else if (t === "optional") optional = true;
      else if (t.startsWith("via ")) source = p.slice(4).trim();
      else if (place === null) place = p;
      else extra.push(p);
    }
    const code = course(`${i.title} ${i.summary ?? ""}`);
    let said: Nearby["said"] = null;
    if (host) {
      // the host's own message about this event: from them, and sharing a word with it (or its course code)
      const words = words4(`${i.title} ${place ?? ""}`);
      const mine = messages
        .filter((x) => x.title.toLowerCase().startsWith(`${host!.toLowerCase()}:`) && x.summary)
        .filter((x) => (code && x.summary!.includes(code)) || [...words4(x.summary!)].some((w) => words.has(w)))
        .sort((a, b) => (b.when ?? "").localeCompare(a.when ?? ""));
      const parsed = mine[0]?.summary ? /^.+? in (.+?): (.+)$/.exec(mine[0].summary) : null;
      if (parsed) said = { where: parsed[1] as string, text: parsed[2] as string };
    }
    // the assignment of that course it is most likely about: the one sharing the most words with the event and what the host said
    const about = words4(`${i.title} ${i.summary ?? ""} ${said?.text ?? ""}`);
    const score = (a: Item) => [...words4(a.title)].filter((w) => about.has(w)).length;
    const helpsWith = code
      ? (assignments.filter((a) => a.title.includes(code)).sort((a, b) => score(b) - score(a) || (a.when ?? "").localeCompare(b.when ?? ""))[0] ?? null)
      : null;
    out.push({
      id: i.external_id,
      title: i.title,
      start,
      end: endOf(start, until),
      until,
      place,
      placeId: placeIdFor(place, packs),
      host,
      friends,
      free,
      optional,
      source,
      extra,
      helpsWith,
      said,
    });
  }
  return out.sort((a, b) => a.start - b.start);
}

// ---------- the person's answer ----------

export type Answer = "going" | "skipped";
const KEY = "atlas.nearby.v1";

function load(): Record<string, Answer> {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? (parsed as Record<string, Answer>) : {};
  } catch {
    return {};
  }
}

// One copy shared by every component on the screen (the panel that asks and the schedule that shows the yes).
const NONE: Record<string, Answer> = {};
let current: Record<string, Answer> | null = null;
const listeners = new Set<() => void>();

const read = () => (current ??= load());
const publish = (next: Record<string, Answer>) => {
  current = next;
  listeners.forEach((fn) => fn());
};
const subscribe = (fn: () => void) => {
  listeners.add(fn);
  const onStorage = (e: StorageEvent) => e.key === KEY && publish(load()); // another tab answered
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(fn);
    window.removeEventListener("storage", onStorage);
  };
};
const answerFor = (id: string, a: Answer | null) => {
  const next = { ...read() };
  if (a) next[id] = a;
  else delete next[id];
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* private window or blocked storage: the answer lasts until the page is closed */
  }
  publish(next);
};

/**
 * Which nearby events this person said yes or no to. Kept in this browser only, as a convenience: nothing is
 * written to the box's calendar or plan (that would be an approval card and is a separate decision), and it
 * does not follow the person to another device. Reads and writes never throw.
 */
export function useGoing(): { answers: Record<string, Answer>; answer: (id: string, a: Answer | null) => void } {
  const answers = useSyncExternalStore(subscribe, read, () => NONE);
  return { answers, answer: answerFor };
}
