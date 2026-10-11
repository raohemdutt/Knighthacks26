/**
 * Words for the voice: what Atlas says aloud, and what a heard phrase means.
 *
 * Everything here is pure (no window, no React), so it can be run and checked
 * outside a browser. The speaking and listening themselves live in voice.tsx.
 */

export interface SpeechTextOptions {
  /** IANA zone for ISO timestamps found in the text. Default: the device's own. */
  timeZone?: string;
  locale?: string;
}

/** Longest answer read aloud, in characters, before it is cut at a sentence end. */
export const MAX_SPOKEN_CHARS = 900; // room for a whole day walked through (the box's voice has the same cap: contexts/voice/text.py); was 420
/** In the offline demo the "answer" is the evidence list itself: read this many lines of it. */
const MAX_FAKE_LINES = 3;
/** Chrome stops in the middle of a long utterance, so speech is fed to it a sentence or two at a time. */
export const CHUNK_CHARS = 180;

const WEEKDAYS: Record<string, string> = {
  mon: "Monday",
  tue: "Tuesday",
  tues: "Tuesday",
  wed: "Wednesday",
  thu: "Thursday",
  thur: "Thursday",
  thurs: "Thursday",
  fri: "Friday",
  sat: "Saturday",
  sun: "Sunday",
};

const ISO_STAMP = /\b(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b/g;
const ISO_DATE = /\b(\d{4})-(\d{2})-(\d{2})\b/g;
const URL = /\b(?:https?:\/\/|www\.)[^\s<>()\]]+/gi;
const FRAME_REF = /\bframes?:[\w.-]+:\d{6,}\b/gi;
const EPISODE_ID = /\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*[-_]\d{10,}\b/gi;
const UUID = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
const LONG_HEX = /\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b/gi;
const LONG_NUMBER = /\b\d{10,}\b/g;
const ID_FIELD = /\b[a-z]+_ids?\b\s*[:=]?\s*/gi; // "episode_id: " left behind once its value is gone

/** Stands in for a removed link or ref until the words around it have been dealt with. */
const REF = "\u0001";
const POINTER = new RegExp(
  `(?:\\b(?:see|visit|open|view|source|sources|link|links|ref|refs|frame|frames|more at|details at|here|at|via|from)\\b[\\s:,]*)*${REF}` +
    `(?:\\s*(?:,|and|or)?\\s*${REF})*(?:\\s+for (?:more|details|more details|more info|more information)\\b)?`,
  "gi",
);

function sayClock(hour24: number, minute: number): string {
  const suffix = hour24 >= 12 ? "PM" : "AM";
  const h = hour24 % 12 === 0 ? 12 : hour24 % 12;
  if (minute === 0 && hour24 === 12) return "noon";
  if (minute === 0 && hour24 === 0) return "midnight";
  return minute === 0 ? `${h} ${suffix}` : `${h}:${String(minute).padStart(2, "0")} ${suffix}`;
}

/** "2026-10-10T15:40:00Z" -> "3:40 PM on Saturday, October 10", in the listener's zone. */
function sayStamp(iso: string, o: SpeechTextOptions): string {
  const normal = iso.replace(" ", "T");
  const d = new Date(/(Z|[+-]\d{2}:?\d{2})$/.test(normal) ? normal : `${normal}Z`);
  if (Number.isNaN(d.getTime())) return "";
  try {
    const parts = new Intl.DateTimeFormat(o.locale ?? "en-US", {
      timeZone: o.timeZone,
      hour: "numeric",
      minute: "2-digit",
      hour12: false,
      weekday: "long",
      month: "long",
      day: "numeric",
    }).formatToParts(d);
    const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
    const clock = sayClock(Number(get("hour")) % 24, Number(get("minute")));
    return `${clock} on ${get("weekday")}, ${get("month")} ${get("day")}`;
  } catch {
    return "";
  }
}

function sayDate(y: number, m: number, d: number, o: SpeechTextOptions): string {
  const date = new Date(Date.UTC(y, m - 1, d, 12));
  if (Number.isNaN(date.getTime())) return "";
  try {
    return new Intl.DateTimeFormat(o.locale ?? "en-US", { timeZone: "UTC", weekday: "long", month: "long", day: "numeric" }).format(date);
  } catch {
    return "";
  }
}

/** Times as a person says them: "15:40" and "3:40PM" -> "3:40 PM", "3:00 PM" -> "3 PM", "Sun 7:30 AM" -> "Sunday 7:30 AM". */
function naturalTimes(text: string, o: SpeechTextOptions): string {
  // Python's timedelta, as the box prints it: "1 day, 0:00:00" -> "1 day", "2 days, 6:30:00" -> "2 days 6 hours 30 minutes"
  let t = text.replace(/\b(\d+) (days?), (\d+):(\d{2}):\d{2}(?:\.\d+)?\b/g, (_m, n: string, unit: string, h: string, min: string) => {
    const hours = Number(h) ? ` ${Number(h)} ${Number(h) === 1 ? "hour" : "hours"}` : "";
    const mins = Number(min) ? ` ${Number(min)} ${Number(min) === 1 ? "minute" : "minutes"}` : "";
    return `${n} ${unit}${hours}${mins}`;
  });
  t = t.replace(ISO_STAMP, (m) => sayStamp(m, o));
  t = t.replace(ISO_DATE, (_m, y: string, mo: string, d: string) => sayDate(Number(y), Number(mo), Number(d), o));
  // 12-hour clock, with or without a space or dots: "3:40pm", "3 p.m.", "12:00 AM"
  t = t.replace(
    /\b(\d{1,2})(?::(\d{2}))?\s*(?:([ap])m\b|([ap])\.m\.(\s+[A-Z]|\s*$)?)/gi,
    (m, h: string, min: string | undefined, a1?: string, a2?: string, end?: string) => {
      const hour = Number(h);
      if (hour < 1 || hour > 12) return m;
      const hour24 = (hour % 12) + ((a1 ?? a2 ?? "a").toLowerCase() === "p" ? 12 : 0);
      return `${sayClock(hour24, Number(min ?? 0))}${end === undefined ? "" : `.${end}`}`; // "p.m." that ends a sentence keeps the full stop
    },
  );
  // 24-hour clock: "15:40", "07:05:00". A bare "10:40" or "2:30" is already how a person says it (the box's answers write a
  // morning time that way: contexts/day/domain/ear.py) and is left as written; appending "AM" to it was a guess.
  t = t.replace(/\b([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?\b(?!\s*[AP]M)/g, (m, h: string, min: string) => {
    const bare = Number(h) >= 1 && Number(h) <= 12 && !h.startsWith("0") && m.split(":").length === 2;
    return bare ? m : sayClock(Number(h), Number(min));
  });
  // a dash between two times or numbers is a range
  t = t.replace(/(\d|[AP]M|noon|midnight)\s*[–—]\s*(?=\d|noon|midnight)/g, "$1 to ");
  t = t.replace(/\b(\d{1,2}(?::\d{2})?)\s*-\s*(\d{1,2}(?::\d{2})? ?[AP]M)\b/g, "$1 to $2");
  // "Sun 7:30 AM" -> "Sunday 7:30 AM"; a bare "Sun" or "Sat" in a sentence is left alone
  t = t.replace(/\b(Mon|Tues?|Wed|Thu(?:rs?)?|Fri|Sat|Sun)\.?(?=,? (?:\d|at\b|noon|midnight|morning|afternoon|evening|night))/g, (m) => {
    return WEEKDAYS[m.replace(".", "").toLowerCase()] ?? m;
  });
  return t;
}

function stripMarkdown(text: string): string {
  let t = text;
  t = t.replace(/```[\s\S]*?(```|$)/g, " "); // code blocks are not read
  t = t.replace(/!\[[^\]]*\]\([^)]*\)/g, " "); // images
  t = t.replace(/\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)/g, "$1"); // [title](url) -> title
  t = t.replace(/\[\^?\d+\]/g, ""); // [1] citation marks
  t = t.replace(/<\/?[a-z][^>]*>/gi, " "); // stray html
  t = t.replace(/^\s{0,3}#{1,6}\s+(.*)$/gm, "$1."); // headings become their own sentence
  t = t.replace(/^\s{0,3}>\s?/gm, "");
  t = t.replace(/^\s*\|?[\s:|-]{3,}\|?\s*$/gm, ""); // table rule lines
  t = t.replace(/\s*\|\s*/g, ", ");
  t = t.replace(/^\s*(?:[-*+•]|\d{1,2}[.)])\s+(.*)$/gm, (_m, item: string) => (/[.!?:;,]$/.test(item.trim()) ? item.trim() : `${item.trim()}.`));
  t = t.replace(/(\*\*|__)(.+?)\1/g, "$2");
  t = t.replace(/(^|[\s(])([*_])(?=\S)([^*_\n]+?)\2(?=[\s).,!?;:]|$)/g, "$1$3");
  t = t.replace(/~~(.+?)~~/g, "$1");
  t = t.replace(/`([^`]*)`/g, "$1");
  t = t.replace(/^\s*([-*_]\s*){3,}$/gm, "");
  return t;
}

/** Links, camera frame refs, episode ids and hashes mean nothing aloud. */
function stripRefs(text: string): string {
  let t = text.replace(URL, REF);
  t = t.replace(FRAME_REF, REF).replace(UUID, REF).replace(EPISODE_ID, REF).replace(LONG_HEX, REF).replace(LONG_NUMBER, REF);
  t = t.replace(ID_FIELD, " ");
  // the words that only pointed at the ref go with it: "See <link> for more.", "(source: <frame>)", "at <link>"
  t = t.replace(POINTER, " ");
  t = t.split(REF).join(" ");
  t = t.replace(/\bcam-([a-z][a-z0-9]*)(?:-([a-z][a-z0-9]*))?\b/gi, (_m, a: string, b?: string) => `the ${b ? `${a} ${b}` : a} camera`);
  t = t.replace(/\b(the|a) the\b/gi, "the");
  // what the removals leave behind: "(see )", "[, ]", "source: ."
  t = t.replace(/[([{]\s*(?:see|source|sources|ref|refs|link|links|via|from|frame|frames|id|and|or|at|,|;|:|\s)*[)\]}]/gi, " ");
  t = t.replace(/\b(?:see|source|sources|ref|refs|link|links|frame|frames)\s*:\s*(?=[.,;!?]|$)/gim, " ");
  return t;
}

function tidy(text: string): string {
  let t = text;
  t = t.replace(/\s*&\s*/g, " and ").replace(/\s*(?:→|->|=>)\s*/g, " to ").replace(/\s*[·•]\s*/g, ", ");
  t = t.replace(/\b(\d+(?:\.\d+)?)[\s-]*min(?:ute)?s?\b\.?(?=\s+(?:walk|drive|ride|bike|bus|break|wait|trip|nap)\b)/gi, "$1 minute");
  t = t.replace(/\b(\d+(?:\.\d+)?)\s*mins?\b\.?/gi, (_m, n: string) => `${n} ${n === "1" ? "minute" : "minutes"}`);
  t = t.replace(/\b(\d+(?:\.\d+)?)\s*(?:hrs?|h)\b\.?(?!\w)/gi, (_m, n: string) => `${n} ${n === "1" ? "hour" : "hours"}`);
  t = t.replace(/\b(\d+)\s*secs?\b\.?/gi, "$1 seconds");
  t = t.replace(/\s*\n+\s*/g, " "); // every line was already ended as a sentence
  t = t.replace(/\s+/g, " ");
  t = t.replace(/\s+([.,;:!?])/g, "$1"); // no space before punctuation
  t = t.replace(/([,;:])(?:\s*[,;:])+/g, "$1"); // ", ," -> ","
  t = t.replace(/[,;:]\s*([.!?])/g, "$1"); // ",." -> "."
  t = t.replace(/([.!?])(?:\s*\.)+/g, "$1"); // ". ." -> "."
  t = t.replace(/\(\s*\)|\[\s*\]/g, "");
  t = t.replace(/^[\s.,;:!?-]+/, "");
  return t.replace(/\s+/g, " ").trim();
}

/**
 * The offline demo model echoes its prompt: "(fake) Evidence:\n- line\n- line\n\nQuestion: ...".
 * The evidence lines are the only content in it, so they are what gets read, without the scaffolding.
 */
function unwrapFake(text: string): { text: string; more: number } {
  const t = text.replace(/^\s*\(fake\)\s*/i, "");
  const m = /^\s*Evidence:\s*\n([\s\S]*?)(?:\n\s*\n\s*(?:Question|Make a plan for)\s*:[\s\S]*)?$/i.exec(t);
  // an echo with no evidence in it is only the question read back: nothing to say
  if (!m) return { text: t === text ? t : t.replace(/^\s*(?:Question|Make a plan for)\s*:[\s\S]*$/i, ""), more: 0 };
  const lines = (m[1] ?? "")
    .split("\n")
    .map((l) => l.replace(/^\s*[-*•]\s*/, "").trim())
    .filter(Boolean);
  return { text: lines.slice(0, MAX_FAKE_LINES).map((l) => `- ${l}`).join("\n"), more: Math.max(0, lines.length - MAX_FAKE_LINES) };
}

export function sentences(text: string): string[] {
  // a full stop after a single capital or a short abbreviation ("Dr.", "p.m.") is not the end of a sentence
  const out: string[] = [];
  let start = 0;
  const re = /[.!?]+["')\]]*\s+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const end = m.index + m[0].length;
    const before = text.slice(start, m.index);
    const lastWord = /(\S+)$/.exec(before)?.[1] ?? "";
    if (m[0].trim().startsWith(".") && /^(?:[A-Za-z]|Dr|Mr|Mrs|Ms|Prof|St|vs|etc|e\.g|i\.e|a\.m|p\.m|No|approx)$/i.test(lastWord)) continue;
    out.push(text.slice(start, end).trim());
    start = end;
  }
  const rest = text.slice(start).trim();
  if (rest) out.push(rest);
  return out;
}

function capAtSentence(text: string, max: number): string {
  if (text.length <= max) return text;
  let out = "";
  for (const s of sentences(text)) {
    if (out && out.length + s.length + 1 > max) break;
    if (!out && s.length > max) return `${s.slice(0, max).replace(/\s+\S*$/, "")}.`;
    out = out ? `${out} ${s}` : s;
  }
  return out;
}

/** Any text from the box as it should be said: no markdown, links, frame refs or ids, and times a person would say. */
export function cleanForSpeech(text: string, options: SpeechTextOptions = {}): string {
  if (typeof text !== "string") return "";
  return tidy(naturalTimes(stripRefs(stripMarkdown(text)), options));
}

/**
 * What Atlas says for an answer: the answer itself, cut to a few sentences, then at most
 * "Based on N pieces of evidence." The evidence stays on screen; it is not read out.
 */
export function speechForAnswer(answer: string, citationCount = 0, options: SpeechTextOptions = {}): string {
  const fake = unwrapFake(typeof answer === "string" ? answer : "");
  let said = capAtSentence(cleanForSpeech(fake.text, options), MAX_SPOKEN_CHARS);
  if (!said) return "I don't have an answer to read out for that.";
  if (!/[.!?]$/.test(said)) said += ".";
  if (citationCount > 0) said += ` Based on ${citationCount} ${citationCount === 1 ? "piece" : "pieces"} of evidence.`;
  return said;
}

/** One short sentence: the first sentence of `text`, cleaned, and cut at `max` characters. */
export function oneSentence(text: string, max = 180, options: SpeechTextOptions = {}): string {
  const first = sentences(cleanForSpeech(text, options))[0] ?? "";
  if (!first) return "";
  const cut = first.length > max ? first.slice(0, max).replace(/\s+\S*$/, "") : first;
  return /[.!?]$/.test(cut) ? cut : `${cut}.`;
}

/** Pieces short enough that Chrome does not cut one off: whole sentences, joined up to `max` characters. */
export function chunkForSpeech(text: string, max = CHUNK_CHARS): string[] {
  const out: string[] = [];
  let current = "";
  const push = (piece: string) => {
    if (!piece) return;
    if (current && current.length + piece.length + 1 > max) {
      out.push(current);
      current = "";
    }
    current = current ? `${current} ${piece}` : piece;
  };
  for (const s of sentences(text.replace(/\s+/g, " ").trim())) {
    if (s.length <= max) {
      push(s);
      continue;
    }
    // one very long sentence: break at commas, then at spaces
    for (const part of s.split(/(?<=[,;:])\s+/)) {
      if (part.length <= max) {
        push(part);
        continue;
      }
      let line = "";
      for (const word of part.split(" ")) {
        if (line && line.length + word.length + 1 > max) {
          push(line);
          line = "";
        }
        line = line ? `${line} ${word}` : word;
      }
      push(line);
    }
  }
  if (current) out.push(current);
  return out;
}

// ---------- things Atlas says without being asked ----------

export type SpokenKind = "alert" | "nudge" | "approval" | "leaving" | "plan" | "coach" | "brief" | "replan" | "review" | "move" | "milestone" | "checkin" | "suggest" | "start";

/** The shortest gap between two spoken events of one kind. Anything sooner is shown as usual but not said. */
export const SPEAK_COOLDOWN_MS: Record<SpokenKind, number> = {
  alert: 20_000,
  nudge: 120_000,
  approval: 30_000,
  leaving: 300_000,
  plan: 120_000,
  coach: 60_000,
  // the day in words (day.narrated): the morning brief, a re-plan, the evening wrap, and "leave now" / "late"
  brief: 120_000,
  replan: 60_000,
  review: 120_000,
  move: 60_000,
  // Waste Watch's milestones (impact.milestone.reached): rare, and never two in one breath
  milestone: 60_000,
  // the check-in (day.narrated, moment "checkin"): the box raises one every ninety minutes at most; said once
  checkin: 300_000,
  // an optional event nearby (moment "suggest") and a planned block starting (moment "start"); the box's arbiter
  // already keeps every unasked line apart, these are the per-kind floor underneath it
  suggest: 300_000,
  start: 60_000,
};

/** Longest narration said unasked, in characters: the narrator keeps the morning brief under this. */
const MAX_NARRATION_CHARS = 560;
const NARRATED_KIND: Record<string, SpokenKind> = { morning: "brief", replan: "replan", coach: "coach", evening: "review", checkin: "checkin", suggest: "suggest", start: "start" };

/**
 * One speaker: a narration the box's own speaker says (`spoken_by: "box"`), or one said inside another narration
 * (`merged_into`), is not said by the phone as well.
 */
export function saidElsewhere(e: { spoken_by?: unknown; merged_into?: unknown } | null | undefined): boolean {
  return !!e && (e.spoken_by === "box" || (typeof e.merged_into === "string" && e.merged_into !== ""));
}

/**
 * What to say for a `day.narrated` event, or null when it is not one Atlas says unasked: the morning brief, a
 * re-plan, the coach and the evening wrap always; a "now" only when it is urgent (leave now, late). Its
 * `spoken` text was written for the ear, so it is said as it is. Approval asks are said by the approval itself.
 */
export function sentenceForNarration(e: Record<string, unknown>): { kind: SpokenKind; text: string } | null {
  if (saidElsewhere(e)) return null;
  const moment = str(e.moment);
  const kind = moment === "now" ? (e.urgent === true ? "move" : null) : (NARRATED_KIND[moment] ?? null);
  const text = capAtSentence(str(e.spoken).replace(/\s+/g, " ").trim(), MAX_NARRATION_CHARS);
  return kind && text ? { kind, text } : null;
}

/** "I'd like to add eight blocks to your calendar for tomorrow. Okay?" for a narrated ask; null when it has none. */
export function narratedAsk(narration: { spoken?: unknown } | null | undefined): string | null {
  const text = str(narration?.spoken).replace(/\s+/g, " ").trim();
  return text ? text : null;
}

const str = (v: unknown): string => (typeof v === "string" ? v : "");

/** "Send blurred keyframes ..." -> "send blurred keyframes ...", but "COP 3502 ..." and "A package was left" keep their meaning. */
function asClause(reason: string): string {
  const r = reason.replace(/[.!?\s]+$/, "");
  return /^[A-Z][a-z]/.test(r) && !/^(A|An|The|I|I'm|Your|You|Nobody|Someone|There|It)\b/.test(r) ? r.charAt(0).toLowerCase() + r.slice(1) : r;
}

/** "I need your approval to send a message to Mom." for a request whose reason starts with a verb; otherwise the reason, then the ask. */
export function approvalSentence(reason: string, options: SpeechTextOptions = {}): string {
  const clean = oneSentence(reason, 160, options).replace(/[.!?]+$/, "");
  if (!clean) return "I need your approval for something.";
  const clause = asClause(clean);
  if (/^(Atlas|I) wants? to\b/i.test(clean)) return `I need your approval to ${clean.replace(/^(Atlas|I) wants? to\s+/i, "")}.`;
  return clause === clean ? `${clean}. I need your approval to act on it.` : `I need your approval to ${clause}.`;
}

/** One sentence for an event from the box, or null for events Atlas stays quiet about. */
export function sentenceForEvent(e: { type: string } & Record<string, unknown>, options: SpeechTextOptions = {}): { kind: SpokenKind; text: string } | null {
  switch (e.type) {
    case "agent.alert.raised": {
      const said = oneSentence(str(e.message), 180, options);
      return { kind: "alert", text: said ? `Alert. ${said}` : "Atlas raised an alert." };
    }
    case "habits.nudge.raised": {
      const said = oneSentence(str(e.message), 180, options);
      return said ? { kind: "nudge", text: said } : null;
    }
    case "workspace.approval.requested":
      return { kind: "approval", text: approvalSentence(str(e.reason), options) };
    case "agent.plan.created": {
      if (str(e.subject).toLowerCase() === "leaving") {
        // the checklist at the door: "Next: Lecture at 3:00 PM; leave by 2:40 PM (12 min walk to ...)."
        const first = oneSentence(str(e.text).replace(/^Next:\s*/i, "Next up, ").replace(/;\s*leave\b/i, ", so leave"), 200, options);
        return first ? { kind: "leaving", text: `Before you go. ${first}` } : null;
      }
      const subject = oneSentence(str(e.subject), 80, options).replace(/[.!?]+$/, "");
      return subject ? { kind: "plan", text: `I made a plan for ${subject}.` } : null;
    }
    case "day.narrated":
      return sentenceForNarration(e);
    case "impact.milestone.reached": {
      // Waste Watch passed a milestone: its `text` is one sentence written to be said ("That's 10 items kept out of landfill this week.")
      const said = oneSentence(str(e.text), 200, options);
      return said ? { kind: "milestone", text: said } : null;
    }
    case "day.coach.noted": {
      // the study coach when a block ends: already one to three short sentences written to be said
      const said = cleanForSpeech(str(e.text), options).slice(0, 260);
      return said ? { kind: "coach", text: said } : null;
    }
    default:
      return null;
  }
}

// ---------- what a heard phrase means ----------

// "Brief me", "what's my day", "what's next" and "how's it going" are NOT commands: they are questions, and they go to
// /api/ask like every other one. The Ask screen used to answer them itself from the day narrator (#171), so the phone and
// the box's voice (which posts straight to /api/ask) could say different things for the same words. One path now: the
// box answers (contexts/agent/application/schedule.py and talk.py). The morning brief itself is the Now screen's button.
export type VoiceCommand = "stop" | "yes" | "no" | null;

const POLITE = /\b(?:please|thanks|thank you|atlas|hey|ok|okay|now)\b/g;
const STOP = /^(?:stop|stop listening|stop talking|that s all|that is all|thats all|that s it|that will be all|never mind|nevermind|forget it|be quiet|quiet|goodbye|bye|i m done|we re done|done)$/;
// "Yes" only ever opens the Approvals screen, so it may be loose ("yes approve it", "yeah go ahead").
// "No" acts, so it has to be the whole utterance.
const YES = /^(?:(?:yes|yeah|yep|yup|sure|approve|approved|do it|go ahead|go for it|confirm|allow|it|that)(?:\s+|$))+$/;
const NO = /^(?:no\s+)*(?:no|nope|nah|deny|deny it|deny that|denied|don t|do not|don t do it|do not do it|reject|reject it|cancel|cancel it)$/;

// Right after Atlas offered an optional event ("Want me to add it?"). A yes only ever raises an approval card, so
// it may be loose ("yeah add it", "sure go ahead"). A no is recorded, so it has to be the whole utterance.
const ADD_IT = /^(?:(?:yes|yeah|yep|yup|sure|ok|okay|please|add|add it|add that|go ahead|do it|let s do it|sounds good|i m in|count me in|it|that)(?:\s+|$))+$/;
const AFFIRM = /\b(?:yes|yeah|yep|yup|sure|ok|okay|add|go ahead|do it|let s do it|sounds good|i m in|count me in)\b/;
const NOT_TODAY = /^(?:no\s+)*(?:no|nope|nah|not today|not now|not this time|no thanks|no thank you|skip it|skip that|pass|i ll pass|maybe later)$/;

/**
 * What a heard phrase means in the half minute after Atlas suggested an optional event: "yes" (add it: that raises
 * the approval card, it never approves), "no" (not today), or null for anything else, which is then heard as usual.
 */
export function suggestionAnswer(transcript: string): "yes" | "no" | null {
  const said = transcript
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\b(?:atlas|hey)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!said) return null;
  if (NOT_TODAY.test(said)) return "no";
  return ADD_IT.test(said) && AFFIRM.test(said) ? "yes" : null;
}

/** What Atlas says back to a spoken yes or no to a suggestion. A yes is only ever a card to approve by a tap. */
export function suggestionReply(answer: "yes" | "no", proposed: boolean): string {
  if (answer === "no") return "Okay, not today.";
  return proposed ? "Okay, there's a card to approve." : "Okay, noted. It is too far ahead to plan yet; I'll bring it up before it starts.";
}

/**
 * A whole utterance that is a command, never a word inside a longer sentence: "stop" stops,
 * "where does the bus stop" is a question. `expecting` says whether Atlas just asked a yes/no question.
 */
export function voiceCommand(transcript: string, expecting: "question" | "yes-no" = "question"): VoiceCommand {
  const raw = transcript
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!raw) return null;
  const bare = raw.replace(POLITE, " ").replace(/\s+/g, " ").trim();
  const said = bare || raw; // "okay" on its own is all politeness: keep it
  if (expecting === "yes-no") {
    if (NO.test(said)) return "no";
    if (YES.test(said) || /^(?:ok|okay)$/.test(raw)) return "yes";
  }
  if (STOP.test(said) || (expecting === "question" && /^(?:cancel|no thanks|no thank you)$/.test(raw))) return "stop";
  return null;
}
