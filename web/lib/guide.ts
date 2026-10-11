/**
 * The look guide: "where do I look for X?" -> a place on a screen, and the pointer's way there.
 *
 * This file is the web-layer half and has no imports, so scripts/check-guide.mjs can run it in node:
 *  - `whereIntent` tells a where-question from any other; `classify` says whether it is about a thing in the room
 *    (the box answers it, the guide follows the answer's evidence) or about the app (resolved here, no backend call);
 *  - `APP_TARGETS` is the registry: one row per thing a person may ask for, with the screen it lives on, how to find
 *    it in the page, how to bring it into view first, and the sentences that walk someone there;
 *  - `resolveApp` matches a question against the registry (ids first, then words, a little spelling tolerance);
 *  - the pointer's path maths, and the words for a direction on screen.
 *
 * The Inspection screen's equipment, places, rounds and findings are resolved on the box
 * (GET /api/spot/guide, contexts/spot/application/guide.py); `spotTarget` turns its answer into the same shape.
 */

// ---------- the question ----------

const LEAD =
  /^(?:(?:hey |ok |okay )?atlas )?(?:please |can you |could you |so |and |ok |okay )*(?:where (?:do|can|should|would|could) (?:i|we|you) (?:look|go|see|find|check|get|start)(?: (?:for|at|to))?|where (?:did|have) (?:i|we) (?:leave|left|put|drop|dropped|see)|where (?:is|are|s|was|were)|wheres|have you seen|did you see|where (?:can|do|could|would) (?:i|we) (?:find|see|get to|get|read|check|change|set|write)|(?:show|point|take|guide|walk|bring|direct|lead) me(?: (?:to|at|where|towards|through))?(?: (?:is|are))?|(?:find|locate|point (?:to|at|out)|look for|guide me to)|how do i (?:get to|find|see|open|reach)|i (?:am looking|m looking|cant find|can t find|need to find|lost)(?: for)?|where (?:do|can|should|would|could) (?:i|we|you))\b ?/;
const TRAIL = /\b(?:please|thanks|thank you|on (?:the|this) (?:screen|page|app)|on screen|for me|right now|again|at|gone|go)$/;

/** Lower case, letters and digits only. */
export function plainText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[’']/g, " ")
    .replace(/[^a-z0-9:\-_ ]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** "Where did I leave my keys?" -> "my keys". null when the sentence does not ask where something is. */
export function whereIntent(text: string): string | null {
  const low = plainText(text);
  if (!LEAD.test(low)) return null;
  let thing = low;
  for (let i = 0; i < 2; i++) thing = thing.replace(LEAD, "").trim().replace(TRAIL, "").trim();
  return thing;
}

const FILLER = new Set("the a an my our your that this those these to of for is are it i me on in at some any do can where look find show please s about be with and or does which what".split(" "));
const SAME: Record<string, string> = { screen: "tab", page: "tab", view: "tab", section: "card", box: "card", button: "card", list: "card", panel: "card", lesson: "class", lecture: "class", meeting: "event", friends: "people", friend: "people", leaderboard: "league", standings: "league", trash: "bin", recycling: "bin", bins: "bin", enroll: "enrol", face: "face", calendar: "schedule" };
const LIGHT = new Set(["tab", "card", "thing", "part", "one", "atlas"]);
const NUMBERS: Record<string, string> = { one: "1", two: "2", three: "3", first: "1", second: "2", third: "3" };

/** Content words, plural and near-synonym folded: "My friends list" -> people card. */
export function words(text: string): string[] {
  const out: string[] = [];
  for (const raw of plainText(text).match(/[a-z]+|\d+/g) ?? []) {
    let w = NUMBERS[raw] ?? raw;
    if (/^\d+$/.test(w)) {
      out.push(String(Number(w)));
      continue;
    }
    if (FILLER.has(w)) continue;
    if (SAME[w]) {
      out.push(SAME[w]!);
      continue;
    }
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
    out.push(SAME[w] ?? w);
  }
  return out;
}

/** Levenshtein distance turned into a 0..1 likeness. */
function likeness(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const row = [i];
    for (let j = 1; j <= n; j++) row[j] = Math.min(prev[j]! + 1, row[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = row;
  }
  return 1 - prev[n]! / Math.max(m, n, 1);
}

function same(a: string, b: string): number {
  if (a === b) return 1;
  if (/^\d/.test(a) || /^\d/.test(b) || Math.min(a.length, b.length) < 4) return 0;
  const r = likeness(a, b);
  return r >= 0.72 ? r : 0;
}

const weight = (w: string) => (/^\d+$/.test(w) ? 1.5 : LIGHT.has(w) ? 0.3 : 1);

/** How well the asked words name a thing: how much of the name was said, and how much of what was said is the name. */
export function similarity(asked: string[], name: string[]): number {
  if (asked.length === 0 || name.length === 0) return 0;
  const gotName = name.reduce((s, n) => s + weight(n) * Math.max(...asked.map((a) => same(n, a))), 0);
  const gotAsked = asked.reduce((s, a) => s + weight(a) * Math.max(...name.map((n) => same(a, n))), 0);
  if (Math.max(gotName, gotAsked) < 0.75) return 0;
  const total = (ws: string[]) => ws.reduce((s, w) => s + weight(w), 0);
  return Math.round((0.55 * (gotName / total(name)) + 0.45 * (gotAsked / total(asked))) * 1000) / 1000;
}

// ---------- the registry ----------

/** How to find an element: a `data-guide` id, a CSS selector, and/or the text of a heading or button (a regular expression). */
export interface Locator {
  id?: string;
  sel?: string;
  text?: string;
  /** With `text` (or `sel`): ring the closest ancestor that matches, e.g. the heading's whole section. */
  up?: string;
  /** Take the last match on the page instead of the first (the newest answer). */
  last?: boolean;
}

/** Something to do before pointing, so a hidden target is on screen. Skipped when it is already so. */
export type Reveal =
  | { op: "press"; at: Locator; key?: string }
  | { op: "open"; at: Locator; key?: string }
  | { op: "select"; at: Locator; value: string; key?: string };

export interface GuideStep {
  /** Said only if this reveal really had to be done ("I am opening the Analysis tab."). */
  reveal?: string;
  say: string;
}

export interface GuideTarget {
  id: string;
  /** The screen it lives on ("/now/"); null: wherever the person is. */
  route: string | null;
  label: string;
  /** Tried in order; the first that is on screen wins. */
  at: Locator[];
  reveal?: Reveal[];
  steps: GuideStep[];
}

interface Row extends Omit<GuideTarget, "steps" | "at"> {
  names: string[];
  at: Locator | Locator[];
  /** Where to look, then what it is: the two sentences of the walk-through. */
  say: [string, string];
}

export const SCREENS: Record<string, string> = {
  "/": "Ask",
  "/now/": "Now",
  "/today/": "Today",
  "/dayrun/": "Day Run",
  "/circle/": "Circle",
  "/impact/": "Impact",
  "/map/": "Room map",
  "/play/": "Sort It",
  "/approvals/": "Approvals",
  "/teach/": "Teach",
  "/lab/": "Learning lab",
  "/privacy/": "Privacy",
  "/people/": "People",
  "/timeline/": "Timeline",
  "/live/": "Live",
  "/zones/": "Zones",
  "/spot/": "Inspection",
};

const HEAD: Locator = { sel: "main h1" };
const section = (text: string): Locator => ({ sel: "main h2, main h3", text, up: "section" });
const tab = (text: string): Locator => ({ sel: "main [role=tab], main [role=group] button", text });

/** Everything a person may ask for by name, outside the Inspection screen's own data. One row per thing. */
const ROWS: Row[] = [
  // Ask
  { id: "ask:input", route: "/", label: "The question box", names: ["question box", "ask box", "ask", "type a question", "ask a question", "ask atlas", "the input"], at: { id: "ask:input" }, say: ["Look at the bottom of the Ask screen.", "Type your question here, or tap the microphone above it."] },
  { id: "ask:mic", route: "/", label: "The microphone", names: ["microphone", "mic", "voice button", "talk to atlas", "talk button", "speak"], at: { sel: "button[aria-label='Talk to Atlas'], button[aria-label='Why the microphone is off'], button[aria-label='Done talking']" }, say: ["Look at the bottom of the Ask screen, in the middle.", "Tap this and ask out loud."] },
  { id: "ask:suggested", route: "/", label: "Questions to try", names: ["suggested questions", "example questions", "examples", "try one", "what can i ask", "questions to try"], at: { sel: "section[aria-label='Suggested questions']" }, say: ["Look under the conversation.", "Tap any of these to ask it."] },
  { id: "ask:answer", route: "/", label: "The last answer", names: ["last answer", "the answer", "evidence", "sources", "the picture", "the frame", "proof"], at: [{ id: "answer", last: true }, HEAD], say: ["Look at the newest card in the conversation.", "The answer is on top, and the frames and links it comes from are under it."] },
  // Now
  { id: "now:next", route: "/now/", label: "What is next", names: ["next class", "next event", "what is next", "my next class", "next thing", "leave by", "when do i leave", "when to leave", "where should i be"], at: [{ id: "now:next" }, { sel: "main section[aria-live='polite']" }, HEAD], say: ["Look at the top of the Now screen.", "This is what is next, and when to leave for it."] },
  { id: "now:action", route: "/now/", label: "What to do now", names: ["what to do", "what to do now", "what should i do", "the one thing", "do now"], at: [{ sel: "main section[aria-label='What to do']" }, HEAD], say: ["Look just under the top of the Now screen.", "This is the one thing to do right now."] },
  { id: "now:plan", route: "/now/", label: "Your day", names: ["my plan", "the plan", "plan ribbon", "ribbon", "my day", "my schedule", "today plan", "day plan", "the blocks"], at: [{ id: "now:plan" }, { sel: "main section[aria-label='Your day']" }, HEAD], say: ["Look at the list called Your day.", "These are today's blocks in order. Tap one to see why it is there."] },
  { id: "now:checkin", route: "/now/", label: "The check-in", names: ["check in", "checkin", "check in card", "how am i", "wellbeing", "the check"], at: [{ sel: "main section[aria-label='Check-in from Atlas']" }, HEAD], say: ["Look above the list called Your day.", "When Atlas has a check-in for you, it is this card."] },
  { id: "now:brief", route: "/now/", label: "Brief me", names: ["brief me", "the brief", "morning brief", "briefing", "read my day"], at: [{ sel: "main button", text: "^(Brief me|Stop)$" }, HEAD], say: ["Look at the top of the Now screen.", "Tap Brief me and Atlas reads your day aloud."] },
  { id: "now:sofar", route: "/now/", label: "Today so far", names: ["today so far", "so far", "what i did", "progress today"], at: [{ sel: "main section[aria-label='Today so far']" }, HEAD], say: ["Look at the strip called Today so far.", "This is what the box measured today."] },
  // Today
  { id: "today:habits", route: "/today/", label: "Habits", names: ["habits", "my habits", "streak", "habit"], at: [section("^Habits"), HEAD], say: ["Look at the part called Habits.", "Each bar is one habit against its target."] },
  { id: "today:due", route: "/today/", label: "Due next", names: ["due next", "what is due", "deadlines", "assignments", "homework", "due this week"], at: [section("^Due next"), HEAD], say: ["Look at the part called Due next.", "Your next deadlines, soonest first."] },
  { id: "today:coming", route: "/today/", label: "Coming up", names: ["coming up", "upcoming", "events today", "what is coming"], at: [section("^Coming up"), HEAD], say: ["Look at the part called Coming up.", "The events on your calendar."] },
  { id: "today:waste", route: "/today/", label: "Waste Watch", names: ["waste watch", "waste", "what i threw away"], at: [section("^Waste Watch"), HEAD], say: ["Look at the part called Waste Watch.", "What went into which bin."] },
  // Day Run
  { id: "dayrun:plan", route: "/dayrun/", label: "Your plan in Day Run", names: ["day run plan", "day run", "plan my day", "the game", "day game"], at: [section("^Your plan"), HEAD], say: ["Look at the part called Your plan.", "Move the blocks here; every change is run on the box."] },
  { id: "dayrun:play", route: "/dayrun/", label: "Play the day", names: ["play", "play the day", "watch the day", "play mode", "watch my day"], at: [{ sel: "main button", text: "^(Watch|Play)" }, HEAD], say: ["Look near the top of Day Run.", "This plays your day, with a figure walking the campus."] },
  { id: "dayrun:map", route: "/dayrun/", label: "Where the day takes you", names: ["day map", "campus map", "where the day takes me", "the route", "campus"], at: [section("^Where the day takes you"), HEAD], say: ["Look at the part called Where the day takes you.", "The map of today's walks."] },
  { id: "dayrun:board", route: "/dayrun/", label: "The board", names: ["the board", "board", "the deadlines board", "risk"], at: [section("^The board"), HEAD], say: ["Look at the part called The board.", "Everything that has to fit into the day."] },
  // Circle
  { id: "circle:people", route: "/circle/", label: "People in your Circle", names: ["people", "my friends", "friends", "who is in my circle", "circle", "my circle"], at: [tab("^People"), HEAD], reveal: [{ op: "press", at: tab("^People"), key: "circle:people" }], say: ["Look at the row of buttons under the heading.", "People: who is in your Circle and where they are, if they shared."] },
  { id: "circle:plans", route: "/circle/", label: "Plans with your Circle", names: ["plans", "circle plans", "plans with friends", "find a time", "conversations"], at: [tab("^Plans"), HEAD], reveal: [{ op: "press", at: tab("^Plans"), key: "circle:plans" }], say: ["Look at the row of buttons under the heading.", "Plans: what is coming up and what the Atlases agreed."] },
  { id: "circle:league", route: "/circle/", label: "The League", names: ["league", "my league", "the league", "standings", "the weekly game", "dares", "score"], at: [tab("^League"), HEAD], reveal: [{ op: "press", at: tab("^League"), key: "circle:league" }], say: ["Look at the row of buttons under the heading.", "League: this week's standings, the stake and the dares."] },
  { id: "circle:sharing", route: "/circle/", label: "What you share", names: ["sharing", "what i share", "what leaves the box"], at: [tab("^Sharing"), HEAD], reveal: [{ op: "press", at: tab("^Sharing"), key: "circle:sharing" }], say: ["Look at the row of buttons under the heading.", "Sharing: exactly what leaves this box, and what never does."] },
  // Impact
  { id: "impact:story", route: "/impact/", label: "The story this week", names: ["the story", "impact", "impact story", "how are we doing", "headline", "my impact"], at: [{ sel: "main section h2" }, HEAD], say: ["Look at the top of the Impact screen.", "This sentence is the week in one line."] },
  { id: "impact:goal", route: "/impact/", label: "This week's goal", names: ["goal", "the goal", "this week goal", "weekly goal", "target"], at: [section("^This week.s goal"), HEAD], say: ["Look at the part called This week's goal.", "One square for each day of the week."] },
  { id: "impact:practice", route: "/impact/", label: "Practise in Sort It", names: ["practice set", "practise", "practice", "practise in sort it", "what i got wrong"], at: [section("^Practise in Sort It"), HEAD], say: ["Look at the part called Practise in Sort It.", "The things that went in the wrong bin, ready to practise."] },
  { id: "impact:where", route: "/impact/", label: "What went where", names: ["what went where", "disposals", "which bin things went"], at: [section("^What went where"), HEAD], say: ["Look at the part called What went where.", "Every disposal the camera saw."] },
  // Room map and Sort It
  { id: "map:room", route: "/map/", label: "The room map", names: ["room map", "the room", "map of the room", "zones map", "the map", "room twin", "3d room"], at: [{ id: "map:room" }, { sel: "main svg[role=img]" }, HEAD], say: ["Look at the drawing at the top of the Room map.", "Each panel is one camera's view. A numbered pin is one of your things."] },
  { id: "map:things", route: "/map/", label: "Where are my things?", names: ["my things", "where are my things", "things list", "last seen", "lost things"], at: [section("^Where are my things"), HEAD], say: ["Look at the part called Where are my things.", "Tap a name to see where the cameras last saw it."] },
  { id: "map:sort", route: "/map/", label: "Which bin?", names: ["which bin", "the bin", "bin route", "way to the bin", "where does this go"], at: [section("^Sort It: which bin"), HEAD], say: ["Look at the part called Sort It: which bin.", "Pick what you are holding and the map draws the way to its bin."] },
  { id: "play:game", route: "/play/", label: "Sort It", names: ["sort it", "sorting game", "sort game", "sorting"], at: [{ sel: "main [role=group][aria-label='Item']" }, HEAD], say: ["Look at the row of things to sort.", "Pick one and the board shows the way to its bin."] },
  { id: "play:practice", route: "/play/", label: "Practise what you got wrong", names: ["practise what i got wrong", "my mistakes", "wrong bin"], at: [{ sel: "main section[aria-label='Practise what you got wrong']" }, HEAD], say: ["Look at the card at the top of Sort It.", "These are the ones that went in the wrong bin."] },
  { id: "play:better", route: "/play/", label: "Are you getting better?", names: ["am i getting better", "getting better", "my score", "sorting score"], at: [section("^Are you getting better"), HEAD], say: ["Look at the part called Are you getting better.", "Your sorting, week by week."] },
  // Approvals
  { id: "approvals:pending", route: "/approvals/", label: "What is waiting for you", names: ["approvals", "pending", "pending card", "approve", "what needs my approval", "waiting for me", "approval"], at: [{ id: "approvals:pending" }, { sel: "main article, main h2" }, HEAD], say: ["Look at the first card on the Approvals screen.", "Nothing is sent or written until you tap Approve here."] },
  // Teach, Learning lab, Privacy, People, and the screens with only a heading to point at
  { id: "teach:home", route: "/teach/", label: "Teach", names: ["teach", "teach atlas", "teach it something", "teaching"], at: HEAD, say: ["Look at the top of the Teach screen.", "This is where you teach Atlas something new."] },
  { id: "lab:home", route: "/lab/", label: "Learning lab", names: ["learning lab", "lab", "what atlas learned", "training runs", "training"], at: [{ sel: "main [role=group][aria-label='Which run']" }, HEAD], say: ["Look at the top of the Learning lab.", "Each run is one night of practice on the lab machine."] },
  { id: "privacy:home", route: "/privacy/", label: "Privacy", names: ["privacy", "what left the box", "delete my data", "my data", "pictures setting"], at: HEAD, say: ["Look at the top of the Privacy screen.", "What is kept, what left the box, and how to delete it."] },
  { id: "people:enrol", route: "/people/", label: "Teach Atlas your face", names: ["enrol", "enrol my face", "add my face", "teach my face", "teach atlas my face", "face enrol"], at: [section("^Teach Atlas your face"), HEAD], say: ["Look at the part called Teach Atlas your face.", "A few looks at the camera and Atlas knows you."] },
  { id: "people:forget", route: "/people/", label: "Forget me", names: ["forget me", "delete my face", "remove my face", "forget my face"], at: [{ sel: "main button", text: "^Forget me" }, section("^Enrolled"), HEAD], say: ["Look at the part called Enrolled.", "Forget me removes your face from this box for good."] },
  { id: "people:enrolled", route: "/people/", label: "Who is enrolled", names: ["enrolled", "who is enrolled", "enrolled people", "who atlas knows"], at: [section("^Enrolled"), HEAD], say: ["Look at the part called Enrolled.", "Everyone this box can recognise."] },
  { id: "timeline:home", route: "/timeline/", label: "Timeline", names: ["timeline", "what happened", "history", "events"], at: HEAD, say: ["Look at the top of the Timeline.", "Everything the box saw, newest first."] },
  { id: "live:home", route: "/live/", label: "Live", names: ["live", "live view", "the cameras", "camera view"], at: HEAD, say: ["Look at the top of the Live screen.", "What the cameras see right now."] },
  { id: "zones:home", route: "/zones/", label: "Zones", names: ["zones", "camera zones", "draw zones"], at: HEAD, say: ["Look at the top of the Zones screen.", "The areas each camera watches."] },
  { id: "spot:home", route: "/spot/", label: "Inspection", names: ["inspection", "spot", "the robot", "plant floor", "nextera"], at: [{ id: "status" }, HEAD], say: ["Look at the top of the Inspection screen.", "Your shift in one line. Ask me where anything is from here."] },
  // anywhere
  { id: "settings", route: null, label: "Settings", names: ["settings", "voice settings", "the token", "use my eyes", "eye tracking", "gaze", "camera switch"], at: [{ sel: "button[aria-label='Settings']" }, { sel: "aside button", text: "^Settings" }], say: ["Look for the Settings button: top right on a phone, at the foot of the list of screens on a computer.", "Settings: the voice switches, and Use my eyes to guide me."] },
];

/** The things the cameras can be asked about. A question that names one goes to the box, never to the app guide. */
export const ROOM_THINGS = "key keys wallet charger laptop phone package parcel bag backpack bottle mug cup glasses headphones earbuds umbrella notebook remote badge jacket shoes pen pencil mouse keyboard tablet ipad purse lunch snack".split(" ");
/** The few the room map keeps a pin for (lib/room.ts WATCH_ITEMS). */
export const PINNED = ["keys", "wallet", "charger", "laptop", "phone"];

function build(row: Row): GuideTarget {
  const { names: _names, say, at, ...rest } = row;
  return { ...rest, at: Array.isArray(at) ? at : [at], steps: [{ say: say[0] }, { say: say[1] }] };
}

export const APP_TARGETS: GuideTarget[] = ROWS.map(build);
const KEYS = ROWS.map((r) => [...r.names, r.label].map(words));

/** The pin of a thing on the room map: opens the map, picks the thing, lands on its pin (or on its card when it has none). */
export function pinTarget(thing: string): GuideTarget {
  const label = thing.charAt(0).toUpperCase() + thing.slice(1);
  const chip: Locator = { sel: "main [role=group][aria-label='Item'] button", text: `^${thing}` };
  return {
    id: `pin:${thing}`,
    route: "/map/",
    label: `${label} on the room map`,
    reveal: [{ op: "press", at: chip, key: "pick" }],
    at: [{ id: `pin:${thing}` }, section("^Where are my things"), HEAD],
    steps: [{ say: "Look at the drawing of the room." }, { say: `The ringed pin is where the cameras last saw your ${thing}. It sits inside its zone, not at the exact spot.` }],
  };
}

export const CONFIDENT = 0.75;

export interface Resolved {
  target: GuideTarget | null;
  confidence: number;
  alternatives: GuideTarget[];
}

/** A question about the app -> the registry row it names. Rows on the current screen win a tie. */
export function resolveApp(question: string, route: string | null = null): Resolved {
  const thing = whereIntent(question) ?? plainText(question);
  const exact = APP_TARGETS.find((t) => t.id === question.trim().toLowerCase() || t.id === thing.replace(/ /g, ""));
  if (exact) return { target: exact, confidence: 1, alternatives: [] };
  const pin = /^pin:([a-z ]+)$/.exec(question.trim().toLowerCase());
  if (pin) return { target: pinTarget(pin[1]!), confidence: 1, alternatives: [] };
  const asked = words(thing);
  const scored = APP_TARGETS.map((t, i) => ({ t, s: Math.max(...KEYS[i]!.map((k) => similarity(asked, k))) + (route && t.route === route ? 0.03 : 0) })).sort((a, b) => b.s - a.s);
  const best = scored[0]!;
  const rest = scored.slice(1).filter((x) => x.s >= 0.34);
  if (best.s >= CONFIDENT) return { target: best.t, confidence: Math.min(1, best.s), alternatives: rest.filter((x) => x.s >= 0.6).slice(0, 2).map((x) => x.t) };
  return { target: null, confidence: best.s, alternatives: best.s >= 0.34 ? [best, ...rest].slice(0, 3).map((x) => x.t) : [] };
}

export type Asked =
  | { kind: "room"; thing: string } // a thing or a person in the room: the box answers; the guide follows the evidence
  | { kind: "app"; target: GuideTarget } // something in the app: resolved here
  | { kind: "unsure"; thing: string } // ask the box as today; guide only if its answer shows a picture
  | null; // not a where-question

/** Conservative: a known thing in the room goes to the box; a known app feature to the guide; anything else is the box's. */
export function classify(question: string, route: string | null = null): Asked {
  const thing = whereIntent(question);
  if (thing === null) return null;
  const said = thing.split(" ");
  if (said.some((w) => ROOM_THINGS.includes(w))) return { kind: "room", thing };
  const app = resolveApp(question, route);
  if (app.target && app.confidence >= 0.8) return { kind: "app", target: app.target };
  return { kind: "unsure", thing };
}

/** The thing a room question names, as the room map knows it: "my keys" -> "keys"; null when the map keeps no pin for it. */
export function pinnedThing(thing: string): string | null {
  const said = thing.split(" ");
  return PINNED.find((p) => said.includes(p) || said.includes(p.replace(/s$/, ""))) ?? null;
}

/** The thing itself, for a sentence: "my keys" -> "keys". */
export function thingName(thing: string): string {
  return thing.split(" ").filter((w) => !FILLER.has(w)).join(" ") || thing;
}

// ---------- the Inspection screen: the box resolves, this maps its answer onto the page ----------

export interface SpotGuide {
  configured: boolean;
  query: string;
  target: string | null;
  label: string | null;
  steps: GuideStep[];
  confidence: number;
  alternatives: { target: string; label: string; confidence: number }[];
}

/** A reveal the box named ("map:floor:2", "round:pick:M017", "open:analysis:findings", "tab:analysis") as something to do in the page. */
export function spotReveal(key: string): Reveal {
  const pick = /^round:pick:(.+)$/.exec(key);
  if (pick) return { op: "select", at: { id: "round:pick" }, value: pick[1]!, key };
  const fold = /^open:(.+)$/.exec(key);
  if (fold) return { op: "open", at: { id: fold[1]! }, key };
  return { op: "press", at: { id: key }, key };
}

export function spotTarget(g: SpotGuide): GuideTarget | null {
  if (!g.target) return null;
  const id = /^round:\d+$/.test(g.target) ? "round:pick" : g.target;
  return {
    id: g.target,
    route: "/spot/",
    label: g.label ?? g.target,
    at: [{ id }],
    reveal: g.steps.filter((s) => s.reveal).map((s) => spotReveal(s.reveal!)),
    steps: g.steps,
  };
}

// ---------- the pointer's path ----------

export interface Pt {
  x: number;
  y: number;
}

/** Slow out, slow in. */
export function ease(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
}

/** A gentle arc from a to b (a quadratic curve bowed to one side by a sixth of its length), at u in 0..1. */
export function pathPoint(a: Pt, b: Pt, u: number): Pt {
  const c = Math.max(0, Math.min(1, u));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const bow = Math.min(120, Math.hypot(dx, dy) / 6);
  const len = Math.hypot(dx, dy) || 1;
  const mid = { x: (a.x + b.x) / 2 - (dy / len) * bow, y: (a.y + b.y) / 2 + (dx / len) * bow };
  const k = 1 - c;
  return { x: k * k * a.x + 2 * k * c * mid.x + c * c * b.x, y: k * k * a.y + 2 * k * c * mid.y + c * c * b.y };
}

/** The fading trail behind the pointer at time t (0..1): points from oldest to newest. */
export function trail(a: Pt, b: Pt, t: number, n = 14, span = 0.28): Pt[] {
  const head = ease(t);
  const tail = Math.max(0, head - span);
  return Array.from({ length: n }, (_, i) => pathPoint(a, b, tail + ((head - tail) * i) / (n - 1)));
}

/** How long the pointer takes: longer for a longer way, between 0.7 and 1.8 seconds. */
export function travelMs(a: Pt, b: Pt): number {
  return Math.round(Math.max(700, Math.min(1800, 500 + Math.hypot(b.x - a.x, b.y - a.y) * 1.1)));
}

// ---------- words for a place and a direction on screen ----------

/** Which cell of a cols x rows grid over the screen a point is in. */
export function cellOf(p: Pt, w: number, h: number, cols = 3, rows = 3): { col: number; row: number } {
  return { col: Math.max(0, Math.min(cols - 1, Math.floor((p.x / Math.max(w, 1)) * cols))), row: Math.max(0, Math.min(rows - 1, Math.floor((p.y / Math.max(h, 1)) * rows))) };
}

/** "the top left", "the middle", "the bottom" of the screen, from thirds. */
export function regionWords(p: Pt, w: number, h: number): string {
  const { col, row } = cellOf(p, w, h);
  const c = ["left", "", "right"][col]!;
  const r = ["top", "", "bottom"][row]!;
  if (!c && !r) return "the middle";
  return `the ${[r, c].filter(Boolean).join(" ")}`;
}

/** "down and a little left", "right", "" when the two points are close (within an eighth of the screen's diagonal). */
export function directionWords(from: Pt, to: Pt, w: number, h: number): string {
  const dx = (to.x - from.x) / Math.max(w, 1);
  const dy = (to.y - from.y) / Math.max(h, 1);
  if (Math.hypot(dx, dy) < 0.18) return "";
  const horizontal = dx > 0 ? "right" : "left";
  const vertical = dy > 0 ? "down" : "up";
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (ax < 0.1 || ax < ay * 0.2) return vertical;
  if (ay < 0.1 || ay < ax * 0.2) return horizontal;
  if (ax >= ay * 1.8) return `${horizontal} and a little ${vertical}`;
  if (ay >= ax * 1.8) return `${vertical} and a little ${horizontal}`;
  return `${vertical} and ${horizontal}`;
}

/** The first sentence when the eyes are known: "From where you are, look down and a little left." */
export function fromGaze(from: Pt, to: Pt, w: number, h: number): string {
  const d = directionWords(from, to, w, h);
  return d ? `From where you are, look ${d}.` : "It is right where you are looking.";
}

/** The sentences of a run, in the order they are said: reveals that happened, then the way, then the arrival. */
export function narration(steps: GuideStep[], done: string[], opened: string | null, gazeLine: string | null): { reveal: string[]; travel: string[]; arrive: string } {
  const reveal = steps.filter((s) => s.reveal && done.includes(s.reveal)).map((s) => s.say);
  if (opened) reveal.unshift(`I am opening ${opened}.`);
  const plain = steps.filter((s) => !s.reveal).map((s) => s.say);
  const arrive = plain.length > 0 ? plain[plain.length - 1]! : "";
  const travel = plain.slice(0, -1);
  if (gazeLine) travel.unshift(gazeLine);
  return { reveal, travel, arrive };
}
