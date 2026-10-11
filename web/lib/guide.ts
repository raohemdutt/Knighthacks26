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
  "/dayrun/": "Today's Plan",
  "/circle/": "Circle",
  "/impact/": "Impact",
  "/home/": "Home",
  "/map/": "the Room screen",
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
  // Today's Plan (the route and the ids keep their old name)
  { id: "dayrun:plan", route: "/dayrun/", label: "Your plan in Today's Plan", names: ["today's plan", "todays plan", "day run plan", "day run", "plan my day", "the game", "day game", "my plan today"], at: [section("^Your plan"), HEAD], say: ["Look at the part called Your plan.", "Tap a block to move it, or tap Add in a gap; every change is tried at once."] },
  { id: "dayrun:play", route: "/dayrun/", label: "Watch the day", names: ["play", "play the day", "watch the day", "play mode", "watch my day"], at: [{ sel: "main button", text: "^(Watch|Pause|Play)" }, HEAD], say: ["Look at the card at the top of Today's Plan.", "This plays your day, with a figure walking the campus."] },
  { id: "dayrun:map", route: "/dayrun/", label: "The map of your plan", names: ["day map", "campus map", "where the day takes me", "the route", "campus", "plan map"], at: [{ sel: "main [role=group][aria-label^='Map of']" }, HEAD], say: ["Look at the map in the card at the top.", "Tap a block of your plan and its walk lights up here."] },
  { id: "dayrun:board", route: "/dayrun/", label: "What is due", names: ["the board", "board", "the deadlines board", "risk", "what is due", "deadlines in the plan"], at: [section("^Due soon"), HEAD], say: ["Look at the part called Due soon.", "Each deadline, how much work is left, and a Why that says where the numbers come from."] },
  { id: "dayrun:tell", route: "/dayrun/", label: "Tell Atlas what to change", names: ["tell atlas", "change the plan", "edit the plan", "say a change", "tell atlas what to change"], at: [{ sel: "main input[aria-label='Tell Atlas what to change']" }, HEAD], say: ["Look at the bottom of the card at the top.", "Type or say a change, like move lunch to one."] },
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
  { id: "map:room", route: "/map/", label: "The room", names: ["room map", "the room", "map of the room", "zones map", "the map", "room twin", "3d room", "walk the room"], at: [{ id: "map:room" }, HEAD], say: ["Look at the room on the Room screen.", "The room, drawn from its floor plan. A pin is one of your things; drag to look around."] },
  { id: "map:now", route: "/map/", label: "Right now in the room", names: ["right now", "who is in view", "who is in the room", "who the cameras see", "in view"], at: [section("^Right now"), HEAD], say: ["Look at the part called Right now.", "Who and what the cameras see at this moment."] },
  { id: "map:cameras", route: "/map/", label: "Camera views", names: ["camera views", "camera panels", "the zones", "cameras and zones"], reveal: [{ op: "press", at: { sel: "main [role=group][aria-label='View'] button", text: "^Camera views" }, key: "cameras" }], at: [{ id: "map:room" }, HEAD], say: ["Look at the room card, switched to Camera views.", "Each panel is one camera, with its zones where they sit in the picture."] },
  { id: "map:things", route: "/map/", label: "Where are my things?", names: ["my things", "where are my things", "things list", "last seen", "lost things"], at: [section("^Where are my things"), HEAD], say: ["Look at the part called Where are my things.", "Tap a name to see where the cameras last saw it."] },
  { id: "map:sort", route: "/map/", label: "Which bin?", names: ["which bin", "the bin", "bin route", "way to the bin", "where does this go"], reveal: [{ op: "open", at: { id: "map:bins" }, key: "bins" }], at: [{ id: "map:bins" }, HEAD], say: ["Look under the room, at Which bin does it go in.", "Pick what you are holding and the way to its bin is drawn in the room."] },
  // Home
  { id: "home:top", route: "/home/", label: "Home", names: ["home", "home screen", "landing", "start screen", "the box", "the device", "atlas box", "exploded view", "what is inside the box"], at: [HEAD], say: ["Look at the top of Home.", "The Atlas box. Scroll and it pulls apart to show what is inside."] },
  { id: "home:what", route: "/home/", label: "What Atlas does", names: ["what atlas does", "what it does", "features", "what can atlas do"], at: [{ sel: "main section[aria-labelledby='what']" }, HEAD], say: ["Look further down Home.", "What Atlas does, in four parts: it sees, remembers, plans the day and asks first."] },
  { id: "home:numbers", route: "/home/", label: "The numbers", names: ["the numbers", "by the numbers", "stats", "figures"], at: [{ sel: "main section[aria-label='By the numbers']" }, HEAD], say: ["Look at the row of numbers on Home.", "The few numbers that matter about the box."] },
  { id: "home:flow", route: "/home/", label: "How a question travels", names: ["how a question travels", "how it works", "the flow", "how atlas works", "see detect remember"], at: [{ sel: "main section[aria-labelledby='flow']" }, HEAD], say: ["Look at the drawing near the foot of Home.", "How a question travels through the box: see, detect, remember, ask, answer, act."] },
  { id: "home:start", route: "/home/", label: "A way in", names: ["a way in", "try it", "get started", "where do i start on home"], at: [{ sel: "main section[aria-labelledby='go']" }, HEAD], say: ["Look at the foot of Home.", "Two ways in: Ask Atlas, or See today."] },
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
    label: `${label} in the room`,
    // picking the thing in the Room screen's own finder focuses its pin in the room and opens its card: the guide drives that, and adds the dot and the words
    reveal: [{ op: "press", at: chip, key: "pick" }],
    at: [{ id: `pin:${thing}` }, { id: `thing:${thing}` }, section("^Where are my things"), HEAD],
    steps: [{ say: "Look at the room." }, { say: `This is where the cameras last saw your ${thing}.` }],
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
  | { kind: "thing"; thing: string } // anything that is not a part of the app: the box answers; the guide then looks for it (answer, cameras, room)
  | { kind: "app"; target: GuideTarget } // something in the app: resolved here
  | null; // not a where-question

/**
 * No list of nouns: a where-question that names a part of the app goes to the app guide, and every other one is about
 * a thing (or a person, or a place) the box is asked for. What can be pointed at is decided after the answer, from what
 * the box really has (`resolveThing`).
 */
export function classify(question: string, route: string | null = null): Asked {
  const thing = whereIntent(question);
  if (thing === null) return null;
  const app = resolveApp(question, route);
  if (app.target && app.confidence >= 0.8) return { kind: "app", target: app.target };
  return { kind: "thing", thing };
}

/** The thing itself, for a sentence: "my keys" -> "keys". */
export function thingName(thing: string): string {
  return thing.split(" ").filter((w) => !FILLER.has(w)).join(" ") || thing;
}

/** The labels the box may know a thing by, most likely first: "my key" -> ["key", "keys"]; "the coffee mugs" -> ["coffee mugs", "coffee mug", "mugs", "mug"]. */
export function labelsFor(thing: string): string[] {
  const name = thingName(thing);
  const forms = (w: string) => (w.length > 3 && w.endsWith("s") ? [w, w.slice(0, -1)] : [w, `${w}s`]);
  const last = name.split(" ").pop() ?? name;
  return [...new Set([...forms(name), ...(last !== name ? forms(last) : [])])].filter(Boolean);
}

/** Was it asked as the person's own thing ("my keys", "our charger")? A name ("Dominick") or a place is not. */
export const ownThing = (thing: string): boolean => /^(?:my|our) /.test(thing);

// ---------- any object: where it is in a picture, exactly or about ----------

export interface BoxN {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** GET /memory/last-seen/<label> */
export interface WhereSeen {
  label: string;
  camera_id: string;
  zone: string | null;
  position: string | null;
  frame_ref: string | null;
  box: BoxN | null;
  seen_at: string;
  anchors: { label: string; box: BoxN }[];
}

export type Relation = "left of" | "right of" | "in front of" | "behind" | "next to" | "on" | "under" | "between";
const RELATIONS: [RegExp, Relation][] = [
  [/\b(?:to the left of|left of|on the left of)\b/, "left of"],
  [/\b(?:to the right of|right of|on the right of)\b/, "right of"],
  [/\bin front of\b/, "in front of"],
  [/\bbehind\b/, "behind"],
  [/\b(?:next to|beside|by|near)\b/, "next to"],
  [/\b(?:under|underneath|below|beneath)\b/, "under"],
  [/\bbetween\b/, "between"],
  [/\b(?:on top of|on|atop)\b/, "on"],
];

/** "left of the closed laptop" -> left of, ["closed laptop"]. The first relation in the sentence wins; null when there is none. */
export function parseRelation(position: string | null): { relation: Relation; things: string[] } | null {
  const low = plainText(position ?? "");
  let best: { at: number; end: number; relation: Relation } | null = null;
  for (const [re, relation] of RELATIONS) {
    const m = re.exec(low);
    if (m && (!best || m.index < best.at)) best = { at: m.index, end: m.index + m[0].length, relation };
  }
  if (!best) return null;
  const rest = low.slice(best.end).split(/\b(?:left of|right of|in front of|behind|next to|beside|under|at|in the)\b/)[0]!.trim();
  const things = (best.relation === "between" ? rest.split(/\band\b/) : [rest]).map((t) => thingName(t.trim())).filter(Boolean);
  return things.length > 0 ? { relation: best.relation, things } : null;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** A soft region for "<relation> the anchor": as big as the anchor is small, never a tight box. In a picture, "behind" is above and "in front of" is below. */
export function aboutRegion(relation: Relation, anchors: BoxN[]): BoxN | null {
  const a = anchors[0];
  if (!a) return null;
  const w = a.x2 - a.x1;
  const h = a.y2 - a.y1;
  const side = Math.max(0.12, Math.min(0.3, w * 0.8));
  const tall = Math.max(0.12, Math.min(0.3, h * 0.8));
  let r: BoxN;
  if (relation === "between" && anchors[1]) {
    const b = anchors[1];
    r = { x1: Math.min(a.x2, b.x2), x2: Math.max(a.x1, b.x1), y1: Math.min(a.y1, b.y1), y2: Math.max(a.y2, b.y2) };
    if (r.x2 - r.x1 < 0.1) r = { ...r, x1: (r.x1 + r.x2) / 2 - 0.08, x2: (r.x1 + r.x2) / 2 + 0.08 };
  } else if (relation === "left of") r = { x1: a.x1 - side, x2: a.x1, y1: a.y1 - 0.04, y2: a.y2 + 0.04 };
  else if (relation === "right of") r = { x1: a.x2, x2: a.x2 + side, y1: a.y1 - 0.04, y2: a.y2 + 0.04 };
  else if (relation === "behind") r = { x1: a.x1 - 0.04, x2: a.x2 + 0.04, y1: a.y1 - tall, y2: a.y1 + h * 0.15 };
  else if (relation === "in front of" || relation === "under") r = { x1: a.x1 - 0.04, x2: a.x2 + 0.04, y1: a.y2 - h * 0.15, y2: a.y2 + tall };
  else if (relation === "on") r = { x1: a.x1, x2: a.x2, y1: a.y1 - h * 0.1, y2: a.y2 };
  else r = { x1: a.x1 - side * 0.7, x2: a.x2 + side * 0.7, y1: a.y1 - 0.06, y2: a.y2 + 0.06 }; // next to: either side
  const out = { x1: clamp01(r.x1), y1: clamp01(r.y1), x2: clamp01(r.x2), y2: clamp01(r.y2) };
  return out.x2 - out.x1 > 0.03 && out.y2 - out.y1 > 0.03 ? out : null;
}

export interface Spotlight {
  /** "box": a tight outline on the thing. "about": a soft region placed from the words. "picture": the whole picture. */
  kind: "box" | "about" | "picture";
  rect: BoxN | null; // null: the whole picture
  precision: Precision;
  label: string; // "Here it is" | "About here"
  words: string | null; // the model's own words for the place, shown underneath
  anchor: string | null; // what the region was placed against
}

/**
 * What to draw on the picture a thing was seen in. A tight outline needs a box for the thing itself: with no box the
 * most this ever returns is a soft region beside something that does have one, and with no such anchor, the picture.
 */
export function spotlightFor(where: Pick<WhereSeen, "box" | "position" | "anchors"> | null, plural = false): Spotlight {
  const words = where?.position?.trim() || null;
  const b = where?.box;
  if (b && [b.x1, b.y1, b.x2, b.y2].every((v) => Number.isFinite(v)) && b.x2 > b.x1 && b.y2 > b.y1) {
    const p = precisionOf("box");
    return { kind: "box", rect: { x1: clamp01(b.x1), y1: clamp01(b.y1), x2: clamp01(b.x2), y2: clamp01(b.y2) }, precision: p, label: spotWords(p, plural), words, anchor: null };
  }
  const p = precisionOf(words ? "words" : "picture");
  const rel = parseRelation(words);
  if (rel && where) {
    const found = rel.things.map((t) => where.anchors.find((a) => labelsFor(t).some((l) => l === a.label || a.label.split(" ").includes(l))));
    const boxes = found.filter((a): a is { label: string; box: BoxN } => Boolean(a)).map((a) => a.box);
    const rect = boxes.length > 0 && (rel.relation !== "between" || boxes.length > 1 || rel.things.length === 1) ? aboutRegion(rel.relation, boxes) : null;
    if (rect) return { kind: "about", rect, precision: p, label: spotWords(p), words, anchor: found.find(Boolean)!.label };
  }
  return { kind: "picture", rect: null, precision: p, label: spotWords(p), words, anchor: null };
}

/** A box in the picture -> where it falls in a frame that shows the picture with object-fit: cover (percent of the frame). */
export function coverRect(box: BoxN, natural: { w: number; h: number }, frame: { w: number; h: number }): { left: number; top: number; width: number; height: number } {
  const scale = Math.max(frame.w / Math.max(natural.w, 1), frame.h / Math.max(natural.h, 1));
  const w = natural.w * scale;
  const h = natural.h * scale;
  const ox = (frame.w - w) / 2;
  const oy = (frame.h - h) / 2;
  const x1 = Math.max(0, ox + box.x1 * w);
  const y1 = Math.max(0, oy + box.y1 * h);
  const x2 = Math.min(frame.w, ox + box.x2 * w);
  const y2 = Math.min(frame.h, oy + box.y2 * h);
  return { left: (100 * x1) / frame.w, top: (100 * y1) / frame.h, width: (100 * Math.max(0, x2 - x1)) / frame.w, height: (100 * Math.max(0, y2 - y1)) / frame.h };
}

export type ThingSource = "answer-box" | "answer-words" | "answer-picture" | "live-track" | "room-pin" | "none";

export interface ThingFound {
  source: ThingSource;
  precision: Precision | null; // null: nothing to point at
  spotlight: Spotlight | null; // for the answer's picture
  say: string; // the sentence for it: "Your keys are here." / "about here: left of the laptop" / "I haven't seen that."
}

/**
 * Where to point for a thing the box was asked about, in this order: the answer's own picture (with the sighting's box
 * or words when the sighting is of that same frame), a live detector track, the room's pin. `citationFrame` is the frame
 * under the answer; `where` is Memory's last sighting; `live` is true when a camera tracks it right now.
 */
export function resolveThing(input: { thing: string; citationFrame: string | null; where: WhereSeen | null; live: boolean }): ThingFound {
  const name = thingName(input.thing);
  const plural = /s$/.test(name);
  const { where, citationFrame } = input;
  if (citationFrame) {
    const same = where && where.frame_ref === citationFrame ? where : null;
    const spot = spotlightFor(same, plural);
    const source: ThingSource = spot.kind === "box" ? "answer-box" : same?.position ? "answer-words" : "answer-picture";
    return { source, precision: spot.precision, spotlight: spot, say: hereSentence(spot.precision, name, plural, source === "answer-picture" ? "somewhere in this picture" : spot.words) };
  }
  if (input.live) {
    const p = precisionOf("box");
    return { source: "live-track", precision: p, spotlight: null, say: `${hereSentence(p, name, plural)} This is the camera's picture, refreshed, not video.` };
  }
  if (where) {
    const p = precisionOf("zone-pin");
    return { source: "room-pin", precision: p, spotlight: null, say: hereSentence(p, name, plural, where.position ?? (where.zone ? `in the ${where.zone} zone` : null)) };
  }
  return { source: "none", precision: null, spotlight: null, say: "I haven't seen that." };
}

// ---------- exact, or about: one rule for the drawing and the words ----------

/**
 * What a target's place is known from. "Exact" is only ever claimed from something measured or drawn by the app itself;
 * anything inferred, worded or unscaled is "about", and both the outline and the sentence must say so.
 */
export type PlaceSource =
  | "element" // a thing on this screen: its own rectangle
  | "box" // an object the box put a box round in a picture
  | "zone" // a zone polygon the box has for that camera
  | "floor" // a calibrated position on the room's floor plan
  | "route-real" // a route the box planned over a measured plan
  | "picture" // "it is somewhere in this picture": a frame with no box for the thing
  | "words" // a position given in words ("left of the laptop")
  | "zone-pin" // a pin placed inside a zone, not at the spot
  | "floor-default" // a floor position from a camera still on the room file's placeholder calibration
  | "route-traced" // a route over a map traced from a picture, or over legs with no geometry
  | "unscaled"; // a length with no defensible scale

export interface Precision {
  precision: "exact" | "about";
  reason: string;
}

const EXACT: Partial<Record<PlaceSource, string>> = {
  element: "it is a thing on this screen",
  box: "the box put a box round it in the picture",
  zone: "the box has that zone's outline for this camera",
  floor: "its place on the floor plan is calibrated",
  "route-real": "the route was planned over a measured plan",
};
const ABOUT: Partial<Record<PlaceSource, string>> = {
  picture: "the box keeps the zone and a description, not where it is in the picture",
  words: "its place is known only in words",
  "zone-pin": "the pin sits inside its zone, not at the exact spot",
  "floor-default": "this camera is not calibrated for the room yet",
  "route-traced": "the route follows what the data shows, not a surveyed path",
  unscaled: "the plan has no scale",
};

/** The precision a source earns. Anything not listed as exact is about: exact is never the default. */
export function precisionOf(source: PlaceSource): Precision {
  const why = EXACT[source];
  return why ? { precision: "exact", reason: why } : { precision: "about", reason: ABOUT[source] ?? "its place is not measured" };
}

/**
 * A pin in the room twin: exact only when the box projected the spot from a camera track AND that camera's calibration
 * was really set (GET /api/room, `calibration[camera].source`). A placeholder calibration makes every position about.
 */
export function roomPrecision(projected: boolean, calibration: "set" | "default" | undefined): Precision {
  if (!projected) return precisionOf("zone-pin");
  return precisionOf(calibration === "default" ? "floor-default" : "floor");
}

/** The two or three words on the spotlight: "Here it is" only when exact, "About here" otherwise. */
export function spotWords(p: Precision, plural = false): string {
  return p.precision === "exact" ? (plural ? "Here they are" : "Here it is") : "About here";
}

/** "Your keys are here." / "Your keys are about here: left of the laptop." */
export function hereSentence(p: Precision, name: string, plural: boolean, detail?: string | null): string {
  const be = plural ? "are" : "is";
  if (p.precision === "exact") return `Your ${name} ${be} here.`;
  return `Your ${name} ${be} about here${detail ? `: ${detail}` : ""}.`;
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
