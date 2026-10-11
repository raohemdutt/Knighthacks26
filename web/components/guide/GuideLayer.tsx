"use client";

import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { Dwell, onTarget } from "@/lib/gaze";
import { gaze } from "@/lib/gazeTracker";
import { SCREENS, classify, ease, fromGaze, narration, pathPoint, pinTarget, pinnedThing, resolveApp, thingName, trail, travelMs, type GuideTarget, type Locator, type Pt, type Reveal } from "@/lib/guide";
import { offer, onGuide, pointAt, type GuideMessage, type GuideRun } from "@/lib/guideBus";
import { along, length as routeLength, pointerRoute, type Box } from "@/lib/guidePath";
import { spotRun } from "@/lib/guideSpot";
import { oneSentence } from "@/lib/speechText";
import { useVoice } from "@/lib/voice";
import { Calibrate } from "./Calibrate";
import g from "./guide.module.css";

/**
 * The look guide, once for the whole app (mounted in app/layout.tsx): a red pointer that travels to the thing someone
 * asked for, a ring around it, the walk-through as captions, and the same sentences spoken when the speaker is on.
 *
 *  - It brings the target into view first: opens the right screen, presses the tab, opens the fold, picks the floor,
 *    scrolls. What it may do is declared per target in lib/guide.ts (and by the box for the Inspection screen).
 *  - It never takes a click: only the caption can be tapped. Escape, a tap anywhere else, or another question ends it.
 *  - The target is measured again on every frame, so a resize or a scroll does not leave the ring behind.
 *  - With "reduce motion" on, the pointer does not travel: the ring and the caption appear at once.
 *  - With eye tracking on (Settings), the pointer starts where the person is looking, the first sentence gives the
 *    direction from there, and resting the eyes on the target for 0.7 s turns the ring green ("That's it.").
 *  - On Ask, an answer to "where is my ...?" is followed to its evidence: the picture under the answer.
 *
 * Links: /spot/?guide=battery&t=0.5 · /?guide=keys&t=1 · /?guide=league   (t holds the pointer at that part of its way)
 * ?gazeDebug=1 draws the screen region the eyes are estimated to be in.
 */

type Phase = "reveal" | "travel" | "arrived" | "found";

interface View {
  run: GuideRun;
  phase: Phase;
  t: number;
  start: Pt;
  /** The way round the cards (and through a walking route's waypoints), in screen points; null: the eased arc. */
  route: Pt[] | null;
  /** Where the target's middle was when the route was planned, so the end can follow the target if it moves. */
  anchor: Pt | null;
  rect: DOMRect | null;
  lines: string[];
  note: string | null;
}

interface Offer {
  say: string;
  options: { label: string; run?: GuideRun; ask?: string }[];
}

const visible = (el: Element): DOMRect | null => {
  let r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0 && el.children.length > 0) {
    // display: contents wrappers have no box of their own: use their children's
    const rs = [...el.children].map((c) => c.getBoundingClientRect()).filter((c) => c.width > 0 && c.height > 0);
    if (rs.length === 0) return null;
    const x0 = Math.min(...rs.map((c) => c.left));
    const y0 = Math.min(...rs.map((c) => c.top));
    r = new DOMRect(x0, y0, Math.max(...rs.map((c) => c.right)) - x0, Math.max(...rs.map((c) => c.bottom)) - y0);
  }
  return r.width > 0 && r.height > 0 ? r : null;
};

/** The element a locator names, if it is on the page and has a size. */
function locate(at: Locator): Element | null {
  let found: Element[] = [];
  if (at.id) found = [...document.querySelectorAll(`[data-guide="${at.id.replace(/["\\]/g, "")}"]`)];
  else if (at.sel) {
    try {
      found = [...document.querySelectorAll(at.sel)];
    } catch {
      found = [];
    }
    if (at.text) {
      const re = new RegExp(at.text, "i");
      found = found.filter((el) => re.test((el.textContent ?? "").trim()));
    }
  }
  if (at.up) found = found.map((el) => el.closest(at.up!) ?? el);
  found = found.filter((el) => visible(el));
  return (at.last ? found[found.length - 1] : found[0]) ?? null;
}

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));
const samePath = (a: string, b: string) => a.replace(/\/+$/, "") === b.replace(/\/+$/, "");

export function GuideLayer() {
  const router = useRouter();
  const path = usePathname() || "/";
  const voice = useVoice();
  const { chat, setChat } = useAtlas();
  const [view, setView] = useState<View | null>(null);
  const [offered, setOffered] = useState<Offer | null>(null);
  const [ok, setOk] = useState(false); // the eyes arrived
  const [calibrating, setCalibrating] = useState(false);
  const [debug, setDebug] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const turn = useRef(0); // bumped by every new run and every cancel
  const el = useRef<Element | null>(null);
  const voiceRef = useRef(voice);
  voiceRef.current = voice;
  const pathRef = useRef(path);
  pathRef.current = path;
  const handled = useRef<string | null>(null); // the newest answer already looked at

  const say = useCallback((text: string) => {
    const v = voiceRef.current;
    if (text && v.canSpeak && v.getSettings().speak) v.say(text, { tag: "guide", raw: true, keepOnNavigate: true });
  }, []);

  const cancel = useCallback(() => {
    turn.current++;
    el.current = null;
    voiceRef.current.cancel("guide");
    setView(null);
    setOffered(null);
    setOk(false);
  }, []);

  const start = useCallback(
    async (run: GuideRun) => {
      cancel();
      const mine = turn.current;
      const alive = () => mine === turn.current;
      const { target } = run;
      const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
      const held = run.t !== null && run.t !== undefined;
      const lines: string[] = [];
      const show = (patch: Partial<View>) => alive() && setView((v) => ({ ...(v ?? { run, phase: "reveal", t: 0, start: { x: 0, y: 0 }, route: null, anchor: null, rect: null, lines: [], note: null }), ...patch }));
      const tell = (text: string) => {
        if (!text || !alive()) return;
        lines.push(text);
        show({ lines: [...lines] });
        say(text);
      };
      show({});

      // 1. the right screen
      let opened: string | null = null;
      if (target.route && !samePath(window.location.pathname, target.route)) {
        opened = SCREENS[target.route] ?? "that screen";
        tell(`I am opening ${opened}.`);
        router.push(target.route);
        for (let i = 0; i < 80 && alive() && !samePath(window.location.pathname, target.route); i++) await sleep(50);
        await sleep(250);
      }
      if (!alive()) return;
      // a sheet over the page would hide the target: close it the way the person would
      if (document.querySelector("[role=dialog]") && !target.at.some((a) => a.id && document.querySelector(`[role=dialog] [data-guide="${a.id}"]`))) {
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
        await sleep(120);
      }

      // 2. what has to be opened first
      const done: string[] = [];
      const find = async (at: Locator, ms: number) => {
        for (let waited = 0; waited <= ms && alive(); waited += 100) {
          const found = locate(at);
          if (found) return found;
          await sleep(100);
        }
        return null;
      };
      for (const r of target.reveal ?? ([] as Reveal[])) {
        const found = await find(r.at, 4000);
        if (!alive()) return;
        if (!found) continue;
        let acted = false;
        if (r.op === "press" && found instanceof HTMLElement && found.getAttribute("aria-selected") !== "true" && found.getAttribute("aria-pressed") !== "true") {
          found.click();
          acted = true;
        } else if (r.op === "open" && found instanceof HTMLDetailsElement && !found.open) {
          found.open = true;
          acted = true;
        } else if (r.op === "select" && found instanceof HTMLSelectElement && found.value !== r.value) {
          Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set?.call(found, r.value);
          found.dispatchEvent(new Event("change", { bubbles: true }));
          acted = true;
        }
        if (acted) {
          if (r.key) {
            done.push(r.key);
            const line = target.steps.find((s) => s.reveal === r.key)?.say;
            if (line) tell(line);
          }
          await sleep(260);
        }
      }

      // 3. the target itself: the first locator gets time to load; the fallbacks only after that
      let found: Element | null = null;
      for (let waited = 0; waited <= 6000 && alive() && !found; waited += 100) {
        found = locate(target.at[0]!);
        if (!found && waited >= 2500) for (const at of target.at.slice(1)) if ((found = locate(at))) break;
        if (!found) await sleep(100);
      }
      if (!alive()) return;
      if (!found) {
        tell("I could not find that on this screen.");
        show({ phase: "found" });
        return;
      }
      el.current = found;
      (found.getBoundingClientRect().width > 0 ? found : found.firstElementChild ?? found).scrollIntoView({ block: "center", inline: "nearest", behavior: still || held ? "auto" : "smooth" });
      let last = visible(found);
      for (let i = 0, same = 0; i < 14 && same < 2 && alive(); i++) {
        await sleep(110);
        const now = visible(found);
        same = now && last && Math.abs(now.top - last.top) < 0.5 && Math.abs(now.left - last.left) < 0.5 ? same + 1 : 0;
        last = now;
      }
      if (!alive() || !last) return;

      // 4. the way there, in words and on screen
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const centre = { x: last.left + last.width / 2, y: last.top + last.height / 2 };
      const eyes = gaze.estimate();
      const from: Pt = eyes ? { x: eyes.x * vw, y: eyes.y * vh } : { x: vw / 2, y: vh / 2 };
      const origin: Pt = eyes ? { x: ((eyes.col + 0.5) / eyes.cols) * vw, y: ((eyes.row + 0.5) / eyes.rows) * vh } : from;
      const gazeLine = eyes ? fromGaze(origin, centre, vw, vh) : null;
      const words = narration(target.steps, done, null, gazeLine);
      const arrive = run.arrive ?? words.arrive;
      const note = run.note?.(found) ?? null;
      // around the cards rather than across their text; through the walking route's waypoints when there is one
      const stops = (run.via ?? []).map((at) => locate(at)).filter((x): x is Element => x !== null).map((x) => visible(x)!).map((r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2 }));
      const first = stops[0] ?? centre;
      const around = pointerRoute(origin, first, obstacles(found), [], vw, vh);
      const route = stops.length > 0 ? [...(around ?? [origin, first]), ...stops.slice(1), centre] : around;
      show({ start: origin, route, anchor: centre, rect: last, note });

      if (held || still) {
        const at = held ? Math.max(0, Math.min(1, run.t!)) : 1;
        for (const line of [...words.travel, ...(run.extra ?? [])]) tell(line);
        if (at >= 1) tell(arrive);
        show({ phase: at >= 1 ? "arrived" : "travel", t: at });
      } else {
        for (const line of [...words.travel, ...(run.extra ?? [])]) tell(line);
        show({ phase: "travel", t: 0 });
        const ms = route ? Math.round(Math.max(900, Math.min(3200, 500 + routeLength(route) * 1.3))) : travelMs(origin, centre);
        const t0 = performance.now();
        await new Promise<void>((resolve) => {
          const frame = (now: number) => {
            if (!alive()) return resolve();
            const t = Math.min(1, (now - t0) / ms);
            show({ t });
            if (t < 1) window.requestAnimationFrame(frame);
            else resolve();
          };
          window.requestAnimationFrame(frame);
        });
        if (!alive()) return;
        tell(arrive);
        show({ phase: "arrived", t: 1 });
      }

      // 5. did the eyes get there?
      if (eyes && !held) {
        const dwell = new Dwell();
        const began = performance.now();
        while (alive() && performance.now() - began < 20_000) {
          await sleep(100);
          const e = gaze.estimate();
          const r = el.current ? visible(el.current) : null;
          const on = e && r ? onTarget(e, e.cols, e.rows, { x0: r.left / window.innerWidth, y0: r.top / window.innerHeight, x1: r.right / window.innerWidth, y1: r.bottom / window.innerHeight }) : null;
          const what = dwell.update(on, performance.now());
          if (what === "arrived") {
            if (!alive()) return;
            setOk(true);
            tell("That's it.");
            break;
          }
          if (what === "repeat" && e && r) tell(fromGaze({ x: ((e.col + 0.5) / e.cols) * window.innerWidth, y: ((e.row + 0.5) / e.rows) * window.innerHeight }, { x: r.left + r.width / 2, y: r.top + r.height / 2 }, window.innerWidth, window.innerHeight));
        }
      }
      if (!held) {
        await sleep(run.action ? 14_000 : 9000);
        if (alive()) cancel();
      }
    },
    [cancel, router, say],
  );

  /** A question or a guide id from a link or a "Did you mean" button, on whatever screen this is. */
  const ask = useCallback(
    async (query: string, t: number | null = null) => {
      const here = pathRef.current;
      if (samePath(here, "/spot/")) {
        try {
          const { run, found } = await spotRun(query, t);
          if (run) return void start(run);
          if (found && found.alternatives.length > 0) return offer(found.steps[0]?.say ?? "Did you mean one of these?", found.alternatives.map((a) => ({ label: a.label, ask: a.target })));
        } catch {
          /* the box is not answering: fall through to what the web layer knows */
        }
      }
      const asked = classify(query, here);
      const room = asked?.kind === "room" ? asked.thing : pinnedThing(query.toLowerCase()) ? query.toLowerCase() : null;
      if (room !== null && samePath(here, "/")) {
        // a thing in the room, from a link: ask the box as the person would, then follow its answer
        const q = asked ? query : `Where are my ${thingName(room)}?`;
        try {
          const answer = await api.ask(q);
          const id = `g${Date.now()}`;
          handled.current = `${id}-a`;
          setChat((turns) => [...turns, { id, role: "you", text: q }, { id: `${id}-a`, role: "atlas", answer }]);
          await sleep(150);
          return void start(evidenceRun(room, answer.answer, answer.citations.some((c) => c.frame_ref), t));
        } catch {
          return;
        }
      }
      const app = resolveApp(query, here);
      if (app.target) return void start({ target: app.target, t });
      if (app.alternatives.length > 0) offer("I am not sure what to point at. Did you mean one of these?", app.alternatives.map((a) => ({ label: a.label, run: { target: a } })));
      else offer("I do not know where that is in the app. Try asking the question on the Ask screen.", []);
    },
    [setChat, start],
  );

  // requests from the screens
  useEffect(
    () =>
      onGuide((m: GuideMessage) => {
        if (m.kind === "run") void start(m.run);
        else if (m.kind === "calibrate") (cancel(), setCalibrating(true));
        else (cancel(), setOffered({ say: m.say, options: m.options }), say(m.say));
      }),
    [start, cancel, say],
  );

  // a link: ?guide=<what>&t=<0..1>, and ?gazeDebug=1
  useEffect(() => {
    gaze.boot();
    const q = new URLSearchParams(window.location.search);
    const what = q.get("guide");
    const t = q.get("t");
    let timer: number | undefined;
    if (what) timer = window.setTimeout(() => void ask(what, t === null || Number.isNaN(Number(t)) ? null : Number(t)), 700);
    let tick: number | undefined;
    if (q.get("gazeDebug")) {
      tick = window.setInterval(() => {
        const e = gaze.estimate();
        setDebug(e ? { x: (e.col / e.cols) * 100, y: (e.row / e.rows) * 100, w: 100 / e.cols, h: 100 / e.rows } : null);
      }, 150);
    }
    return () => (window.clearTimeout(timer), window.clearInterval(tick));
    // once per page load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ask: follow the answer to a where-question to its evidence
  const seenFirst = useRef(false);
  useEffect(() => {
    const lastTurn = chat[chat.length - 1];
    if (!seenFirst.current) {
      // whatever is in the conversation when the app opens is not a new answer
      seenFirst.current = true;
      handled.current = lastTurn?.id ?? "";
      return;
    }
    if (!lastTurn || lastTurn.role !== "atlas" || handled.current === lastTurn.id) return;
    handled.current = lastTurn.id;
    const before = chat[chat.length - 2];
    if (!before || before.role !== "you" || !samePath(pathRef.current, "/")) return;
    const asked = classify(before.text, "/");
    const framed = lastTurn.answer.citations.some((c) => c.frame_ref);
    if (!asked || asked.kind === "app" || (asked.kind === "unsure" && !framed)) return;
    const run = evidenceRun(asked.thing, lastTurn.answer.answer, framed, null);
    const timer = window.setTimeout(() => void start(run), 250);
    return () => window.clearTimeout(timer);
  }, [chat, start]);

  // Escape, or a tap anywhere but the caption, ends it
  const active = view !== null || offered !== null;
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => e.isTrusted && e.key === "Escape" && cancel();
    const onDown = (e: PointerEvent) => {
      if (e.target instanceof Element && e.target.closest("[data-guide-caption]")) return;
      cancel();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [active, cancel]);

  // keep the ring on the target through scrolls and resizes
  const [, redraw] = useState(0);
  const watching = view !== null && view.rect !== null;
  useEffect(() => {
    if (!watching) return;
    let raf = 0;
    let before = "";
    const loop = () => {
      const r = el.current ? visible(el.current) : null;
      const key = r ? `${r.left.toFixed(1)},${r.top.toFixed(1)},${r.width.toFixed(1)},${r.height.toFixed(1)}` : "";
      if (key !== before) {
        before = key;
        redraw((n) => n + 1);
      }
      raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [watching]);

  const debugBox = debug ? <div className={g.gazeDot} style={{ left: `${debug.x}%`, top: `${debug.y}%`, width: `${debug.w}%`, height: `${debug.h}%` }} data-gaze-debug /> : null;
  if (calibrating) return <Calibrate onDone={() => setCalibrating(false)} />;
  if (!view && !offered) return debugBox ? <div className={g.layer}>{debugBox}</div> : null;

  const vw = typeof window === "undefined" ? 1024 : window.innerWidth;
  const vh = typeof window === "undefined" ? 768 : window.innerHeight;
  const rect = view && el.current ? visible(el.current) ?? view.rect : view?.rect ?? null;
  const centre = rect ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 } : null;
  const moving = view && centre && view.phase === "travel";
  // on a planned route the end follows the target: the whole way is nudged by however far the target has moved since
  const onRoute = (u: number): Pt => {
    const p = along(view!.route!, u);
    const k = Math.max(0, Math.min(1, u));
    return { x: p.x + (centre!.x - view!.anchor!.x) * k, y: p.y + (centre!.y - view!.anchor!.y) * k };
  };
  const routed = Boolean(view?.route && view.anchor && centre);
  const at = view && centre ? (routed ? onRoute(ease(view.t)) : pathPoint(view.start, centre, ease(view.t))) : null;
  const tail =
    moving && view.t > 0 && view.t < 1
      ? routed
        ? Array.from({ length: 18 }, (_, i) => onRoute(Math.max(0, ease(view.t) - 0.22 + (0.22 * i) / 17)))
        : trail(view.start, centre!, view.t)
      : [];
  const landed = view?.phase === "arrived";
  const small = rect ? Math.max(rect.width, rect.height) < 64 : false;
  const pad = small ? 8 : 6;

  // the caption sits under the target when there is room, above it otherwise; in the middle when there is no target
  const capW = Math.min(330, vw - 24);
  let capX = (vw - capW) / 2;
  let capY = vh * 0.3;
  if (rect) {
    capX = Math.max(12, Math.min(vw - capW - 12, rect.left + rect.width / 2 - capW / 2));
    const below = rect.bottom + pad + 14;
    capY = below + 150 < vh ? below : Math.max(12, rect.top - pad - 14 - 150);
    if (rect.height > vh * 0.6) capY = Math.max(12, Math.min(vh - 170, rect.top + 24)); // a tall target: inside its top
  }
  const lines = view?.lines ?? [];

  return (
    <div className={g.layer} aria-live="polite" data-guide-layer={view?.phase ?? "offer"} data-guide-target={view?.run.target.id}>
      {debugBox}
      {tail.length > 1 ? (
        <svg className={g.trail} aria-hidden="true">
          {tail.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={2 + (5 * i) / tail.length} opacity={(0.55 * (i + 1)) / tail.length} />
          ))}
        </svg>
      ) : null}
      {rect && landed ? (
        <div
          className={`${g.ring} ${small ? g.ringRound : ""} ${ok ? g.ringOk : ""} ${view?.run.t != null ? g.ringStill : ""}`}
          style={{ transform: `translate(${rect.left - pad}px, ${rect.top - pad}px)`, width: rect.width + 2 * pad, height: rect.height + 2 * pad }}
        />
      ) : null}
      {at && !landed ? <div className={g.dot} style={{ transform: `translate(${at.x}px, ${at.y}px)` }} /> : null}
      <div className={`${g.caption} ${ok ? g.captionOk : ""}`} style={{ transform: `translate(${capX}px, ${capY}px)` }} data-guide-caption role="status">
        <p className={g.label}>
          <span>{view ? view.run.target.label : "Where is it?"}</span>
          <button type="button" className={g.close} onClick={cancel} aria-label="Stop pointing">
            Close
          </button>
        </p>
        {offered ? <p className={g.line}>{offered.say}</p> : null}
        {lines.slice(-2).map((line, i, shown) => (
          <p key={`${lines.length}-${i}`} className={`${g.line} ${i < shown.length - 1 ? g.lineOld : ""}`}>
            {line}
          </p>
        ))}
        {view?.note && landed ? <p className={g.note}>{view.note}</p> : null}
        {offered && offered.options.length > 0 ? (
          <div className={g.acts}>
            {offered.options.map((o) => (
              <button key={o.label} type="button" className={g.act} onClick={() => (o.run ? pointAt(o.run) : o.ask ? void ask(o.ask) : undefined)}>
                {o.label}
              </button>
            ))}
          </div>
        ) : null}
        {view?.run.action && landed ? (
          <div className={g.acts}>
            <button type="button" className={g.act} onClick={() => pointAt(view.run.action!.run)}>
              {view.run.action.label}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** What the pointer should go round: the cards and panels on the page, except any that hold the target. */
function obstacles(target: Element): Box[] {
  const out: Box[] = [];
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  for (const el of document.querySelectorAll("main section, main article, main figure, main [class*='card'], main [class*='Card']")) {
    if (el.contains(target) || target.contains(el)) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 80 || r.height < 60 || r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) continue;
    if (r.width > vw * 0.9 && r.height > vh * 0.9) continue; // a wrapper around everything is not a card
    out.push({ left: r.left, top: r.top, right: r.right, bottom: r.bottom });
  }
  return out;
}

/** The pointer's run for an answer about a thing in the room: to the picture under the answer, or to the answer when it has none. */
function evidenceRun(thing: string, answer: string, framed: boolean, t: number | null): GuideRun {
  const name = thingName(thing);
  const pinned = pinnedThing(thing);
  const newest = "main article[data-guide='answer']:last-of-type";
  const target: GuideTarget = {
    id: "evidence",
    route: "/",
    label: `Where your ${name} ${/s$/.test(name) ? "are" : "is"}`,
    at: framed ? [{ sel: `${newest} [data-guide='frame']` }, { sel: newest }] : [{ sel: newest }],
    steps: [{ say: framed ? "Under my answer: the picture it comes from." : "This is my answer. I have no picture to show for it." }, { say: "" }],
  };
  return {
    target,
    t,
    arrive: oneSentence(answer, 170) || "That is all I know about it.",
    note: (found) =>
      !framed
        ? "No camera frame is attached to this answer."
        : found.querySelector("[role=img]")
          ? "The picture for that moment is not stored on this box, so this is where it would be."
          : "The box keeps the zone and a description, not the spot inside the picture, so the whole picture is ringed.",
    action: pinned ? { label: "Show me in the room", run: { target: pinTarget(pinned) } } : undefined,
  };
}
