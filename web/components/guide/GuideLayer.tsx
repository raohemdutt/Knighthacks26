"use client";

import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { INSPECTION_SPEAKS, boxLines } from "@/lib/boxSpeech";
import { Dwell, landed as eyesLanded, snap, type Thing } from "@/lib/gaze";
import { gaze, type GazeEstimate } from "@/lib/gazeTracker";
import { FIXATION_MS, attention, simScript } from "@/lib/guideAttention";
import { SCREENS, classify, ease, fromGaze, labelsFor, narration, ownThing, pathPoint, pinTarget, precisionOf, resolveApp, resolveThing, spotWords, thingName, trail, travelMs, type GuideTarget, type ThingFound, type Locator, type Pt, type Reveal } from "@/lib/guide";
import { offer, onGuide, pointAt, type GuideMessage, type GuideRun } from "@/lib/guideBus";
import { along, length as routeLength, pointerRoute, type Box } from "@/lib/guidePath";
import { alongStops, sayMs, stepAt, walkFraction } from "@/lib/guideWalk";
import { binRun } from "@/lib/guideBin";
import { spotRun } from "@/lib/guideSpot";
import { findLive, frameSpots, whereSeen } from "@/lib/guideSpotlight";
import { cameraOfZone } from "@/lib/room";
import { oneSentence } from "@/lib/speechText";
import { useVoice } from "@/lib/voice";
import { Calibrate } from "./Calibrate";
import { Spotting, type SpottingAsk } from "./Spotting";
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

type Phase = "reveal" | "travel" | "walk" | "arrived" | "found";

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
  /** A paced walk (run.pace): the route's points on the page and each one's share of the way. In phase "walk", `t` is the share of the walk's time. */
  walk?: { els: Element[]; at: number[] } | null;
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
  const found = locateAll(at);
  return found === "hidden" ? null : found;
}

/** The same, telling "it is on the page but not shown at this width" ("hidden") from "not there (yet)" (null). */
function locateAll(at: Locator): Element | "hidden" | null {
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
  const there = found.length > 0;
  found = found.filter((el) => visible(el));
  return (at.last ? found[found.length - 1] : found[0]) ?? (there ? "hidden" : null);
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
  const [spotting, setSpotting] = useState<SpottingAsk | null>(null); // a camera's refreshed picture with the thing (or the bin) outlined
  const [sim, setSim] = useState(false); // ?gazeSim: the camera is replaced by a script, and the screen says so
  const [debug, setDebug] = useState<{ on: boolean; e: GazeEstimate | null; snapped: Thing[]; several: boolean; why: string }>({ on: false, e: null, snapped: [], several: false, why: "" });
  const turn = useRef(0); // bumped by every new run and every cancel
  const el = useRef<Element | null>(null);
  const voiceRef = useRef(voice);
  voiceRef.current = voice;
  const pathRef = useRef(path);
  pathRef.current = path;
  const handled = useRef<string | null>(null); // the newest answer already looked at

  const unsaid = useRef(0); // guide lines handed to the browser's voice and not yet said
  /** Is a guide line still waiting or being said (on the box's speaker or in this browser)? */
  const talking = useCallback(() => boxLines.busy() || unsaid.current > 0, []);

  const say = useCallback((text: string) => {
    const v = voiceRef.current;
    if (!text || !v.getSettings().speak) return;
    if (!INSPECTION_SPEAKS && samePath(pathRef.current, "/spot/")) return; // the Inspection screen is silent: the caption carries the words
    const here = (line: string) => {
      if (!voiceRef.current.canSpeak) return;
      unsaid.current++;
      voiceRef.current.say(line, { tag: "guide", raw: true, keepOnNavigate: true, onDone: () => (unsaid.current = Math.max(0, unsaid.current - 1)) });
    };
    // on the Inspection screen the lines come out of the box's own speaker when it has one working, in order; else, and on every other screen, out of this browser
    if (samePath(pathRef.current, "/spot/")) boxLines.say(text, here);
    else here(text);
  }, []);

  const cancel = useCallback(() => {
    turn.current++;
    el.current = null;
    voiceRef.current.cancel("guide");
    boxLines.cancel();
    unsaid.current = 0;
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
      // On the Inspection screen the pointer keeps step with the voice: it does not start a stretch, arrive or leave while an
      // earlier line is still being said. `quiet` waits for the voice (never for ever); with the speaker off it returns at once.
      const paced = () => samePath(window.location.pathname, "/spot/");
      const quiet = async (most: number) => {
        for (let waited = 0; alive() && paced() && talking() && waited < most; waited += 50) await sleep(50);
      };
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
        // a locator whose element exists but is not shown at this width (the phone's Settings button on a desktop) is passed over at once
        for (const at of target.at) {
          const got = locateAll(at);
          if (got && got !== "hidden") {
            found = got;
            break;
          }
          if (got !== "hidden" && waited < 2500) {
            break;
          }
        }
        if (!found) await sleep(100);
      }
      if (!alive()) return;
      if (!found) {
        tell("I could not find that on this screen.");
        show({ phase: "found" });
        if (run.onArrived) {
          await quiet(8000);
          if (alive()) run.onArrived();
        }
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
      const onPage = (run.via ?? []).map((at, i) => ({ el: locate(at), share: run.pace?.at[i] ?? 0 })).filter((x): x is { el: Element; share: number } => x.el !== null);
      const stops = onPage.map((x) => visible(x.el)!).map((r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2 }));
      const first = stops[0] ?? centre;
      const around = pointerRoute(origin, first, obstacles(found), [], vw, vh);
      const pace = run.pace && onPage.length > 1 ? run.pace : null;
      const route = pace ? around ?? [origin, first] : stops.length > 0 ? [...(around ?? [origin, first]), ...stops.slice(1), centre] : around;
      show({ start: origin, route, anchor: centre, rect: last, note, walk: pace ? { els: onPage.map((x) => x.el), at: onPage.map((x) => x.share) } : null });

      if (pace) {
        // a walk: to its start quickly, then along it, one instruction per stretch
        for (const line of words.travel) tell(line);
        let said = -1;
        const upTo = (tf: number) => {
          const now = tf <= 0 ? 0 : stepAt(pace, Math.min(tf, 0.9999));
          for (; said < now; said++) tell(pace.steps[said + 1]!.say);
        };
        if (held || still) {
          const tf = held ? Math.max(0, Math.min(1, run.t!)) : 1;
          upTo(tf);
          if (tf >= 1) tell(arrive);
          show({ phase: tf >= 1 ? "arrived" : "walk", t: tf });
        } else {
          const animate = (ms: number, each: (t: number) => void) =>
            new Promise<void>((resolve) => {
              const t0 = performance.now();
              const frame = (now: number) => {
                if (!alive()) return resolve();
                const t = Math.min(1, (now - t0) / ms);
                each(t);
                if (t < 1) window.requestAnimationFrame(frame);
                else resolve();
              };
              window.requestAnimationFrame(frame);
            });
          show({ phase: "travel", t: 0 });
          await animate(Math.round(Math.max(500, Math.min(1100, 300 + routeLength(route ?? [origin, first]) * 1.1))), (t) => show({ t }));
          if (!alive()) return;
          show({ phase: "walk", t: 0 });
          if (paced() && voiceRef.current.getSettings().speak && (voiceRef.current.canSpeak || talking())) {
            // Spoken, on Inspection: the dot and the words are one unit. It stands where a stretch starts while that stretch's
            // line is said to its end, waits a moment, walks the stretch, and only then is the next line said.
            await quiet(15_000);
            for (let i = 0; i < pace.steps.length && alive(); i++) {
              const st = pace.steps[i]!;
              tell(st.say);
              said = i;
              await quiet(25_000);
              await sleep(STOP_PAUSE_MS);
              if (!alive()) return;
              await animate(Math.round(Math.max(600, Math.min(3000, (st.upto - st.from) * 9000))), (k) => show({ t: st.t0 + (st.t1 - st.t0) * k }));
            }
            if (!alive()) return;
            await sleep(STOP_PAUSE_MS);
          } else {
            // not spoken: the timed walk, one instruction per stretch as the dot starts on it
            await animate(pace.ms, (t) => (upTo(t), show({ t })));
          }
          if (!alive()) return;
          tell(arrive);
          show({ phase: "arrived", t: 1 });
        }
      } else if (held || still) {
        const at = held ? Math.max(0, Math.min(1, run.t!)) : 1;
        for (const line of [...words.travel, ...(run.extra ?? [])]) tell(line);
        if (at >= 1) tell(arrive);
        show({ phase: at >= 1 ? "arrived" : "travel", t: at });
      } else {
        for (const line of [...words.travel, ...(run.extra ?? [])]) tell(line);
        show({ phase: "travel", t: 0 });
        const way = route ? Math.round(Math.max(900, Math.min(3200, 500 + routeLength(route) * 1.3))) : travelMs(origin, centre);
        // on Inspection the way takes as long as its sentences, so the dot arrives as the voice gets to the last one
        const ms = paced() && talking() ? Math.max(way, Math.min(12_000, lines.reduce((n, line) => n + sayMs(line), 0))) : way;
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
        await quiet(6000);
        if (!alive()) return;
        tell(arrive);
        show({ phase: "arrived", t: 1 });
      }

      // 5. did the eyes get there? They have when they rest on the target: it is the one marked thing within the measured
      //    miss of the estimate ("That's it."), or one of a few that close (region-level: "or right next to it").
      if (eyes && !held) {
        const dwell = new Dwell();
        const began = performance.now();
        while (alive() && performance.now() - began < 20_000) {
          await sleep(80);
          const e = gaze.estimate();
          const r = el.current ? visible(el.current) : null;
          const hit = e && r && el.current ? eyesLanded(snap(restPoint(e), e.radius, things(el.current)), TARGET) : null;
          const what = dwell.update(e && r ? hit !== null && e.rest.ms >= 200 : null, performance.now());
          if (what === "arrived") {
            if (!alive()) return;
            setOk(true);
            tell(hit === "near" ? "That's it, or right next to it." : "That's it.");
            break;
          }
          if (what === "repeat" && e && r) tell(fromGaze({ x: e.x * window.innerWidth, y: e.y * window.innerHeight }, { x: r.left + r.width / 2, y: r.top + r.height / 2 }, window.innerWidth, window.innerHeight));
        }
      }
      if (!held) {
        await quiet(30_000); // the caption stays until its last line has been said
        if (alive()) run.onArrived?.();
        await sleep(run.action ? 14_000 : 9000);
        if (alive()) cancel();
      }
    },
    [cancel, router, say, talking],
  );

  /**
   * A thing the box was asked about, after its answer: resolved against the answer's own picture, then a live camera
   * track, then the room's pin (lib/guide.ts `resolveThing`). Nothing found: "I haven't seen that" for the person's
   * own thing, and nothing at all for a name or a place (the answer stands as it is).
   */
  const followThing = useCallback(
    async (thing: string, answer: string, frame: string | null, citations: number, t: number | null) => {
      const where = await whereSeen(thing);
      const live = frame ? null : await findLive(thing);
      const found = resolveThing({ thing, citationFrame: frame, where, live: live !== null && live !== "off" });
      if (found.source === "none") {
        if (ownThing(thing) && citations === 0) offer(found.say, []);
        return;
      }
      if (frame && found.spotlight) frameSpots.set(frame, found.spotlight);
      if (found.source === "live-track" && live && live !== "off") {
        setSpotting({ camera: live.camera, labels: labelsFor(thing), name: thingName(thing) });
        await sleep(350);
        const name = thingName(thing);
        return void start({
          target: { id: "spotting", route: null, label: `Your ${name}, on camera`, at: [{ id: "spotting:box" }, { id: "spotting" }], steps: [{ say: "Look at the camera picture, bottom right." }, { say: found.say }] },
          t,
          precision: found.precision ?? undefined,
          spot: { label: spotWords(found.precision ?? precisionOf("picture"), /s$/.test(name)) },
        });
      }
      await sleep(frame ? 250 : 0); // the picture draws its spotlight first, so the pointer can land on it
      void start(evidenceRun(thing, answer, found, t, live === "off"));
    },
    [start],
  );

  /** A question or a guide id from a link or a "Did you mean" button, on whatever screen this is. */
  const ask = useCallback(
    async (query: string, t: number | null = null) => {
      const here = pathRef.current;
      const bin = await binRun(query, t).catch(() => null);
      if (bin) {
        // where a camera runs, its picture with the right bin's zone outlined and the others dimmed
        if (bin.zone) void findLive("").then((live) => (live === "off" ? undefined : setSpotting({ camera: cameraOfZone(bin.zone!) ?? null, zone: bin.zone, name: bin.zone! })));
        return void start(bin);
      }
      if (samePath(here, "/spot/")) {
        try {
          const { run, found } = await spotRun(query, t, INSPECTION_SPEAKS && Boolean(voiceRef.current.canSpeak && voiceRef.current.getSettings().speak));
          if (run) return void start(run);
          if (found && found.alternatives.length > 0) return offer(found.steps[0]?.say ?? "Did you mean one of these?", found.alternatives.map((a) => ({ label: a.label, ask: a.target })));
        } catch {
          /* the box is not answering: fall through to what the web layer knows */
        }
      }
      const asked = classify(query, here);
      // a bare word that is no part of the app ("keys") is a thing to look for, like the question would be
      const thing = asked?.kind === "thing" ? asked.thing : asked === null && resolveApp(query, here).target === null && /^[a-z ]{2,30}$/i.test(query.trim()) ? `my ${query.trim().toLowerCase()}` : null;
      if (thing !== null && samePath(here, "/")) {
        // from a link: ask the box as the person would, then follow its answer
        const q = asked ? query : `Where are my ${thingName(thing)}?`;
        try {
          const answer = await api.ask(q);
          const id = `g${Date.now()}`;
          handled.current = `${id}-a`;
          setChat((turns) => [...turns, { id, role: "you", text: q }, { id: `${id}-a`, role: "atlas", answer }]);
          await sleep(150);
          return void followThing(thing, answer.answer, answer.citations.find((c) => c.frame_ref)?.frame_ref ?? null, answer.citations.length, t);
        } catch {
          return;
        }
      }
      const app = resolveApp(query, here);
      if (app.target) return void start(app.target.id.startsWith("pin:") ? { target: app.target, t, precision: precisionOf("zone-pin"), spot: { label: spotWords(precisionOf("zone-pin")) } } : { target: app.target, t }); // a pin from a link: its calibration is not known here, so about
      if (app.alternatives.length > 0) offer("I am not sure what to point at. Did you mean one of these?", app.alternatives.map((a) => ({ label: a.label, run: { target: a } })));
      else offer("I do not know where that is in the app. Try asking the question on the Ask screen.", []);
    },
    [setChat, start, followThing],
  );

  // requests from the screens
  useEffect(
    () =>
      onGuide((m: GuideMessage) => {
        if (m.kind === "run") void start(m.run);
        else if (m.kind === "calibrate") (cancel(), setCalibrating(true));
        else if (m.kind === "cancel") cancel();
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
        const all = e ? things(null) : [];
        const s = e ? snap(restPoint(e), e.radius, all) : { kind: "none" as const };
        const ids = s.kind === "one" ? [s.id] : s.kind === "several" ? s.ids : [];
        const st = gaze.get();
        const why = e ? "" : st.status !== "on" ? `camera ${st.status}` : st.face !== "ok" ? (st.face === "many" ? "more than one face" : "no face") : !st.calibration?.validation ? "not calibrated and checked" : st.calibration.quality === "poor" ? "calibration too poor to use" : "no estimate";
        setDebug({ on: true, e, snapped: all.filter((t) => ids.includes(t.id)), several: s.kind === "several", why });
      }, 100);
    }
    // the attention check: what the eyes rest on (a fixation that snaps to exactly one marked thing) is noted, in memory only
    const watch = window.setInterval(() => {
      attention.setMode(gaze.simulated() ? "sim" : gaze.usable() ? "gaze" : "off");
      const e = gaze.estimate();
      if (!e || e.rest.ms < FIXATION_MS || !samePath(window.location.pathname, "/spot/")) return;
      const all = things(null);
      const s = snap(restPoint(e), e.radius, all);
      const hit = s.kind === "one" ? all.find((t) => t.id === s.id) : undefined;
      if (hit) attention.note(hit.id, 100, { left: hit.left + window.scrollX, top: hit.top + window.scrollY, right: hit.right + window.scrollX, bottom: hit.bottom + window.scrollY });
    }, 100);
    // ?gazeSim=id,id,...: a scripted stand-in for the camera. Each id is "looked at" for 0.9 s, in order, through the same rules.
    const script = simScript(q.get("gazeSim"));
    let simAlive = script.length > 0;
    if (simAlive) {
      setSim(true);
      gaze.simulate(null);
      void (async () => {
        await sleep(1800);
        for (const id of script) {
          let found: Element | null = null;
          for (let i = 0; i < 40 && simAlive && !(found = locate({ id })); i++) await sleep(100);
          if (!found || !simAlive) continue;
          (found.getBoundingClientRect().width > 0 ? found : found.firstElementChild ?? found).scrollIntoView({ block: "center" });
          await sleep(250);
          for (let k = 0; k <= 18 && simAlive; k++) {
            const r = visible(found);
            if (r) {
              const p = { x: (r.left + r.width / 2) / window.innerWidth, y: (r.top + r.height / 2) / window.innerHeight };
              gaze.simulate({ ...p, raw: p, col: Math.floor(p.x * 6), row: Math.floor(p.y * 4), cols: 6, rows: 4, radius: 12, rest: { ms: k * 50, ...p }, at: performance.now() });
            }
            await sleep(50);
          }
          gaze.simulate(null);
          await sleep(150);
        }
        document.documentElement.setAttribute("data-gaze-sim", "done");
      })();
    }
    return () => {
      window.clearTimeout(timer);
      window.clearInterval(tick);
      window.clearInterval(watch);
      simAlive = false;
    };
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
    if (!asked || asked.kind === "app") return;
    const a = lastTurn.answer;
    const timer = window.setTimeout(() => void followThing(asked.thing, a.answer, a.citations.find((c) => c.frame_ref)?.frame_ref ?? null, a.citations.length, null), 250);
    return () => window.clearTimeout(timer);
  }, [chat, followThing]);

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

  // ?gazeDebug=1: the raw estimate (small ring), the filtered one (dot), the measured-miss radius, the grid cell, and what it snapped to
  const dw = typeof window === "undefined" ? 1 : window.innerWidth;
  const dh = typeof window === "undefined" ? 1 : window.innerHeight;
  const de = debug.e;
  const debugBox = debug.on ? (
    <>
      {de ? (
        <>
          <div className={g.gazeCell} style={{ left: `${(de.col / de.cols) * 100}%`, top: `${(de.row / de.rows) * 100}%`, width: `${100 / de.cols}%`, height: `${100 / de.rows}%` }} />
          <div className={g.gazeRadius} style={{ left: de.x * dw - de.radius, top: de.y * dh - de.radius, width: 2 * de.radius, height: 2 * de.radius }} />
          <div className={g.gazeRaw} style={{ left: de.raw.x * dw - 6, top: de.raw.y * dh - 6 }} />
          <div className={g.gazeFiltered} style={{ left: de.x * dw - 5, top: de.y * dh - 5 }} />
          {debug.snapped.map((t) => (
            <div key={t.id} className={`${g.gazeSnap} ${debug.several ? g.gazeSnapSeveral : ""}`} style={{ left: t.left - 3, top: t.top - 3, width: t.right - t.left + 6, height: t.bottom - t.top + 6 }} />
          ))}
        </>
      ) : null}
      <div className={g.gazeReadout} data-gaze-debug>
        {de
          ? `gaze: ${de.cols}x${de.rows} grid, cell ${de.col + 1},${de.row + 1} · miss radius ${Math.round(de.radius)} px · resting ${Math.round(de.rest.ms)} ms · ${debug.snapped.length === 0 ? "on nothing marked" : debug.several ? `one of ${debug.snapped.length}: ${debug.snapped.map((t) => t.id).join(", ")}` : `on ${debug.snapped[0]!.id}`}`
          : `gaze: not in use (${debug.why})`}
      </div>
    </>
  ) : null;
  if (calibrating) return <Calibrate onDone={() => setCalibrating(false)} />;
  const simBanner = sim ? (
    <div className={g.simBanner} data-gaze-sim-banner>
      SIMULATED GAZE (?gazeSim): a script stands in for the camera. No camera is in use.
    </div>
  ) : null;
  const panel = spotting ? <Spotting ask={spotting} onClose={() => setSpotting(null)} /> : null;
  if (!view && !offered) return debugBox || simBanner || panel ? <div className={g.layer}>{debugBox}{simBanner}{panel}</div> : null;

  const vw = typeof window === "undefined" ? 1024 : window.innerWidth;
  const vh = typeof window === "undefined" ? 768 : window.innerHeight;
  const rect = view && el.current ? visible(el.current) ?? view.rect : view?.rect ?? null;
  const centre = rect ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 } : null;
  const moving = view && centre && view.phase === "travel";
  // a paced walk: the route's points are measured again on every frame, so a scroll does not leave the dot behind
  const walk = view?.phase === "walk" && view.walk && view.run.pace ? view.walk : null;
  const walkPts = walk ? walk.els.map((e, i) => ({ r: visible(e), share: walk.at[i]! })).filter((x) => x.r).map((x) => ({ x: x.r!.left + x.r!.width / 2, y: x.r!.top + x.r!.height / 2, share: x.share })) : [];
  const walkAt = (u: number): Pt => alongStops(walkPts, walkPts.map((p) => p.share), u);
  const walked = walk && view?.run.pace ? walkFraction(view.run.pace, view.t) : 0;
  // on a planned route the end follows the target: the whole way is nudged by however far the target has moved since
  const onRoute = (u: number): Pt => {
    const p = along(view!.route!, u);
    const k = Math.max(0, Math.min(1, u));
    return { x: p.x + (centre!.x - view!.anchor!.x) * k, y: p.y + (centre!.y - view!.anchor!.y) * k };
  };
  const routed = Boolean(view?.route && view.anchor && centre);
  // on the approach to a walk the way ends at the walk's first point, not at the target
  const approach = Boolean(view?.walk && view.run.pace && view.phase === "travel" && view.route);
  const at = walk && walkPts.length > 1 ? walkAt(walked) : approach ? along(view!.route!, ease(view!.t)) : view && centre ? (routed ? onRoute(ease(view.t)) : pathPoint(view.start, centre, ease(view.t))) : null;
  const tail =
    walk && walkPts.length > 1 && view!.t > 0
      ? Array.from({ length: 14 }, (_, i) => walkAt(Math.max(0, walked - 0.05 + (0.05 * i) / 13)))
      : approach
        ? []
        : moving && view.t > 0 && view.t < 1
          ? routed
            ? Array.from({ length: 18 }, (_, i) => onRoute(Math.max(0, ease(view.t) - 0.22 + (0.22 * i) / 17)))
            : trail(view.start, centre!, view.t)
          : [];
  const landed = view?.phase === "arrived";
  const pointer = samePath(path, "/spot/"); // the red pointer and its trail are drawn on the Inspection screen only; everywhere else the ring and the words do the work
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
    // the thing lights up by itself (a bin in the room): keep the caption off it, at the foot of the screen
    if (view?.run.spot?.ring === false) {
      capX = Math.max(12, Math.min(vw - capW - 12, rect.right - capW - 16));
      capY = Math.max(12, Math.min(vh - 210, rect.bottom - 200));
    }
  }
  // a walk on a plan: the caption keeps off the plan, so the line and the dot stay in view (beside it, else under it, else at the foot)
  const plan = view?.run.pace && el.current ? el.current.closest("svg")?.getBoundingClientRect() ?? null : null;
  if (plan) {
    if (vw - plan.right >= capW + 24) {
      capX = plan.right + 12;
      capY = Math.max(12, Math.min(vh - 210, plan.top));
    } else if (plan.bottom + 180 < vh) {
      capX = Math.max(12, Math.min(vw - capW - 12, plan.left));
      capY = plan.bottom + 10;
    } else {
      capX = (vw - capW) / 2;
      capY = Math.max(12, vh - 200);
    }
  }
  const lines = view?.lines ?? [];

  return (
    <div className={g.layer} aria-live="polite" data-guide-layer={view?.phase ?? "offer"} data-guide-target={view?.run.target.id} data-guide-walked={walk ? walked.toFixed(3) : undefined}>
      {debugBox}
      {simBanner}
      {panel}
      {pointer && tail.length > 1 ? (
        <svg className={g.trail} aria-hidden="true">
          {tail.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={2 + (5 * i) / tail.length} opacity={(0.55 * (i + 1)) / tail.length} />
          ))}
        </svg>
      ) : null}
      {rect && landed && view?.run.spot?.ring !== false ? (
        <div
          data-precision={view?.run.precision?.precision ?? "exact"}
          className={`${g.ring} ${view?.run.precision?.precision === "about" ? g.ringAbout : ""} ${small ? g.ringRound : ""} ${ok ? g.ringOk : ""} ${view?.run.t != null ? g.ringStill : ""}`}
          style={{ transform: `translate(${rect.left - pad}px, ${rect.top - pad}px)`, width: rect.width + 2 * pad, height: rect.height + 2 * pad }}
        >
          {view?.run.spot?.label ? <span className={g.spotLabel}>{view.run.spot.label}</span> : null}
        </div>
      ) : null}
      {pointer && at && !landed ? <div className={g.dot} style={{ transform: `translate(${at.x}px, ${at.y}px)` }} /> : null}
      <div className={`${g.caption} ${ok ? g.captionOk : ""}`} style={{ transform: `translate(${capX}px, ${capY}px)` }} data-guide-caption role="status">
        <p className={g.label}>
          <span>{view ? (landed && view.run.spot?.ring === false && view.run.spot.label ? `${view.run.spot.label}: ${view.run.target.label}` : view.run.target.label) : "Where is it?"}</span>
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

const TARGET = "\u0000target";
/** The pause between the end of a stop's line and the pointer moving on. */
const STOP_PAUSE_MS = 250;

/** Where the eyes are resting when they rest, else the filtered estimate; in pixels. */
function restPoint(e: GazeEstimate): { x: number; y: number } {
  const p = e.rest.ms >= 150 ? e.rest : e;
  return { x: p.x * window.innerWidth, y: p.y * window.innerHeight };
}

/**
 * The marked things on screen that gaze may snap to: every `data-guide` element in view that holds no other one (a card
 * and the items in it would always tie). With a target, the target is one thing whatever it holds, and its own parts and
 * the cards around it are left out.
 */
function things(target: Element | null): Thing[] {
  const out: Thing[] = [];
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const box = (id: string, r: DOMRect) => out.push({ id, left: r.left, top: r.top, right: r.right, bottom: r.bottom });
  if (target) {
    const r = visible(target);
    if (r) box(TARGET, r);
  }
  for (const el of document.querySelectorAll("[data-guide]")) {
    if (target && (el === target || el.contains(target) || target.contains(el))) continue;
    if (el.querySelector("[data-guide]") || el.closest("[data-guide-caption]")) continue;
    const r = visible(el);
    if (!r || r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw || r.width < 6 || r.height < 6) continue;
    box(el.getAttribute("data-guide") ?? "", r);
  }
  return out;
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

/**
 * The pointer's run for an answer about a thing: to the spotlight on the picture under the answer (a tight outline when
 * the box has a box for it, a soft region from the words, or the whole picture), or to the answer when it has no picture.
 * The drawing and the sentence both follow `found.precision`.
 */
function evidenceRun(thing: string, answer: string, found: ThingFound, t: number | null, off = false): GuideRun {
  const name = thingName(thing);
  const framed = found.spotlight !== null;
  const newest = "main article[data-guide='answer']:last-of-type";
  const about = precisionOf("zone-pin");
  const target: GuideTarget = {
    id: "evidence",
    route: "/",
    label: `Where your ${name} ${/s$/.test(name) ? "are" : "is"}`,
    at: framed ? [...(found.spotlight?.rect ? [{ sel: `${newest} [data-guide='spot']` }] : []), { sel: `${newest} [data-guide='frame']` }, { sel: newest }] : [{ sel: newest }],
    steps: [{ say: framed ? "Under my answer: the picture it comes from." : "This is my answer. I have no picture to show for it." }, { say: "" }],
  };
  const spot = found.spotlight;
  return {
    target,
    t,
    arrive: framed ? found.say : oneSentence(answer, 170) || found.say,
    note: (el) =>
      !framed
        ? `No camera frame is attached to this answer.${off ? " The camera is not running on this box right now, so there is no live picture either." : ""}`
        : el.closest("[data-guide='frame']")?.querySelector("[role=img]") ?? el.querySelector("[role=img]")
          ? "The picture for that moment is not stored on this box, so this is where it would be."
          : spot?.kind === "box"
            ? "Exact: the box put a box round it in this picture."
            : spot?.kind === "about"
              ? `About: placed from the box's words beside the ${spot.anchor}. There is no box for it.`
              : "About: the box keeps the zone and a description, not the spot inside the picture, so the whole picture is ringed.",
    action: found.source !== "none" ? { label: "Show me in the room", run: { target: pinTarget(labelsFor(thing)[0] ?? name), precision: about, spot: { label: spotWords(about) } } } : undefined,
    precision: found.precision ?? precisionOf("words"),
    spot: { label: spot ? spot.label : undefined },
  };
}
