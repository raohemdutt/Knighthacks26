"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RoomLive, RoomPlan, RoomRouteAnswer } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { FsrInfo } from "@/lib/fsr";
import type { TwinPin } from "@/lib/roomTwin";
import { useVoice } from "@/lib/voice";
import ui from "../ui.module.css";
import { BENCH, BENCH_FRAMES, MODES, STORE_KEY, type BenchRow, type Mode, dims, remembered, scaleOf } from "./Room3D";
import { ROUTE_COLORS, RoomScene, type SceneRoute, type TwinView } from "./roomScene";
import css from "./roomtwin.module.css";

/**
 * The room twin: the room from config/rooms/<name>.yaml built in 3D, with the people and things the cameras see
 * walking around in it, memory pins where things were last seen, and Sort It played on the floor.
 *
 * Everything here is driven by what the box reports (lib/roomTwin.ts): the plan, the tracks projected onto the
 * floor, the round with its routes. Nothing is placed by this component's own guess, except a memory pin whose
 * item has no projected position: that pin stands in its zone, and its card says so.
 *
 * Links for presenting and testing (the render ones are the same as the old 3D view's):
 *   ?top=1                               start in the top view
 *   ?fsr=off|quality|performance         render mode;  &compare=1 (&split=0.3) for the bilinear / FSR split
 *   ?gl=1 | ?gl=badshader                force the fallback to normal rendering
 *   ?fsrbench=1                          time 120 frames in each mode (with the shadow map redrawn every frame)
 */

export interface RoomTwinProps {
  plan: RoomPlan;
  live: RoomLive | null;
  pins?: TwinPin[];
  /** A route to show while no round is on: the Sort It screen's "what if I held this?". */
  preview?: RoomRouteAnswer | null;
  /** The render-mode bar, its caption and the hint line (the Room map shows them; the other screens do not). "tucked" folds the render controls away under a disclosure. */
  controls?: boolean | "tucked";
  /** "phone": a wide stage for the angled view. "tall": squarer, for the view from above. "fill": the parent's whole box (the operator screen). */
  shape?: "phone" | "tall" | "fill";
  /** A line for the result card, e.g. "3 of 4 right this sitting". */
  tally?: string | null;
  /** Say the "which bin?" hint and the round's item aloud. */
  speak?: boolean;
  /** The pin in focus, when the screen keeps that itself (the Room map's item chips); else the twin does. */
  picked?: string | null;
  onPick?: (label: string | null) => void;
  /** Where the camera starts. Sort It starts from the top: the floor, the route and who is where are what matter. */
  start?: TwinView;
}

const RESULT_FOR_MS = 15_000;

function seconds(n: number): string {
  return `${n.toFixed(1)} s`;
}

/** Elapsed seconds of the round, ticking while it is on. */
function useElapsed(started: string | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!started) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, [started]);
  return started ? Math.max(0, (now - Date.parse(started)) / 1000) : 0;
}

function binWords(bin: string, color: string | null | undefined): string {
  const name = bin === "trash" ? "the trash" : bin;
  return color ? `${name} (the ${color} bin)` : name;
}

/** The room from above without WebGL: the floor plan as a flat drawing, with who and what the cameras see and the way to the bin. */
function FlatPlan({ plan, live, pins, why, routes, target }: { plan: RoomPlan; live: RoomLive | null; pins: TwinPin[]; why: string; routes: [number, number][][]; target: string | null }) {
  const [w, h] = plan.size;
  const line = { fill: "none", stroke: "var(--muted)", strokeWidth: 0.03 };
  return (
    <div data-room={plan.name} data-view="flat">
      <p className={ui.stateTitle}>3D view is not available on this device</p>
      <p className={`${ui.muted} ${ui.small}`}>Here is the same room from above, drawn from its floor plan.</p>
      <svg viewBox={`-0.1 -0.1 ${w + 0.2} ${h + 0.2}`} role="img" aria-label={`${plan.title} from above`} style={{ display: "block", width: "100%", maxHeight: "78vh", margin: "10px 0" }}>
        <rect x={0} y={0} width={w} height={h} {...line} strokeWidth={0.06} />
        {plan.furniture.map((f) => (
          <g key={f.id} transform={`translate(${f.at[0]} ${f.at[1]}) rotate(${f.rot})`}>
            <rect x={-f.size[0] / 2} y={-f.size[1] / 2} width={f.size[0]} height={f.size[1]} rx={0.04} {...line} strokeDasharray={f.type === "rug" ? "0.08 0.06" : undefined} fill={f.type === "bin" ? (f.color ?? "var(--muted)") : "none"} stroke={f.id === target ? "var(--accent)" : line.stroke} strokeWidth={f.id === target ? 0.08 : line.strokeWidth} />
          </g>
        ))}
        {plan.furniture.map((f) =>
          f.type === "rug" ? null : (
            <text key={f.id} x={f.at[0]} y={f.at[1] + (f.type === "bin" ? 0.38 : 0.06)} textAnchor="middle" fontSize={0.17} fill="var(--text)">
              {f.label ?? (f.bin ? humanize(f.bin) : humanize(f.type))}
            </text>
          ),
        )}
        {routes.map((points, i) => (
          <polyline key={i} points={points.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={0.07} strokeLinecap="round" strokeLinejoin="round" strokeDasharray="0.02 0.14" />
        ))}
        {pins.filter((p) => p.exact).map((p) => (
          <circle key={p.label} cx={p.x} cy={p.y} r={0.07} fill="var(--accent)">
            <title>{humanize(p.label)}</title>
          </circle>
        ))}
        {(live?.tracks ?? []).map((t) => (
          <circle key={t.key} cx={t.x} cy={t.y} r={t.kind === "person" ? 0.16 : 0.08} fill={t.kind === "person" ? "var(--text)" : "var(--accent)"}>
            <title>{humanize(t.label)}</title>
          </circle>
        ))}
      </svg>
      <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>
        Large dots are people, small ones are things, and the dotted line is the way to the bin. Why there is no 3D: {why.replace(/\.+$/, "")}.
      </p>
    </div>
  );
}

export default function RoomTwin({ plan, live, pins = [], preview = null, controls = false, shape = "phone", tally = null, speak = false, picked: pickedProp, onPick, start = "iso" }: RoomTwinProps) {
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const scene = useRef<RoomScene | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [view, setView] = useState<TwinView>("iso");
  const [mode, setMode] = useState<Mode>("quality");
  const [compare, setCompare] = useState(false);
  const [info, setInfo] = useState<FsrInfo | null>(null);
  const [bench, setBench] = useState<BenchRow[] | "running" | null>(null);
  const [pickedOwn, setPickedOwn] = useState<string | null>(null);
  const picked = pickedProp !== undefined ? pickedProp : pickedOwn;
  const pickRef = useRef<(id: string) => void>(() => undefined);
  const setPicked = (label: string | null) => (onPick ? onPick(label) : setPickedOwn(label));
  const [resultShown, setResultShown] = useState(false);
  const choice = useRef<{ mode: Mode; compare: boolean }>({ mode: "quality", compare: false });
  const voice = useVoice();

  // The renderer and scene: made once.
  useEffect(() => {
    const el = host.current;
    const over = overlay.current;
    if (!el || !over) return;
    choice.current = remembered();
    setMode(choice.current.mode);
    setCompare(choice.current.compare);
    const q = new URLSearchParams(window.location.search);
    const gl = q.get("gl");
    const split = Number(q.get("split") ?? "0.5");
    const startView: TwinView = q.get("top") === "1" ? "top" : start;
    setView(startView);
    let made: RoomScene;
    try {
      made = new RoomScene(el, over, {
        scale: scaleOf(choice.current.mode),
        simulate: gl === "1" ? "webgl1" : gl === "badshader" ? "badshader" : null,
        compare: choice.current.compare,
        split: Number.isFinite(split) ? split : 0.5,
        calm: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
        view: startView,
        classes: { label: css.label ?? "", person: css.person ?? "", thing: css.thing ?? "", bin: css.bin ?? "", zone: css.zone ?? "", pin: css.pin ?? "", pinOn: css.pinOn ?? "", target: css.target ?? "", dot: css.dot ?? "" },
        onInfo: setInfo,
        onPick: (label) => pickRef.current(label),
      });
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
      return;
    }
    scene.current = made;
    setReady(true);
    let timer = 0;
    if (q.get("fsrbench") === "1") {
      timer = window.setTimeout(async () => {
        setBench("running");
        const rows = await made.bench(BENCH, BENCH_FRAMES, { scale: scaleOf(choice.current.mode), compare: choice.current.compare });
        setBench(rows);
        console.info("fsrbench", JSON.stringify(rows), "draw calls", made.drawCalls());
      }, 2500);
    }
    return () => {
      window.clearTimeout(timer);
      made.dispose();
      scene.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    if (ready) scene.current?.setPlan(plan);
  }, [plan, ready]);

  // Who is playing: the tracks the round's routes start from.
  const round = live?.round ?? null;
  const assist = round ? null : (live?.assist ?? null);
  const players = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of round?.routes ?? []) if (r.track) map.set(r.track, r.player);
    return map;
  }, [round]);

  // Pins on the stage. A pin with a projected spot is its own chip. Pins that only know their zone share one chip
  // per zone ("Wallet +2"): tapping it steps through them, so a desk full of things does not bury the room in names.
  const chips = useMemo(() => {
    const out: { id: string; text: string; x: number; y: number; z: number; labels: string[] }[] = [];
    const groups = new Map<string, TwinPin[]>();
    for (const p of pins) {
      if (p.exact) out.push({ id: p.label, text: humanize(p.label), x: p.x, y: p.y, z: p.z, labels: [p.label] });
      else groups.set(`zone:${p.sighting.camera_id}:${p.sighting.zone_id}`, [...(groups.get(`zone:${p.sighting.camera_id}:${p.sighting.zone_id}`) ?? []), p]);
    }
    for (const [id, list] of groups) {
      const first = list[0] as TwinPin;
      const at = list.findIndex((p) => p.label === picked);
      const text = at >= 0 ? `${humanize(picked ?? "")}${list.length > 1 ? ` · ${at + 1} of ${list.length}` : ""}` : list.length > 1 ? `${humanize(first.label)} +${list.length - 1}` : humanize(first.label);
      out.push({ id, text, x: first.x, y: first.y, z: first.z, labels: list.map((p) => p.label) });
    }
    return out;
  }, [pins, picked]);
  pickRef.current = (id) => {
    const chip = chips.find((c) => c.id === id);
    if (!chip) return;
    const at = chip.labels.indexOf(picked ?? "");
    setPicked(at === chip.labels.length - 1 ? null : (chip.labels[at + 1] ?? null)); // step through, then close
  };

  const tracks = live?.tracks;
  const pinned = useMemo(() => new Set(pins.filter((p) => p.exact).map((p) => p.label)), [pins]);
  const pinnedKey = [...pinned].join("|");
  useEffect(() => {
    if (ready) scene.current?.setTracks(tracks ?? [], players, pinned);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, players, ready, pinnedKey]);

  // Routes: the round's own, else the "which bin?" hint (the bin only), else the screen's preview.
  const binOf = (name: string | undefined) => plan.furniture.find((f) => f.type === "bin" && f.bin === name);
  const routeKey = JSON.stringify([round?.round_id, round?.routes, assist?.bin, assist?.label, preview?.item, preview?.points, plan.name, resultShown]);
  useEffect(() => {
    if (!ready) return;
    let routes: SceneRoute[] = [];
    let target: string | null = null;
    let text = "";
    if (round) {
      routes = round.routes.filter((r) => r.reachable).map((r) => ({ player: r.player, track: r.track, points: r.points.map((p) => [p.x, p.y] as [number, number]) }));
      target = binOf(round.bin)?.id ?? null;
      text = `${humanize(round.item)} goes here`;
    } else if (assist) {
      target = assist.bin_id;
      text = `${humanize(assist.label)} goes here`;
    } else if (preview && !resultShown) {
      if (preview.reachable) routes = [{ player: 1, track: preview.from.track, points: preview.points }];
      target = preview.bin_id;
      text = `${humanize(preview.item)} goes here`;
    }
    scene.current?.setRoutes(routes, target, text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeKey, ready]);

  const pinKey = JSON.stringify(chips);
  useEffect(() => {
    if (ready) scene.current?.setPins(chips, chips.find((c) => c.labels.includes(picked ?? ""))?.id ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pinKey, picked, ready]);

  // The result stays up for a while, then leaves the room clear again.
  const result = live?.result ?? null;
  useEffect(() => {
    if (!result) {
      setResultShown(false);
      return;
    }
    const left = RESULT_FOR_MS - (Date.now() - Date.parse(result.ts));
    setResultShown(left > 0);
    if (left <= 0) return;
    const id = window.setTimeout(() => setResultShown(false), left);
    return () => window.clearTimeout(id);
  }, [result?.round_id, result?.ts]); // eslint-disable-line react-hooks/exhaustive-deps

  // Say it: the hint outside a round, the item when a round starts, and how it went.
  const said = useRef("");
  const sayOnce = (key: string, text: string) => {
    if (!speak || said.current === key || !voice.getSettings().speak) return;
    said.current = key;
    voice.say(text, { tag: "sort-it", replace: true });
  };
  useEffect(() => {
    if (assist) sayOnce(`assist:${assist.track}:${assist.bin}`, assist.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assist?.track, assist?.bin]);
  useEffect(() => {
    if (round) sayOnce(`round:${round.round_id}`, `${humanize(round.item)} goes in ${binWords(round.bin, binOf(round.bin)?.color)}. Follow the line.`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [round?.round_id]);
  useEffect(() => {
    if (result && resultShown) sayOnce(`result:${result.round_id}`, result.correct ? `Right bin, in ${result.seconds.toFixed(1)} seconds.` : `That was the wrong bin. ${humanize(result.item)} goes in ${binWords(result.bin, binOf(result.bin)?.color)}.`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result?.round_id, resultShown]);

  const elapsed = useElapsed(round?.started ?? null);

  const choose = (next: { mode?: Mode; compare?: boolean }) => {
    choice.current = { ...choice.current, ...next };
    setMode(choice.current.mode);
    setCompare(choice.current.compare);
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(choice.current));
    } catch {
      // nowhere to remember it; the choice still holds for this visit
    }
    scene.current?.setRender(scaleOf(choice.current.mode), choice.current.compare);
  };

  const look = (next: TwinView) => {
    setView(next);
    scene.current?.setView(next);
  };

  if (failed) {
    return (
      <FlatPlan
        plan={plan}
        live={live}
        pins={pins}
        why={failed}
        routes={round ? round.routes.filter((r) => r.reachable).map((r) => r.points.map((p) => [p.x, p.y] as [number, number])) : preview?.reachable ? [preview.points] : []}
        target={round ? (binOf(round.bin)?.id ?? null) : (assist?.bin_id ?? preview?.bin_id ?? null)}
      />
    );
  }

  const fsrOn = info?.active ?? false;
  const splitOn = info?.compare ?? false;
  const unavailable = info !== null && !info.supported;
  const shown: Mode = unavailable ? "native" : mode;
  const pin = pins.find((p) => p.label === picked) ?? null;
  const renderBar = controls ? <RenderControls shown={shown} unavailable={unavailable} choose={choose} compare={compare} fsrOn={fsrOn} splitOn={splitOn} info={info} bench={bench} /> : null;
  const par = round?.routes.find((r) => r.reachable)?.par_s ?? null;
  const people = (tracks ?? []).filter((t) => t.kind === "person").length;
  const things = (tracks ?? []).filter((t) => t.kind === "object").length;
  const targetBin = round ? binOf(round.bin) : null;

  return (
    <>
      <div ref={host} className={`${css.stage} ${shape === "fill" ? css.fill : shape === "tall" ? css.tall : ""}`} data-room={plan.name} data-view={view}>
        <div ref={overlay} className={css.labels} />

        <div className={css.topBar}>
          <span className={css.chip} data-keepclear data-live={people + things > 0 ? "on" : "off"}>
            <span className={css.liveDot} aria-hidden="true" />
            {plan.title}
            <span className={css.chipSub}>{people + things === 0 ? "nobody in view" : `${people} ${people === 1 ? "person" : "people"} · ${things} ${things === 1 ? "thing" : "things"}`}</span>
          </span>
          <div className={css.views} data-keepclear role="group" aria-label="Camera">
            <button type="button" className={`${css.viewBtn} ${view === "iso" ? css.viewOn : ""}`} aria-pressed={view === "iso"} onClick={() => look("iso")}>
              Room
            </button>
            <button type="button" className={`${css.viewBtn} ${view === "top" ? css.viewOn : ""}`} aria-pressed={view === "top"} onClick={() => look("top")}>
              Top
            </button>
          </div>
        </div>

        {splitOn && info ? (
          <>
            {info.split > 0.2 ? (
              <span className={css.side} style={{ right: `calc(${(1 - info.split) * 100}% + 8px)` }}>
                Bilinear
              </span>
            ) : null}
            {info.split < 0.8 ? (
              <span className={css.side} style={{ left: `calc(${info.split * 100}% + 8px)` }}>
                FSR 1
              </span>
            ) : null}
          </>
        ) : null}

      </div>
      <div className={shape === "fill" ? css.bottom : css.under} aria-live="polite">
          {round ? (
            <div className={css.round} data-keepclear data-hud="round">
              <div className={css.roundHead}>
                <span className={css.roundTitle}>
                  <span className={css.swatch} style={{ background: `#${(ROUTE_COLORS[0] as number).toString(16).padStart(6, "0")}` }} aria-hidden="true" />
                  {humanize(round.item)} → {humanize(round.bin)}
                  {targetBin?.color ? <span className={css.roundSub}> the {targetBin.color} bin</span> : null}
                </span>
                <span className={css.clock}>
                  {seconds(elapsed)}
                  {par !== null ? <span className={css.roundSub}> / par {seconds(par)}</span> : null}
                </span>
              </div>
              {par !== null ? (
                <div className={css.meter} role="progressbar" aria-label="Time against par" aria-valuemin={0} aria-valuemax={Math.round(par * 2)} aria-valuenow={Math.round(Math.min(elapsed, par * 2))}>
                  <div className={`${css.meterFill} ${elapsed > par ? css.meterOver : ""}`} style={{ width: `${Math.min(100, (elapsed / (par * 2)) * 100)}%` }} />
                  <span className={css.meterPar} aria-hidden="true" />
                </div>
              ) : (
                <p className={css.roundSub}>No way to the bin right now: something is blocking every path.</p>
              )}
              {round.routes.length > 1 ? (
                <p className={css.legend}>
                  {round.routes.map((r) => (
                    <span key={r.player}>
                      <span className={css.swatch} style={{ background: `#${(ROUTE_COLORS[(r.player - 1) % ROUTE_COLORS.length] as number).toString(16).padStart(6, "0")}` }} aria-hidden="true" />
                      Player {r.player}
                      {r.reachable ? ` · ${r.distance_m.toFixed(1)} m` : " · no way"}
                    </span>
                  ))}
                </p>
              ) : par !== null ? (
                <p className={css.roundSub}>{(round.routes[0]?.distance_m ?? 0).toFixed(1)} m to walk. Follow the line on the floor.</p>
              ) : null}
            </div>
          ) : assist ? (
            <div className={css.assist} data-keepclear data-hud="assist">
              <p className={css.resultTitle}>Which bin?</p>
              <p className={css.resultLine}>{assist.text}</p>
            </div>
          ) : result && resultShown ? (
            <div className={`${css.result} ${result.correct ? css.resultGood : css.resultBad}`} data-keepclear data-hud="result">
              <p className={css.resultTitle}>{result.correct ? "Right bin" : "Wrong bin"}</p>
              <p className={css.resultLine}>
                {humanize(result.item)} goes in {result.bin === "trash" ? "the trash" : result.bin}. {seconds(result.seconds)}
                {result.par_s !== null ? ` against a par of ${seconds(result.par_s)}` : ""} · {Math.round(result.score)} points
              </p>
              <p className={css.roundSub}>{tally ?? `${result.right} of ${result.rounds} right this sitting`}</p>
            </div>
          ) : pin ? (
            <div className={css.pinCard} data-keepclear data-hud="pin" data-guide={`pin:${pin.label}`}>
              <div>
                <p className={css.resultTitle}>{pin.title}</p>
                <p className={css.resultLine}>{pin.detail}</p>
                {pin.exact ? null : <p className={css.roundSub}>The box knows the zone, not the exact spot: the pin stands inside it.</p>}
              </div>
              <button type="button" className={css.close} aria-label="Close" onClick={() => setPicked(null)}>
                ×
              </button>
            </div>
          ) : preview && !resultShown ? (
            <div className={css.assist} data-keepclear data-hud="preview">
              <p className={css.resultLine}>
                <strong>{humanize(preview.item)}</strong> goes in {preview.bin === "trash" ? "the trash" : preview.bin}
                {preview.reachable ? `: ${preview.distance_m.toFixed(1)} m, about ${seconds(preview.par_s)}${preview.from.track ? ", from where the camera sees you" : ""}.` : ". There is no clear way to it right now."}
              </p>
            </div>
          ) : null}
      </div>

      {controls ? (
        <>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <p className={`${ui.muted} ${ui.small}`}>Drag to rotate and tilt. Pinch or scroll to zoom. Tap a pin to see where that thing was last seen.</p>
            <button type="button" className={css.pill} onClick={() => scene.current?.home(true)}>
              Reset the view
            </button>
          </div>
          {controls === "tucked" ? (
            <details className={css.tuck}>
              <summary>
                <span>Rendering: AMD FidelityFX Super Resolution</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </summary>
              <div className={css.tuckBody}>{renderBar}</div>
            </details>
          ) : (
            renderBar
          )}
        </>
      ) : null}
    </>
  );
}

/** The render-mode bar, its caption and the benchmark rows, as one block so the Room map can fold it away. */
function RenderControls({ shown, unavailable, choose, compare, fsrOn, splitOn, info, bench }: { shown: Mode; unavailable: boolean; choose: (next: { mode?: Mode; compare?: boolean }) => void; compare: boolean; fsrOn: boolean; splitOn: boolean; info: FsrInfo | null; bench: BenchRow[] | "running" | null }) {
  return (
    <>
          <div className={css.bar}>
            <span className={css.barName} id="render-mode">
              Render
            </span>
            <div className={`${ui.segmented} ${css.modes}`} role="group" aria-labelledby="render-mode">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  className={`${ui.seg} ${css.mode} ${shown === m.id ? ui.segOn : ""}`}
                  aria-pressed={shown === m.id}
                  aria-label={m.id === "native" ? "Native" : `${m.name} (${m.sub})`}
                  disabled={unavailable && m.id !== "native"}
                  onClick={() => choose({ mode: m.id })}
                >
                  <span>{m.name}</span>
                  <span className={css.modeSub}>{m.sub}</span>
                </button>
              ))}
            </div>
            <button type="button" className={`${css.pill} ${compare && fsrOn ? css.pillOn : ""}`} aria-pressed={compare && fsrOn} disabled={!fsrOn} onClick={() => choose({ compare: !compare })}>
              Compare
            </button>
          </div>
          <p className={css.caption} data-fsr-caption>
            {info === null
              ? "Starting the 3D view."
              : unavailable
                ? `FSR is not available here, so the room is drawn normally at ${dims(info.output)}: ${info.reason}.`
                : fsrOn
                  ? `Upscaled with AMD FidelityFX Super Resolution 1 (EASU + RCAS): ${dims(info.internal)} → ${dims(info.output)}${splitOn ? ". Left half: the same low-resolution picture, stretched the plain (bilinear) way." : ""}`
                  : `Native rendering at ${dims(info.output)}. FSR is off.`}
          </p>
          {bench === "running" ? (
            <p className={css.caption} data-fsr-bench="running">
              Timing {BENCH_FRAMES} frames in each mode.
            </p>
          ) : bench ? (
            <ul className={css.bench} data-fsr-bench="done" aria-label={`Average time for one frame, over ${BENCH_FRAMES} frames`}>
              {bench.map((row) => (
                <li key={row.name}>
                  {row.name}: {row.ms.toFixed(2)} ms a frame ({dims(row.internal)} → {dims(row.output)})
                </li>
              ))}
            </ul>
          ) : null}
    </>
  );
}
