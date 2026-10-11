"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { AnalysisLayer } from "@/components/spot/AnalysisLayer";
import { EquipmentList } from "@/components/spot/EquipmentList";
import { EXPLAINS, type ExplainId, type Force } from "@/components/spot/Explainers";
import { FloorMap } from "@/components/spot/FloorMap";
import o from "@/components/spot/op.module.css";
import { SpotMic } from "@/components/spot/SpotMic";
import { Round } from "@/components/spot/Round";
import { SpotApprovals } from "@/components/spot/SpotApprovals";
import { ShiftRun, type RunLink } from "@/components/spot/ShiftRun";
import { AssetSheet, DoFirst, LooksFine, ORDERS_CHANGED, StatusLine, TrustLine, WatchOut } from "@/components/spot/Shift";
import type { Replay, Target } from "@/components/spot/types";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { onGuide } from "@/lib/guideBus";
import { spotWalk, useTraced } from "@/lib/guideSpot";
import { goSpotView, useSpotView } from "@/lib/spotView";
import { useFetch } from "@/lib/useFetch";

/**
 * Inspection (SPOT), for the person working the shift. Three layers:
 *
 *  1. "Your shift": one status line, three cards (do this first, watch out, looks fine) and the floor plan with one mark per
 *     piece of equipment. Everything is a sentence from GET /api/spot/shift; no statistics.
 *  2. "A round, replayed": one mission as a row of steps with SPOT on the plan, after the round or as it happened.
 *  3. "Analysis": everything the analyst screen showed, behind the Analysis tab and every "Why?".
 *
 * Only layers 1 and 2 load with the page (one request, plus the live picture). The analysis payloads are fetched when the
 * Analysis tab or a "Why?" is first opened.
 *
 * The Shift Run tab is a training game on the same plan: plan a round, run it through a toy model, compare with Atlas's.
 *
 * Inspection is a workspace of views (lib/spotView.ts; the sidebar lists them): My shift, Equipment, Rounds, Training
 * (Shift Run), Analysis, Handoff and Approvals (Inspection's own cards). The address names the view and the page renders only that view's sections. The look
 * guide's "tab:" hooks sit on the sidebar's My shift, Training and Analysis items (components/Shell.tsx).
 *
 * Deep links: ?tab=run&route=last&turn=40&show=result opens Shift Run with a plan already run, ?asset=HX-202 opens that equipment's panel, ?round=M017&step=6 puts that round, at that step, at the top of the page, ?why=battery plays that finding's explainer (&at=end or &at=mid draws it at that step), ?numbers=battery opens its rows, ?tab=analysis opens the analysis, ?floor=2 picks the floor, ?all=1 lists all equipment.
 */
type Tab = "shift" | "run" | "analysis";

export default function SpotPage() {
  const shift = useFetch((sig) => api.spotShift(sig));
  const orders = useFetch((sig) => api.spotWorkOrders(sig));
  // a button on this screen wrote something: read the list again (the live event does the same when the socket is up)
  const reloadOrders = orders.reload;
  useEffect(() => {
    window.addEventListener(ORDERS_CHANGED, reloadOrders);
    return () => window.removeEventListener(ORDERS_CHANGED, reloadOrders);
  }, [reloadOrders]);
  const [live, setLive] = useState<Replay | null>(null);
  const view = useSpotView();
  const tab: Tab = view === "training" ? "run" : view === "analysis" ? "analysis" : "shift"; // the three layers
  const [asset, setAsset] = useState<string | null>(null);
  const [target, setTarget] = useState<Target | null>(null);
  const [explain, setExplain] = useState<ExplainId | null>(null);
  const [force, setForce] = useState<Force>(null);
  const [evidence, setEvidence] = useState(false); // becomes true the first time the analysis is needed, and stays
  const walking = useSyncExternalStore(spotWalk.subscribe, spotWalk.get, spotWalk.get); // the look guide's walking route, and "you are here"
  const [link, setLink] = useState<{ round: string | null; step: number | null; floor: number | null; run: RunLink } | null>(null);
  const traced = useTraced(); // the floor traced from the plan picture: where a tap lands, and the faint overlay
  const [tapping, setTapping] = useState(false); // the next tap on the plan sets "You are here"
  const [tapMsg, setTapMsg] = useState<string | null>(null);
  const [showFloor, setShowFloor] = useState(false);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const up = (k: string) => q.get(k)?.toUpperCase() ?? null;
    if (up("asset")) setAsset(up("asset"));
    const at = q.get("at");
    if (at === "end" || at === "mid") setForce(at);
    const asked = q.get("why");
    if (asked) (EXPLAINS[asked] ? setExplain(EXPLAINS[asked]!) : setTarget({ kind: asked === "validation" ? "validation" : "finding", id: asked }), setEvidence(true));
    if (q.get("numbers")) (setTarget({ kind: "finding", id: q.get("numbers")! }), setEvidence(true));
    if (q.get("walkable")) setShowFloor(true);
    const here = q.get("here"); // ?here=WP04, ?here=xy:0.30,0.57 or ?here=spot (where SPOT is in the round on show): where the walk starts
    if (here?.toLowerCase() === "spot") fromSpot.current = true;
    else if (here) spotWalk.setHere(/^xy:/i.test(here) ? here.toLowerCase() : here.toUpperCase());
    setLink({ round: up("round"), step: q.get("step") ? Number(q.get("step")) : null, floor: q.get("floor") ? Number(q.get("floor")) : null,
      run: { route: q.get("route"), turn: q.get("turn") ? Number(q.get("turn")) : null, show: q.get("show"), hard: Boolean(q.get("hard")) },
    });
  }, []);

  // The live picture: one reader, at most four reads a second.
  const reading = useRef(false);
  const again = useRef(false);
  const readLive = useCallback(() => {
    if (reading.current) {
      again.current = true;
      return;
    }
    reading.current = true;
    api
      .spotReplay()
      .then(setLive)
      .catch(() => undefined)
      .finally(() => {
        window.setTimeout(() => {
          reading.current = false;
          if (again.current) {
            again.current = false;
            readLive();
          }
        }, 250);
      });
  }, []);
  useEffect(readLive, [readLive]);
  const fromSpot = useRef(false);
  const liveAt = live?.robot?.location_id ?? null;
  useEffect(() => {
    if (!fromSpot.current || !liveAt) return;
    fromSpot.current = false;
    spotWalk.setHere(liveAt);
  }, [liveAt]);
  useAtlasEvents(["spot."], (e) => (e.type === "spot.work_order.created" ? orders.reload() : readLive()));
  const playing = Boolean(live?.playing);
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(readLive, 1000);
    return () => window.clearInterval(t);
  }, [playing, readLive]);

  // "Why?" on a finding plays its explainer when it has one; "Show the numbers" there (and every other Why?) opens the rows
  const numbers = useCallback((t: Target | null) => {
    if (t) setEvidence(true);
    setTarget(t);
  }, []);
  const why = useCallback(
    (t: Target | null) => {
      const id = t && (t.kind === "finding" || t.kind === "validation") ? EXPLAINS[t.id] : undefined;
      if (!id) return numbers(t);
      setEvidence(true);
      setExplain(id);
    },
    [numbers],
  );
  const pick = (t: Tab) => {
    if (t === "analysis") setEvidence(true);
    goSpotView(t === "run" ? "training" : t === "analysis" ? "analysis" : "shift");
  };
  useEffect(() => {
    if (view === "analysis") setEvidence(true);
  }, [view]);

  // The look guide reveals a thing by pressing a "tab:" hook, which is now a sidebar menu item (hidden on a phone). Open
  // the view that holds what it is about to point at here, so it lands at any width: Analysis, Training, the round, or
  // the shift cards and the plan.
  useEffect(
    () =>
      onGuide((m) => {
        if (m.kind !== "run") return;
        const keys = (m.run.target.reveal ?? []).map((r) => r.key);
        if (keys.includes("tab:analysis")) (setEvidence(true), goSpotView("analysis"));
        else if (keys.includes("tab:shift-run")) goSpotView("training");
        else if (keys.includes("tab:shift")) goSpotView(/^round(:|$)/.test(m.run.target.id) ? "rounds" : "shift");
      }),
    [],
  );

  // Handoff: the note is written by the existing button, as soon as the view opens.
  const ready = Boolean(shift.data && link);
  useEffect(() => {
    if (!ready || view !== "handoff") return;
    const t = window.setTimeout(() => document.querySelector<HTMLElement>('[data-guide="handoff"]')?.click(), 60);
    return () => window.clearTimeout(t);
  }, [view, ready]);

  if (shift.error) return <ErrorBox what="the inspection data" error={shift.error} onRetry={shift.reload} />;
  if (!shift.data || link === null) return <Loading what="your shift" />;
  const d = shift.data;
  if (!d.configured || !d.map) {
    return (
      <div className={o.page}>
        <div className={ui.pageHead}>
          <h1>Inspection</h1>
        </div>
        <Empty title="No inspection data on this box yet">{d.note}</Empty>
      </div>
    );
  }
  const written = orders.data?.work_orders ?? [];
  const open = asset ? d.assets.find((a) => a.asset_id === asset) : undefined;
  const robotAt = playing ? live?.robot?.location_id ?? null : null;
  const spotAt = live?.robot?.location_id ?? null; // where SPOT is in the round on show, playing or not
  const tapped = /^xy:([\d.]+),([\d.]+)$/.exec(walking.here ?? "");
  const stand = walking.here && !tapped ? traced.floor?.waypoints[walking.here] : undefined;
  const hereAt = tapped
    ? { floor: Number(tapped[1]) > (d.map.plan?.panels["2"]?.x0 ?? 2) ? 2 : 1, x: Number(tapped[1]), y: Number(tapped[2]) }
    : stand?.stand
      ? { floor: stand.floor, ...stand.stand }
      : null;
  const onTap = (at: { floor: number; x: number; y: number; moved: number } | null) => {
    setTapping(false);
    if (!at) return setTapMsg("That spot is not on the floor I traced. Tap a corridor or a room.");
    setTapMsg(at.moved > 0 ? "Moved to the nearest floor you can stand on." : null);
    spotWalk.setHere(`xy:${at.x.toFixed(4)},${at.y.toFixed(4)}`);
  };

  return (
    <div className={o.page}>
      <div className={o.top}>
        <StatusLine shift={d} />
      </div>

      {view === "shift" ? (
        <>
          {d.trust ? <TrustLine trust={d.trust} onWhy={why} /> : null}

          <div className={o.layout}>
          <div className={o.first}>
            <div className={o.mapCol}>
              <FloorMap
                map={d.map}
                assets={d.assets}
                title="The plant floor"
                guide="map"
                walk={walking.walk}
                hereAt={hereAt}
                onTap={tapping ? onTap : null}
                showFloor={showFloor}
                route={d.map.route}
                robotAt={robotAt}
                selected={asset}
                onPick={setAsset}
                initialFloor={link.floor}
                note={
                  <>
                    <p className={o.note}>Tap a mark to see what SPOT found there. The picture is the kit's sample floor plan; it is practice data, not a real plant.</p>
                    <p className={o.note} data-guide="walk">
                      <label>
                        You are here:{" "}
                        <select className={ui.input} style={{ display: "inline-block", width: "auto", padding: "4px 8px" }} value={walking.here ?? ""} onChange={(e) => (setTapMsg(null), spotWalk.setHere(e.target.value || null))} aria-label="Where you are on the plan">
                          <option value="">where SPOT's last round started</option>
                          {walking.here?.startsWith("xy:") ? <option value={walking.here}>the spot you tapped</option> : null}
                          {d.map.places.map((p) => (
                            <option key={p.location_id} value={p.location_id}>
                              {p.name} (floor {p.floor})
                            </option>
                          ))}
                        </select>
                      </label>{" "}
                      {traced.floor ? (
                        <button type="button" className={o.link} aria-pressed={tapping} data-guide="walk:tap" onClick={() => (setTapMsg(null), setTapping((v) => !v))}>
                          {tapping ? "Now tap the plan (or cancel)" : "Tap the plan to set it"}
                        </button>
                      ) : null}{" "}
                      {spotAt ? (
                        <button type="button" className={o.link} data-guide="walk:from-spot" onClick={() => (setTapMsg(null), spotWalk.setHere(spotAt))}>
                          From SPOT ({d.map.places.find((p) => p.location_id === spotAt)?.name ?? spotAt})
                        </button>
                      ) : null}
                      {tapMsg ? <b> {tapMsg}</b> : null}
                    </p>
                    <p className={o.note} data-guide="walk:words">
                      {walking.walk?.found && walking.walk.routed && walking.walk.routed.floors.length > 0 ? (
                        <>
                          <b>{walking.walk.summary}</b> The way is about, not exact: {walking.walk.precision_reason ?? "traced from the picture"}, and the picture has no scale, so no distances.{" "}
                          <button type="button" className={o.link} onClick={spotWalk.clear}>
                            Hide the route
                          </button>
                        </>
                      ) : walking.walk?.found && walking.walk.path.length > 1 ? (
                        <>
                          <b>{walking.walk.summary}</b> {walking.walk.note}{" "}
                          <button type="button" className={o.link} onClick={spotWalk.clear}>
                            Hide the route
                          </button>
                        </>
                      ) : walking.walk ? (
                        walking.walk.summary
                      ) : (
                        "Ask “where is…” and the red dot walks you there on the plan."
                      )}
                    </p>
                    {walking.walk?.found && walking.walk.steps && walking.walk.steps.length > 1 ? (
                      <ol className={o.note} data-guide="walk:steps" style={{ margin: 0, paddingLeft: 22 }}>
                        {walking.walk.steps.map((st, i) => (
                          <li key={i}>{st.say}</li>
                        ))}
                      </ol>
                    ) : null}
                    {traced.floor ? (
                      <p className={o.note} data-guide="walk:caption">
                        {traced.floor.caption}{" "}
                        <label>
                          <input type="checkbox" checked={showFloor} onChange={(e) => setShowFloor(e.target.checked)} /> Show the traced floor
                        </label>
                      </p>
                    ) : null}
                  </>
                }
              />
            </div>
            <div className={o.sideCol}>
              <DoFirst shift={d} orders={written} onAsset={setAsset} onWhy={why} onAll={() => goSpotView("equipment")} />
              <div className={o.onlyWide}>
                <WatchOut items={d.watch} onWhy={why} />
              </div>
            </div>
          </div>

          <div className={o.second}>
            <div className={o.onlyNarrow}>
              <WatchOut items={d.watch} onWhy={why} />
            </div>
            <LooksFine shift={d} onAll={() => goSpotView("equipment")} onPick={setAsset} />
          </div>

          </div>

          <button type="button" className={o.more} onClick={() => pick("analysis")}>
            <span>
              <b>The analysis behind this</b>
              How sure each sentence is, the tables, the method and what the data cannot say.
            </span>
            <span aria-hidden="true">›</span>
          </button>
        </>
      ) : null}

      {view === "equipment" ? <EquipmentList shift={d} onPick={setAsset} /> : null}

      {view === "approvals" ? <SpotApprovals /> : null}

      {view === "rounds" ? (
        <div id="round" data-guide="round" className={link.round ? o.roundFirst : undefined}>
          <Round shift={d} live={live} onLive={setLive} onWhy={why} startRound={link.round} startStep={link.step} />
        </div>
      ) : null}

      {tab === "run" ? <ShiftRun shift={d} orders={written} link={link.run} /> : null}

      {evidence ? <AnalysisLayer shift={d} show={tab === "analysis"} target={target} setTarget={numbers} explain={explain} setExplain={setExplain} force={force} live={live} onLive={setLive} orders={written} /> : null}

      {/* The Atlas mic, docked at the foot of every view: the one place to ask on Inspection, so the guide's "ask" hook stays unique. */}
      <SpotMic chips={d.ask ?? []} onWhy={why} jobs={d.do_first} guideChips={[d.do_first[0] ? `Where is ${d.do_first[0].name}?` : "Where is the plan?", "Where do I look for the battery?", "Show me what to do first"]} />
      {open ? <AssetSheet asset={open} orders={written} onClose={() => setAsset(null)} onWhy={why} /> : null}
    </div>
  );
}
