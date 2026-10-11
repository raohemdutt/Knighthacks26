"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import dr from "@/components/dayrun/dayrun.module.css";
import { Keyframe } from "@/components/Keyframe";
import css from "@/components/map/map.module.css";
import { TwinStage } from "@/components/map/TwinStage";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type FloorTrack, type PlayRoute, type Sighting } from "@/lib/api";
import { cameraName, dayTime, humanize, relative } from "@/lib/format";
import { hereSentence, pinTarget, roomPrecision, spotWords } from "@/lib/guide";
import { pointAt } from "@/lib/guideBus";
import { CAMERAS, SORT_ITEMS, WATCH_ITEMS, cameraOfZone, placePins, routePoints, type Point, type RoomPin } from "@/lib/room";
import { useRoomTwin, useTwinPins } from "@/lib/roomTwin";
import { useFetch } from "@/lib/useFetch";

/**
 * The Room: find a thing, see who the cameras see right now, and get pointed the right way.
 *
 * Presenter links: /map/?view=cameras  ·  /map/?item=wallet  ·  /map/?sort=can
 *
 * What is real here and what is not:
 *  - The room in 3D is the room twin (components/map/RoomTwin.tsx) when the box has a room plan (GET /api/room):
 *    the furnished room in metres, with the cameras' tracks projected onto its floor and a pin where each watch
 *    item was last seen. Without a plan the per-camera panels (Room3D) stand in.
 *  - A sighting (GET /memory/objects/{label}/last-seen) has a camera, a zone and a text position; the pin stands
 *    at the floor point the box projected for it, or inside its zone when that is all the box knows.
 *  - Who and what is in view (GET /api/room/live) is the same feed the twin walks its figures on.
 *  - Zone polygons come from config/zones.yaml, embedded at build time, for the camera panels.
 *  - The route to a bin is GET /api/play/route, drawn in the bin camera's view.
 */

// three.js lives in this chunk only, fetched when the room is first shown.
const Room3D = dynamic(() => import("@/components/map/Room3D"), { ssr: false, loading: () => <Loading what="the 3D view" /> });

const PANEL_W = 360;
const PANEL_H = 270; // 4:3, the shape of the Sort It grid (32 x 24 cells)
const HEAD = 26;
const GAP = 14;
const ZONE_CHAR = 8; // about one letter of a zone's name (13 px bold), to tell when a name is wider than its zone
const HOUR_MS = 3_600_000;

interface Seen {
  label: string;
  sighting: Sighting | null; // null: the box has never seen it
  error?: ApiError;
}

async function loadSightings(signal: AbortSignal): Promise<Seen[]> {
  return Promise.all(
    WATCH_ITEMS.map(async (label): Promise<Seen> => {
      try {
        return { label, sighting: await api.lastSeen(label, signal) };
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return { label, sighting: null };
        if (err instanceof ApiError) return { label, sighting: null, error: err };
        throw err;
      }
    }),
  );
}

/** How fresh a sighting is: within the hour, earlier today, or older. */
function freshness(s: Sighting | null, now: number): "now" | "today" | "old" {
  if (!s) return "old";
  const age = now - Date.parse(s.ts);
  if (age < HOUR_MS) return "now";
  return new Date(s.ts).toDateString() === new Date(now).toDateString() ? "today" : "old";
}

/** One sentence for where a thing is, from what the box recorded. */
function whereSentence(s: Sighting): string {
  const zone = s.zone_id ? `in the ${humanize(s.zone_id).toLowerCase()} zone` : null;
  if (s.position && zone) return `${humanize(s.position)}, ${zone}.`;
  if (s.position) return `${humanize(s.position)}.`;
  if (zone) return `Somewhere ${zone}.`;
  return "Seen, but the box did not record where.";
}

const ICON_PERSON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="8.5" r="3.5" />
    <path d="M5 20c.6-4 3.4-6 7-6s6.4 2 7 6" />
  </svg>
);
const ICON_THING = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M4 8l8-4 8 4v8l-8 4-8-4z" />
    <path d="M4 8l8 4 8-4M12 12v8" />
  </svg>
);
const ICON_POINT = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
    <circle cx="12" cy="12" r="4" />
  </svg>
);
const ICON_ASK = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M4 5.5h16v10.5H10l-4.5 3.5V16H4z" />
  </svg>
);
const ICON_CHEVRON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M6 9l6 6 6-6" />
  </svg>
);

/** What a track is doing, in a few words: "Someone at the desk, seated", "Bottle on the table". */
function trackWords(t: FloorTrack): { what: string; sub: string } {
  const on = t.on ? `${t.kind === "person" ? "at" : "on"} the ${t.on.replace(/-/g, " ")}` : null;
  if (t.kind === "person") {
    const who = t.member_id ? humanize(t.member_id) : "Someone";
    return { what: [who, on].filter(Boolean).join(" "), sub: [t.seated ? "seated" : "standing", cameraName(t.camera).toLowerCase()].join(" · ") };
  }
  const held = t.held_by ? "in someone's hand" : t.in_bin ? `in the ${t.in_bin.replace(/-/g, " ")}` : on;
  return { what: [humanize(t.label), held].filter(Boolean).join(" "), sub: cameraName(t.camera).toLowerCase() };
}

export default function RoomMapPage() {
  const [view, setView] = useState<"room" | "cameras">("room");
  const [item, setItem] = useState<string | null>(null);
  const [sort, setSort] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const seen = useFetch(loadSightings);
  const route = useFetch<PlayRoute | null>((s) => (sort ? api.playRoute(sort, s) : Promise.resolve(null)), sort ?? "");
  const twin = useRoomTwin(true);
  const twinPins = useTwinPins(twin.plan, twin.live);
  const hasPlan = twin.plan !== null && twin.plan !== undefined;

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("view") === "cameras" || q.get("view") === "2d") setView("cameras");
    const i = q.get("item");
    if (i) setItem(i);
    const s = q.get("sort");
    if (s) setSort(s);
  }, []);

  // "12 min ago" stays honest while the screen is open.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  const { pins, unplaced } = useMemo(() => placePins((seen.data ?? []).flatMap((s) => (s.sighting ? [s.sighting] : []))), [seen.data]);
  // Until something is picked, the thing seen most recently is the one in focus.
  const latest = useMemo(() => (seen.data ?? []).filter((s) => s.sighting).sort((a, b) => Date.parse(b.sighting!.ts) - Date.parse(a.sighting!.ts))[0]?.label ?? null, [seen.data]);
  const selected = item ?? latest;
  const selectedSeen = seen.data?.find((s) => s.label === selected);

  const routeCamera = route.data ? cameraOfZone(route.data.bin) : undefined;
  const drawn = useMemo(
    () => (route.data && routeCamera && route.data.path.length > 0 ? { camera: routeCamera, points: routePoints(route.data), bin: route.data.bin } : null),
    [route.data, routeCamera],
  );

  const tracks = twin.live?.tracks ?? [];
  const people = tracks.filter((t) => t.kind === "person").length;
  const things = tracks.filter((t) => t.kind === "object" && !t.held_by).length;
  const seenCount = (seen.data ?? []).filter((s) => s.sighting).length;
  const liveOn = tracks.length > 0;

  // Cameras still on the points the room file shipped with: a placeholder, not a calibration of the real camera.
  const placeholder = Object.entries(twin.plan?.calibration ?? {}).filter(([, c]) => c.source === "default").map(([camera]) => camera);
  // The look guide lands on the pin this screen already focuses; exact only for a projected spot from a calibrated camera.
  const point = (label: string) => {
    const pin = twinPins.find((p) => p.label === label);
    const s = seen.data?.find((x) => x.label === label)?.sighting;
    const precision = roomPrecision(Boolean(pin?.exact), s ? twin.plan?.calibration?.[s.camera_id]?.source : undefined);
    setItem(label);
    pointAt({ target: pinTarget(label), precision, spot: { label: spotWords(precision, /s$/.test(label)) }, arrive: hereSentence(precision, label, /s$/.test(label), s?.position ? humanize(s.position).toLowerCase() : null) });
  };

  return (
    <>
      <div className={css.head}>
        <h1>Room</h1>
        <p className={css.status}>
          <span>
            <span className={css.liveDot} data-on={liveOn} aria-hidden="true" />
            {liveOn ? (
              <>
                <strong>{people === 0 ? "Nobody" : people === 1 ? "One person" : `${people} people`}</strong> in view
                {things > 0 ? `, ${things} ${things === 1 ? "thing" : "things"} tracked` : ""}
              </>
            ) : (
              "Nobody in view"
            )}
          </span>
          <span>
            <strong>{CAMERAS.length}</strong> {CAMERAS.length === 1 ? "camera" : "cameras"}
          </span>
          {seen.data ? (
            <span>
              <strong>{seenCount}</strong> of {WATCH_ITEMS.length} things placed
            </span>
          ) : null}
          {twin.live?.ts ? <span>updated {relative(twin.live.ts, new Date(now))}</span> : null}
        </p>
      </div>

      <section className={ui.section} aria-labelledby="things">
        <h2 id="things" className={ui.sectionTitle}>
          Where are my things?
        </h2>
        {seen.error ? (
          <ErrorBox what="your things" error={seen.error} onRetry={seen.reload} />
        ) : !seen.data ? (
          <Loading what="your things" />
        ) : (
          <div className={css.finder} role="group" aria-label="Item">
            {seen.data.map((s) => {
              const on = selected === s.label;
              const fresh = freshness(s.sighting, now);
              return (
                <button key={s.label} type="button" data-guide={`find:${s.label}`} className={`${css.thing} ${on ? css.thingOn : ""}`} aria-pressed={on} onClick={() => setItem(s.label)}>
                  <span className={css.fresh} data-fresh={fresh} aria-hidden="true" />
                  <span className={css.thingName}>{humanize(s.label)}</span>
                  {s.sighting ? (
                    <>
                      <span className={css.thingWhere}>{s.sighting.zone_id ? `${humanize(s.sighting.zone_id)} zone` : cameraName(s.sighting.camera_id)}</span>
                      <span className={css.thingWhen}>{relative(s.sighting.ts, new Date(now))}</span>
                    </>
                  ) : (
                    <span className={css.thingWhen}>not seen yet</span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </section>

      <div className={css.layout}>
        <div className={css.room}>
          {CAMERAS.length === 0 && !hasPlan ? (
            <Empty title="No room to draw">This box has no room plan and no camera zones yet. Both come from the config when the app is built.</Empty>
          ) : (
            <div className={`${ui.card} ${css.stageCard}`} data-guide="map:room">
              <div className={css.viewRow}>
                <div className={css.seg} role="group" aria-label="View">
                  <button type="button" className={`${css.segBtn} ${view === "room" ? css.segOn : ""}`} aria-pressed={view === "room"} onClick={() => setView("room")}>
                    The room
                  </button>
                  <button type="button" className={`${css.segBtn} ${view === "cameras" ? css.segOn : ""}`} aria-pressed={view === "cameras"} onClick={() => setView("cameras")} disabled={CAMERAS.length === 0}>
                    Camera views
                  </button>
                </div>
                <p className={`${ui.muted} ${ui.small}`}>{view === "room" ? "Drag to look around. Tap a pin to focus it." : "Each panel is one camera, with its zones where they sit in the picture."}</p>
              </div>
              {view === "cameras" && CAMERAS.length > 0 ? (
                <Room2D pins={pins} selected={selected} route={drawn} onPick={setItem} />
              ) : hasPlan ? (
                <TwinStage twin={twin} pins={twinPins} previewItem={sort} controls="tucked" picked={selected} onPick={setItem} />
              ) : twin.plan === undefined ? (
                <Loading what="the room" />
              ) : (
                <Room3D cameras={CAMERAS} pins={pins} selected={selected} route={drawn} />
              )}
              {view === "cameras" && pins.length > 0 ? (
                <ol className={dr.pinList} aria-label="What each numbered pin is">
                  {pins.map((p) => (
                    <li key={p.label}>
                      <button type="button" className={`${dr.pinItem} ${selected === p.label ? dr.pinItemOn : ""}`} aria-pressed={selected === p.label} onClick={() => setItem(p.label)}>
                        <span className={dr.pinNum} aria-hidden="true">
                          {p.n}
                        </span>
                        <span>
                          <span className={dr.srOnly}>Pin {p.n}: </span>
                          {humanize(p.label)}
                          <span className={`${ui.muted} ${ui.small}`}> · {humanize(p.zone).toLowerCase()} zone</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              ) : null}
              {view !== "cameras" && hasPlan && placeholder.length > 0 ? (
                <p className={`${ui.muted} ${ui.small}`} data-guide="map:calibration">
                  Not calibrated for {placeholder.length === 1 ? "this camera" : "these cameras"} yet ({placeholder.map((c) => cameraName(c).toLowerCase()).join(", ")}): positions are approximate. To calibrate, mark
                  four spots on the floor and send them once: <code>PUT /api/room/calibration/{placeholder[0]}</code> with the four picture points and the same four in room metres.
                </p>
              ) : null}
              {view !== "cameras" && hasPlan ? (
                <p className={`${ui.muted} ${ui.small}`}>
                  Drawn from the floor plan, in metres. A person or thing stands where its camera puts it on the floor; a pin stands where its item was last seen, or in its zone when that is all the
                  box knows.
                </p>
              ) : null}
            </div>
          )}

          <details className={css.bin} open={sort !== null} data-guide="map:bins">
            <summary>
              <span>Which bin does it go in?</span>
              {ICON_CHEVRON}
            </summary>
            <div className={css.binBody}>
              <p className={`${ui.muted} ${ui.small}`}>Pick what you are holding and the way to its bin is drawn in the room.</p>
              <div className={ui.chips} role="group" aria-label="Item to sort">
                {SORT_ITEMS.map((i) => (
                  <button key={i} type="button" className={`${ui.chip} ${sort === i ? ui.chipOn : ""}`} aria-pressed={sort === i} onClick={() => setSort(sort === i ? null : i)}>
                    {humanize(i)}
                  </button>
                ))}
                {sort && !SORT_ITEMS.includes(sort) ? (
                  <button type="button" className={`${ui.chip} ${ui.chipOn}`} aria-pressed="true" onClick={() => setSort(null)}>
                    {humanize(sort)}
                  </button>
                ) : null}
              </div>
              {!sort ? null : route.error ? (
                route.error.notConfigured ? (
                  <Empty title="Sort It is not set up on this box">It needs a camera that can see the bins and the sorting rules.</Empty>
                ) : (
                  <ErrorBox what="the route" error={route.error} onRetry={route.reload} />
                )
              ) : !route.data ? (
                <Loading what="the route" />
              ) : (
                <p>
                  {humanize(route.data.item)} goes in <strong>{humanize(route.data.bin).toLowerCase()}</strong>.{" "}
                  {route.data.path.length === 0
                    ? "The box found no way there, so nothing is drawn."
                    : !routeCamera
                      ? `The box's route ends in a zone (${route.data.bin}) this map does not have, so it is not drawn.`
                      : `The route is drawn in the ${cameraName(routeCamera).toLowerCase()}'s view: ${route.data.path.length} cells of its ${route.data.cols} by ${route.data.rows} grid.`}
                </p>
              )}
            </div>
          </details>
        </div>

        <aside className={css.rail} aria-label="Details">
          {selectedSeen ? (
            <ThingCard key={selectedSeen.label} seen={selectedSeen} placed={pins.some((p) => p.label === selectedSeen.label) || twinPins.some((p) => p.label === selectedSeen.label)} now={now} onPoint={() => point(selectedSeen.label)} />
          ) : selected ? (
            <p className={ui.muted}>{humanize(selected)} is not on this box&apos;s watch list.</p>
          ) : seen.data ? (
            <Empty title="Nothing placed yet">The box has not seen any of the things on its list. Pins appear here once it does.</Empty>
          ) : null}
          {unplaced.length > 0 ? (
            <p className={`${ui.muted} ${ui.small}`}>
              Seen, but with no zone this map has, so not pinned: {unplaced.map((s) => `${humanize(s.label)} (${cameraName(s.camera_id)})`).join(", ")}.
            </p>
          ) : null}

          <div className={ui.card}>
            <h2 className={ui.sectionTitle}>Right now</h2>
            <div className={css.now}>
              {tracks.length === 0 ? (
                <p className={css.quiet}>{twin.live?.ts ? `Nobody in view. The cameras last saw movement ${relative(twin.live.ts, new Date(now))}.` : "Nobody in view."}</p>
              ) : (
                tracks
                  .filter((t) => t.kind === "person" || !t.held_by)
                  .map((t) => {
                    const w = trackWords(t);
                    return (
                      <div key={t.key} className={css.track}>
                        <span className={`${css.trackIcon} ${t.kind === "person" ? css.trackIconPerson : ""}`}>{t.kind === "person" ? ICON_PERSON : ICON_THING}</span>
                        <span className={css.trackText}>
                          <span>{w.what}</span>
                          <span className={css.trackSub}>{w.sub}</span>
                        </span>
                      </div>
                    );
                  })
              )}
            </div>
          </div>

          {CAMERAS.length > 0 ? (
            <div className={ui.card}>
              <h2 className={ui.sectionTitle}>Cameras and zones</h2>
              <div className={css.cameras}>
                {CAMERAS.map((cam) => (
                  <div key={cam.id} className={css.camera}>
                    <span className={css.cameraName}>{cameraName(cam.id)}</span>
                    <div className={css.zones}>
                      {cam.zones.map((z) => {
                        const on = selectedSeen?.sighting?.camera_id === cam.id && selectedSeen.sighting.zone_id === z.id;
                        return (
                          <span key={z.id} className={`${css.zone} ${on ? css.zoneOn : ""}`}>
                            {humanize(z.id)}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </aside>
      </div>
    </>
  );
}

/** The thing in focus: where it is in one sentence, when, the frame that proves it, and what to do about it. */
function ThingCard({ seen, placed, now, onPoint }: { seen: Seen; placed: boolean; now: number; onPoint: () => void }) {
  const s = seen.sighting;
  if (seen.error) return <ErrorBox what={`where the ${seen.label} are`} error={seen.error} />;
  if (!s)
    return (
      <div className={`${ui.card} ${css.detail}`}>
        <p className={css.detailTitle}>{humanize(seen.label)}</p>
        <p className={css.where}>Not seen yet.</p>
        <p className={css.whereNote}>No camera has caught {seen.label} so far. Leave {seen.label === "keys" ? "them" : "it"} in view of a camera once and Atlas will remember the spot.</p>
        <div className={css.actions}>
          <Link href={`/?q=${encodeURIComponent(`Where did I leave my ${seen.label}?`)}`} className={css.action}>
            {ICON_ASK} Ask Atlas
          </Link>
        </div>
      </div>
    );
  const fresh = freshness(s, now);
  return (
    <div className={`${ui.card} ${css.detail}`} data-thing={s.label} data-guide={`thing:${s.label}`}>
      <div className={css.detailHead}>
        <p className={css.detailTitle}>{humanize(s.label)}</p>
        <span className={`${ui.tag} ${fresh === "now" ? ui.tagAccent : ""}`}>{relative(s.ts, new Date(now))}</span>
      </div>
      <p className={css.where}>{whereSentence(s)}</p>
      <p className={css.whereNote}>
        {cameraName(s.camera_id)} · {dayTime(s.ts)}
        {placed ? " · pinned in the room" : " · no pin: the zone is not on this map"}
      </p>
      <div className={css.actions}>
        <button type="button" className={`${css.action} ${css.actionPrimary}`} onClick={onPoint}>
          {ICON_POINT} Point me to it
        </button>
        <Link href={`/?q=${encodeURIComponent(`Where did I leave my ${s.label}?`)}`} className={css.action}>
          {ICON_ASK} Ask Atlas
        </Link>
      </div>
      <Keyframe frameRef={s.frame_ref} caption={`${cameraName(s.camera_id)} · ${dayTime(s.ts)}`} />
    </div>
  );
}

function Room2D({ pins, selected, route, onPick }: { pins: RoomPin[]; selected: string | null; route: { camera: string; points: Point[]; bin: string } | null; onPick: (label: string) => void }) {
  const height = CAMERAS.length * (HEAD + PANEL_H) + (CAMERAS.length - 1) * GAP;
  const selectedPin = pins.find((p) => p.label === selected);
  return (
    <svg className={dr.room} viewBox={`0 0 ${PANEL_W} ${height}`} role="img" aria-label={`Camera zones: ${CAMERAS.map((c) => `${cameraName(c.id)}: ${c.zones.map((z) => z.id).join(", ")}`).join(". ")}.`}>
      {CAMERAS.map((cam, i) => {
        const top = i * (HEAD + PANEL_H + GAP);
        const px = (p: Point): Point => [p[0] * PANEL_W, top + HEAD + p[1] * PANEL_H];
        const mine = pins.filter((p) => p.camera === cam.id);
        return (
          <g key={cam.id}>
            <text className={dr.cameraLabel} x={10} y={top + 18}>
              {cameraName(cam.id)}
            </text>
            <line className={ui.gridLine} x1={0} x2={PANEL_W} y1={top + HEAD} y2={top + HEAD} />
            {cam.zones.map((z) => {
              const on = (selectedPin?.camera === cam.id && selectedPin.zone === z.id) || (route?.camera === cam.id && route.bin === z.id);
              const xs = z.polygon.map((p) => p[0]);
              const ys = z.polygon.map((p) => p[1]);
              const cx = ((Math.min(...xs) + Math.max(...xs)) / 2) * PANEL_W;
              const labelY = top + HEAD + Math.min(...ys) * PANEL_H + 18;
              // A name wider than its zone is squeezed to fit, so it never runs into the zone next door.
              const room = (Math.max(...xs) - Math.min(...xs)) * PANEL_W - 8;
              const tight = humanize(z.id).length * ZONE_CHAR > room;
              return (
                <g key={z.id}>
                  <polygon className={on ? dr.zoneOn : dr.zone} points={z.polygon.map((p) => px(p).join(",")).join(" ")} />
                  <text className={dr.zoneLabel} x={cx} y={labelY} textAnchor="middle" textLength={tight ? room : undefined} lengthAdjust={tight ? "spacingAndGlyphs" : undefined}>
                    {humanize(z.id)}
                  </text>
                </g>
              );
            })}
            {route && route.camera === cam.id && route.points.length > 1 ? (
              <>
                <polyline className={dr.roomRoute} points={route.points.map((p) => px(p).join(",")).join(" ")} />
                <circle className={ui.startMark} cx={px(route.points[0] as Point)[0]} cy={px(route.points[0] as Point)[1]} r={6} />
                <text className={dr.pinLabel} x={px(route.points[0] as Point)[0]} y={px(route.points[0] as Point)[1] - 11} textAnchor="middle">
                  You
                </text>
              </>
            ) : null}
            {mine.map((p) => {
              const [x, y] = px(p.at);
              const on = p.label === selected;
              // Every pin carries its number; its name is written here only where the zone has room for
              // every name in it. The list under the map names them all.
              return (
                <g key={p.label} data-guide={`pin:${p.label}`} onClick={() => onPick(p.label)} style={{ cursor: "pointer" }}>
                  <title>{humanize(p.label)}</title>
                  <circle className={`${dr.pin} ${on ? dr.pinOn : ""}`} cx={x} cy={y} r={on ? 10.5 : 9.5} />
                  <text className={`${dr.pinNumber} ${on ? dr.pinNumberOn : ""}`} x={x} y={y + 4} textAnchor="middle">
                    {p.n}
                  </text>
                  {p.named ? (
                    <text className={dr.pinLabel} x={x} y={y + 25} textAnchor="middle">
                      {humanize(p.label)}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
