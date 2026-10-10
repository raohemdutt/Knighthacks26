"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import dr from "@/components/dayrun/dayrun.module.css";
import { Keyframe } from "@/components/Keyframe";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type PlayRoute, type Sighting } from "@/lib/api";
import { cameraName, dayTime, humanize, relative } from "@/lib/format";
import { CAMERAS, SORT_ITEMS, WATCH_ITEMS, cameraOfZone, placePins, routePoints, type Point, type RoomPin } from "@/lib/room";
import { useFetch } from "@/lib/useFetch";

/**
 * Room map: the camera zones, where the watch items were last seen, and the
 * Sort It route to the right bin.
 *
 * Presenter links: /map/?view=3d  ·  /map/?item=wallet  ·  /map/?sort=can
 *
 * What is real here and what is not:
 *  - Zone polygons come from config/zones.yaml, embedded at build time
 *    (lib/zones.generated.ts) because no endpoint returns them. They are in each
 *    camera's own image, so each camera is its own panel; the box has no floor plan.
 *  - A sighting (GET /memory/objects/{label}/last-seen) has a camera, a zone and
 *    a text position, never coordinates. Pins sit inside their zone, on a grid when
 *    several share it; each has a number, and the list under the map names them.
 *  - The route is the existing GET /api/play/route, drawn in the bin camera's view.
 */

// three.js lives in this chunk only, fetched when the 3D view is opened.
const Room3D = dynamic(() => import("@/components/map/Room3D"), { ssr: false, loading: () => <Loading what="the 3D view" /> });

const PANEL_W = 360;
const PANEL_H = 270; // 4:3, the shape of the Sort It grid (32 x 24 cells)
const HEAD = 26;
const GAP = 14;
const ZONE_CHAR = 8; // about one letter of a zone's name (13 px bold), to tell when a name is wider than its zone

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

export default function RoomMapPage() {
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [item, setItem] = useState<string | null>(null);
  const [sort, setSort] = useState<string | null>(null);
  const seen = useFetch(loadSightings);
  const route = useFetch<PlayRoute | null>((s) => (sort ? api.playRoute(sort, s) : Promise.resolve(null)), sort ?? "");

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("view") === "3d") setView("3d");
    const i = q.get("item");
    if (i) setItem(i);
    const s = q.get("sort");
    if (s) setSort(s);
  }, []);

  const { pins, unplaced } = useMemo(() => placePins((seen.data ?? []).flatMap((s) => (s.sighting ? [s.sighting] : []))), [seen.data]);
  // Until something is picked, the first item the box has seen is the one in focus.
  const selected = item ?? pins[0]?.label ?? null;
  const selectedSeen = seen.data?.find((s) => s.label === selected);

  const routeCamera = route.data ? cameraOfZone(route.data.bin) : undefined;
  const drawn = useMemo(
    () => (route.data && routeCamera && route.data.path.length > 0 ? { camera: routeCamera, points: routePoints(route.data), bin: route.data.bin } : null),
    [route.data, routeCamera],
  );

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Room map</h1>
        <p className={ui.muted}>The zones the cameras watch, where your things were last seen, and the way to the right bin.</p>
      </div>

      {CAMERAS.length === 0 ? (
        <Empty title="No zones to draw">This build has no camera zones. They come from config/zones.yaml when the app is built.</Empty>
      ) : (
        <>
          <div className={ui.segmented} role="group" aria-label="View">
            <button type="button" className={`${ui.seg} ${view === "2d" ? ui.segOn : ""}`} aria-pressed={view === "2d"} onClick={() => setView("2d")}>
              2D
            </button>
            <button type="button" className={`${ui.seg} ${view === "3d" ? ui.segOn : ""}`} aria-pressed={view === "3d"} onClick={() => setView("3d")}>
              3D
            </button>
          </div>

          <div className={ui.card}>
            {view === "2d" ? <Room2D pins={pins} selected={selected} route={drawn} onPick={setItem} /> : <Room3D cameras={CAMERAS} pins={pins} selected={selected} route={drawn} />}
            {pins.length > 0 ? (
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
            <p className={dr.legend}>
              <span>
                <span className={dr.pinKey} aria-hidden="true" />a numbered pin is an item, placed inside its zone, not at its exact spot
              </span>
              {drawn ? (
                <span>
                  <span className={`${dr.lineKey} ${dr.lineKeyOn}`} aria-hidden="true" />
                  the way to the {humanize(drawn.bin).toLowerCase()} bin
                </span>
              ) : null}
            </p>
            <p className={`${ui.muted} ${ui.small}`}>
              Each panel is one camera&apos;s own view, with its zones where they sit in the picture. The box has no floor plan, so the panels are side by side rather than joined into one room
              {view === "3d" ? ", and the heights are only there to make the zones readable" : ""}.
            </p>
          </div>
        </>
      )}

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Where are my things?</h2>
        {seen.error ? (
          <ErrorBox what="your things" error={seen.error} onRetry={seen.reload} />
        ) : !seen.data ? (
          <Loading what="your things" />
        ) : (
          <>
            <div className={ui.chips} role="group" aria-label="Item">
              {seen.data.map((s) => (
                <button key={s.label} type="button" className={`${ui.chip} ${selected === s.label ? ui.chipOn : ""}`} aria-pressed={selected === s.label} onClick={() => setItem(s.label)}>
                  {humanize(s.label)}
                  {s.sighting ? "" : " · not seen"}
                </button>
              ))}
            </div>
            {selectedSeen ? <SeenCard seen={selectedSeen} placed={pins.some((p) => p.label === selectedSeen.label)} /> : selected ? <p className={ui.muted}>{humanize(selected)} is not on this box&apos;s watch list.</p> : <p className={ui.muted}>The box has not seen any of these yet.</p>}
            {unplaced.length > 0 ? (
              <p className={`${ui.muted} ${ui.small}`}>
                Seen, but with no zone this map has, so not pinned: {unplaced.map((s) => `${humanize(s.label)} (${cameraName(s.camera_id)})`).join(", ")}.
              </p>
            ) : null}
          </>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Sort It: which bin?</h2>
        <p className={`${ui.muted} ${ui.small}`}>Pick what you are holding to draw the shortest way to its bin on the map above.</p>
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
      </section>
    </>
  );
}

function SeenCard({ seen, placed }: { seen: Seen; placed: boolean }) {
  const s = seen.sighting;
  if (seen.error) return <ErrorBox what={`where the ${seen.label} are`} error={seen.error} />;
  if (!s) return <Empty title={`${humanize(seen.label)}: not seen yet`}>No camera has caught {seen.label} so far, so there is no pin for it.</Empty>;
  return (
    <div className={ui.card}>
      <div className={`${ui.row} ${ui.wrap}`}>
        <h3>{humanize(s.label)}</h3>
        <span className={ui.tag}>{s.zone_id ? `${humanize(s.zone_id)} zone` : "no zone"}</span>
        <span className={ui.tag}>{cameraName(s.camera_id)}</span>
      </div>
      <p className={ui.reason}>{s.position ? humanize(s.position) : s.zone_id ? `Somewhere in the ${s.zone_id} zone` : "No position recorded"}</p>
      <p className={`${ui.muted} ${ui.small}`}>
        Last seen {dayTime(s.ts)} · {relative(s.ts)}
      </p>
      <p className={ui.nudge}>
        {placed
          ? `The box knows the zone${s.position ? " and the description above" : ""}, not coordinates. The pin sits inside the ${s.zone_id} zone, not at the exact spot.`
          : "The box has no zone on this map for it, so there is no pin."}
      </p>
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
                <g key={p.label} onClick={() => onPick(p.label)} style={{ cursor: "pointer" }}>
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
