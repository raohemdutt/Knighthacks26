"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, snapshotUrl, viewTicketReady, type ZoneSet } from "@/lib/api";
import { cameraName, humanize } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import z from "./zones.module.css";

/**
 * Zones: set where things count, from a phone or laptop, in the room you walked into (the cold-room step).
 *
 * Pick the camera, pick a zone, drag a box on the picture, Save. The box rewrites config/zones.yaml and the zone engine
 * uses the new boxes at once (PUT /api/zones/{camera}): no restart. "Use demo zones" applies big, forgiving boxes
 * (config/zones.demo.yaml) when there is no time to draw. Only zones that already exist can be moved, because rules and
 * the game refer to their names; Sort It reads the bin boxes at start, so restart the gateway after moving the bins.
 */

interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const MIN_SIDE = 0.03; // a smaller drag is a tap, not a box
const clamp = (v: number) => Math.min(1, Math.max(0, v));
const round = (v: number) => Math.round(v * 1000) / 1000;
const polygon = (r: Rect): [number, number][] => [
  [r.x0, r.y0],
  [r.x1, r.y0],
  [r.x1, r.y1],
  [r.x0, r.y1],
];

function normalize(a: [number, number], b: [number, number]): Rect {
  return { x0: round(Math.min(a[0], b[0])), y0: round(Math.min(a[1], b[1])), x1: round(Math.max(a[0], b[0])), y1: round(Math.max(a[1], b[1])) };
}

export default function ZonesPage() {
  const zones = useFetch((signal) => api.zones(signal));
  const status = useFetch((signal) => api.captureStatus(signal));
  const [camera, setCamera] = useState("");
  const [zone, setZone] = useState("");
  const [picture, setPicture] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, Rect>>({});
  const [drag, setDrag] = useState<{ from: [number, number]; to: [number, number] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [saved, setSaved] = useState("");
  const stage = useRef<HTMLDivElement>(null);

  const all: ZoneSet = zones.data ?? {};
  const cameras = Object.keys(all);
  const names = Object.keys(all[camera] ?? {});
  const flowing = (id: string) => (status.data?.cameras[id]?.status ?? "down") !== "down";

  // Start on the first camera that is delivering pictures, else the first one with zones
  useEffect(() => {
    if (!camera && cameras.length) setCamera(cameras.find(flowing) ?? cameras[0] ?? "");
  }, [cameras, camera, status.data]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!names.includes(zone)) setZone(names[0] ?? "");
  }, [names, zone]);

  const grab = useCallback(() => {
    // Frozen: a person walking by must not move the picture under the drag. It is taken once this screen knows
    // whether it holds the owner's view ticket, so one frozen picture is not the blurred one by accident of timing.
    if (camera) void viewTicketReady().then(() => setPicture(snapshotUrl(camera, Date.now())));
  }, [camera]);
  useEffect(() => {
    setDrafts({});
    setDrag(null);
    setSaved("");
    grab();
  }, [camera, grab]);

  const at = (e: React.PointerEvent): [number, number] => {
    const r = stage.current!.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };
  const down = (e: React.PointerEvent) => {
    if (!zone) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = at(e);
    setDrag({ from: p, to: p });
    setSaved("");
  };
  const move = (e: React.PointerEvent) => drag && setDrag({ ...drag, to: at(e) });
  const up = () => {
    if (!drag) return;
    const r = normalize(drag.from, drag.to);
    if (r.x1 - r.x0 >= MIN_SIDE && r.y1 - r.y0 >= MIN_SIDE) setDrafts((d) => ({ ...d, [zone]: r }));
    setDrag(null);
  };

  async function run(action: () => Promise<ZoneSet>, done: string) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setDrafts({});
      setSaved(done);
      zones.reload();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  }

  const save = () =>
    run(
      () => api.setZones(camera, Object.fromEntries(Object.entries(drafts).map(([name, r]) => [name, polygon(r)]))),
      "Saved. The box is using the new zones now.",
    );
  const shown: Record<string, Rect> = drag ? { ...drafts, [zone]: normalize(drag.from, drag.to) } : drafts;
  const pending = Object.keys(drafts).length;
  const saved_ = all[camera] ?? {};

  if (zones.loading && !zones.data) return <Loading what="zones" />;
  if (zones.error && !zones.data) return <ErrorBox what="zones" error={zones.error} onRetry={zones.reload} />;

  return (
    <div className={ui.section}>
      <div className={ui.pageHead}>
        <h1>Zones</h1>
        <p className={`${ui.muted} ${ui.small}`}>Mark where things count in this room.</p>
      </div>

      <div className={ui.chips} role="group" aria-label="Camera">
        {cameras.map((id) => (
          <button key={id} type="button" className={`${ui.chip} ${id === camera ? ui.chipOn : ""}`} aria-pressed={id === camera} onClick={() => setCamera(id)}>
            {cameraName(id)}
            {flowing(id) ? "" : " (no picture)"}
          </button>
        ))}
      </div>

      <div className={ui.chips} role="group" aria-label="Zone to draw">
        {names.map((name) => (
          <button key={name} type="button" className={`${ui.chip} ${name === zone ? ui.chipOn : ""}`} aria-pressed={name === zone} onClick={() => setZone(name)}>
            {humanize(name)}
            {drafts[name] ? " ✓" : ""}
          </button>
        ))}
      </div>

      <div
        ref={stage}
        className={z.stage}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={() => setDrag(null)}
        role="img"
        aria-label={`Latest picture from the ${cameraName(camera).toLowerCase()}. Drag on it to draw the ${humanize(zone).toLowerCase()} zone.`}
      >
        {picture && flowing(camera) ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={picture} alt="" />
        ) : (
          <div className={z.empty}>{flowing(camera) ? "Getting a picture…" : "No picture from this camera right now. Is it plugged in?"}</div>
        )}
        <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
          {Object.entries(saved_)
            .filter(([name]) => !shown[name])
            .map(([name, poly]) => (
              <polygon key={name} className={z.saved} points={poly.map(([x, y]) => `${x},${y}`).join(" ")} />
            ))}
          {Object.entries(shown).map(([name, r]) => (
            <polygon key={name} className={z.draft} points={polygon(r).map(([x, y]) => `${x},${y}`).join(" ")} />
          ))}
        </svg>
        {Object.keys(saved_).map((name) => {
          const [x, y] = shown[name] ? [shown[name].x0, shown[name].y0] : (saved_[name]?.[0] ?? [0, 0]);
          return (
            <span key={name} className={z.tag} style={{ left: `${x * 100}%`, top: `${y * 100}%` }}>
              {humanize(name)}
            </span>
          );
        })}
      </div>

      {error ? <WriteError error={error} /> : null}
      {saved ? (
        <p className={`${z.ok} ${ui.small}`} role="status">
          {saved}
        </p>
      ) : null}

      <div className={`${ui.row} ${ui.wrap}`}>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !pending} onClick={save}>
          {pending > 1 ? `Save ${pending} zones` : "Save zone"}
        </button>
        <button type="button" className={ui.btn} disabled={busy || !camera} onClick={grab}>
          New picture
        </button>
        <button type="button" className={ui.btn} disabled={busy || !zone} onClick={() => setDrafts((d) => ({ ...d, [zone]: { x0: 0.02, y0: 0.02, x1: 0.98, y1: 0.98 } }))}>
          Whole picture
        </button>
        <button type="button" className={ui.btn} disabled={busy} onClick={() => run(api.demoZones, "Demo zones are on: big boxes for every zone.")}>
          Use demo zones
        </button>
      </div>

      <ol className={z.steps}>
        <li>Pick the camera and the zone (Desk, Door…), then drag a box over that part of the picture.</li>
        <li>Draw each zone you need (a ✓ marks the ones drawn), then Save.</li>
        <li>No time? &ldquo;Use demo zones&rdquo; puts big forgiving boxes over most of the picture.</li>
      </ol>
      <p className={`${ui.muted} ${ui.small}`}>
        A thing counts as in a zone when the bottom middle of its box is inside. Sort It reads the bin boxes when the app starts, so restart the gateway after moving Recycling or Trash.
      </p>
    </div>
  );
}
