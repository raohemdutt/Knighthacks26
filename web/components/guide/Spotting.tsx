"use client";

import { useEffect, useState } from "react";
import { api, snapshotUrl, type ZoneSet } from "@/lib/api";
import { cameraName, humanize } from "@/lib/format";
import type { BoxN } from "@/lib/guide";
import g from "./guide.module.css";

/**
 * "Spotting": a camera's picture with the thing outlined where the detector has it now, or the right bin's zone
 * outlined and the others dimmed. The picture is the camera's blurred snapshot (GET /capture/<camera>/snapshot.jpg),
 * fetched again every 1.5 s with the newest tracks (GET /capture/<camera>/tracks): refreshed stills, not video, and the
 * panel says so. Where no camera runs (the seeded laptop) it says that instead of showing an empty frame.
 */

const EVERY_MS = 1500;

export interface SpottingAsk {
  camera: string | null; // null: no camera is running on this box
  /** The labels a track may carry for the thing ("keys", "key"). */
  labels?: string[];
  /** A zone of that camera to outline (a bin); the camera's other zones are dimmed. */
  zone?: string;
  name: string; // "keys", "Recycling"
}

export function Spotting({ ask, onClose }: { ask: SpottingAsk; onClose: () => void }) {
  const [stamp, setStamp] = useState(() => Date.now());
  const [box, setBox] = useState<BoxN | null>(null);
  const [state, setState] = useState<"loading" | "on" | "lost" | "off">(ask.camera ? "loading" : "off");
  const [zones, setZones] = useState<ZoneSet | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const camera = ask.camera;
    if (!camera) return;
    let alive = true;
    let misses = 0;
    const tick = async () => {
      try {
        const got = await api.captureTracks(camera);
        if (!alive) return;
        misses = 0;
        const hit = ask.labels ? got.tracks.find((t) => ask.labels!.includes(t.label)) : undefined;
        setBox(hit?.box ?? null);
        setState(ask.labels && !hit ? "lost" : "on");
        setStamp(Date.now());
      } catch {
        if (alive && ++misses >= 2) setState("off");
      }
    };
    void tick();
    const id = window.setInterval(tick, EVERY_MS);
    if (ask.zone) void api.zones().then((z) => alive && setZones(z), () => undefined);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [ask.camera, ask.labels, ask.zone]);

  const polys = ask.camera && ask.zone && zones ? Object.entries(zones[ask.camera] ?? {}) : [];
  const pts = (p: [number, number][]) => p.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");
  return (
    <aside className={g.spotting} data-guide="spotting" data-guide-caption data-state={state} aria-label={`Spotting ${ask.name}`}>
      <p className={g.label}>
        <span>Spotting: {humanize(ask.name)}</span>
        <button type="button" className={g.close} onClick={onClose} aria-label="Close the camera picture">
          Close
        </button>
      </p>
      {state === "off" || !ask.camera ? (
        <p className={g.line} data-spotting="off">
          The camera is not running on this box right now, so there is no live picture. The room shows where it was last seen.
        </p>
      ) : (
        <>
          <div className={g.spotFrame}>
            {failed ? <p className={g.note}>No picture from the {cameraName(ask.camera).toLowerCase()} yet.</p> : <img src={snapshotUrl(ask.camera, stamp)} alt={`${cameraName(ask.camera)}, newest picture`} onError={() => setFailed(true)} onLoad={() => setFailed(false)} />}
            {box ? <span className={g.spotBox} data-guide="spotting:box" data-precision="exact" style={{ left: `${box.x1 * 100}%`, top: `${box.y1 * 100}%`, width: `${(box.x2 - box.x1) * 100}%`, height: `${(box.y2 - box.y1) * 100}%` }} /> : null}
            {polys.length > 0 ? (
              <svg className={g.spotZones} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                {polys.map(([id, poly]) => (
                  <polygon key={id} points={pts(poly)} data-zone={id} data-on={id === ask.zone} className={id === ask.zone ? g.zoneOn : g.zoneDim} />
                ))}
              </svg>
            ) : null}
          </div>
          <p className={g.line}>{state === "lost" ? `The camera does not see your ${ask.name} right now.` : ask.zone ? `${humanize(ask.zone)}: throw it here.` : box ? "Here it is." : "Looking…"}</p>
          <p className={g.note}>
            {cameraName(ask.camera)}: the camera&apos;s picture, refreshed every {EVERY_MS / 1000} s with faces blurred. It is not video.
          </p>
        </>
      )}
    </aside>
  );
}
