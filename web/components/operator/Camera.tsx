"use client";

import { useEffect, useState } from "react";
import { api, snapshotUrl, type CameraStatus, type LiveTracks } from "@/lib/api";
import { cameraName, humanize } from "@/lib/format";
import { ZONES } from "@/lib/zones.generated";
import op from "./operator.module.css";

const REFRESH_MS = 400;
const W = 1600; // overlay coordinates: the 16:9 frame
const H = 900;

function useClock(ms: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

/** The newest picture, swapped in only once it has loaded, so the view never flashes empty between frames. */
function useSnapshot(camera: string, stamp: number, enabled: boolean): string | null {
  const [shown, setShown] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const url = snapshotUrl(camera, stamp);
    const img = new Image();
    img.onload = () => setShown(url);
    img.src = url;
  }, [camera, stamp, enabled]);
  return enabled ? shown : null;
}

function useTracks(camera: string, stamp: number, enabled: boolean): LiveTracks | null {
  const [tracks, setTracks] = useState<LiveTracks | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const ctl = new AbortController();
    api
      .liveTracks(camera, ctl.signal)
      .then(setTracks)
      .catch(() => undefined);
    return () => ctl.abort();
  }, [camera, stamp, enabled]);
  return enabled ? tracks : null;
}

/** One camera for the big display: the blurred picture from the box with boxes, track ids and zone outlines on top. */
export function OperatorCamera({ camera, status }: { camera: string; status: CameraStatus }) {
  const stamp = useClock(REFRESH_MS);
  const flowing = status.status !== "down";
  const picture = useSnapshot(camera, stamp, flowing);
  const tracks = useTracks(camera, stamp, flowing);
  const zones = ZONES[camera] ?? {};
  const seen = tracks?.tracks ?? [];

  return (
    <div className={op.camera}>
      <div className={op.cameraHead}>
        <span className={op.cameraName}>{cameraName(camera)}</span>
        <span className={`${op.pill} ${flowing ? op.pillOk : op.pillBad}`}>
          {flowing ? `${status.fps_actual.toFixed(0)} fps` : "No picture"}
        </span>
      </div>
      <div className={op.view}>
        {picture ? <img src={picture} alt={`Latest picture from the ${cameraName(camera).toLowerCase()}`} /> : null}
        {!flowing ? <div className={op.viewEmpty}>No picture from this camera right now</div> : null}
        {flowing && !picture ? <div className={op.viewEmpty}>Waiting for the first picture…</div> : null}
        <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
          {Object.entries(zones).map(([name, poly]) => {
            const [x0, y0] = poly[0] ?? [0, 0];
            return (
              <g key={name}>
                <polygon className={op.zone} points={poly.map(([x, y]) => `${x * W},${y * H}`).join(" ")} />
                <text className={op.zoneLabel} x={x0 * W + 10} y={y0 * H + 40}>
                  {humanize(name)}
                </text>
              </g>
            );
          })}
          {seen.map((t) => (
            <g key={t.track_id}>
              <rect className={op.box} x={t.box.x1 * W} y={t.box.y1 * H} width={(t.box.x2 - t.box.x1) * W} height={(t.box.y2 - t.box.y1) * H} />
              <text className={op.boxLabel} x={t.box.x1 * W + 8} y={Math.max(t.box.y1 * H - 10, 44)}>
                {t.label} #{t.track_id}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
