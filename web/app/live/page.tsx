"use client";

import { ScanPrompt } from "@/components/people/ScanPrompt";
import { useEffect, useState } from "react";
import live from "@/components/live/live.module.css";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, snapshotUrl, type CameraStatus, type LiveTracks } from "@/lib/api";
import { cameraName, humanize } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { liveViewLine, useLiveView } from "@/lib/useLiveView";
import { useZones } from "@/lib/useZones";

/**
 * Live: every camera the box reads, with what the detector sees drawn on top.
 *
 *  - The picture is GET /capture/{camera}/snapshot.jpg, refreshed about twice a second. The box blurs the faces in
 *    it for everyone except the owner's own signed-in screen while "Your live view" is on (Privacy); the line under
 *    the heading says which of the two this screen is getting. Saved pictures are blurred either way. It is a
 *    still, not a video stream: the MJPEG endpoint (C5) can replace the <img> later.
 *  - Boxes and track ids are GET /capture/{camera}/tracks (normalized 0..1). Zone outlines come from
 *    config/zones.yaml, read from GET /api/zones (the build-time copy, lib/zones.generated.ts, until it answers).
 *  - Tap a camera to make it full width.
 */

const REFRESH_MS = 500;
const W = 1600; // overlay coordinates: the 16:9 frame
const H = 900;

const STATUS_TEXT: Record<CameraStatus["status"], string> = { ok: "Live", degraded: "Slow", down: "No picture" };

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

function Tile({ camera, status, open, onToggle }: { camera: string; status: CameraStatus; open: boolean; onToggle: () => void }) {
  const stamp = useClock(REFRESH_MS);
  const flowing = status.status !== "down";
  const picture = useSnapshot(camera, stamp, flowing);
  const tracks = useTracks(camera, stamp, flowing);
  const zones = useZones()[camera] ?? {};
  const seen = tracks?.tracks ?? [];
  const summary = seen.length ? seen.map((t) => `${t.label} #${t.track_id}`).join(", ") : "Nothing in view";

  return (
    <button type="button" className={`${live.tile} ${open ? live.tileOpen : ""}`} onClick={onToggle} aria-expanded={open}>
      <div className={live.head}>
        <span className={live.name}>{cameraName(camera)}</span>
        <span className={`${live.status} ${flowing ? "" : live.statusBad}`}>{STATUS_TEXT[status.status]}</span>
      </div>
      <div className={live.view}>
        {picture ? <img src={picture} alt={`Latest picture from the ${cameraName(camera).toLowerCase()}`} /> : null}
        {!flowing ? <div className={live.empty}>No picture from this camera right now. Is it plugged in?</div> : null}
        {flowing && !picture ? <div className={live.empty}>Waiting for the first picture…</div> : null}
        <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
          {Object.entries(zones).map(([name, poly]) => {
            const [x0, y0] = poly[0] ?? [0, 0];
            return (
              <g key={name}>
                <polygon className={live.zone} points={poly.map(([x, y]) => `${x * W},${y * H}`).join(" ")} />
                <text className={live.zoneLabel} x={x0 * W + 8} y={y0 * H + 30}>
                  {humanize(name)}
                </text>
              </g>
            );
          })}
          {seen.map((t) => (
            <g key={t.track_id}>
              <rect className={live.box} x={t.box.x1 * W} y={t.box.y1 * H} width={(t.box.x2 - t.box.x1) * W} height={(t.box.y2 - t.box.y1) * H} />
              <text className={live.boxLabel} x={t.box.x1 * W + 6} y={Math.max(t.box.y1 * H - 8, 34)}>
                {t.label} #{t.track_id}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className={live.foot}>
        {summary}
        {open ? ` · ${status.fps_actual.toFixed(0)} fps in, ${status.detected} frames checked` : ""}
      </div>
    </button>
  );
}

export default function LivePage() {
  const cameras = useFetch((signal) => api.captureStatus(signal));
  const clock = useClock(3000);
  const [open, setOpen] = useState<string | null>(null);
  const view = useLiveView();
  const { reload } = cameras;
  useEffect(() => reload(), [clock, reload]);

  const entries = Object.entries(cameras.data?.cameras ?? {});
  return (
    <>
      <div className={ui.pageHead}>
        <h1>Live</h1>
        <p className={ui.muted} data-live-view={view}>{liveViewLine(view)}</p>
      </div>
      <ScanPrompt />
      {cameras.loading && !cameras.data ? <Loading what="the cameras" /> : null}
      {cameras.error && !cameras.data && !cameras.error.notConfigured ? <ErrorBox what="the cameras" error={cameras.error} onRetry={cameras.reload} /> : null}
      {(cameras.data && entries.length === 0) || (cameras.error && !cameras.data && cameras.error.notConfigured) ? (
        <Empty title="No cameras on this box">The box is not reading any camera yet. Cameras are listed in config/cameras.yaml and start with the gateway.</Empty>
      ) : null}
      <div className={live.grid}>
        {entries.map(([id, status]) => (
          <Tile key={id} camera={id} status={status} open={open === id} onToggle={() => setOpen(open === id ? null : id)} />
        ))}
      </div>
    </>
  );
}
