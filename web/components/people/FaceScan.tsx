"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type EnrollPhotosResult, type ScanPose, type ScanStatus } from "@/lib/api";
import p from "./people.module.css";
import { still } from "./photos";

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

const EVERY_MS = 700;
const PROMPTS: Record<ScanPose, string> = {
  front: "Look straight",
  left: "Turn a little left",
  right: "A little right",
  up: "Chin slightly up",
};

/** Can this page open the phone's camera at all? Browsers allow it only on https (or localhost). */
export function canScan(): boolean {
  return typeof window !== "undefined" && window.isSecureContext && typeof navigator.mediaDevices?.getUserMedia === "function";
}

/**
 * A guided scan with the phone's front camera. A still goes to the box about every 700 ms; the box answers each
 * one (taken, or why not, and which way to look next). Nothing is kept on the phone, and the box keeps only the
 * template it makes at the end. The camera is switched off when the scan ends and whenever the page is hidden.
 */
export function FaceScan({ who, consent, onDone, onClose }: {
  who: { name?: string; member_id?: string; circle_member_id?: string };
  consent: boolean;
  onDone: (result: EnrollPhotosResult) => void;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const [status, setStatus] = useState<ScanStatus | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const [phase, setPhase] = useState<"starting" | "scanning" | "saving" | "stopped">("starting");

  const stopCamera = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }, []);

  // Open the camera and the scan together; close both when this leaves the screen.
  useEffect(() => {
    let gone = false;
    (async () => {
      try {
        const media = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
        if (gone) return media.getTracks().forEach((t) => t.stop());
        stream.current = media;
        if (video.current) {
          video.current.srcObject = media;
          await video.current.play().catch(() => undefined);
        }
        const started = await api.scanStart(who, consent);
        if (gone) return void api.scanCancel(started.session).catch(() => undefined);
        session.current = started.session;
        setStatus(started);
        setPhase("scanning");
      } catch (err) {
        if (gone) return;
        stopCamera();
        const denied = err instanceof DOMException && (err.name === "NotAllowedError" || err.name === "SecurityError");
        setError(denied ? new ApiError(0, "The camera was not allowed. Allow it for this page, or upload photos instead.") : asApiError(err));
        setPhase("stopped");
      }
    })();
    return () => {
      gone = true;
      stopCamera();
      if (session.current) void api.scanCancel(session.current).catch(() => undefined);
      session.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one scan per mount: `who` and `consent` are fixed by then
  }, []);

  // A hidden page must not keep a camera open: the scan stops, and what it had is dropped on the box.
  useEffect(() => {
    const onHide = () => {
      if (document.visibilityState !== "hidden") return;
      stopCamera();
      if (session.current) void api.scanCancel(session.current).catch(() => undefined);
      session.current = null;
      setPhase((ph) => (ph === "saving" ? ph : "stopped"));
      setError((e) => e ?? new ApiError(0, "The scan stopped because the page was hidden. Nothing was saved."));
    };
    document.addEventListener("visibilitychange", onHide);
    return () => document.removeEventListener("visibilitychange", onHide);
  }, [stopCamera]);

  // One still at a time: the next is taken only after the box answered the last.
  useEffect(() => {
    if (phase !== "scanning") return;
    const ctl = new AbortController();
    let timer = 0;
    const tick = async () => {
      const v = video.current;
      const id = session.current;
      if (!v || !id) return;
      const frame = v.readyState >= 2 ? still(v, v.videoWidth, v.videoHeight, 960) : null;
      if (frame) {
        try {
          const answer = await api.scanFrame(id, frame, ctl.signal);
          if (ctl.signal.aborted) return;
          setStatus(answer);
          if (answer.done) {
            setPhase("saving");
            stopCamera();
            session.current = null;
            try {
              onDone(await api.scanFinish(id));
            } catch (err) {
              setError(asApiError(err));
              setPhase("stopped");
            }
            return;
          }
        } catch (err) {
          if (ctl.signal.aborted) return;
          if (err instanceof ApiError && err.status !== 0) {
            // the box closed the scan (it took too long, or recognition was switched off)
            stopCamera();
            session.current = null;
            setError(err);
            setPhase("stopped");
            return;
          }
          /* a still that did not get through: the next one is 700 ms away */
        }
      }
      timer = window.setTimeout(tick, EVERY_MS);
    };
    timer = window.setTimeout(tick, EVERY_MS);
    return () => {
      ctl.abort();
      window.clearTimeout(timer);
    };
  }, [phase, onDone, stopCamera]);

  const count = status?.count ?? 0;
  const needed = status?.needed ?? 8;
  const round = 2 * Math.PI * 26;
  const prompt = phase === "saving" ? "Got it" : status?.want ? PROMPTS[status.want] : phase === "starting" ? "Starting the camera…" : "Hold still";
  const feedback = status?.accepted === false ? status.message : status?.accepted ? "Good." : null;

  return (
    <div className={`${ui.card} ${p.stack}`}>
      {phase !== "stopped" ? (
        <>
          <div className={p.stage}>
            <video ref={video} className={p.video} playsInline muted autoPlay aria-label="Your phone's front camera" />
            <svg className={`${p.outline} ${status?.accepted ? p.outlineGood : ""}`} viewBox="0 0 75 100" preserveAspectRatio="none" aria-hidden="true">
              <ellipse cx="37.5" cy="46" rx="21" ry="29" />
            </svg>
            <svg className={p.ring} viewBox="0 0 64 64" role="img" aria-label={`${count} of ${needed} taken`}>
              <circle cx="32" cy="32" r="26" />
              <circle
                className={p.ringDone}
                cx="32"
                cy="32"
                r="26"
                transform="rotate(-90 32 32)"
                strokeDasharray={round}
                strokeDashoffset={round * (1 - Math.min(count / Math.max(needed, 1), 1))}
              />
              <text x="32" y="32">
                {count}/{needed}
              </text>
            </svg>
          </div>
          <div role="status" aria-live="polite">
            <p className={p.prompt}>{prompt}</p>
            <p className={`${p.feedback} ${status?.accepted === false ? p.feedbackBad : ""}`}>{phase === "saving" ? "Saving…" : feedback}</p>
          </div>
          <p className={`${ui.muted} ${ui.small}`}>
            Fit your face in the outline, in good light. The stills go to your box one at a time and are thrown away as soon as it has looked at
            them; only an encrypted template stays on the box.
          </p>
        </>
      ) : null}
      {error ? <WriteError error={error} /> : null}
      <button type="button" className={ui.btn} onClick={onClose} disabled={phase === "saving"}>
        {phase === "stopped" ? "Close" : "Cancel"}
      </button>
    </div>
  );
}
