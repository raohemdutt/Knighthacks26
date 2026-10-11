"use client";

import { useState } from "react";
import { cameraOfFrame, keyframeUrl } from "@/lib/api";
import { cameraName } from "@/lib/format";
import { coverRect } from "@/lib/guide";
import { useFrameSpot } from "@/lib/guideSpotlight";
import ui from "./ui.module.css";

/**
 * The camera frame behind a frame ref. When the box has no image for the ref
 * (GET /memory/keyframes/{ref} answers 404, as it does for demo fixtures) a
 * neutral placeholder takes its place, so the evidence slot never looks broken.
 * `compact` is for small thumbnails.
 *
 * The look guide may leave a spotlight for this frame (lib/guideSpotlight.ts): a tight solid outline only when the box
 * has a box for the thing ("Here it is"), a soft dashed region placed from the words otherwise ("About here"), with the
 * model's own words underneath. With neither, nothing is drawn here and the guide rings the whole picture.
 */
export function Keyframe({ frameRef, caption, compact = false }: { frameRef: string; caption?: string; compact?: boolean }) {
  const [missing, setMissing] = useState(false);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const spot = useFrameSpot(frameRef);
  const camera = cameraOfFrame(frameRef);
  const alt = camera ? `Frame from the ${cameraName(camera).toLowerCase()}` : "Camera frame";
  // the frame is shown 4:3 with the picture covering it; a stored placeholder stands in for the whole picture
  const at = spot?.rect ? coverRect(spot.rect, missing || !natural ? { w: 4, h: 3 } : natural, { w: 4, h: 3 }) : null;
  const exact = spot?.kind === "box" && spot.precision.precision === "exact"; // a tight outline needs the thing's own box
  return (
    <figure className={ui.frame} data-guide="frame">
      <div className={ui.frameBox}>
        {missing ? (
          <div className={ui.framePlaceholder} role="img" aria-label={`${alt}: image not stored on this box`}>
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <rect x="3" y="6" width="18" height="13" rx="2.5" />
              <circle cx="12" cy="12.5" r="3.4" />
              <path d="M8 6l1.4-2h5.2L16 6" />
            </svg>
            <span>{compact ? "No frame stored" : "Frame not stored on this box"}</span>
          </div>
        ) : (
          <img className={ui.frameImg} src={keyframeUrl(frameRef)} alt={alt} loading="lazy" decoding="async" onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} onError={() => setMissing(true)} />
        )}
        {spot && at && at.width > 0 && at.height > 0 ? (
          <span
            data-guide="spot"
            data-spot={spot.kind}
            data-precision={spot.precision.precision}
            aria-label={`${spot.label}${spot.words ? `: ${spot.words}` : ""}`}
            style={{
              position: "absolute",
              left: `${at.left}%`,
              top: `${at.top}%`,
              width: `${at.width}%`,
              height: `${at.height}%`,
              boxSizing: "border-box",
              pointerEvents: "none",
              border: exact ? "3px solid var(--danger)" : "3px dashed var(--danger)",
              borderRadius: exact ? 6 : 22,
              background: exact ? "transparent" : "var(--danger-soft)",
            }}
          />
        ) : null}
      </div>
      {caption ? <figcaption className={ui.frameCaption}>{caption}</figcaption> : null}
      {spot && !compact ? (
        <figcaption className={ui.frameCaption} data-guide="spot:words" data-precision={spot.precision.precision}>
          <b>{spot.label}.</b> {spot.words ? `The box's words: “${spot.words}”.` : spot.kind === "box" ? "" : "The box has no words for the place."}{" "}
          {spot.kind === "box" ? "Outlined where the box found it." : spot.kind === "about" ? `Placed from those words beside the ${spot.anchor}; there is no box for it.` : "There is no box for it, so the whole picture is marked."}
        </figcaption>
      ) : null}
    </figure>
  );
}
