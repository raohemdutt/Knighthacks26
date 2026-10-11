"use client";

import { useState } from "react";
import { cameraOfFrame, keyframeUrl } from "@/lib/api";
import { cameraName } from "@/lib/format";
import ui from "./ui.module.css";

/**
 * The camera frame behind a frame ref. When the box has no image for the ref
 * (GET /memory/keyframes/{ref} answers 404, as it does for demo fixtures) a
 * neutral placeholder takes its place, so the evidence slot never looks broken.
 * `compact` is for small thumbnails.
 */
export function Keyframe({ frameRef, caption, compact = false }: { frameRef: string; caption?: string; compact?: boolean }) {
  const [missing, setMissing] = useState(false);
  const camera = cameraOfFrame(frameRef);
  const alt = camera ? `Frame from the ${cameraName(camera).toLowerCase()}` : "Camera frame";
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
          <img className={ui.frameImg} src={keyframeUrl(frameRef)} alt={alt} loading="lazy" decoding="async" onError={() => setMissing(true)} />
        )}
      </div>
      {caption ? <figcaption className={ui.frameCaption}>{caption}</figcaption> : null}
    </figure>
  );
}
