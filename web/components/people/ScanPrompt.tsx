"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ui from "@/components/ui.module.css";
import { api, viewTicketReady, viewTicketState } from "@/lib/api";
import p from "./people.module.css";

/**
 * The first-run prompt on Live: "Scan your face so Atlas knows it's you" (the People screen's own "You" card says
 * the same thing in the same state, so the prompt is not repeated above it).
 *
 * Shown only to the owner's signed-in screen (it holds a view ticket) on a box where recognition exists and nobody
 * has done the face scan. Dismissing it is remembered in this browser; where storage is not allowed the prompt
 * simply comes back next time, and the page works the same. A screen without a ticket never sees it.
 */
const KEY = "atlas:scan-prompt:dismissed";

function dismissed(): boolean {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function ScanPrompt({ here = false }: { /** on a screen that already holds the scan button: point at it */ here?: boolean }) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    let alive = true;
    if (dismissed()) return;
    Promise.all([api.identitySettings(), api.members(), viewTicketReady()])
      .then(([settings, members]) => {
        const scanned = members.some((m) => m.source !== "circle" && m.scanned !== false);
        if (alive && viewTicketState().held && settings.available && !scanned) setShow(true);
      })
      .catch(() => undefined); // recognition is not set up here, or the box did not answer: nothing to prompt for
    return () => {
      alive = false;
    };
  }, []);
  if (!show) return null;
  const close = () => {
    setShow(false);
    try {
      window.localStorage.setItem(KEY, "1");
    } catch {
      // no storage in this browser: it will be offered again next time
    }
  };
  return (
    <aside className={p.scanPrompt} role="note" aria-label="Scan your face" data-guide="people:scan-prompt">
      <div className={ui.grow}>
        <strong>Scan your face so Atlas knows it&apos;s you</strong>
        <p className={`${ui.muted} ${ui.small}`}>A few looks at your phone&apos;s camera. Only numbers are kept, never a picture.</p>
      </div>
      <Link className={`${ui.btn} ${ui.btnPrimary}`} href={here ? "#you" : "/people/#you"} onClick={here ? undefined : close}>
        Scan my face
      </Link>
      <button type="button" className={ui.btn} onClick={close}>
        Not now
      </button>
    </aside>
  );
}
