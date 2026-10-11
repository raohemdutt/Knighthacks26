"use client";

import { useEffect, useState } from "react";
import { api, viewTicketReady, viewTicketState } from "./api";

/**
 * What the live picture on this screen shows, for the line under it.
 *
 *  - "clear": this is the owner's signed-in screen and "Your live view" is on (Privacy), so faces show here.
 *  - "blurred": everyone else, and the owner's screen while that switch is off. Also the answer until the box
 *    has been asked, so the line never promises faces it may not show.
 *  - "unblurred": an older choice some box still holds: saved pictures are not blurred, and neither is the live
 *    picture, for anyone. The Privacy screen offers the one button that ends it.
 *
 * The box decides what is in the picture; this only reads the same two facts it decides on.
 */
export type LiveView = "clear" | "blurred" | "unblurred";

export function useLiveView(): LiveView {
  const [view, setView] = useState<LiveView>("blurred");
  useEffect(() => {
    let alive = true;
    Promise.all([api.vaultPolicy(), viewTicketReady()])
      .then(([policy]) => {
        if (!alive) return;
        if (!policy.blur_faces) setView("unblurred");
        else setView(policy.live_faces === "clear" && viewTicketState().held ? "clear" : "blurred");
      })
      .catch(() => undefined); // no answer: say the careful thing
    return () => {
      alive = false;
    };
  }, []);
  return view;
}

/** One sentence that is true in each state, in plain words. */
export function liveViewLine(view: LiveView): string {
  if (view === "clear") return "Faces show on this signed-in screen only. Saved pictures and anything sent off this box are blurred.";
  if (view === "unblurred") return "Faces are not blurred on this box right now. You can change that on the Privacy screen.";
  return "Faces are blurred here. Saved pictures and anything sent off this box are blurred too.";
}
