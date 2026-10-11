"use client";

import { useEffect, useRef } from "react";
import type { Loadable } from "./useFetch";

/**
 * Ask again every `ms` while `active`, but never while the last ask is still on its way.
 *
 * useFetch's reload aborts the request in flight and sends a new one. On a timer that is a trap: a box that takes
 * longer than the beat to answer is cut off and asked again on every beat, so it never gets to answer and the screen
 * sits on its loading line for good. The Learning lab's 3 s beat did exactly that on the Jetson. A beat that finds the
 * last request still out simply waits for the next beat.
 */
export function usePoll(loads: Loadable<unknown>[], active: boolean, ms = 3000): void {
  const ref = useRef(loads);
  ref.current = loads;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => {
      for (const l of ref.current) if (!l.loading) l.reload();
    }, ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
}
