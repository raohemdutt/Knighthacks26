"use client";

import { useEffect, useState } from "react";
import { api, type ZoneSet } from "./api";
import { ZONES } from "./zones.generated";

const REFRESH_MS = 10_000;

/**
 * The zones to draw on a camera: what the box has in config/zones.yaml right now (they can be reset on site from the
 * Zones page), or the copy embedded at build time until the box answers or when it cannot.
 */
export function useZones(): ZoneSet {
  const [zones, setZones] = useState<ZoneSet>(ZONES);
  useEffect(() => {
    let live = true;
    const load = () =>
      api
        .zones()
        .then((z) => live && setZones(z))
        .catch(() => undefined); // keep what is on screen
    load();
    const id = window.setInterval(load, REFRESH_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, []);
  return zones;
}
