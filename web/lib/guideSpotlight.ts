/**
 * The spotlight on a picture: which frame carries one, and what a live camera tracks right now.
 *
 * The look guide works out a `Spotlight` for the frame under an answer (lib/guide.ts `resolveThing`) and leaves it here;
 * the Keyframe showing that frame draws it. Nothing is stored: it lasts for this visit.
 */

import { useEffect, useState } from "react";
import { api } from "./api";
import { labelsFor, type BoxN, type Spotlight, type WhereSeen } from "./guide";

const spots = new Map<string, Spotlight>();
const listeners = new Set<() => void>();

export const frameSpots = {
  get: (frameRef: string): Spotlight | null => spots.get(frameRef) ?? null,
  set(frameRef: string, spot: Spotlight | null): void {
    if (spot) spots.set(frameRef, spot);
    else spots.delete(frameRef);
    listeners.forEach((fn) => fn());
  },
};

export function useFrameSpot(frameRef: string): Spotlight | null {
  const [spot, setSpot] = useState<Spotlight | null>(() => frameSpots.get(frameRef));
  useEffect(() => {
    const read = () => setSpot(frameSpots.get(frameRef));
    read();
    listeners.add(read);
    return () => void listeners.delete(read);
  }, [frameRef]);
  return spot;
}

/** Memory's last sighting of a thing, under whichever of its names the box knows; null when it has never seen it. */
export async function whereSeen(thing: string): Promise<WhereSeen | null> {
  for (const label of labelsFor(thing)) {
    try {
      return await api.whereSeen(label);
    } catch {
      /* 404: not under this name */
    }
  }
  return null;
}

export interface LiveTrack {
  camera: string;
  label: string;
  box: BoxN;
}

/** A camera that tracks the thing right now. "off": no camera is running on this box (the seeded laptop). */
export async function findLive(thing: string): Promise<LiveTrack | "off" | null> {
  let cameras: string[];
  try {
    cameras = Object.keys((await api.captureStatus()).cameras ?? {});
  } catch {
    return "off";
  }
  if (cameras.length === 0) return "off";
  const names = labelsFor(thing);
  for (const camera of cameras) {
    try {
      const hit = (await api.captureTracks(camera)).tracks.find((t) => names.includes(t.label));
      if (hit) return { camera, label: hit.label, box: hit.box };
    } catch {
      /* this camera is not answering */
    }
  }
  return null;
}
