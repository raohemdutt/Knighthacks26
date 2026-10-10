"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, type RoomLive, type RoomPlan, type Sighting } from "./api";
import { useAtlas } from "./atlas";
import { dayTime, humanize } from "./format";
import { WATCH_ITEMS } from "./room";

/**
 * Data for the room twin (components/map/RoomTwin.tsx).
 *
 *  - The plan (GET /api/room) is read once and re-read every ten seconds: zones and the calibration can be reset
 *    on site, and the picture follows without a reload.
 *  - What is in the room (GET /api/room/live) is read again whenever the box says something moved: every
 *    `perception.observation.recorded`, zone, face or `play.*` event on /ws asks for the newest state. The box does
 *    the projection onto the floor (one place for the maths). When the socket is down the same call is polled.
 *  - Memory pins: where each watch item was last seen (GET /memory/objects/<label>/last-seen), placed at the floor
 *    point the box last projected for that label when it has one, else in the middle of its zone.
 */

const PLAN_EVERY_MS = 10_000;
const POLL_LIVE_MS = 5_000; // a safety net while /ws is up
const POLL_OFFLINE_MS = 1_500; // the only source while it is down
const COALESCE_MS = 70; // two cameras report in the same instant: one fetch for both
const MOVES = ["perception.observation", "perception.zone", "perception.person", "perception.faces", "play.round", "play.route"];

export interface RoomTwinData {
  /** undefined while loading; null when this box has no room plan. */
  plan: RoomPlan | null | undefined;
  live: RoomLive | null;
  error: ApiError | null;
}

export function useRoomTwin(enabled = true): RoomTwinData {
  const { subscribe, link } = useAtlas();
  const [plan, setPlan] = useState<RoomPlan | null | undefined>(undefined);
  const [live, setLive] = useState<RoomLive | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const planText = useRef("");
  const liveText = useRef("");

  useEffect(() => {
    if (!enabled) return;
    let on = true;
    const load = () =>
      api
        .room()
        .then((p) => {
          if (!on) return;
          const text = JSON.stringify(p);
          if (text !== planText.current) {
            planText.current = text;
            setPlan(p);
          }
          setError(null);
        })
        .catch((err: unknown) => {
          if (!on) return;
          if (err instanceof ApiError && err.status === 404) setPlan(null);
          else if (err instanceof ApiError) setError(err); // keep the room that is on screen
        });
    load();
    const id = window.setInterval(load, PLAN_EVERY_MS);
    return () => {
      on = false;
      window.clearInterval(id);
    };
  }, [enabled]);

  const busy = useRef(false);
  const again = useRef(false);
  const fetchLive = useCallback(() => {
    if (busy.current) {
      again.current = true;
      return;
    }
    busy.current = true;
    api
      .roomLive()
      .then((state) => {
        const text = JSON.stringify(state);
        if (text !== liveText.current) {
          liveText.current = text;
          setLive(state);
        }
      })
      .catch(() => undefined) // keep what is on screen
      .finally(() => {
        busy.current = false;
        if (again.current) {
          again.current = false;
          fetchLive();
        }
      });
  }, []);

  useEffect(() => {
    if (plan === null || !enabled) return;
    let timer = 0;
    fetchLive();
    const off = subscribe((e) => {
      if (!MOVES.some((p) => e.type.startsWith(p)) || timer) return;
      timer = window.setTimeout(() => {
        timer = 0;
        fetchLive();
      }, COALESCE_MS);
    });
    const poll = window.setInterval(fetchLive, link === "live" ? POLL_LIVE_MS : POLL_OFFLINE_MS);
    return () => {
      off();
      window.clearTimeout(timer);
      window.clearInterval(poll);
    };
  }, [subscribe, fetchLive, link, plan === null, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  return { plan, live, error };
}

/** A memory pin on the floor plan. */
export interface TwinPin {
  label: string;
  x: number;
  y: number;
  z: number;
  /** True when the box projected this spot from a camera track; false when the pin only stands in its zone. */
  exact: boolean;
  title: string; // "Your keys are here"
  detail: string; // "Left of the laptop · seen 9:12 AM"
  sighting: Sighting;
}

function centre(poly: [number, number][]): [number, number] {
  return [poly.reduce((s, p) => s + p[0], 0) / poly.length, poly.reduce((s, p) => s + p[1], 0) / poly.length];
}

const PLURAL = /s$/;

/** Pins for the sightings: at the last projected floor point when there is one, else spread inside the zone. */
export function buildPins(plan: RoomPlan, live: RoomLive | null, sightings: Sighting[]): TwinPin[] {
  const pins: TwinPin[] = [];
  for (const s of sightings) {
    const seen = live?.seen[s.label];
    const title = `Your ${s.label} ${PLURAL.test(s.label) ? "are" : "is"} here`;
    const when = `seen ${dayTime(s.ts).replace(/^Today /, "today at ")}`;
    if (seen && seen.camera === s.camera_id) {
      const detail = [s.position ? humanize(s.position) : seen.on ? `On the ${seen.on.replace(/-/g, " ")}` : null, when].filter(Boolean).join(" · ");
      pins.push({ label: s.label, x: seen.x, y: seen.y, z: seen.z, exact: true, title, detail, sighting: s });
      continue;
    }
    const poly = s.zone_id ? plan.zones[s.camera_id]?.[s.zone_id] : undefined;
    if (!poly || poly.length < 3) continue;
    const [x, y] = centre(poly); // the twin draws one pin per zone for these, and steps through them
    const under = plan.furniture.filter((f) => f.type !== "rug" && inside(f, x, y)).sort((a, b) => b.height - a.height)[0];
    const detail = [s.position ? humanize(s.position) : `Somewhere in the ${s.zone_id} zone`, when].join(" · ");
    pins.push({ label: s.label, x, y, z: under ? under.height : 0, exact: false, title, detail, sighting: s });
  }
  return pins;
}

function inside(f: RoomPlan["furniture"][number], x: number, y: number): boolean {
  const t = (f.rot * Math.PI) / 180;
  const dx = x - f.at[0];
  const dy = y - f.at[1];
  const lx = dx * Math.cos(t) + dy * Math.sin(t);
  const ly = -dx * Math.sin(t) + dy * Math.cos(t);
  return Math.abs(lx) <= f.size[0] / 2 && Math.abs(ly) <= f.size[1] / 2;
}

/** The watch items' last sightings, refreshed when Memory or the cameras say something new. */
export function useSightings(): Sighting[] {
  const { subscribe } = useAtlas();
  const [found, setFound] = useState<Sighting[]>([]);
  useEffect(() => {
    let on = true;
    let last = 0;
    const load = () => {
      last = Date.now();
      Promise.all(WATCH_ITEMS.map((label) => api.lastSeen(label).catch(() => null))).then((all) => {
        if (on) setFound(all.filter((s): s is Sighting => s !== null));
      });
    };
    load();
    const off = subscribe((e) => {
      // a new episode or scene may have moved something; not more often than every few seconds
      if ((e.type.startsWith("memory.") || e.type.startsWith("perception.scene") || e.type.startsWith("perception.zone")) && Date.now() - last > 4000) load();
    });
    const id = window.setInterval(load, 20_000);
    return () => {
      on = false;
      off();
      window.clearInterval(id);
    };
  }, [subscribe]);
  return found;
}

export function useTwinPins(plan: RoomPlan | null | undefined, live: RoomLive | null): TwinPin[] {
  const sightings = useSightings();
  return useMemo(() => (plan ? buildPins(plan, live, sightings) : []), [plan, live, sightings]);
}
