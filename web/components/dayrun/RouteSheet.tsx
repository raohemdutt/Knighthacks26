"use client";

import { useEffect, useRef, useState } from "react";
import { api, type PlaceRoute } from "@/lib/api";
import { minutes, ms } from "@/lib/dayrun";
import { useFetch } from "@/lib/useFetch";
import { RouteFacts, RouteMap, RoutePlaces } from "../map/RouteMap";
import { Sheet as BottomSheet } from "../Portal";
import { ErrorBox, Loading } from "../States";
import ui from "../ui.module.css";
import cm from "./campusmap.module.css";

/**
 * "Get me there": a Route button that opens the campus map with the walk from
 * one place to another along the real footpaths, and when to leave.
 *
 *   <RouteSheet from={status.seen_at ?? "here"} to={next.place} arriveBy={next.start} now={data.now} />
 *
 * `from` and `to` are what GET /api/places/route takes: a pack id ("ucf.eng2"),
 * a place's name, "friend:<member>", or "here" (where the box last saw you).
 * With `arriveBy` the sheet counts down to the time to leave; without it, it
 * says how long the walk is from now. `now` is the box's own time when the
 * screen was loaded (a replayed or scripted day is not on the phone's clock).
 *
 * The map, the facts under it and the list of places are the shared walk drawing (components/map/RouteMap.tsx),
 * the same one the Circle draws the walk to a friend with.
 */

export function RouteSheet({
  from = "here",
  to,
  arriveBy,
  now,
  label = "Route",
  what,
  walkMinutes,
}: {
  from?: string;
  to: string;
  arriveBy?: string | null;
  now?: string | null;
  label?: string;
  /** What is at the other end, for the heading: "Knight Hacks: judging and demos". */
  what?: string;
  /** The length of the walk, when the screen already knows it: with 0 there is nowhere to go, so no button is offered. */
  walkMinutes?: number;
}) {
  const [open, setOpen] = useState(false);
  if (walkMinutes !== undefined && walkMinutes <= 0) return null;
  return (
    <>
      <button type="button" className={cm.routeBtn} aria-haspopup="dialog" onClick={() => setOpen(true)}>
        {label}
      </button>
      {open ? <Sheet from={from} to={to} arriveBy={arriveBy ?? null} now={now ?? null} what={what} onClose={() => setOpen(false)} /> : null}
    </>
  );
}

/** "leave in 6 min", "leave now", "you should have left 3 min ago". */
export function leaveText(route: PlaceRoute, at: number): { text: string; tone: "ok" | "soon" | "late" } {
  if (route.walk_minutes === 0) return { text: "You are already there", tone: "ok" };
  if (route.mode !== "arrive_by") return { text: `${minutes(route.walk_minutes)} on foot`, tone: "ok" };
  const left = (ms(route.depart) - at) / 60000;
  if (left >= 1) return { text: `Leave in ${minutes(Math.floor(left))}`, tone: left <= 5 ? "soon" : "ok" };
  if (left > -1) return { text: "Leave now", tone: "soon" };
  if (at < ms(route.arrive)) return { text: `Leave now: ${minutes(Math.floor(-left))} behind`, tone: "late" };
  return { text: `It started ${minutes(Math.floor((at - ms(route.arrive)) / 60000))} ago`, tone: "late" };
}

function Sheet({ from, to, arriveBy, now, what, onClose }: { from: string; to: string; arriveBy: string | null; now: string | null; what?: string; onClose: () => void }) {
  const route = useFetch((s) => api.placeRoute({ from, to, arrive_by: arriveBy ?? undefined, depart: arriveBy ? undefined : (now ?? undefined) }, s), `${from}|${to}|${arriveBy}|${now}`);
  const packs = useFetch((s) => api.places(s));
  const r = route.data;

  // The box's clock, carried forward by the time this sheet has been open.
  const opened = useRef({ wall: Date.now(), box: now ? ms(now) : Date.now() });
  const [at, setAt] = useState(opened.current.box);
  useEffect(() => {
    const id = window.setInterval(() => setAt(opened.current.box + (Date.now() - opened.current.wall)), 5000);
    return () => window.clearInterval(id);
  }, []);

  const leave = r ? leaveText(r, at) : null;

  // The shared sheet: over the viewport (a portal), page scroll locked, focus kept inside and handed back on close.
  return (
    <BottomSheet label={r ? `Route to ${r.to_name}` : "Route"} onClose={onClose}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2 className={ui.breakWords}>{r ? `To ${r.to_name}` : "Route"}</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {route.error ? (
          <ErrorBox what="the route" error={route.error} onRetry={route.reload} />
        ) : !r || !leave ? (
          <Loading what="the route" />
        ) : (
          <>
            <div>
              <p className={`${cm.leave} ${leave.tone === "soon" ? cm.leaveSoon : leave.tone === "late" ? cm.leaveLate : ""}`} role="status">
                {leave.text}
              </p>
              <p className={ui.breakWords}>
                {what ? `${what}: ` : ""}
                {r.sentence}.
              </p>
            </div>
            <RouteFacts route={r} />
            <RouteMap route={r} packs={packs.data} />
            <RoutePlaces route={r} packs={packs.data} />
          </>
        )}
    </BottomSheet>
  );
}
