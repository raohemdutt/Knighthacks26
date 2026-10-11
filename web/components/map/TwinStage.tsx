"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { Loading } from "@/components/States";
import { api, type RoomRouteAnswer } from "@/lib/api";
import { type RoomTwinData, type TwinPin, useRoomTwin, useTwinPins } from "@/lib/roomTwin";
import type { RoomTwinProps } from "./RoomTwin";
import css from "./roomtwin.module.css";

// three.js lives in this chunk only, fetched when a screen first shows the room.
const RoomTwin = dynamic(() => import("./RoomTwin"), { ssr: false, loading: () => <Loading what="the room" /> });

const PREVIEW_EVERY_MS = 2000; // the person the route starts from may be walking

/** The walk to the right bin for `item` from wherever a camera sees someone, kept fresh. Null when there is nothing to ask. */
function usePreviewRoute(item: string | null, on: boolean): RoomRouteAnswer | null {
  const [route, setRoute] = useState<RoomRouteAnswer | null>(null);
  useEffect(() => {
    if (!item || !on) {
      setRoute(null);
      return;
    }
    let live = true;
    const load = () =>
      api
        .roomRoute(item)
        .then((r) => live && setRoute((old) => (JSON.stringify(old) === JSON.stringify(r) ? old : r)))
        .catch(() => live && setRoute(null));
    load();
    const id = window.setInterval(load, PREVIEW_EVERY_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [item, on]);
  return route;
}

export interface TwinStageProps extends Pick<RoomTwinProps, "controls" | "shape" | "tally" | "speak" | "picked" | "onPick" | "start"> {
  twin: RoomTwinData;
  pins: TwinPin[];
  /** Draw the way to the right bin for this item while no round is on. */
  previewItem?: string | null;
}

/** The room twin once its plan is in. Renders nothing when the box has no room plan (the caller shows its fallback). */
export function TwinStage({ twin, pins, previewItem = null, ...rest }: TwinStageProps) {
  const preview = usePreviewRoute(previewItem, Boolean(twin.plan) && !twin.live?.round);
  if (twin.plan === undefined) return <Loading what="the room" />;
  if (twin.plan === null) return null;
  return <RoomTwin plan={twin.plan} live={twin.live} pins={pins} preview={preview} {...rest} />;
}

/** The whole thing for a screen that only wants the room: the operator's panel. */
export function OperatorRoom() {
  const twin = useRoomTwin();
  const pins = useTwinPins(twin.plan, twin.live);
  if (twin.plan === null) return null; // no room plan on this box: no panel
  return (
    <div className={css.opPanel} data-panel="room">
      <TwinStage twin={twin} pins={pins} shape="fill" />
    </div>
  );
}
