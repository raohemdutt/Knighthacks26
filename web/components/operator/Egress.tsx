"use client";

import op from "@/components/operator/operator.module.css";
import type { Egress } from "@/lib/api";
import { bytes, groupLabel, internetSummary, picturesLine } from "@/lib/egress";

const SHOWN = 4;

/**
 * "Leaving the box" (V7): what went out to the internet and where, from the firewall's own counters, and how many
 * camera pictures were cleared to leave (approved exports only). Video is never sent: the box keeps no clips.
 */
export function EgressPanel({ egress }: { egress: Egress | null }) {
  if (!egress) return null;
  const pictures = picturesLine(egress);
  if (!egress.enforced) {
    return (
      <div className={op.panel}>
        <h2>Leaving the box</h2>
        <p className={op.big}>Video sent: none</p>
        {pictures ? <p className={op.muted}>{pictures}</p> : null}
        <p className={op.muted}>No firewall on this machine, so bytes per destination are not counted here.</p>
      </div>
    );
  }
  const { total, groups } = internetSummary(egress);
  const offline = egress.mode === "offline";
  return (
    <div className={op.panel}>
      <h2>Leaving the box</h2>
      <p className={op.big}>Video sent: none</p>
      {pictures ? <p className={op.muted}>{pictures}</p> : null}
      <p className={op.muted}>
        {offline ? "Offline: nothing may reach the internet" : `To the internet: ${bytes(total)}`}
        {egress.blocked ? ` · ${egress.blocked} blocked` : ""}
        {egress.stale ? " · numbers are old" : ""}
      </p>
      {!offline && groups.length ? (
        <p className={op.scores}>
          {groups.slice(0, SHOWN).map((g) => (
            <span key={g.name}>
              {groupLabel(g.name)} <b>{bytes(g.bytes)}</b>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}
