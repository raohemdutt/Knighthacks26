"use client";

import { useEffect, useState } from "react";
import { BuddiesStage } from "@/components/lab/LearningBuddies";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { type PairMember, type SharedBrainPair, linkLine, modelLine, pairState, servingLine, wattsText } from "@/lib/sharedBrain";
import c from "./circle.module.css";

/**
 * The top of the Shared brain card (G6): the brain as two friends. One 122B model split 24 and 24 layers across the
 * lab's two Strix Halo pods, drawn as the Learning lab's two cartoon boxes (LearningBuddies' stage): bumping under a
 * shared glow while both are online, standing apart with the offline one dimmed otherwise. Under them each pod's live
 * watts and online dot, the tunnel between them, and how fast the brain answers.
 *
 * GET /api/circle/shared-brain is polled every 2 s (the box caches its own reads of the pods for 2 s, so phones
 * add no load on the pods). The honest empty state, "No friends online", shows when the box has no keys for the
 * pods or neither pod answers.
 */

const POLL_MS = 2000;

function Member({ m }: { m: PairMember }) {
  return (
    <div className={ui.stat}>
      <div className={c.pairWho}>
        <span className={`${c.pairDot} ${m.online ? c.pairDotOn : ""}`} aria-hidden="true" />
        <b>{m.name}</b>
        <span className={ui.muted}>{m.online ? "online" : "offline"}</span>
      </div>
      <div className={`${ui.statNum} ${c.brainWatts}`}>{wattsText(m.watts_now)}</div>
      <div className={ui.statLabel}>
        {m.watts_now === null ? (m.online ? "watts not read" : "no reading") : "right now"}
        {m.watts_idle !== null ? `, idle ${m.watts_idle} W` : ""}
      </div>
      <div className={`${ui.small} ${ui.muted}`}>
        {m.role === "head" ? "the head" : "the other half"}: {m.layers} layers{m.vram_gb !== null ? `, ${m.vram_gb} GiB of the model` : ""}
      </div>
    </div>
  );
}

export function BrainPair() {
  const [pair, setPair] = useState<SharedBrainPair | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      if (!document.hidden) {
        try {
          const p = await api.circleSharedBrain();
          if (stop) return;
          setPair(p);
          setFailed(false);
        } catch {
          if (!stop) setFailed(true); // an older gateway without the route, or the box is away: the empty state
        }
      }
      if (!stop) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, []);

  const b = failed ? null : pair;
  const state = pairState(b);
  const [head, half] = b?.members ?? [];
  const on = (b?.members ?? []).filter((m) => m.online);
  const away = (b?.members ?? []).filter((m) => !m.online);
  const label =
    state === "both"
      ? `${on.map((m) => m.name).join(" and ")} together, sharing one brain`
      : state === "one"
        ? `${on[0]?.name} is online${away[0] ? `; ${away[0].name} is not` : ""}`
        : "No friends online";
  const loading = !pair && !failed;

  return (
    <div className={c.pair}>
      <BuddiesStage
        state={state === "both" ? "with-friend" : "idle"}
        apart={state !== "both"}
        off={{ me: !head?.online, friend: !half?.online }}
        label={loading ? "Looking for the two friends" : label}
      />
      {!loading && state === "none" ? (
        <div role="status">
          <p className={c.pairEmpty}>No friends online</p>
          <p className={`${ui.small} ${ui.muted}`}>
            {!b ? "The box did not answer for the shared brain." : !b.configured ? (b.reason ?? "This box has no keys for the pods.") : "Neither pod answered the box just now."}
          </p>
        </div>
      ) : null}
      {b ? (
        <>
          <div>
            <p className={c.brainWhere}>{modelLine(b)}</p>
            <p className={`${ui.small} ${ui.muted}`}>
              {b.model.params_b}B parameters, {b.model.active_b}B active for each token, {b.model.quant}
            </p>
          </div>
          <div className={ui.stats}>
            {b.members.map((m) => (
              <Member key={m.name} m={m} />
            ))}
          </div>
          <dl className={c.pairLines}>
            <div>
              <dt>Between them</dt>
              <dd>{linkLine(b)}</dd>
            </div>
            <div>
              <dt>Answering</dt>
              <dd>{servingLine(b)}</dd>
            </div>
          </dl>
        </>
      ) : loading ? (
        <p className={`${ui.small} ${ui.muted}`}>Looking for the two friends...</p>
      ) : null}
    </div>
  );
}
