"use client";

import Link from "next/link";
import { useState } from "react";
import { Empty, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type FaceShareFriend, type FaceShareStatus, type Member } from "@/lib/api";
import { dayLabel } from "@/lib/format";
import p from "./people.module.css";

/**
 * Face share on the People screen (GET /api/circle/face-share; rendered only while the box has it switched on).
 *
 *   Friends who can recognise you   per friend: Share, or "Shared until <date>" with Revoke; the key behind a tap
 *   Friends this box recognises     what friends' boxes shared with this one: from whom, until when, Forget
 *
 * Share only raises an approval card on this box: nothing leaves it until the card is approved. A friend's
 * offer is a card too; "Review" raises it again when it ran out. No template, vector or key material is ever
 * in an answer from the box: names, dates, states and the short key fingerprints two friends compare.
 */

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

/** "today", "tomorrow" or "Sat, Nov 10" */
function until(iso: string | null | undefined): string {
  const label = dayLabel(iso);
  return label === "Today" || label === "Tomorrow" || label === "Yesterday" ? label.toLowerCase() : label;
}

function useAct(onChanged: () => void) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const act = async (what: () => Promise<unknown>) => {
    setBusy(true);
    setError(undefined);
    try {
      await what();
      onChanged();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, act };
}

export function FaceShare({ status, received, onChanged }: {
  status: FaceShareStatus;
  /** GET /identity/members rows with source "circle": what friends' boxes shared with this one. */
  received: Member[];
  onChanged: () => void;
}) {
  const offers = status.friends.filter((f) => f.incoming.state === "offered" || f.renewal_offered);
  return (
    <>
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Friends who can recognise you</h2>
        <div className={ui.card}>
          <p>{status.explanation}</p>
          {status.me.can_share ? (
            <p className={`${ui.muted} ${ui.small}`}>
              What goes is the template Atlas made of {status.me.member?.name ?? "you"} on this box: numbers, never a photo. It lasts {status.expiry_days} days
              unless you renew it, and their box deletes it when you take it back.
            </p>
          ) : (
            <p className={ui.nudge} role="status">
              Enroll yourself on this box first: under “Teach Atlas your face”, choose “This is me”. A box can only share its own person, never someone else
              who enrolled here.
            </p>
          )}
          {status.friends.length === 0 ? (
            <p className={ui.muted}>There is nobody else in this Circle yet.</p>
          ) : (
            <ul className={p.friends}>
              {status.friends.map((f) => (
                <Outgoing key={f.member_id} friend={f} mine={status.me.fingerprint} canShare={status.me.can_share} onChanged={onChanged} />
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Friends this box recognises</h2>
        {received.length === 0 && offers.length === 0 ? (
          <Empty title="No friend has shared their face with this box">
            When a friend lets your Atlas recognise them, you are asked first. Nothing is stored until you accept.
          </Empty>
        ) : (
          <div className={ui.card}>
            {!status.recognition_enabled ? (
              <p className={ui.nudge} role="status">
                Recognition is switched off on this box, so nobody is recognised right now. What friends shared is kept, unused, until it is switched on.
              </p>
            ) : null}
            <ul className={p.friends}>
              {offers.map((f) => (
                <Offer key={`offer-${f.member_id}`} friend={f} onChanged={onChanged} />
              ))}
              {received.map((m) => (
                <Received key={m.member_id} member={m} friend={status.friends.find((f) => f.member_id === m.shared_by)} onChanged={onChanged} />
              ))}
            </ul>
          </div>
        )}
      </section>
    </>
  );
}

/** One friend, the way out: may their box recognise me? */
function Outgoing({ friend: f, mine, canShare, onChanged }: { friend: FaceShareFriend; mine: string; canShare: boolean; onChanged: () => void }) {
  const { busy, error, act } = useAct(onChanged);
  const [asking, setAsking] = useState(false);
  const shared = f.outgoing.state === "shared";
  const waiting = f.outgoing.action_id !== null;
  const blocked = f.key.known ? null : `${f.display_name}'s box has not switched face sharing on yet, so there is nothing to send it to.`;

  return (
    <li className={p.friend}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <h3 className={ui.breakWords}>{f.display_name}</h3>
          <p className={`${ui.muted} ${ui.small}`}>
            {shared ? `Shared until ${until(f.outgoing.expires_at)}` : waiting ? "Waiting for your OK. Nothing has been sent." : "Their Atlas does not recognise you."}
          </p>
        </div>
        {shared ? (
          asking ? (
            <div className={ui.row}>
              <button type="button" className={ui.btn} disabled={busy} onClick={() => setAsking(false)}>
                Keep
              </button>
              <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => act(() => api.faceShareRevoke(f.member_id)).then(() => setAsking(false))}>
                {busy ? "Taking it back…" : "Take it back"}
              </button>
            </div>
          ) : (
            <div className={ui.row}>
              <button type="button" className={ui.btn} disabled={busy || waiting || !canShare || f.key.changed} onClick={() => act(() => api.faceShareAsk(f.member_id))}>
                Renew
              </button>
              <button type="button" className={`${ui.btn} ${ui.btnDeny}`} onClick={() => setAsking(true)}>
                Revoke
              </button>
            </div>
          )
        ) : (
          <button
            type="button"
            className={`${ui.btn} ${ui.btnPrimary}`}
            disabled={busy || waiting || !canShare || !f.key.known || f.key.changed}
            onClick={() => act(() => api.faceShareAsk(f.member_id))}
          >
            {busy ? "Asking…" : "Share"}
          </button>
        )}
      </div>
      {asking ? (
        <p className={`${ui.muted} ${ui.small}`}>
          {f.display_name}&apos;s box deletes your template the next time it checks the Circle (about half a minute when it is on). Until then it still has it.
        </p>
      ) : null}
      {waiting ? (
        <p className={ui.nudge} role="status">
          A card is waiting in <Link href="/approvals">Approvals</Link>. Your template is sent only when you approve it.
        </p>
      ) : null}
      {blocked ? <p className={`${ui.muted} ${ui.small}`}>{blocked}</p> : null}
      {f.key.changed ? <KeyChanged friend={f} onChanged={onChanged} /> : null}
      {f.key.known ? (
        <details className={p.keys}>
          <summary>Check the key</summary>
          <dl>
            <dt>{f.display_name}&apos;s box</dt>
            <dd>
              <span className={p.fingerprint}>{f.key.fingerprint}</span>
            </dd>
            <dt>This box</dt>
            <dd>
              <span className={p.fingerprint}>{mine}</span>
            </dd>
          </dl>
          <p className={`${ui.muted} ${ui.small}`}>
            Ask {f.display_name} to open People on their Atlas. Their “This box” must be the first line here, and what they see for you must be the second.
            If they differ, do not share: someone between your boxes may have swapped a key.
          </p>
        </details>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}

/** Another key appeared under a friend's name. Nothing is sent to it until the two friends compared it. */
function KeyChanged({ friend: f, onChanged }: { friend: FaceShareFriend; onChanged: () => void }) {
  const { busy, error, act } = useAct(onChanged);
  return (
    <div className={p.stack} role="alert">
      <p>
        <strong>{f.display_name}&apos;s box key changed.</strong> That happens when a box is set up again, and it is also what a swapped key looks like. Sharing
        with {f.display_name} is paused. Ask them what their People screen shows for “This box”:
      </p>
      <p>
        <span className={p.fingerprint}>{f.key.new_fingerprint}</span>
      </p>
      <div className={ui.row}>
        <button type="button" className={ui.btn} disabled={busy || !f.key.new_fingerprint} onClick={() => act(() => api.faceShareTrust(f.member_id, f.key.new_fingerprint ?? ""))}>
          {busy ? "Saving…" : "It matches: use this key"}
        </button>
      </div>
      {error ? <WriteError error={error} /> : null}
    </div>
  );
}

/** A friend asked this box to recognise them (or to keep doing so for longer). The answer is an approval card. */
function Offer({ friend: f, onChanged }: { friend: FaceShareFriend; onChanged: () => void }) {
  const { busy, error, act } = useAct(onChanged);
  const [raised, setRaised] = useState(false);
  return (
    <li className={p.friend}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <h3 className={ui.breakWords}>{f.display_name}</h3>
          <p className={`${ui.muted} ${ui.small}`}>
            {f.renewal_offered ? "Wants your Atlas to keep recognising them for longer." : "Wants your Atlas to recognise them."} Nothing is stored until you
            accept.
          </p>
        </div>
        <div className={ui.row}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={() => act(() => api.faceShareReview(f.member_id)).then(() => setRaised(true))}>
            Review
          </button>
          {!f.renewal_offered ? (
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => act(() => api.faceShareForget(f.member_id))}>
              Decline
            </button>
          ) : null}
        </div>
      </div>
      {raised ? (
        <p className={ui.nudge} role="status">
          The card is in <Link href="/approvals">Approvals</Link>: accept or decline it there.
        </p>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}

/** A template a friend's box shared with this one: from whom, until when, and Forget. */
function Received({ member: m, friend, onChanged }: { member: Member; friend: FaceShareFriend | undefined; onChanged: () => void }) {
  const { busy, error, act } = useAct(onChanged);
  const [asking, setAsking] = useState(false);
  const from = friend?.display_name ?? m.shared_by ?? "a friend";
  const forget = () => (m.shared_by ? api.faceShareForget(m.shared_by) : api.removeMember(m.member_id));
  return (
    <li className={p.friend}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <h3 className={ui.breakWords}>{m.name}</h3>
          <p className={`${ui.muted} ${ui.small}`}>
            Shared by {from}&apos;s Atlas {dayLabel(m.received_at).toLowerCase()} · until {until(m.expires_at)}
            {friend && friend.incoming.usable === false ? " · not used right now" : ""}
          </p>
        </div>
        {asking ? (
          <div className={ui.row}>
            <button type="button" className={ui.btn} disabled={busy} onClick={() => setAsking(false)}>
              Keep
            </button>
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => act(forget)}>
              {busy ? "Forgetting…" : "Forget for good"}
            </button>
          </div>
        ) : (
          <button type="button" className={`${ui.btn} ${ui.btnDeny}`} onClick={() => setAsking(true)}>
            Forget
          </button>
        )}
      </div>
      {asking ? (
        <p className={`${ui.muted} ${ui.small}`}>
          This deletes {m.name}&apos;s template from this box at once. {from} can share it again, and you would be asked again.
        </p>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}
