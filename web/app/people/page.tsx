"use client";

import { useCallback, useEffect, useState } from "react";
import { ExpressionStrip } from "@/components/people/ExpressionStrip";
import { LooksRemembered } from "@/components/people/LooksRemembered";
import { FaceShare } from "@/components/people/FaceShare";
import { FaceScan, canScan } from "@/components/people/FaceScan";
import { PhotoEnroll } from "@/components/people/PhotoEnroll";
import p from "@/components/people/people.module.css";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CircleMember, type EnrollPhotosResult, type EnrollStatus, type IdentitySettings, type Member } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

/** What the box wants to see next, from GET /identity/enroll/status `hint`. */
const HINTS: Record<string, string> = {
  no_face: "Atlas can't see a face yet. Sit where the camera can see you.",
  one_person_only: "More than one face is in view. Only the person enrolling should be in front of the camera.",
  look_at_camera: "Look straight at the camera.",
  hold_still: "Hold still, looking at the camera.",
};

type Who = { name?: string; member_id?: string; circle_member_id?: string };
type Way = "scan" | "photos" | "box";

/** True once the page is on the phone and may open its camera (https or localhost). False until then, and on plain http. */
function useCanScan(): boolean {
  const [can, setCan] = useState(false);
  useEffect(() => setCan(canScan()), []);
  return can;
}

export default function PeoplePage() {
  const settings = useFetch((s) => api.identitySettings(s));
  const people = useFetch((s) => api.members(s));
  const circle = useFetch((s) => api.circleMembers(s));
  // Face share: null while it is switched off on this box (the default), and then nothing about it is shown.
  const sharing = useFetch((s) => api.faceShare(s));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);

  useAtlasEvents(["perception.identity"], () => {
    settings.reload();
    people.reload();
    sharing.reload();
  });
  useAtlasEvents(["circle.face_share", "agent.action.proposed", "workspace.action.completed"], () => {
    sharing.reload();
    people.reload();
  });
  const reloadSharing = useCallback(() => {
    sharing.reload();
    people.reload();
  }, [sharing.reload, people.reload]);

  const toggle = async () => {
    if (!settings.data) return;
    setSaving(true);
    setError(undefined);
    try {
      await api.setIdentity(!settings.data.identity_enabled);
      settings.reload();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setSaving(false);
    }
  };

  const on = settings.data?.identity_enabled ?? false;
  const friends = circle.data?.configured ? circle.data.members : [];
  const enrolled = (people.data ?? []).filter((m) => m.source !== "circle"); // people who enrolled on this box
  const received = (people.data ?? []).filter((m) => m.source === "circle"); // templates friends' boxes shared with it
  const saved = people.data ?? [];
  const noModel = settings.data ? !settings.data.available : false; // no face model on this machine: nothing is recognised live
  const demo = saved.some((m) => m.demo); // written by the demo seed, not enrolled by anyone

  return (
    <>
      <div className={ui.pageHead}>
        <h1>People</h1>
        <p className={ui.muted}>Who Atlas may tell apart. Off unless you switch it on.</p>
      </div>

      {settings.error ? (
        settings.error.notConfigured ? (
          <Empty title="Recognition is not set up on this box">
            Atlas does not tell people apart here: habits describe this desk, not a person, and every face in a stored picture is blurred.
          </Empty>
        ) : (
          <ErrorBox what="the recognition setting" error={settings.error} onRetry={settings.reload} />
        )
      ) : !settings.data ? (
        <Loading what="the recognition setting" />
      ) : (
        <>
          <StateLine on={on} noModel={noModel} saved={saved} demo={demo} />

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Recognise enrolled people</h2>
            <div className={ui.card}>
              <p>
                When this is on, Atlas counts habits only for people who enrolled their face here and for nobody else; a face is kept as an
                encrypted template on this box, never as a picture,{" "}
                {sharing.data
                  ? "it leaves the box only when you share your own with a friend below (encrypted for their box alone),"
                  : "it never leaves the box,"}{" "}
                and stored pictures stay blurred for everyone.
              </p>
              <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
                <span className={on ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{on ? "On" : "Off"}</span>
                <button
                  type="button"
                  role="switch"
                  aria-checked={on}
                  className={on ? ui.btn : `${ui.btn} ${ui.btnPrimary}`}
                  disabled={saving || (!on && !settings.data.available)}
                  onClick={toggle}
                >
                  {saving ? "Saving…" : on ? "Switch off" : "Switch on"}
                </button>
              </div>
              {noModel && saved.length === 0 ? (
                <p className={`${ui.muted} ${ui.small}`}>
                  This box cannot recognise anyone yet: {settings.data.reason ?? "the face recognition model is not installed"}.
                </p>
              ) : null}
              {error ? <WriteError error={error} /> : null}
            </div>
          </section>

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Teach Atlas your face</h2>
            {on ? (
              <Teach settings={settings.data} friends={friends} onDone={people.reload} />
            ) : (
              <Empty title={noModel ? "Nobody can be enrolled on this machine" : "Switch recognition on to enroll"}>
                {noModel
                  ? "Enrolling needs the face model, which this machine does not have. On the Atlas box this is where the face scan starts."
                  : "Nobody can be enrolled, and nobody is recognised, while it is off."}
              </Empty>
            )}
          </section>

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Enrolled</h2>
            {people.error ? (
              <ErrorBox what="who is enrolled" error={people.error} onRetry={people.reload} />
            ) : !people.data ? (
              <Loading what="who is enrolled" />
            ) : enrolled.length === 0 ? (
              <Empty title="Not scanned yet">
                Atlas can&apos;t tell you from anyone else, so it reads no expressions and learns no face. Until someone does the face scan, habits
                describe this desk as a whole.
              </Empty>
            ) : (
              <ul className={ui.list}>
                {enrolled.map((m) => (
                  <Person key={m.member_id} member={m} settings={settings.data!} friends={friends} on={on} onChanged={people.reload} />
                ))}
              </ul>
            )}
          </section>

          {sharing.data ? (
            <FaceShare status={sharing.data} received={received} onChanged={reloadSharing} />
          ) : received.length > 0 ? (
            <section className={ui.section}>
              <h2 className={ui.sectionTitle}>Friends this box recognises</h2>
              <ul className={ui.list}>
                {received.map((m) => (
                  <LeftOver key={m.member_id} member={m} friends={friends} onChanged={people.reload} />
                ))}
              </ul>
            </section>
          ) : null}

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Room mood from everyone</h2>
            <div className={`${ui.card} ${p.stack}`}>
              <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
                <span className={ui.tag}>Off</span>
                <button type="button" role="switch" aria-checked={false} className={ui.btn} disabled>
                  Not available
                </button>
              </div>
              <p className={`${ui.muted} ${ui.small}`}>
                Atlas reads expressions only for someone who has done the face scan and switched it on for themselves. It no longer reads everyone in the
                room.
              </p>
            </div>
          </section>
        </>
      )}
    </>
  );
}

/** "Demo data": these rows were written by the demo seed (made-up numbers), not by anyone enrolling. */
function DemoTag() {
  return (
    <span className={`${ui.tag} ${p.demoTag}`} data-guide="people:demo" title="Written by the demo seed with made-up numbers. Nobody was enrolled and no face was used.">
      Demo data
    </span>
  );
}

/** One plain line about where things stand, above everything else; and the "Demo data" tag when the list is the demo's. */
function StateLine({ on, noModel, saved, demo }: { on: boolean; noModel: boolean; saved: Member[]; demo: boolean }) {
  const count = (n: number, what: string) => (n > 0 ? `${n} ${what}` : null);
  const local = saved.filter((m) => m.source !== "circle");
  const parts = [
    count(local.filter((m) => m.scanned !== false).length, "scanned"),
    count(saved.length - local.length, "through the Circle"),
    count(local.filter((m) => m.scanned === false).length, "to finish"),
  ].filter(Boolean);
  const line =
    noModel && saved.length > 0
      ? "This machine has no face model, so nobody will be recognised live. What you see is saved from before."
      : noModel
        ? "This machine has no face model, so nobody will be recognised live."
        : on
          ? "Recognition is on."
          : "Recognition is off, so nobody is recognised right now.";
  return (
    <div className={p.stateLine} role="status" data-guide="people:state">
      <p>{line}</p>
      {saved.length > 0 ? (
        <p className={`${ui.muted} ${ui.small}`}>
          {saved.length} {saved.length === 1 ? "person" : "people"} saved: {parts.join(", ")}.
        </p>
      ) : null}
      {demo ? (
        <p className={p.demoLine}>
          <DemoTag />
          <span className={`${ui.muted} ${ui.small}`}>These people come from the demo seed: made-up numbers, nobody&apos;s face, nobody enrolled.</span>
        </p>
      ) : null}
    </div>
  );
}

/** Who is being enrolled, their agreement, then one of three ways: a scan on this phone, photos, or the box's own camera. */
function Teach({ settings, friends, onDone }: { settings: IdentitySettings; friends: CircleMember[]; onDone: () => void }) {
  const scan = useCanScan();
  const [name, setName] = useState("");
  const [other, setOther] = useState(false);
  const [friend, setFriend] = useState("");
  const [consent, setConsent] = useState(false);
  const [way, setWay] = useState<Way | undefined>(undefined);
  const [known, setKnown] = useState<EnrollPhotosResult | undefined>(undefined);
  const photos = settings.photos_available !== false;
  const mine = friends.find((f) => f.mine)?.member_id;
  const link = other ? friend || undefined : mine;
  const ready = name.trim().length > 0 && consent;
  const who: Who = { name: name.trim(), ...(link ? { circle_member_id: link } : {}) };

  const done = useCallback(
    (result: EnrollPhotosResult) => {
      setWay(undefined);
      if (result.state === "done") {
        setKnown(result);
        setName("");
        setConsent(false);
        setFriend("");
        onDone();
      } else setKnown(result);
    },
    [onDone],
  );
  const boxDone = useCallback(() => {
    setName("");
    setConsent(false);
    onDone();
  }, [onDone]);

  if (way === "scan") return <FaceScan who={who} consent={consent} onDone={done} onClose={() => setWay(undefined)} />;

  return (
    <div className={`${ui.card} ${p.stack}`}>
      {known?.state === "done" && known.scanned === false ? (
        <div role="status">
          <h3>Photos saved</h3>
          <p className={ui.muted}>
            Not scanned yet. Atlas can&apos;t tell {other ? "them" : "you"} from anyone else, so it reads no expressions and learns no face. Finish the face
            scan next to the name below; the photos then add to it.
          </p>
        </div>
      ) : known?.state === "done" ? (
        <div role="status">
          <h3>Atlas knows {other ? "them" : "you"}</h3>
          <p className={ui.muted}>
            From {known.accepted} {known.accepted === 1 ? "look" : "looks"}. From now on their habits count as theirs. More photos later (another day, other
            light, glasses) make it steadier: use “Add more photos” next to the name below.
          </p>
        </div>
      ) : known ? (
        <p role="status">
          <strong>
            {known.error === "already_enrolled"
              ? "This face is already enrolled as someone else here, so nothing was saved."
              : `Only ${known.accepted} good ${known.accepted === 1 ? "look" : "looks"} came through, and ${known.needed} are needed. Nothing was saved; try again in better light.`}
          </strong>
        </p>
      ) : (
        <p className={ui.muted}>
          Atlas learns a face from a handful of looks and keeps an encrypted template on this box. It never keeps the pictures, and nothing is sent anywhere.
        </p>
      )}

      <div className={ui.segmented} role="group" aria-label="Who is this">
        <button type="button" className={other ? ui.seg : `${ui.seg} ${ui.segOn}`} aria-pressed={!other} onClick={() => setOther(false)}>
          This is me
        </button>
        <button type="button" className={other ? `${ui.seg} ${ui.segOn}` : ui.seg} aria-pressed={other} onClick={() => setOther(true)}>
          Someone else
        </button>
      </div>

      <label htmlFor="enroll-name">
        <h3>Name</h3>
      </label>
      <input
        id="enroll-name"
        className={ui.input}
        autoComplete="off"
        maxLength={60}
        placeholder="Shown only in this app"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />

      {other && friends.some((f) => !f.mine) ? (
        <>
          <label htmlFor="enroll-friend">
            <h3>Which friend is this?</h3>
          </label>
          <select id="enroll-friend" className={ui.select} value={friend} onChange={(e) => setFriend(e.target.value)}>
            <option value="">Not someone in the Circle</option>
            {friends
              .filter((f) => !f.mine)
              .map((f) => (
                <option key={f.member_id} value={f.member_id}>
                  {f.display_name}
                </option>
              ))}
          </select>
          <p className={`${ui.muted} ${ui.small}`}>
            Optional. Only the link is kept, on this box: their face is never sent to the Circle, and nothing is fetched from their box.
          </p>
        </>
      ) : null}

      <label className={p.check}>
        <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        <span>
          {other
            ? "They are here with me and agree to be enrolled. They can be forgotten at any time from this screen."
            : "This is my own face and I want Atlas to recognise me. I can be forgotten at any time from this screen."}
        </span>
      </label>

      {scan && photos ? (
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} data-guide="people:scan" disabled={!ready} onClick={() => setWay("scan")}>
          {other ? "Scan their face" : "Scan my face"}
        </button>
      ) : photos ? (
        <p className={`${ui.muted} ${ui.small}`}>
          The live scan needs the box&apos;s https address: a phone only opens its camera for a secure page. Start the box with ATLAS_HTTPS=1 and open this app
          at https://, or use photos below.
        </p>
      ) : (
        <p className={`${ui.muted} ${ui.small}`}>This box cannot read pictures yet, so only its own camera can enroll someone.</p>
      )}

      {photos ? (
        <div className={ui.segmented} role="group" aria-label="Other ways">
          <button type="button" className={way === "photos" ? `${ui.seg} ${ui.segOn}` : ui.seg} aria-pressed={way === "photos"} onClick={() => setWay(way === "photos" ? undefined : "photos")}>
            Upload photos
          </button>
          <button type="button" className={way === "box" ? `${ui.seg} ${ui.segOn}` : ui.seg} aria-pressed={way === "box"} onClick={() => setWay(way === "box" ? undefined : "box")}>
            Use the box&apos;s camera
          </button>
        </div>
      ) : null}
      {!ready && way ? <p className={`${ui.muted} ${ui.small}`}>Type the name and tick the box above first.</p> : null}
      {way === "photos" && photos ? <PhotoEnroll who={who} consent={ready} least={3} onDone={done} /> : null}
      {way === "box" || !photos ? <BoxCamera name={name.trim()} consent={ready} circleMemberId={link} onDone={boxDone} /> : null}
    </div>
  );
}

/** Sit in front of the box's own camera for a few looks; progress comes from polling the box. */
function BoxCamera({ name, consent, circleMemberId, onDone }: { name: string; consent: boolean; circleMemberId?: string; onDone: () => void }) {
  const [status, setStatus] = useState<EnrollStatus | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const [starting, setStarting] = useState(false);
  const collecting = status?.state === "collecting";

  useEffect(() => {
    if (!collecting) return;
    const ctl = new AbortController();
    const timer = window.setInterval(() => {
      api
        .enrollStatus(ctl.signal)
        .then((s) => {
          setStatus(s);
          if (s.state === "done") onDone();
        })
        .catch(() => undefined); // a missed poll is not worth a message: the next one is half a second away
    }, 500);
    return () => {
      ctl.abort();
      window.clearInterval(timer);
    };
  }, [collecting, onDone]);

  const start = async () => {
    setStarting(true);
    setError(undefined);
    try {
      setStatus(await api.enrollStart(name, consent, circleMemberId));
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setStarting(false);
    }
  };

  if (collecting) {
    const done = status.frames_done ?? 0;
    return (
      <div role="status" aria-live="polite">
        <h3>Looking for {name || "you"}</h3>
        <p>
          <span className={ui.habitValue}>{done}</span> <span className={ui.muted}>of {status.frames_needed} looks</span>
        </p>
        <div className={ui.meter} role="img" aria-label={`${done} of ${status.frames_needed} looks`}>
          <div className={ui.meterFill} style={{ width: `${Math.round((done / Math.max(status.frames_needed, 1)) * 100)}%` }} />
        </div>
        <p className={ui.muted}>{(status.hint && HINTS[status.hint]) || "Look at the camera."}</p>
      </div>
    );
  }

  return (
    <div className={p.stack}>
      {status?.state === "done" ? (
        <p>
          <strong>Enrolled.</strong> <span className={ui.muted}>From now on Atlas counts their habits as theirs.</span>
        </p>
      ) : status?.state === "failed" ? (
        <p className={ui.muted}>
          {status.error === "timeout" ? "Atlas did not get a clear look in time, so nothing was saved. Try again." : "That did not work, and nothing was saved. Try again."}
        </p>
      ) : (
        <p className={ui.muted}>Sit alone in front of the box&apos;s camera and start. It takes a few seconds.</p>
      )}
      <button type="button" className={ui.btn} disabled={starting || !consent || !name} onClick={start}>
        {starting ? "Starting…" : "Start at the box"}
      </button>
      {error ? <WriteError error={error} /> : null}
    </div>
  );
}

/** "today", "yesterday" or "on Sat, Oct 10" */
function enrolledOn(iso: string): string {
  const label = dayLabel(iso);
  return label === "Today" || label === "Yesterday" ? label.toLowerCase() : `on ${label}`;
}

function Person({ member, settings, friends, on, onChanged }: {
  member: Member;
  settings: IdentitySettings;
  friends: CircleMember[];
  on: boolean;
  onChanged: () => void;
}) {
  const scan = useCanScan();
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState<"photos" | "scan" | undefined>(undefined);
  const [agreed, setAgreed] = useState(false);
  const [added, setAdded] = useState<EnrollPhotosResult | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const reads = member.expressions === true;
  const scanned = member.scanned !== false; // an older box does not say: then it is as before
  const friend = friends.find((f) => f.member_id === member.circle_member_id);

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
  const moreDone = useCallback(
    (result: EnrollPhotosResult) => {
      setAdded(result);
      setAdding(undefined);
      setAgreed(false);
      onChanged();
    },
    [onChanged],
  );

  return (
    <li className={`${ui.card} ${p.stack}`}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <div className={p.nameRow}>
            <h3 className={ui.breakWords}>{member.name}</h3>
            <span className={scanned ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{scanned ? "Scanned" : "Scan to finish"}</span>
            {member.demo ? <DemoTag /> : null}
          </div>
          <p className={`${ui.muted} ${ui.small}`}>
            {scanned ? "Enrolled" : "Photos saved"} {enrolledOn(member.enrolled_at)}
            {member.photos ? ` · ${member.photos} ${member.photos === 1 ? "look" : "looks"}` : ""}
            {friend ? ` · ${friend.display_name} in the Circle` : ""}
          </p>
          {!scanned ? (
            <p className={ui.small} role="status" data-scan="needed" data-guide="people:scan-to-finish">
              {member.recognised === false
                ? "Not scanned yet. Atlas can't tell you from anyone else, so it reads no expressions and learns no face."
                : `Scan to finish. Atlas still recognises ${member.name} from earlier photos, but reads no expressions until the face scan is done.`}
            </p>
          ) : null}
        </div>
        {asking ? (
          <div className={ui.row}>
            <button type="button" className={ui.btn} disabled={busy} onClick={() => setAsking(false)}>
              Keep
            </button>
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => act(() => api.removeMember(member.member_id))}>
              {busy ? "Forgetting…" : "Forget for good"}
            </button>
          </div>
        ) : (
          <button type="button" className={`${ui.btn} ${ui.btnDeny}`} onClick={() => setAsking(true)}>
            Forget me
          </button>
        )}
      </div>
      {asking ? (
        <p className={`${ui.muted} ${ui.small}`}>
          This removes their face template and their settings from this box at once, and their habits and expression history from this app. What
          Atlas already wrote in its log about them (their id next to times, never a picture) stays in the log until you wipe everything on the
          Privacy screen.
        </p>
      ) : null}

      {on ? (
        <>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`} data-guide="people:expressions">
            <div className={ui.grow}>
              <h3>Read my expressions</h3>
              <p className={`${ui.muted} ${ui.small}`}>
                While a camera sees {member.name}, Atlas notes how their face looked (tense, tired, upbeat, neutral), added up over minutes. It is a rough
                signal for the study coach, not a reading of feelings or health. Off for everyone else.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={reads}
              className={reads ? ui.btn : `${ui.btn} ${ui.btnPrimary}`}
              disabled={busy || (!reads && (!scanned || settings.expressions_available === false))}
              onClick={() => act(() => api.updateMember(member.member_id, { expressions: !reads }))}
            >
              {reads ? "Switch off" : "Switch on"}
            </button>
          </div>
          {!scanned ? (
            <p className={`${ui.muted} ${ui.small}`}>
              {reads ? "Switched on, but nothing is read until the face scan is done." : "Finish the face scan first."}
            </p>
          ) : settings.expressions_available === false ? (
            <p className={`${ui.muted} ${ui.small}`}>The expression model is not installed on this box yet.</p>
          ) : reads ? (
            <>
              <ExpressionStrip memberId={member.member_id} />
              <LooksRemembered memberId={member.member_id} />
            </>
          ) : null}

          {adding === "scan" ? (
            <FaceScan who={{ member_id: member.member_id }} consent={agreed} onDone={moreDone} onClose={() => setAdding(undefined)} />
          ) : (
            <>
              <label className={p.check}>
                <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
                <span>{member.name} agrees to Atlas learning more of their face.</span>
              </label>
              <div className={`${ui.row} ${ui.wrap}`}>
                {settings.photos_available !== false ? (
                  <button type="button" className={ui.btn} disabled={!agreed} onClick={() => setAdding(adding === "photos" ? undefined : "photos")}>
                    Add more photos
                  </button>
                ) : null}
                {scan && settings.photos_available !== false ? (
                  <button type="button" className={scanned ? ui.btn : `${ui.btn} ${ui.btnPrimary}`} disabled={!agreed} onClick={() => setAdding("scan")}>
                    {scanned ? "Scan again" : "Finish the face scan"}
                  </button>
                ) : null}
              </div>
              {adding === "photos" ? <PhotoEnroll who={{ member_id: member.member_id }} consent={agreed} least={1} onDone={moreDone} /> : null}
            </>
          )}
          {added?.state === "done" ? (
            <p role="status" className={ui.muted}>
              Added {added.accepted}. The template is now made of {added.exemplars} looks (the newest 12 are kept).
              {added.scanned === false ? " Still not scanned: finish the face scan and Atlas will start to recognise them." : ""}
            </p>
          ) : added ? (
            <p role="status" className={ui.muted}>
              Nothing was added: none of those looked like {member.name} clearly enough.
            </p>
          ) : null}
        </>
      ) : !settings.available ? (
        // No face model here: nothing can be switched or read, so the row shows what was saved, and what was read before.
        <>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`} data-guide="people:expressions">
            <div className={ui.grow}>
              <h3>Read my expressions</h3>
              <p className={`${ui.muted} ${ui.small}`}>
                {!scanned
                  ? "Off until the face scan is done."
                  : reads
                    ? `Saved as on for ${member.name}. Nothing new is read on this machine; below is what was read before.`
                    : `Saved as off for ${member.name}.`}
              </p>
            </div>
            <span className={reads && scanned ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{reads && scanned ? "On" : "Off"}</span>
          </div>
          {reads && scanned ? <ExpressionStrip memberId={member.member_id} hours={48} /> : null}
          {reads && scanned ? <LooksRemembered memberId={member.member_id} /> : null}
        </>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}

/** A template a friend's box shared, on a box where face sharing has since been switched off: it can only be forgotten. */
function LeftOver({ member, friends, onChanged }: { member: Member; friends: CircleMember[]; onChanged: () => void }) {
  const friend = friends.find((f) => f.member_id === member.shared_by);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const forget = async () => {
    setBusy(true);
    setError(undefined);
    try {
      await api.removeMember(member.member_id);
      onChanged();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className={`${ui.card} ${p.stack}`}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <div className={p.nameRow}>
            <h3 className={ui.breakWords}>{member.name}</h3>
            <span className={ui.tag}>Via the Circle</span>
            {member.demo ? <DemoTag /> : null}
          </div>
          {member.demo ? (
            <p className={`${ui.muted} ${ui.small}`}>
              Recognised through the Circle: {friend?.display_name ?? member.name}&apos;s own Atlas shared this, sealed so only this box can read it.
              {member.expires_at ? ` It runs out ${enrolledOn(member.expires_at)} unless they renew it.` : ""}
            </p>
          ) : (
            <p className={`${ui.muted} ${ui.small}`}>
              Shared by a friend&apos;s Atlas. Face sharing is switched off on this box now, so this is deleted the next time the box checks the Circle.
            </p>
          )}
        </div>
        <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={forget}>
          {busy ? "Forgetting…" : "Forget now"}
        </button>
      </div>
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}
