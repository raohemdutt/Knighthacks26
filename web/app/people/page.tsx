"use client";

import { useEffect, useState, type FormEvent } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type EnrollStatus, type Member } from "@/lib/api";
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

export default function PeoplePage() {
  const settings = useFetch((s) => api.identitySettings(s));
  const people = useFetch((s) => api.members(s));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);

  useAtlasEvents(["perception.identity"], () => {
    settings.reload();
    people.reload();
  });

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
          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Recognise enrolled people</h2>
            <div className={ui.card}>
              <p>
                When this is on, Atlas counts habits only for people who enrolled their face here and for nobody else; a face is kept as an
                encrypted template on this box, never as a picture, it never leaves the box, and stored pictures stay blurred for everyone.
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
              {!settings.data.available ? (
                <p className={`${ui.muted} ${ui.small}`}>
                  This box cannot recognise anyone yet: {settings.data.reason ?? "the face recognition model is not installed"}.
                </p>
              ) : null}
              {error ? <WriteError error={error} /> : null}
            </div>
          </section>

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Enroll someone</h2>
            {on ? (
              <Enroll onDone={people.reload} />
            ) : (
              <Empty title="Switch recognition on to enroll">Nobody can be enrolled, and nobody is recognised, while it is off.</Empty>
            )}
          </section>

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Enrolled</h2>
            {people.error ? (
              <ErrorBox what="who is enrolled" error={people.error} onRetry={people.reload} />
            ) : !people.data ? (
              <Loading what="who is enrolled" />
            ) : people.data.length === 0 ? (
              <Empty title="Nobody is enrolled">
                Until someone enrolls, habits describe this desk as a whole. Everyone who has not enrolled stays anonymous.
              </Empty>
            ) : (
              <ul className={ui.list}>
                {people.data.map((m) => (
                  <Person key={m.member_id} member={m} onRemoved={people.reload} />
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </>
  );
}

/** Name, start, then a few looks at the camera; progress comes from polling the box. */
function Enroll({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
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
          if (s.state === "done") {
            setName(""); // ready for the next person
            onDone();
          }
        })
        .catch(() => undefined); // a missed poll is not worth a message: the next one is half a second away
    }, 500);
    return () => {
      ctl.abort();
      window.clearInterval(timer);
    };
  }, [collecting, onDone]);

  const start = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setStarting(true);
    setError(undefined);
    try {
      setStatus(await api.enrollStart(name.trim()));
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setStarting(false);
    }
  };

  if (collecting) {
    const done = status.frames_done ?? 0;
    return (
      <div className={ui.card} role="status" aria-live="polite">
        <h3>Looking for {name.trim() || "you"}</h3>
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
    <form className={ui.card} onSubmit={start}>
      {status?.state === "done" ? (
        <p>
          <strong>Enrolled.</strong> <span className={ui.muted}>From now on Atlas counts their habits as theirs.</span>
        </p>
      ) : status?.state === "failed" ? (
        <p className={ui.muted}>
          {status.error === "timeout" ? "Atlas did not get a clear look in time, so nothing was saved. Try again." : "That did not work, and nothing was saved. Try again."}
        </p>
      ) : (
        <p className={ui.muted}>Sit alone in front of the camera, type your name and start. It takes a few seconds.</p>
      )}
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
      <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={starting || !name.trim()}>
        {starting ? "Starting…" : "Start"}
      </button>
      {error ? <WriteError error={error} /> : null}
    </form>
  );
}

/** "today", "yesterday" or "on Sat, Oct 10" */
function enrolledOn(iso: string): string {
  const label = dayLabel(iso);
  return label === "Today" || label === "Yesterday" ? label.toLowerCase() : `on ${label}`;
}

function Person({ member, onRemoved }: { member: Member; onRemoved: () => void }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);

  const remove = async () => {
    setBusy(true);
    setError(undefined);
    try {
      await api.removeMember(member.member_id);
      onRemoved();
    } catch (err) {
      setError(asApiError(err));
      setBusy(false);
    }
  };

  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <h3 className={ui.breakWords}>{member.name}</h3>
          <p className={`${ui.muted} ${ui.small}`}>Enrolled {enrolledOn(member.enrolled_at)}</p>
        </div>
        {asking ? (
          <div className={ui.row}>
            <button type="button" className={ui.btn} disabled={busy} onClick={() => setAsking(false)}>
              Keep
            </button>
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={remove}>
              {busy ? "Deleting…" : "Delete for good"}
            </button>
          </div>
        ) : (
          <button type="button" className={`${ui.btn} ${ui.btnDeny}`} onClick={() => setAsking(true)}>
            Delete
          </button>
        )}
      </div>
      {asking ? <p className={`${ui.muted} ${ui.small}`}>This removes their face template from this box and their habits from this app.</p> : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}
