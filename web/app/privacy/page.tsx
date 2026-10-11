"use client";

import { useEffect, useState } from "react";
import priv from "@/components/privacy/privacy.module.css";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, viewTicketReady, viewTicketState, type Egress, type VaultPolicy, type WipeResult, type WipeScope, type WipeToken } from "@/lib/api";
import { bytes, groupLabel, internetSummary, picturesLine } from "@/lib/egress";
import { cameraName } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { ZONES } from "@/lib/zones.generated";

/**
 * Privacy: what the box keeps, what it may send, and the delete button.
 *
 * Everything here is the Vault API: GET/PUT /vault/policy (the live view, paused cameras, offline, retention),
 * GET /vault/egress (what the firewall let out, per group) and the two-step POST /vault/wipe/request + /vault/wipe.
 * "Offline" also pauses connected apps; the box keeps seeing, answering and alerting because that is all local.
 *
 * Five calm sections, each a heading over one card, one idea per row: Faces, Cameras, What is kept, What has left
 * this box, Erase. Every row and section carries a data-guide hook (web/lib/guide.ts, web/lib/guide-hooks.json).
 */

/**
 * PROFILE HEADER SLOT (for Xander). A profile header for this screen (an avatar, the owner's name, "signed in on
 * this screen") goes here, between the page heading and the first section. It is left empty on purpose: the avatar
 * and any new visual identity are Xander's to design. Render it from this component and nothing else has to move.
 */
function ProfileHeaderSlot() {
  return null;
}

const KEEP_FOREVER = 91; // the slider's last stop
const SCOPES: { id: WipeScope; label: string; what: string }[] = [
  { id: "all", label: "Everything", what: "all pictures, the memory, exports and the event log" },
  { id: "keyframes", label: "Pictures only", what: "the saved camera pictures" },
  { id: "memory", label: "Memory only", what: "what Atlas remembers (the memory database)" },
];

function Switch({ on, label, onChange, disabled }: { on: boolean; label: string; onChange: (next: boolean) => void; disabled?: boolean }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-label={label} disabled={disabled} className={`${priv.switch} ${on ? priv.switchOn : ""}`} onClick={() => onChange(!on)} />
  );
}

/** One idea: its name, a line of help under it, and one control beside it (`stack`: under it, for a wide button). */
function Row({ title, help, note, guide, stack, children }: { title: string; help?: string; note?: string; guide?: string; stack?: boolean; children?: React.ReactNode }) {
  return (
    <div className={`${priv.row} ${stack ? priv.rowStack : ""}`} data-guide={guide}>
      <div className={priv.rowText}>
        <div className={priv.rowTitle}>{title}</div>
        {help ? <p className={priv.rowHelp}>{help}</p> : null}
        {note ? <p className={priv.rowHelp}>{note}</p> : null}
      </div>
      {children ? <div className={priv.rowControl}>{children}</div> : null}
    </div>
  );
}

/** Faces, Cameras and What is kept: the switches of GET/PUT /vault/policy. `change` saves one of them. */
function Controls({ policy, cameras, busy, change }: { policy: VaultPolicy; cameras: string[]; busy: boolean; change: (body: Partial<VaultPolicy>) => void }) {
  const [days, setDays] = useState(policy.keep_days ?? KEEP_FOREVER);
  useEffect(() => setDays(policy.keep_days ?? KEEP_FOREVER), [policy.keep_days]);

  // Whether this screen could get the owner's view ticket: it decides what the first row can honestly promise.
  const [ticket, setTicket] = useState(() => viewTicketState());
  useEffect(() => {
    let alive = true;
    void viewTicketReady().then(() => alive && setTicket(viewTicketState()));
    return () => {
      alive = false;
    };
  }, []);
  const liveHelp =
    "Show faces on my own signed-in screen. Everyone else sees the live view blurred." +
    (ticket.problem === "not_signed_in"
      ? " This screen is not signed in, so it shows the blurred view."
      : ticket.problem === "box_has_no_token"
        ? " This box has no sign-in token yet, so only a screen plugged into the box itself can show faces."
        : "");

  const togglePause = (camera: string, pause: boolean) =>
    change({ paused_cameras: pause ? [...policy.paused_cameras, camera] : policy.paused_cameras.filter((c) => c !== camera) });

  return (
    <>
      <section className={ui.section} data-guide="privacy:faces">
        <h2 className={ui.sectionTitle}>Faces</h2>
        <div className={priv.card}>
          <Row title="Your live view" help={liveHelp} guide="privacy:live-view">
            <Switch on={policy.live_faces === "clear"} label="Show faces on my own signed-in screen" disabled={busy} onChange={(v) => change({ live_faces: v ? "clear" : "blurred" })} />
          </Row>
          {policy.blur_faces ? (
            <Row title="Saved pictures" help="Always blurred before they are saved." guide="privacy:saved-pictures">
              <span className={priv.pill}>Always</span>
            </Row>
          ) : (
            <Row
              title="Saved pictures"
              help="This box is saving pictures with faces showing. Pictures it already saved stay as they are until you erase them below."
              guide="privacy:saved-pictures"
              stack
            >
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={() => change({ blur_faces: true })}>
                Blur saved pictures from now on
              </button>
            </Row>
          )}
          <Row title="Anything that leaves this box" help="Always blurred." guide="privacy:leaves">
            <span className={priv.pill}>Always</span>
          </Row>
        </div>
      </section>

      <section className={ui.section} data-guide="privacy:cameras">
        <h2 className={ui.sectionTitle}>Cameras</h2>
        <div className={priv.card}>
          {cameras.map((camera) => (
            <Row key={camera} title={`Pause the ${cameraName(camera).toLowerCase()}`} help="A paused camera is not looked at or saved. Live shows nothing for it.">
              <Switch on={policy.paused_cameras.includes(camera)} label={`Pause ${cameraName(camera)}`} disabled={busy} onChange={(v) => togglePause(camera, v)} />
            </Row>
          ))}
        </div>
      </section>

      <section className={ui.section} data-guide="privacy:kept">
        <h2 className={ui.sectionTitle}>What is kept</h2>
        <div className={priv.card}>
          <div className={`${priv.row} ${priv.rowStack}`}>
            <div className={priv.rowText}>
              <div className={priv.rowTitle}>
                Keep saved pictures for <span className={priv.sliderValue}>{days >= KEEP_FOREVER ? "ever" : `${days} day${days === 1 ? "" : "s"}`}</span>
              </div>
              <p className={priv.rowHelp}>Slide all the way right to keep them forever.</p>
            </div>
            <input
              className={priv.slider}
              type="range"
              min={1}
              max={KEEP_FOREVER}
              value={days}
              disabled={busy}
              aria-label="Days to keep"
              onChange={(e) => setDays(Number(e.target.value))}
              onPointerUp={() => change({ keep_days: days >= KEEP_FOREVER ? null : days })}
              onKeyUp={() => change({ keep_days: days >= KEEP_FOREVER ? null : days })}
            />
            <p className={priv.rowHelp}>Older pictures are removed once an hour. What Atlas wrote down about them stays until you erase memory below.</p>
          </div>
        </div>
      </section>
    </>
  );
}

function EgressCard({ egress }: { egress: Egress }) {
  if (!egress.enforced) {
    return <Empty title="No firewall on this box">Where the firewall runs (the Jetson), this lists what each group sent and what was blocked.</Empty>;
  }
  const sent = egress.allowed.filter((g) => g.bytes > 0 || !["lan", "ntp", "dns"].includes(g.name));
  const pictures = picturesLine(egress);
  return (
    <div className={priv.card} data-guide="privacy:proof">
      <Row title={egress.mode === "offline" ? "Offline: only your own network" : "Online: only listed places"} help={egress.stale ? "The numbers are old: the firewall service has stopped reporting. Its rules still hold." : undefined}>
        <span className={`${priv.pill} ${egress.mode === "offline" ? priv.pillWarn : ""}`}>{egress.mode === "offline" ? "Offline" : "Online"}</span>
      </Row>
      <Row title="Video sent: none" help={`${pictures ?? "Camera pictures leave only in a training export you approve"}. To the internet in total: ${bytes(internetSummary(egress).total)}.`}>
        <span className={priv.pill}>Proof</span>
      </Row>
      <div className={`${priv.row} ${priv.rowStack}`}>
        <div className={priv.rowTitle}>Where it went</div>
      <table className={priv.table}>
        <tbody>
          {sent.map((g) => (
            <tr key={g.name}>
              <td>
                {groupLabel(g.name)}
                {g.hosts.length ? <span className={priv.hosts}>{g.hosts.join(", ")}</span> : null}
              </td>
              <td className={priv.num}>{bytes(g.bytes)}</td>
            </tr>
          ))}
          <tr>
            <td>
              Blocked
              {egress.recent_blocks?.length ? (
                <span className={priv.hosts}>
                  {egress.recent_blocks
                    .slice(0, 4)
                    .map((b) => `${b.dst}:${b.port}`)
                    .join(", ")}
                </span>
              ) : null}
            </td>
            <td className={priv.num}>{egress.blocked} tries</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
  );
}

function DeleteSection() {
  const [scope, setScope] = useState<WipeScope>("all");
  const [token, setToken] = useState<WipeToken | null>(null);
  const [result, setResult] = useState<WipeResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const what = SCOPES.find((s) => s.id === scope)?.what ?? "";

  async function run<T>(fn: () => Promise<T>, then: (v: T) => void) {
    setBusy(true);
    setError(null);
    try {
      then(await fn());
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
      setToken(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className={ui.section} data-guide="privacy:erase">
      <h2 className={ui.sectionTitle}>Erase</h2>
      <div className={`${priv.card} ${priv.danger} ${priv.erase}`}>
        <div className={priv.rowText}>
          <div className={priv.rowTitle}>Delete what Atlas has stored</div>
          <p className={priv.rowHelp}>This cannot be undone. You confirm in a second step.</p>
        </div>
        <div className={ui.chips} role="radiogroup" aria-label="What to delete">
          {SCOPES.map((s) => (
            <button key={s.id} type="button" role="radio" aria-checked={scope === s.id} className={`${ui.chip} ${scope === s.id ? ui.chipOn : ""}`} disabled={busy || token !== null} onClick={() => setScope(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        {error ? <WriteError error={error} /> : null}
        {!token && !result ? (
          <div className={priv.actions}>
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => run(() => api.wipeRequest(scope), setToken)}>
              Delete…
            </button>
          </div>
        ) : null}
        {token ? (
          <div className={priv.dangerBox} role="alertdialog" aria-label="Confirm delete">
            <p>
              This will permanently delete <strong>{what}</strong>. The confirmation is valid for two minutes.
            </p>
            <div className={priv.actions}>
              <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy} onClick={() => run(() => api.wipe(token.token), (r) => { setResult(r); setToken(null); })}>
                Yes, delete for good
              </button>
              <button type="button" className={ui.btn} disabled={busy} onClick={() => setToken(null)}>
                Cancel
              </button>
            </div>
          </div>
        ) : null}
        {result ? (
          <div className={priv.dangerBox} role="status">
            <p>
              {result.done ? "Deleted." : "The delete was accepted but no report came back."}{" "}
              {result.removed ? Object.entries(result.removed).map(([k, n]) => `${k}: ${n} files`).join(", ") : ""}
            </p>
            <p className={priv.rowHelp}>Restart the box&apos;s app so it forgets what it had already loaded.</p>
            <div className={priv.actions}>
              <button type="button" className={ui.btn} onClick={() => setResult(null)}>
                Done
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function PrivacyPage() {
  const policy = useFetch((signal) => api.vaultPolicy(signal));
  const egress = useFetch((signal) => api.egress(signal));
  const [current, setCurrent] = useState<VaultPolicy | null>(null);
  const shown = current ?? policy.data ?? null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  /** Save one switch of the policy; every section's controls go through here, so one is saved at a time. */
  async function change(body: Partial<VaultPolicy>) {
    setBusy(true);
    setError(null);
    try {
      setCurrent(await api.setVaultPolicy(body));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  }

  // The cameras to offer: the ones in the zone file (camera ids in config/cameras.yaml must match it), plus any already paused.
  const cameras = Array.from(new Set([...Object.keys(ZONES), ...(shown?.paused_cameras ?? [])]));

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Privacy</h1>
        <p className={ui.muted}>What Atlas keeps, what it may send, and how to delete it.</p>
      </div>
      <ProfileHeaderSlot />
      {policy.loading && !shown ? <Loading what="your privacy settings" /> : null}
      {policy.error && !shown ? <ErrorBox what="your privacy settings" error={policy.error} onRetry={policy.reload} /> : null}
      {error ? <WriteError error={error} /> : null}
      {shown ? <Controls policy={shown} cameras={cameras} busy={busy} change={change} /> : null}
      <section className={ui.section} data-guide="privacy:left">
        <h2 className={ui.sectionTitle}>What has left this box</h2>
        {shown ? (
          <div className={priv.card}>
            <Row
              title="Offline mode"
              help="Nothing leaves your own network. Atlas keeps seeing, answering and alerting; connected apps and the learning lab pause."
              guide="privacy:offline"
            >
              <Switch on={shown.offline} label="Offline mode" disabled={busy} onChange={(v) => change({ offline: v })} />
            </Row>
          </div>
        ) : null}
        {egress.data ? <EgressCard egress={egress.data} /> : egress.error ? <ErrorBox what="the firewall numbers" error={egress.error} onRetry={egress.reload} /> : <Loading what="the firewall numbers" />}
      </section>
      <DeleteSection />
    </>
  );
}
