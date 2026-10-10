"use client";

import { useEffect, useState } from "react";
import priv from "@/components/privacy/privacy.module.css";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Egress, type VaultPolicy, type WipeResult, type WipeScope, type WipeToken } from "@/lib/api";
import { cameraName } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { ZONES } from "@/lib/zones.generated";

/**
 * Privacy: what the box keeps, what it may send, and the delete button.
 *
 * Everything here is the Vault API: GET/PUT /vault/policy (blur, paused cameras, offline, retention),
 * GET /vault/egress (what the firewall let out, per group) and the two-step POST /vault/wipe/request + /vault/wipe.
 * "Offline" also pauses connected apps; the box keeps seeing, answering and alerting because that is all local.
 */

const KEEP_FOREVER = 91; // the slider's last stop
const SCOPES: { id: WipeScope; label: string; what: string }[] = [
  { id: "all", label: "Everything", what: "all pictures, the memory, exports and the event log" },
  { id: "keyframes", label: "Pictures only", what: "the saved camera pictures" },
  { id: "memory", label: "Memory only", what: "what Atlas remembers (the memory database)" },
];

const GROUP_LABEL: Record<string, string> = {
  lan: "Your own network",
  ntp: "Clock sync",
  dns: "Name lookups",
  hub: "Learning lab",
  system: "System check",
  neon: "Circle (shared state)",
  ntfy: "Phone alerts relay",
};

function groupLabel(name: string): string {
  if (name.startsWith("connector:")) return `${name.slice(10).replace(/^./, (c) => c.toUpperCase())} app`;
  if (name.startsWith("extra:")) return name.slice(6);
  return GROUP_LABEL[name] ?? name;
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function Switch({ on, label, onChange, disabled }: { on: boolean; label: string; onChange: (next: boolean) => void; disabled?: boolean }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-label={label} disabled={disabled} className={`${priv.switch} ${on ? priv.switchOn : ""}`} onClick={() => onChange(!on)} />
  );
}

function Row({ title, help, children }: { title: string; help?: string; children: React.ReactNode }) {
  return (
    <div className={priv.row}>
      <div className={priv.rowText}>
        <div className={priv.rowTitle}>{title}</div>
        {help ? <p className={priv.rowHelp}>{help}</p> : null}
      </div>
      {children}
    </div>
  );
}

function Controls({ policy, cameras, onPolicy }: { policy: VaultPolicy; cameras: string[]; onPolicy: (p: VaultPolicy) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [days, setDays] = useState(policy.keep_days ?? KEEP_FOREVER);
  useEffect(() => setDays(policy.keep_days ?? KEEP_FOREVER), [policy.keep_days]);

  async function change(body: Partial<VaultPolicy>) {
    setBusy(true);
    setError(null);
    try {
      onPolicy(await api.setVaultPolicy(body));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  }

  const togglePause = (camera: string, pause: boolean) =>
    change({ paused_cameras: pause ? [...policy.paused_cameras, camera] : policy.paused_cameras.filter((c) => c !== camera) });

  return (
    <>
      {error ? <WriteError error={error} /> : null}
      <section className={priv.section}>
        <h2>Pictures</h2>
        <div className={priv.card}>
          <Row title="Blur faces before saving" help="Faces are blurred on the box before a picture is written to disk. Nothing else is changed.">
            <Switch on={policy.blur_faces} label="Blur faces" disabled={busy} onChange={(v) => change({ blur_faces: v })} />
          </Row>
          {cameras.map((camera) => (
            <Row key={camera} title={`Pause the ${cameraName(camera).toLowerCase()}`} help="A paused camera is not looked at or saved. Live shows nothing for it.">
              <Switch on={policy.paused_cameras.includes(camera)} label={`Pause ${cameraName(camera)}`} disabled={busy} onChange={(v) => togglePause(camera, v)} />
            </Row>
          ))}
          <div className={priv.row}>
            <div className={priv.rowText} style={{ flex: 1 }}>
              <div className={priv.rowTitle}>
                Keep saved pictures for <span className={priv.sliderValue}>{days >= KEEP_FOREVER ? "ever" : `${days} day${days === 1 ? "" : "s"}`}</span>
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
              <p className={priv.rowHelp}>The right-hand end means forever. Older pictures are removed once an hour; what Atlas wrote down about them stays until you delete memory below.</p>
            </div>
          </div>
        </div>
      </section>

      <section className={priv.section}>
        <h2>The internet</h2>
        <div className={priv.card}>
          <Row title="Offline mode" help="Nothing leaves your own network. Atlas keeps seeing, answering and alerting; connected apps and the learning lab pause.">
            <Switch on={policy.offline} label="Offline mode" disabled={busy} onChange={(v) => change({ offline: v })} />
          </Row>
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
  return (
    <div className={priv.card}>
      <Row title={egress.mode === "offline" ? "Offline: only your own network" : "Online: only listed places"} help={egress.stale ? "The numbers are old: the firewall service has stopped reporting. Its rules still hold." : undefined}>
        <span className={`${priv.pill} ${egress.mode === "offline" ? priv.pillWarn : ""}`}>{egress.mode === "offline" ? "Offline" : "Online"}</span>
      </Row>
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
    <section className={priv.section}>
      <h2>Delete</h2>
      <div className={`${priv.card} ${priv.danger}`}>
        <Row title="Delete what Atlas has stored" help="This cannot be undone. You confirm in a second step.">
          <span />
        </Row>
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

  // The cameras to offer: the ones in the zone file (camera ids in config/cameras.yaml must match it), plus any already paused.
  const cameras = Array.from(new Set([...Object.keys(ZONES), ...(shown?.paused_cameras ?? [])]));

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Privacy</h1>
        <p className={ui.muted}>What Atlas keeps, what it may send, and how to delete it.</p>
      </div>
      {policy.loading && !shown ? <Loading what="your privacy settings" /> : null}
      {policy.error && !shown ? <ErrorBox what="your privacy settings" error={policy.error} onRetry={policy.reload} /> : null}
      {shown ? <Controls policy={shown} cameras={cameras} onPolicy={setCurrent} /> : null}
      <section className={priv.section}>
        <h2>What left the box</h2>
        {egress.data ? <EgressCard egress={egress.data} /> : egress.error ? <ErrorBox what="the firewall numbers" error={egress.error} onRetry={egress.reload} /> : <Loading what="the firewall numbers" />}
      </section>
      <DeleteSection />
    </>
  );
}
