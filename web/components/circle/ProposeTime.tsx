"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";

/**
 * "Propose a time": start a plan with the Circle from the phone.
 *
 * POST /api/circle/proposals only raises an approval card on this box. Nothing is sent to friends
 * until that card is approved; after that each friend's Atlas answers free, busy or another time
 * from its own calendar, and the agreed event shows up under "Shared calendar".
 */

const HOURS = [1, 2, 3];

export function ProposeTime() {
  const [title, setTitle] = useState("");
  const [start, setStart] = useState("");
  const [hours, setHours] = useState(1);
  const [location, setLocation] = useState("");
  const [invitees, setInvitees] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [proposed, setProposed] = useState<string | null>(null);

  const begins = start ? new Date(start) : null; // datetime-local: the phone's own time zone
  const ready = Boolean(title.trim()) && begins !== null && !Number.isNaN(begins.getTime());

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready || !begins || busy) return;
    setBusy(true);
    setError(null);
    setProposed(null);
    try {
      await api.proposeTime({
        title: title.trim(),
        start: begins.toISOString(),
        end: new Date(begins.getTime() + hours * 3_600_000).toISOString(),
        location: location.trim() || undefined,
        invitees: invitees.split(",").map((m) => m.trim()).filter(Boolean),
      });
      setProposed(title.trim());
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Propose a time</h2>
      <form className={ui.card} onSubmit={submit}>
        <p className={`${ui.muted} ${ui.small}`}>
          Friends&apos; boxes check their own calendars and answer free, busy or another time. You approve the plan before anything is sent.
        </p>
        <input className={ui.input} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="What, e.g. Study session" aria-label="What" required />
        <input className={ui.input} type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} aria-label="Starts" required />
        <div className={ui.chips} role="group" aria-label="For how long">
          {HOURS.map((h) => (
            <button key={h} type="button" className={`${ui.chip} ${hours === h ? ui.chipOn : ""}`} aria-pressed={hours === h} onClick={() => setHours(h)}>
              {h} {h === 1 ? "hour" : "hours"}
            </button>
          ))}
        </div>
        <input className={ui.input} value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Where (optional)" aria-label="Where" />
        <input
          className={ui.input}
          value={invitees}
          onChange={(e) => setInvitees(e.target.value)}
          placeholder="Who, by member id: dom, xander"
          aria-label="Who to invite, member ids separated by commas"
        />
        <p className={`${ui.muted} ${ui.small}`}>The plan is agreed once everyone named here answers free.</p>
        {error ? <WriteError error={error} /> : null}
        {proposed ? (
          <p className={ui.nudge} role="status">
            <strong>{proposed}</strong> is waiting for you. Nothing is sent to the Circle until you approve it: open{" "}
            <Link href="/approvals">Approvals</Link>.
          </p>
        ) : null}
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !ready}>
          {busy ? "Proposing…" : "Propose this time"}
        </button>
      </form>
    </section>
  );
}
