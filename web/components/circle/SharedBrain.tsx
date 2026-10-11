"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type BrainLastAnswer, type BrainPower, type CircleMembers } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { BrainPair } from "./BrainPair";
import c from "./circle.module.css";

/**
 * Shared brain: the Circle's big model, which every friend's box can ask (G4's group brain). At the top the brain as
 * two friends (BrainPair, G6): one 122B model split across the lab's two AMD Strix Halo pods, each pod's watts, the
 * tunnel between them and how fast it answers. Below, how many friends' boxes are on the network, the head pod's
 * watts as a scrolling trace with its idle floor, and what the last answer cost.
 *
 * GET /api/circle/brain/power is polled once a second: the box relays the pod's own power sampler and keeps the hub
 * token to itself. The last answer's joules are the pod's power integrated over the group brain's own window
 * (circle.brain.asked to circle.brain.answered), gross; the line updates when a question is answered.
 */

const SPAN_MS = 120_000;
const POLL_MS = 1000;
const fmt = (n: number, d = 0) => n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

type Point = { t: number; w: number };

export function SharedBrain({ members }: { members: CircleMembers | undefined }) {
  const status = useFetch((s) => api.circleBrain(s).catch(() => null));
  const [power, setPower] = useState<BrainPower | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const lastSeq = useRef<number | undefined>(undefined);
  useAtlasEvents(["circle.brain."], status.reload);

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const p = await api.circleBrainPower(lastSeq.current);
        if (stop) return;
        setPower(p);
        if (p.samples.length) {
          lastSeq.current = p.samples[p.samples.length - 1]!.seq;
          const fresh = p.samples.map((s) => ({ t: Date.parse(s.ts), w: s.w }));
          setPoints((old) => {
            const all = [...old, ...fresh];
            const newest = all[all.length - 1]!.t;
            return all.filter((x) => x.t >= newest - SPAN_MS - 5000);
          });
        } else if (p.last_seq !== null && lastSeq.current === undefined) lastSeq.current = p.last_seq;
      } catch {
        // the next tick tries again; the card says the trace is stale
      }
      if (!stop) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, []);

  const friends = (members?.members ?? []).filter((m) => !m.mine);
  const online = friends.filter((m) => m.has_box || m.place).length;
  const now = points.length ? points[points.length - 1] : null;
  const idle = power?.idle_w ?? 10.27;
  const last = power?.last_answer ?? null;
  const optedIn = status.data?.opted_in ?? false;
  // What the Circle already shares (named places with their end time) goes along as facts; nothing else does.
  const facts = (members?.members ?? [])
    .filter((m) => m.place)
    .map((m) => `${m.display_name} is at ${m.place!.place} until ${time(m.place!.expires_at)}`);

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      if (!optedIn) {
        await api.circleBrainConsent(true);
        status.reload();
      }
      const r = await api.circleBrainAsk(q, facts);
      setAnswer(r.answer);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={ui.section} aria-labelledby="shared-brain">
      <h2 className={ui.sectionTitle} id="shared-brain">
        Shared brain
      </h2>
      <div className={`${ui.card} ${c.brain}`}>
        <BrainPair />
        <div className={ui.stats}>
          <div className={ui.stat}>
            <div className={ui.statNum}>
              {online} <span className={c.brainUnit}>of {friends.length}</span>
            </div>
            <div className={ui.statLabel}>friends&apos; boxes on the network</div>
          </div>
          <div className={ui.stat}>
            <div className={`${ui.statNum} ${c.brainWatts}`}>
              {now ? fmt(now.w) : "--"} <span className={c.brainUnit}>W</span>
            </div>
            <div className={ui.statLabel}>the head pod right now (idle floor {idle} W)</div>
          </div>
        </div>
        <Trace points={points} idle={idle} last={last} offset={power?.offset_s ?? null} />
        <CostLine last={last} />
        {power?.error ? <p className={`${ui.small} ${ui.muted}`}>{power.error}</p> : null}
        <form className={c.brainAsk} onSubmit={ask}>
          <input className={ui.input} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Where should we meet tonight?" aria-label="Ask the shared brain" maxLength={600} />
          <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !question.trim() || status.data?.configured === false}>
            {busy ? "Asking..." : "Ask the brain"}
          </button>
        </form>
        {!optedIn && status.data?.configured ? <p className={`${ui.small} ${ui.muted}`}>Asking turns on sending your question, as text only, to the shared brain, with the places friends already share.</p> : null}
        {answer ? (
          <p className={c.brainAnswer}>
            {answer} <span className={ui.muted}>{status.data?.badge ?? "Asked the group brain"}</span>
          </p>
        ) : null}
        {error ? <WriteError error={error} /> : null}
      </div>
    </section>
  );
}

function CostLine({ last }: { last: BrainLastAnswer | null }) {
  if (!last) return <p className={c.brainCost}>No answer from the brain yet.</p>;
  if (last.pending) return <p className={c.brainCost}>Measuring the last answer...</p>;
  if (last.joules == null) return <p className={c.brainCost}>{last.reason ?? "The last answer could not be measured."}</p>;
  return (
    <div>
      <p className={c.brainCost}>that answer cost {fmt(last.joules)} J</p>
      <p className={`${ui.small} ${ui.muted}`}>
        {fmt(last.seconds ?? 0, 1)} s at {fmt(last.mean_w ?? 0)} W average on the head pod, gross; {fmt(Math.max(0, last.above_idle_j ?? 0))} J above its {last.idle_w} W idle floor.
        {last.low_confidence && last.reasons?.length ? ` Low confidence: ${last.reasons.join("; ")}.` : ""}
      </p>
    </div>
  );
}

function Trace({ points, idle, last, offset }: { points: Point[]; idle: number; last: BrainLastAnswer | null; offset: number | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const css = getComputedStyle(canvas);
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const g = canvas.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const muted = css.getPropertyValue("--muted").trim() || "#98a5c3";
    const line = css.getPropertyValue("--line").trim() || "rgba(150,178,255,0.14)";
    const accent = css.getPropertyValue("--accent").trim() || "#5b8cff";
    const padR = 44, padB = 18, padT = 6;
    const nowT = points.length ? points[points.length - 1]!.t : Date.now();
    const vis = points.filter((p) => p.t >= nowT - SPAN_MS);
    const maxW = Math.max(40, ...vis.map((p) => p.w)) * 1.12;
    const x = (t: number) => (W - padR) * (1 - (nowT - t) / SPAN_MS);
    const y = (w: number) => padT + (H - padT - padB) * (1 - w / maxW);
    g.font = "12px system-ui, sans-serif";
    g.fillStyle = muted;
    g.strokeStyle = line;
    g.lineWidth = 1;
    const step = maxW > 80 ? 50 : 20;
    for (let w = 0; w <= maxW; w += step) {
      g.beginPath();
      g.moveTo(0, y(w));
      g.lineTo(W - padR, y(w));
      g.stroke();
      g.fillText(`${w} W`, W - padR + 6, y(w) + 4);
    }
    g.fillText("2 min ago", 0, H - 4);
    g.fillText("now", W - padR - 22, H - 4);
    if (last && !last.pending && last.asked && last.answered && offset !== null) {
      const a = Date.parse(last.asked) + offset * 1000;
      const b = Date.parse(last.answered) + offset * 1000;
      if (b >= nowT - SPAN_MS) {
        g.fillStyle = "rgba(255, 214, 102, 0.18)";
        g.fillRect(Math.max(0, x(a)), padT, Math.max(2, x(b) - Math.max(0, x(a))), H - padT - padB);
      }
    }
    g.setLineDash([5, 5]);
    g.strokeStyle = muted;
    g.beginPath();
    g.moveTo(0, y(idle));
    g.lineTo(W - padR, y(idle));
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = muted;
    g.fillText(`idle ${idle} W`, 4, y(idle) - 4);
    if (vis.length > 1) {
      g.beginPath();
      vis.forEach((p, i) => (i ? g.lineTo(x(p.t), y(p.w)) : g.moveTo(x(p.t), y(p.w))));
      g.strokeStyle = accent;
      g.lineWidth = 2.5;
      g.stroke();
      g.lineTo(x(vis[vis.length - 1]!.t), y(0));
      g.lineTo(x(vis[0]!.t), y(0));
      g.closePath();
      g.globalAlpha = 0.14;
      g.fillStyle = accent;
      g.fill();
      g.globalAlpha = 1;
    }
  }, [points, idle, last, offset]);
  return <canvas ref={ref} className={c.brainTrace} role="img" aria-label="The head pod's power over the last two minutes" />;
}
