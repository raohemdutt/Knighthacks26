"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Keyframe } from "@/components/Keyframe";
import ui from "@/components/ui.module.css";
import {
  CHANGE_LABEL,
  CHAPTERS,
  type ChapterKey,
  type LabChange,
  type LabItem,
  type LabMeasured,
  type LabNum,
  type LabStory,
  LIVE_STAGES,
  PHONE_ASSUMPTION,
  PLAY_SECONDS,
  STATUS_LABEL,
  type StageState,
  chapterAt,
  chipShort,
  clamp01,
  clock,
  duration,
  elapsedSeconds,
  energyUpTo,
  fileSize,
  isLive,
  isNum,
  stageStates,
  pct,
  phoneCharges,
  show,
  within,
} from "@/lib/lab";
import { GateChart, TrainingChart } from "./Charts";
import lab from "./lab.module.css";
import { LearningBuddies } from "./LearningBuddies";

/* ---------- small pieces ---------- */

/** Where a measured trainer row ran, by the chip the row itself gives: never an R9700 unless it was one. */
function smokeChip(m: LabMeasured | undefined): string {
  if (!m?.other_chip) return "the lab's trainer";
  if (m.chip?.role === "earlier") return `one ${m.chip.name}, the lab's earlier card`;
  return m.chip ? `${chipShort(m.chip)}, not the lab's own hardware` : "another chip, not the lab's own hardware";
}

/** One number as a tile. A number the log does not hold says so, with the reason, instead of showing a dash. */
function Stat({ n, label, value, tone }: { n: LabNum; label?: string; value?: string; tone?: "good" | "bad" }) {
  const has = isNum(n);
  return (
    <div className={lab.stat} title={has ? `Source: ${n.source?.ref ?? ""}` : (n.reason ?? "")}>
      <span className={`${has ? lab.statNum : lab.statNone} ${tone === "good" ? lab.good : tone === "bad" ? lab.bad : ""}`}>
        {has ? (value ?? show(n)) : "Not in the log"}
        {has && n.low_confidence ? <span className={lab.low}>low confidence</span> : null}
      </span>
      <span className={lab.statLabel}>{label ?? n.label}</span>
      {!has && n.reason ? <span className={lab.note}>{n.reason}</span> : null}
    </div>
  );
}

function Section({ n, title, now, children, className }: { n: number; title: string; now: boolean; children: React.ReactNode; className?: string }) {
  return (
    <section className={`${lab.sec} ${now ? lab.secNow : ""} ${className ?? ""}`} aria-labelledby={`lab-sec-${n}`}>
      <div className={lab.secHead}>
        <span className={lab.secNum} aria-hidden="true">
          {n}
        </span>
        <h2 id={`lab-sec-${n}`}>{title}</h2>
      </div>
      {children}
    </section>
  );
}

export function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return createPortal(  // over the viewport: a transformed page section would otherwise trap position: fixed
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>{title}</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
    , document.body);
}

/* ---------- 1. taught from ---------- */

/** A teacher label as one line: {"tool_calls":[{"name":"last_seen","arguments":{"label":"keys"}}]} -> last_seen(label="keys"). Other labels as they are. */
function label(target: string): string {
  try {
    const calls = (JSON.parse(target) as { tool_calls?: { name: string; arguments: Record<string, unknown> }[] }).tool_calls;
    if (Array.isArray(calls) && calls.length)
      return calls
        .map(
          (c) =>
            `${c.name}(${Object.entries(c.arguments ?? {})
              .map(([k, v]) => `${k}="${k === "start" || k === "end" ? String(v).slice(0, 16) : String(v)}"`)
              .join(", ")})`,
        )
        .join("; ");
  } catch {
    /* free text */
  }
  return target;
}

function TaughtFrom({ story }: { story: LabStory }) {
  const d = story.dataset;
  const rows = d.rows ?? [];
  const train = isNum(d.train) ? d.train.value : null;
  const held = isNum(d.held_out) ? d.held_out.value : null;
  return (
    <>
      <p className={lab.lead}>
        {isNum(d.examples) ? (
          <>
            <b>{d.examples.value}</b> examples left the box for this task
          </>
        ) : (
          "The examples this run trained on"
        )}
        {d.approval.approved ? (
          <>
            . You approved this export at <b>{clock(d.approval.approved)}</b>.
          </>
        ) : (
          <>. {d.approval.approved_reason ?? d.examples.reason}</>
        )}
      </p>
      {d.approval.reason ? <p className={lab.note}>The approval card said: “{d.approval.reason}”.</p> : null}
      {train !== null && held !== null ? (
        <>
          <div className={lab.split} role="img" aria-label={`${train} examples to learn from, ${held} held back for the test`}>
            <div className={lab.splitTrain} style={{ flex: train }}>
              {train} to learn from
            </div>
            <div className={lab.splitHeld} style={{ flex: held }}>
              {held} held back
            </div>
          </div>
          <p className={lab.note}>
            The held-back questions are the test. The model never trains on them
            {isNum(d.held_out_groups) ? `, and they cover ${d.held_out_groups.value} different cases, each asked in several wordings` : ""}.
          </p>
        </>
      ) : (
        <p className={lab.gap}>
          <b>No split to show.</b> {d.train.reason ?? d.held_out.reason}
        </p>
      )}
      {rows.length ? (
        <div className={lab.cards} aria-label="Examples the model learned from">
          {rows.map((r) => (
            <div key={r.id} className={lab.example}>
              {r.frame_ref.startsWith("frames:") ? <Keyframe frameRef={r.frame_ref} compact /> : null}
              <span className={lab.exampleQ}>{r.context ?? "No caption"}</span>
              {r.target ? <span className={`${lab.code} ${lab.exampleA}`}>{label(r.target)}</span> : <span className={lab.note}>Not labeled yet</span>}
              {r.verdict ? <span className={lab.note}>You marked it thumbs {r.verdict}</span> : null}
            </div>
          ))}
        </div>
      ) : (
        <p className={lab.gap}>
          <b>No example rows.</b> {d.rows_reason}
        </p>
      )}
      {rows.length ? <p className={lab.note}>{rows.length} of the examples it learned from. Pictures, when a task has them, are the box&apos;s blurred keyframes.</p> : null}
    </>
  );
}

/* ---------- 2. training ---------- */

function Training({ story, frac, measured }: { story: LabStory; frac: number; measured: LabMeasured[] }) {
  const t = story.training;
  const p = story.power;
  const [hint, setHint] = useState(false);
  const tMax = Math.max(isNum(t.wall_s) ? t.wall_s.value : 0, t.points.length ? t.points[t.points.length - 1]!.t : 0, 1);
  const tNow = frac * tMax;
  const seen = t.points.filter((x) => x.t <= tNow + 1e-6);
  const head = seen[seen.length - 1];
  const joules = p.points.length ? energyUpTo(p.points, tNow) : null;
  const wh = joules !== null ? joules / 3600 : null;
  const speed = measured.find((m) => m.metric === "trainer.samples_per_s");
  const first = measured.find((m) => m.metric === "trainer.smoke_loss_first");
  const last = measured.find((m) => m.metric === "trainer.smoke_loss_last");
  return (
    <>
      <p className={lab.lead}>
        {t.base ? (
          <>
            A small add-on (LoRA{t.lora_rank ? `, rank ${t.lora_rank}` : ""}) is trained on top of <b>{t.base}</b>
            {t.node ? ` on ${t.node}` : ""}. The base model is not changed.
          </>
        ) : (
          (t.points_reason ?? "The model practises on the examples; the loss falls as it gets them right.")
        )}
      </p>
      <div className={lab.stats}>
        <Stat n={t.steps} label={isNum(t.of) ? `of ${t.of.value} steps` : "steps so far"} value={head?.step != null ? String(head.step) : frac > 0 && isNum(t.steps) ? String(Math.round(frac * t.steps.value)) : "0"} />
        <Stat n={t.loss_last} label="loss now" value={head?.loss != null ? head.loss.toFixed(2) : "…"} />
        <Stat n={t.samples_per_s} label="training speed" />
        <Stat n={t.wall_s} label="training time" value={duration(tNow)} />
      </div>
      <TrainingChart training={t} power={p} frac={frac} />
      {wh !== null ? (
        <div className={lab.energy}>
          <span>
            <b>{wh.toFixed(1)} Wh</b> so far, about the energy of <b>{phoneCharges(wh)}</b>
            {p.energy_j.low_confidence ? <span className={lab.low}>low confidence</span> : null}
          </span>
          <button type="button" className={lab.hint} aria-expanded={hint} aria-label="How the phone comparison is worked out" title={PHONE_ASSUMPTION} onClick={() => setHint((h) => !h)}>
            ?
          </button>
          {hint ? <span className={lab.note}>{PHONE_ASSUMPTION}</span> : null}
        </div>
      ) : null}
      {p.points.length ? (
        <div className={lab.stats}>
          <Stat n={p.energy_j} label="training energy, whole run" />
          <Stat n={p.j_per_sample} label="energy per training sample" />
          <Stat n={p.mean_w} label="average power" />
        </div>
      ) : null}
      {speed || first ? (
        <div className={lab.measured}>
          <span className={lab.measuredTag}>Measured on our hardware</span>
          <span>
            A 20-step smoke run on {smokeChip(speed ?? first)}
            {speed?.node ? ` (${speed.node})` : ""}
            {speed ? (
              <>
                : <b>{speed.value} samples/s</b>
              </>
            ) : null}
            {first && last ? (
              <>
                , loss <b>{first.value}</b> to <b>{last.value}</b>
              </>
            ) : null}
            .
          </span>
          <span className={lab.note}>
            {[speed, first, last]
              .filter((m): m is LabMeasured => !!m)
              .map((m) => m.source?.ref)
              .join(", ")}
            . Not this run: a real measurement, shown so the run above can be judged against it. {(speed ?? first)?.chip?.role === "earlier" ? "The lab has since moved to" : "The lab is"} two Strix Halo pods (Radeon 8060S, gfx1151).
          </span>
        </div>
      ) : null}
    </>
  );
}

/* ---------- 3. before / after ---------- */

const ORDER: LabChange[] = ["broke", "fixed", "still_wrong", "still_right"];
type Filter = "changed" | "broke" | "fixed" | "still_wrong" | "all";

function BeforeAfter({ story, frac }: { story: LabStory; frac: number }) {
  const ev = story.eval;
  const [filter, setFilter] = useState<Filter>("changed");
  const [open, setOpen] = useState(false);
  // regressions first, so they are never under the fold; the playhead reveals answers in this same order
  const ordered = useMemo(() => [...(ev.items ?? [])].sort((a, b) => ORDER.indexOf(a.change) - ORDER.indexOf(b.change)), [ev.items]);
  if (!ev.items || !ev.counts) {
    return (
      <>
        <div className={lab.stats}>
          <Stat n={ev.scores.current} label="score before" />
          <Stat n={ev.scores.candidate} label="score after" />
          <Stat n={ev.scores.teacher} label="teacher's score" />
        </div>
        <p className={lab.gap}>
          <b>No per-question answers.</b> {ev.items_reason}
        </p>
      </>
    );
  }
  const n = ordered.length;
  const shown = Math.round(frac * n);
  const index = new Map(ordered.map((it, i) => [it.id, i]));
  const revealed = ordered.slice(0, shown);
  const count = (c: LabChange) => revealed.filter((i) => i.change === c).length;
  const total = (c: LabChange) => ordered.filter((i) => i.change === c).length;
  const beforeRight = ev.counts.before_right.value ?? 0;
  const afterRight = count("fixed") + count("still_right");
  const list = ordered.filter((i) => (filter === "all" ? true : filter === "changed" ? i.change === "fixed" || i.change === "broke" : i.change === filter));
  const visible = open ? list : list.slice(0, 6);
  const filters: [Filter, string][] = [
    ["changed", `Flipped (${total("fixed") + total("broke")})`],
    ["broke", `Regressions (${total("broke")})`],
    ["fixed", `Fixed (${total("fixed")})`],
    ["still_wrong", `Still wrong (${total("still_wrong")})`],
    ["all", `All (${n})`],
  ];
  return (
    <>
      <p className={lab.lead}>
        The same {n} held-back questions, answered by the model before training and by the new one.
        {shown < n ? ` Scoring: ${shown} of ${n}.` : ""}
      </p>
      <div className={lab.bars}>
        <span>Before</span>
        <div className={lab.bar} role="img" aria-label={`Before: ${beforeRight} of ${n} right`}>
          <div className={lab.barFill} style={{ width: `${(beforeRight / n) * 100}%` }} />
        </div>
        <b>
          {beforeRight} of {n}
        </b>
        <span>After</span>
        <div className={lab.bar} role="img" aria-label={`After: ${afterRight} of ${n} right so far`}>
          <div className={`${lab.barFill} ${lab.barAfter}`} style={{ width: `${(afterRight / n) * 100}%` }} />
        </div>
        <b>{shown === 0 ? "not scored yet" : `${afterRight} of ${shown < n ? shown : n}`}</b>
      </div>
      <div className={lab.tally}>
        <div className={`${lab.tallyCard} ${lab.tallyGood}`}>
          <span className={`${lab.statNum} ${lab.good}`}>+{count("fixed")}</span>
          <span className={lab.statLabel}>fixed: wrong before, right now</span>
        </div>
        <div className={`${lab.tallyCard} ${lab.tallyBad}`}>
          <span className={`${lab.statNum} ${lab.bad}`}>−{count("broke")}</span>
          <span className={lab.statLabel}>regressions: right before, wrong now</span>
        </div>
        <div className={lab.tallyCard}>
          <span className={lab.statNum}>{count("still_wrong")}</span>
          <span className={lab.statLabel}>still wrong</span>
        </div>
        <div className={lab.tallyCard}>
          <span className={lab.statNum}>{count("still_right")}</span>
          <span className={lab.statLabel}>still right</span>
        </div>
      </div>
      <div className={ui.chips} role="group" aria-label="Which questions to list">
        {filters.map(([key, label]) => (
          <button key={key} type="button" className={`${ui.chip} ${filter === key ? ui.chipOn : ""}`} aria-pressed={filter === key} onClick={() => setFilter(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className={lab.rows}>
        {list.length === 0 ? <p className={lab.waiting}>No question is in this group.</p> : null}
        {visible.map((it) => (
          <Question key={it.id} it={it} scored={(index.get(it.id) ?? 0) < shown} />
        ))}
      </div>
      {list.length > 6 ? (
        <button type="button" className={ui.btn} onClick={() => setOpen((o) => !o)}>
          {open ? "Show fewer" : `Show all ${list.length}`}
        </button>
      ) : null}
    </>
  );
}

function Question({ it, scored }: { it: LabItem; scored: boolean }) {
  const tone = !scored ? "" : it.change === "fixed" ? lab.qaFixed : it.change === "broke" ? lab.qaBroke : "";
  return (
    <div className={`${lab.qa} ${tone}`}>
      <div className={lab.qaHead}>
        <span>{it.question}</span>
        {scored ? (
          <span key="scored" className={`${lab.change} ${lab.flipIn} ${it.change === "fixed" ? lab.changeFixed : it.change === "broke" ? lab.changeBroke : ""}`}>{CHANGE_LABEL[it.change]}</span>
        ) : (
          <span className={lab.change}>Not scored yet</span>
        )}
      </div>
      {it.expected ? (
        <span className={lab.note}>
          Right answer: <span className={lab.code}>{it.expected}</span>
        </span>
      ) : null}
      <div className={lab.pair}>
        <div className={`${lab.ans} ${it.before.correct ? lab.ansRight : lab.ansWrong}`}>
          <span className={lab.ansLabel}>Before · {it.before.correct ? "right" : "wrong"}</span>
          <span className={lab.code}>{it.before.answer}</span>
        </div>
        {scored ? (
          <div key="after" className={`${lab.ans} ${lab.flipIn} ${it.after.correct ? lab.ansRight : lab.ansWrong}`}>
            <span className={lab.ansLabel}>After · {it.after.correct ? "right" : "wrong"}</span>
            <span className={lab.code}>{it.after.answer}</span>
          </div>
        ) : (
          <div className={`${lab.ans} ${lab.ansWait}`}>
            <span className={lab.ansLabel}>After</span>
            <span>…</span>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- 4. the gate ---------- */

function Gate({ story, on }: { story: LabStory; on: boolean }) {
  const ev = story.eval;
  if (!ev.verdict) return <p className={lab.waiting}>{ev.verdict_detail[0] ?? "The model has not been tested yet."}</p>;
  if (!on) return <p className={lab.waiting}>The gate decides once every held-back question has been scored.</p>;
  const good = ev.decision === "deploy" && story.status !== "rolled back";
  const alpha = ev.alpha ?? 0.05;
  return (
    <>
      <p className={`${lab.verdict} ${good ? "" : lab.verdictBad} ${lab.flipIn}`} role="status">
        {ev.verdict}
        {ev.decision === "reject" ? <span className={lab.verdictSub}>The model on the box did NOT change.</span> : null}
      </p>
      <GateChart ev={ev} />
      <div className={lab.stats}>
        <Stat n={ev.n_items} label={isNum(ev.n_groups) ? `questions, in ${ev.n_groups.value} groups` : "questions scored"} value={isNum(ev.n_items) ? String(ev.n_items.value) : undefined} />
        <Stat n={ev.noise} label="run-to-run noise" value={isNum(ev.noise) ? `±${pct(ev.noise.value, 1)}` : undefined} />
        <Stat
          n={ev.p_value}
          label={`p-value (needs to be below ${alpha})`}
          value={isNum(ev.p_value) ? String(Number(ev.p_value.value.toPrecision(2))) : undefined}
          tone={isNum(ev.p_value) ? (ev.p_value.value < alpha ? "good" : "bad") : undefined}
        />
      </div>
      <ul className={lab.why}>
        {ev.verdict_detail.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {ev.test || ev.decode ? (
        <p className={lab.note}>
          {ev.test ? `Test: ${ev.test}. ` : ""}
          {ev.decode ? `All three models were scored the same way: ${ev.decode}.` : ""}
        </p>
      ) : null}
    </>
  );
}

/* ---------- 5. on the box ---------- */

function OnTheBox({ story, frac, gated }: { story: LabStory; frac: number; gated: boolean }) {
  const { deploy, box, package: pkg } = story;
  if (!gated && story.eval.verdict) return <p className={lab.waiting}>The swap happens only after the gate clears the model.</p>;
  const packageLine = pkg.gguf ? (
    <div className={lab.stats}>
      <Stat n={pkg.bytes} label={`model file${pkg.quantization ? `, ${pkg.quantization}` : ""}`} value={isNum(pkg.bytes) ? fileSize(pkg.bytes.value) : undefined} />
      <div className={lab.stat}>
        <span className={`${lab.code} ${ui.breakWords}`}>{pkg.gguf}</span>
        <span className={lab.statLabel}>{pkg.sha256 ? `sha256 ${pkg.sha256.slice(0, 12)}…` : "no checksum in the log"}</span>
      </div>
    </div>
  ) : (
    <p className={lab.gap}>
      <b>No package.</b> {pkg.reason}
    </p>
  );
  if (!deploy.deployed) {
    return (
      <>
        <p className={`${lab.verdict} ${story.status === "not deployed" ? lab.verdictBad : ""}`}>
          {story.status === "not deployed" ? "Nothing changed on the box." : "Not on the box yet."}
          <span className={lab.verdictSub}>{deploy.deployed_reason}</span>
        </p>
        {packageLine}
      </>
    );
  }
  if (frac <= 0) return <p className={lab.waiting}>The swap happens only after the gate clears the model.</p>;
  const done = frac >= 0.5;
  return (
    <>
      <div className={lab.swap}>
        <div className={`${lab.model} ${done ? lab.modelOff : lab.modelOn}`}>
          <span className={lab.statLabel}>{done ? "was serving" : "serving"}</span>
          <span className={lab.modelName}>{deploy.previous_version ?? "the model before"}</span>
        </div>
        <svg className={lab.swapArrow} viewBox="0 0 44 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M4 12h34M30 5l8 7-8 7" />
        </svg>
        <div className={`${lab.model} ${done ? lab.modelOn : lab.modelOff}`}>
          <span className={lab.statLabel}>{done ? `serving since ${clock(deploy.deployed)}` : "loading"}</span>
          <span className={lab.modelName}>{deploy.model_version}</span>
        </div>
      </div>
      <div className={lab.swapBar} role="progressbar" aria-label="Hot-swap" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(Math.min(1, frac * 2) * 100)}>
        <div className={lab.swapFill} style={{ width: `${Math.min(1, frac * 2) * 100}%` }} />
      </div>
      {deploy.rolled_back ? (
        <p className={`${lab.verdict} ${lab.verdictBad}`}>
          Rolled back to {deploy.rolled_back.to ?? "the old model"} at {clock(deploy.rolled_back.ts)}.{deploy.rolled_back.reason ? <span className={lab.verdictSub}>{deploy.rolled_back.reason}</span> : null}
        </p>
      ) : null}
      <div className={lab.stats}>
        <Stat n={deploy.swap_s} label="hot-swap time (no answers meanwhile)" value={isNum(deploy.swap_s) ? `${deploy.swap_s.value.toFixed(1)} s` : undefined} />
      </div>
      {packageLine}
      {box.before && box.after ? (
        <table className={ui.table}>
          <thead>
            <tr>
              <th>On the box</th>
              <th className={ui.num}>Before</th>
              <th className={ui.num}>After</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Energy per answer</td>
              <BoxCell n={box.before.j_per_answer} />
              <BoxCell n={box.after.j_per_answer} show={done} />
            </tr>
            <tr>
              <td>Tokens per second</td>
              <BoxCell n={box.before.tok_s} />
              <BoxCell n={box.after.tok_s} show={done} />
            </tr>
            <tr>
              <td>Answers measured</td>
              <td className={ui.num}>{box.before.answers}</td>
              <td className={ui.num}>{done ? box.after.answers : "…"}</td>
            </tr>
          </tbody>
        </table>
      ) : null}
      <p className={lab.note}>The new model is the same size as the old one, so speed and energy per answer should barely move. What changes is how often it is right.</p>
    </>
  );
}

function BoxCell({ n, show: visible = true }: { n: LabNum; show?: boolean }) {
  if (!visible) return <td className={ui.num}>…</td>;
  return (
    <td className={ui.num} title={isNum(n) ? `Source: ${n.source?.ref ?? ""}` : (n.reason ?? "")}>
      {isNum(n) ? `${n.value} ${n.unit}` : <span className={lab.note}>not in the log</span>}
      {isNum(n) && n.low_confidence ? (
        <>
          <br />
          <span className={lab.low}>low confidence</span>
        </>
      ) : null}
    </td>
  );
}

/* ---------- drawers ---------- */

function Sources({ story }: { story: LabStory }) {
  const known = story.sources.filter((n) => n.value !== null);
  return (
    <>
      <p className={ui.muted}>
        {story.sample
          ? "Every number below comes from the sample run file, not from a measurement. The ids are the sample file's own."
          : "Every number on this screen, and the event or file it was read from. Nothing is filled in."}
      </p>
      <div className={lab.sourceList}>
        {known.map((n, i) => (
          <div key={`${n.label}-${i}`} className={lab.sourceRow}>
            <div className={lab.sourceTop}>
              <span>{n.label}</span>
              <b>
                {show(n)}
                {n.low_confidence ? <span className={lab.low}>low confidence</span> : null}
              </b>
            </div>
            <span className={`${lab.note} ${ui.breakWords}`}>
              {n.source?.kind === "derived" ? "Worked out: " : n.source?.kind === "file" ? "File: " : "Event: "}
              {n.source?.ref}
              {n.source?.kind === "derived" && n.source.events ? ` (${n.source.events} events)` : ""}
            </span>
          </div>
        ))}
        {story.measured.map((m) => (
          <div key={m.metric} className={lab.sourceRow}>
            <div className={lab.sourceTop}>
              <span>{m.label} (measured, not this run)</span>
              <b>
                {m.value} {m.unit}
              </b>
            </div>
            <span className={`${lab.note} ${ui.breakWords}`}>File: {m.source?.ref}</span>
          </div>
        ))}
      </div>
      {story.gaps.length ? (
        <>
          <h3>Not in the log</h3>
          <div className={lab.sourceList}>
            {story.gaps.map((g, i) => (
              <div key={`${g.label}-${i}`} className={lab.sourceRow}>
                <span>{g.label}</span>
                <span className={lab.note}>{g.reason}</span>
              </div>
            ))}
          </div>
        </>
      ) : null}
    </>
  );
}

const HOW_TO: string[] = [
  "Press Play, or drag the bar. Everything below follows it, from the examples to the swap.",
  "Loss falling means the model is getting its training examples right. It does not prove the model got better.",
  "The proof is the held-back questions: ones it never trained on, answered before and after. Regressions are listed first.",
  "The gate only lets a model through when the gain is larger than run-to-run noise and unlikely to be chance (p below 0.05).",
  "A number marked “Not in the log” was not recorded. Open Sources to see where every other number came from.",
];

/* ---------- a run that is still going (L30) ---------- */

const STAGE_WORD: Record<StageState, string> = { done: "done", now: "now", waiting: "waiting", failed: "failed" };

/** The four stages of a live run, each lit as it reports. Shown only while the run is still going. */
function LiveStages({ story }: { story: LabStory }) {
  const states = stageStates(story);
  return (
    <ol className={lab.stages} aria-label="Where the run is">
      {LIVE_STAGES.map((s) => (
        <li key={s.key} className={`${lab.stage} ${lab[`stage_${states[s.key]}`]}`}>
          <span className={lab.stageDot} aria-hidden="true" />
          {s.title}
          <span className={lab.srOnly}>: {STAGE_WORD[states[s.key]]}</span>
        </li>
      ))}
    </ol>
  );
}

/** The box hopping while the run trains: its steps and loss as they arrive, a burst of sparks per new step. */
function LiveBuddies({ story, forceReduced }: { story: LabStory; forceReduced: boolean }) {
  const loss = story.training.points.map((p) => p.loss).filter((v): v is number => typeof v === "number");
  const steps = isNum(story.training.steps) ? story.training.steps.value : story.training.points.length;
  const training = story.status === "training";
  return (
    <LearningBuddies
      state={training ? "alone" : "idle"}
      steps={steps}
      loss={loss}
      joulesPerExample={isNum(story.power.j_per_sample) ? story.power.j_per_sample.value : null}
      friends={0}
      forceReduced={forceReduced}
      caption={training ? "Each spark is a training step the trainer just reported." : "Training is over; packaging and the swap are still to report."}
    />
  );
}

/** "Running · 1 min 12 s", counting up once a second on its own so the rest of the page does not redraw with it. */
function Running({ story }: { story: LabStory }) {
  const [now, setNow] = useState<number | null>(null); // null until mounted, so the static page and the client agree
  useEffect(() => {
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const elapsed = now === null ? null : elapsedSeconds(story, now);
  return <>Running{elapsed !== null ? ` · ${duration(elapsed)}` : ""}</>;
}

/* ---------- the run ---------- */

export function LabRun({ story, initial, forceReduced, picker }: { story: LabStory; initial: number | null; forceReduced: boolean; picker?: React.ReactNode }) {
  const [u, setU] = useState(initial ?? 1);
  const [playing, setPlaying] = useState(false);
  const [reduced, setReduced] = useState(forceReduced);
  const [drawer, setDrawer] = useState<"how" | "sources" | null>(null);
  const [live, setLive] = useState(false); // played or scrubbed at least once: changes may animate from now on
  const uRef = useRef(u);
  uRef.current = u;

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(forceReduced || mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [forceReduced]);

  // The clock: one animation frame at a time. With reduced motion nothing animates; Play steps chapter by chapter.
  useEffect(() => {
    if (!playing || reduced) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const next = clamp01(uRef.current + (now - last) / 1000 / PLAY_SECONDS);
      last = now;
      setU(next);
      if (next >= 1) setPlaying(false);
      else raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [playing, reduced]);

  const toggle = useCallback(() => {
    if (reduced) {
      setLive(true);
      setU((v) => (v >= 1 ? CHAPTERS[0]!.to : (CHAPTERS.find((c) => c.to > v + 1e-6) ?? CHAPTERS[CHAPTERS.length - 1]!).to));
      return;
    }
    setLive(true);
    setPlaying((p) => {
      if (!p && uRef.current >= 1) setU(0);
      return !p;
    });
  }, [reduced]);

  const closeDrawer = useCallback(() => setDrawer(null), []);
  const chapter = chapterAt(u);
  // While it plays, the section the playhead is in comes into view, so a viewer never watches the wrong part.
  const chapterN = chapter.n;
  useEffect(() => {
    if (!playing && !(reduced && live)) return;
    document.getElementById(`lab-sec-${chapterN}`)?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }, [chapterN, playing, reduced, live]);
  const is = (k: ChapterKey) => chapter.key === k;
  const trainFrac = within(u, "training");
  const good = story.status === "deployed";
  const bad = story.status === "not deployed" || story.status === "rolled back" || story.status === "failed";
  const running = story.status === "training";
  const when: Record<ChapterKey, string | null> = {
    taught: story.marks.exported ?? story.marks.taught,
    training: story.marks.training_start,
    answers: story.marks.evaluated,
    gate: story.marks.evaluated,
    box: story.marks.deployed,
  };
  const nowLabel =
    chapter.key === "training" && story.marks.training_start && isNum(story.training.wall_s)
      ? clock(new Date(new Date(story.marks.training_start).getTime() + trainFrac * story.training.wall_s.value * 1000).toISOString(), true)
      : clock(when[chapter.key]);

  return (
    <div className={`${lab.page} ${live ? lab.live : ""}`}>
      <div className={ui.pageHead}>
        <div className={lab.headTop}>
          <h1>What Atlas learned</h1>
          <span className={`${lab.status} ${good ? lab.statusGood : bad ? lab.statusBad : ""} ${running ? lab.statusLive : ""}`}>
            {running && !story.sample ? <Running story={story} /> : STATUS_LABEL[story.status]}
          </span>
        </div>
        {isLive(story) ? <LiveStages story={story} /> : null}
        {isLive(story) ? <LiveBuddies story={story} forceReduced={forceReduced} /> : null}
        <p className={lab.task}>
          {story.task.instruction ? (
            <>
              <span className={ui.muted}>The task{story.task.slug ? ` (${story.task.slug})` : ""}: </span>
              {story.task.instruction}
            </>
          ) : (
            <span className={ui.muted}>{story.task.instruction_reason}</span>
          )}
        </p>
        <div className={lab.meta}>
          {picker}
          <button type="button" className={lab.linkBtn} onClick={() => setDrawer("how")}>
            How to read this
          </button>
          <button type="button" className={lab.linkBtn} onClick={() => setDrawer("sources")}>
            Sources
          </button>
          {isNum(story.wall_s) ? <span className={`${ui.muted} ${ui.small}`}>Start to finish: {duration(story.wall_s.value)}</span> : null}
        </div>
      </div>

      <div className={lab.dock}>
        {story.sample ? (
          <div className={lab.banner} role="note">
            <strong>Sample run: not measured on our hardware.</strong>
            <span>{story.sample_reason === "asked for with ?sample=1" ? "Shown because you asked for the sample." : "Shown because this box has not trained anything yet."}</span>
          </div>
        ) : null}
        <div className={lab.player}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${lab.play}`} onClick={toggle}>
            <svg viewBox="0 0 24 24" aria-hidden="true">{playing ? <path d="M8 5v14M16 5v14" fill="none" /> : <path d="M8 5l11 7-11 7z" />}</svg>
            {reduced ? "Next" : playing ? "Pause" : u >= 1 ? "Replay" : u > 0 ? "Resume" : "Play"}
          </button>
          <div className={lab.scrub}>
            <div className={lab.track} aria-hidden="true">
              {CHAPTERS.map((c) => (
                <div key={c.key} className={lab.seg} style={{ flex: c.to - c.from }}>
                  <div className={lab.segFill} style={{ width: `${within(u, c.key) * 100}%` }} />
                  <span className={lab.segLabel}>
                    {c.n}
                    <span className={lab.segTitle}> {c.title}</span>
                  </span>
                </div>
              ))}
            </div>
            <input
              className={lab.range}
              type="range"
              min={0}
              max={1000}
              value={Math.round(u * 1000)}
              aria-label="Position in the run"
              aria-valuetext={`${chapter.n}. ${chapter.title}`}
              onChange={(e) => {
                setPlaying(false);
                setLive(true);
                setU(Number(e.target.value) / 1000);
              }}
            />
            <div className={lab.now}>
              <span>
                <b>
                  {chapter.n}. {chapter.title}
                </b>
              </span>
              <span>{nowLabel}</span>
            </div>
          </div>
        </div>
      </div>
      {reduced ? <p className={`${ui.muted} ${ui.small}`}>Reduced motion is on: Next moves one step at a time and nothing animates. The bar still scrubs.</p> : null}

      <div className={lab.grid}>
        <div className={lab.col}>
          <Section n={1} title="Taught from" now={is("taught")} className={lab.o1}>
            <TaughtFrom story={story} />
          </Section>
          <Section n={3} title="Before / after" now={is("answers")} className={lab.o3}>
            <BeforeAfter story={story} frac={within(u, "answers")} />
          </Section>
        </div>
        <div className={lab.col}>
          <Section n={2} title="Training" now={is("training")} className={lab.o2}>
            <Training story={story} frac={trainFrac} measured={story.measured} />
          </Section>
          <Section n={4} title="The gate" now={is("gate")} className={lab.o4}>
            <Gate story={story} on={u >= CHAPTERS[3]!.from - 1e-6} />
          </Section>
          <Section n={5} title="On the box" now={is("box")} className={lab.o5}>
            <OnTheBox story={story} frac={within(u, "box")} gated={u >= CHAPTERS[3]!.from - 1e-6} />
          </Section>
        </div>
      </div>

      {drawer === "how" ? (
        <Sheet title="How to read this" onClose={closeDrawer}>
          <ol className={lab.howto}>
            {HOW_TO.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ol>
        </Sheet>
      ) : null}
      {drawer === "sources" ? (
        <Sheet title="Sources" onClose={closeDrawer}>
          <Sources story={story} />
        </Sheet>
      ) : null}
    </div>
  );
}
