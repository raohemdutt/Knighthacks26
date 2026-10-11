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
  onOurHardware,
  isNum,
  lowConfidence,
  missingLine,
  type RunTone,
  stageStates,
  storyIsStandIn,
  pct,
  phoneCharges,
  show,
  within,
} from "@/lib/lab";
import { answerWords } from "@/lib/labWords";
import { GateChart, TrainingChart } from "./Charts";
import { DeployButton } from "./DeployButton";
import lab from "./lab.module.css";
import { LearningBuddies } from "./LearningBuddies";

/* ---------- small pieces ---------- */

/** Where a measured trainer row ran: only rows from our hardware get here (onOurHardware). */
function smokeChip(m: LabMeasured | undefined): string {
  if (m?.chip?.key === "jetson") return "the box (Jetson)";
  return m?.chip ? `the lab's trainer, ${chipShort(m.chip)}` : "the lab's trainer";
}

/**
 * One number as a tile. A number the log does not hold is no tile at all: the card names it once, in its one muted
 * line (Missing). The "low confidence" badge is for power and energy figures only.
 */
function Stat({ n, label, value, tone }: { n: LabNum; label?: string; value?: string; tone?: "good" | "bad" }) {
  if (!isNum(n)) return null;
  return (
    <div className={lab.stat} title={`Source: ${n.source?.ref ?? ""}`}>
      <span className={`${lab.statNum} ${tone === "good" ? lab.good : tone === "bad" ? lab.bad : ""}`}>
        {value ?? show(n)}
        {lowConfidence(n) ? <span className={lab.low}>low confidence</span> : null}
      </span>
      <span className={lab.statLabel}>{label ?? n.label}</span>
    </div>
  );
}

/** The figures a card's log lacks, in one small muted line; nothing when it lacks none. Reasons are in Sources. */
function Missing({ nums = [], more = [], tail }: { nums?: (LabNum | null | undefined)[]; more?: string[]; tail?: string }) {
  const labels = [...nums.filter((n): n is LabNum => !!n && !isNum(n)).map((n) => n.label), ...more];
  const line = missingLine(labels);
  if (!line && !tail) return null;
  return (
    <p className={lab.missing} title={nums.filter((n): n is LabNum => !!n && !isNum(n) && !!n.reason).map((n) => `${n.label}: ${n.reason}`).join("\n")}>
      {[line, tail].filter(Boolean).join("; ")}.
    </p>
  );
}

/** A card's one headline number, large, and at most one sentence under it. */
function Headline({ big, line, why }: { big: React.ReactNode; line?: React.ReactNode; why?: string | null }) {
  return (
    <div className={lab.headline}>
      <div className={lab.headBig}>{big}</div>
      {why ? <Why text={why} /> : null}
      {line ? <p className={lab.headLine}>{line}</p> : null}
    </div>
  );
}

/** Why a card matters, in under ten words, under its headline number. */
function Why({ text }: { text: string }) {
  return <p className={lab.whyLine}>{text}</p>;
}

/** Every explanation sits in one of these, closed until asked for. */
function More({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className={lab.more}>
      <summary>{title}</summary>
      <div className={lab.moreBody}>{children}</div>
    </details>
  );
}

/** The one honest line a stand-in gate gets, in place of its fake items, p-value and chart. */
function StandIn({ realRun, onPickRun }: { realRun?: { id: string; label: string } | null; onPickRun?: (id: string) => void }) {
  return (
    <p className={lab.standIn} role="note">
      This loop run used a stand-in gate; no held-out score.
      {realRun ? (
        <>
          {" "}
          The real gate ran on{" "}
          <a
            href={`?run=${encodeURIComponent(realRun.id)}`}
            onClick={(e) => {
              if (!onPickRun) return;
              e.preventDefault();
              onPickRun(realRun.id);
            }}
          >
            {realRun.label}
          </a>
          .
        </>
      ) : null}
    </p>
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
  const total = train ?? (isNum(d.examples) ? d.examples.value : null);
  return (
    <>
      <Headline
        big={
          total !== null ? (
            <>
              {total.toLocaleString()} <small>examples {train !== null ? "to learn from" : "left the box"}</small>
            </>
          ) : (
            <small>No example count posted</small>
          )
        }
        why={d.approval.approved ? "Real examples from this box, approved by you." : null}
        line={held !== null ? `${held.toLocaleString()} more held back for the test; the model never trains on them.` : null}
      />
      {train !== null && held !== null ? (
        <div className={lab.split} role="img" aria-label={`${train} examples to learn from, ${held} held back for the test`}>
          <div className={lab.splitTrain} style={{ flex: train }}>
            {train} to learn from
          </div>
          <div className={lab.splitHeld} style={{ flex: held }}>
            {held} held back
          </div>
        </div>
      ) : null}
      {rows.length ? (
        <div className={lab.cards} aria-label="Examples the model learned from">
          {rows.slice(0, 6).map((r) => (
            <div key={r.id} className={lab.example}>
              {r.frame_ref.startsWith("frames:") ? <Keyframe frameRef={r.frame_ref} compact /> : null}
              <span className={lab.exampleQ}>{r.context ?? "No caption"}</span>
              {r.target ? <span className={lab.exampleA}>{answerWords(r.target)}</span> : <span className={lab.note}>Not labeled yet</span>}
            </div>
          ))}
        </div>
      ) : null}
      <Missing more={[...(train === null || held === null ? ["the train and held-out split"] : []), ...(rows.length ? [] : ["the example rows"])]} />
      <More title="About these examples">
        {isNum(d.examples) ? <p>{d.examples.value} examples left the box for this task.</p> : null}
        {d.approval.approved ? <p>You approved this export at {clock(d.approval.approved)}.</p> : null}
        {d.approval.reason ? <p>The approval card said: “{d.approval.reason}”.</p> : null}
        {isNum(d.held_out_groups) ? <p>The held-back questions cover {d.held_out_groups.value} different cases, each asked in several wordings.</p> : null}
        {rows.length ? <p>{Math.min(6, rows.length)} of the examples it learned from are shown. Pictures, when a task has them, are the box&apos;s blurred keyframes.</p> : null}
      </More>
    </>
  );
}

/* ---------- 2. training ---------- */

function Training({ story, frac, measured }: { story: LabStory; frac: number; measured: LabMeasured[] }) {
  const t = story.training;
  const p = story.power;
  const tMax = Math.max(isNum(t.wall_s) ? t.wall_s.value : 0, t.points.length ? t.points[t.points.length - 1]!.t : 0, 1);
  const tNow = frac * tMax;
  const seen = t.points.filter((x) => x.t <= tNow + 1e-6);
  const head = seen[seen.length - 1];
  const joules = p.points.length ? energyUpTo(p.points, tNow) : null;
  const wh = joules !== null ? joules / 3600 : null;
  // Only rows measured on our hardware (the Strix Halo pods or the box); an R9700 row is never shown as ours.
  const ours = measured.filter(onOurHardware);
  const speed = ours.find((m) => m.metric === "trainer.samples_per_s");
  const first = ours.find((m) => m.metric === "trainer.smoke_loss_first");
  const last = ours.find((m) => m.metric === "trainer.smoke_loss_last");
  const loss = (v: number) => (Math.abs(v) >= 0.1 ? v.toFixed(2) : String(Number(v.toPrecision(2))));
  const steps = head?.step ?? (isNum(t.steps) ? t.steps.value : null);
  return (
    <>
      <Headline
        big={
          steps !== null ? (
            <>
              {steps} <small>{isNum(t.of) ? `of ${t.of.value} steps` : "steps"}</small>
            </>
          ) : (
            <small>No steps posted</small>
          )
        }
        why="A small add-on learns; the base model stays."
        line={
          isNum(t.loss_first) && isNum(t.loss_last)
            ? `Loss went from ${loss(t.loss_first.value)} to ${loss(t.loss_last.value)}${isNum(t.wall_s) ? ` in ${duration(t.wall_s.value)}` : ""}.`
            : null
        }
      />
      <TrainingChart training={t} power={p} frac={frac} />
      <div className={lab.stats}>
        <Stat n={t.samples_per_s} label="training speed" />
        <Stat n={t.wall_s} label="training time" value={duration(tNow)} />
        {p.points.length ? <Stat n={p.energy_j} label="training energy, whole run" /> : null}
        {p.points.length ? <Stat n={p.j_per_sample} label="energy per training sample" /> : null}
        {p.points.length ? <Stat n={p.mean_w} label="average power" /> : null}
      </div>
      <Missing
        nums={[t.samples_per_s, t.wall_s, ...(p.points.length ? [p.energy_j, p.j_per_sample, p.mean_w] : [])]}
        more={p.points.length ? [] : ["power readings"]}
      />
      <More title="About this training">
        <p>
          {t.base ? (
            <>
              A small add-on (LoRA{t.lora_rank ? `, rank ${t.lora_rank}` : ""}) is trained on top of {t.base}
              {t.node ? ` on ${t.node}` : ""}. The base model is not changed.
            </>
          ) : (
            (t.points_reason ?? "The model practises on the examples; the loss falls as it gets them right.")
          )}
        </p>
        <p>Loss falling means the model is getting its training examples right. It does not prove the model got better; the held-back questions do.</p>
        {wh !== null ? (
          <p>
            {wh.toFixed(1)} Wh so far, about the energy of {phoneCharges(wh)}
            {lowConfidence(p.energy_j) ? <span className={lab.low}>low confidence</span> : null}. {PHONE_ASSUMPTION}
          </p>
        ) : null}
        {speed || first ? (
          <p>
            Measured on our hardware, not this run: a 20-step smoke run on {smokeChip(speed ?? first)}
            {speed?.node ? ` (${speed.node})` : ""}
            {speed ? `: ${speed.value} samples/s` : ""}
            {first && last ? `, loss ${first.value} to ${last.value}` : ""}.
          </p>
        ) : (
          <p>No smoke run on our own hardware yet.</p>
        )}
      </More>
    </>
  );
}

/* ---------- 3. before / after ---------- */

const ORDER: LabChange[] = ["broke", "fixed", "still_wrong", "still_right"];
type Filter = "changed" | "broke" | "fixed" | "still_wrong" | "all";

function BeforeAfter({ story, frac, realRun, onPickRun }: { story: LabStory; frac: number; realRun?: { id: string; label: string } | null; onPickRun?: (id: string) => void }) {
  const ev = story.eval;
  const [filter, setFilter] = useState<Filter>("fixed");
  const [open, setOpen] = useState(false);
  // regressions first, so they are never under the fold; the playhead reveals answers in this same order
  const ordered = useMemo(() => [...(ev.items ?? [])].sort((a, b) => ORDER.indexOf(a.change) - ORDER.indexOf(b.change)), [ev.items]);
  if (storyIsStandIn(story)) return <StandIn realRun={realRun} onPickRun={onPickRun} />;
  if (!ev.items || !ev.counts) {
    const { current, candidate } = ev.scores;
    return (
      <>
        <Headline
          big={isNum(current) && isNum(candidate) ? <>{pct(current.value)} → {pct(candidate.value)} <small>right</small></> : <small>No test result posted</small>}
          line="Before training and after, on held-back questions the model never trained on."
        />
        <Missing nums={[current, candidate]} more={["the per-question answers"]} />
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
  const visible = open ? list : list.slice(0, 5);
  const filters: [Filter, string][] = [
    ["fixed", `Fixed (${total("fixed")})`],
    ["broke", `Regressions (${total("broke")})`],
    ["still_wrong", `Still wrong (${total("still_wrong")})`],
    ["changed", `Flipped (${total("fixed") + total("broke")})`],
    ["all", `All (${n})`],
  ];
  return (
    <>
      <Headline
        big={
          <>
            {beforeRight} → {shown === 0 ? "…" : afterRight} <small>of {n} right</small>
          </>
        }
        why={beforeRight > 0 && (ev.counts.after_right.value ?? 0) >= 1.6 * beforeRight ? "Right answers almost doubled on unseen questions." : null}
        line="Before training and after, on held-back questions the model never trained on."
      />
      <div className={lab.tally}>
        <div className={`${lab.tallyCard} ${lab.tallyGood}`}>
          <span className={`${lab.statNum} ${lab.good}`}>+{count("fixed")}</span>
          <span className={lab.statLabel}>fixed</span>
        </div>
        <div className={`${lab.tallyCard} ${lab.tallyBad}`}>
          <span className={`${lab.statNum} ${lab.bad}`}>−{count("broke")}</span>
          <span className={lab.statLabel}>got worse</span>
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
          <button
            key={key}
            type="button"
            className={`${ui.chip} ${filter === key ? ui.chipOn : ""}`}
            aria-pressed={filter === key}
            onClick={() => {
              setFilter(key);
              setOpen(false);
            }}
          >
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
      {list.length > 5 ? (
        <button type="button" className={ui.btn} onClick={() => setOpen((o) => !o)}>
          {open ? "Show fewer" : `Show all ${list.length}`}
        </button>
      ) : null}
    </>
  );
}

/** One held-back question in plain words: the question, the right answer, and the two answers as one-line pills. */
function Question({ it, scored }: { it: LabItem; scored: boolean }) {
  const tone = !scored ? "" : it.change === "fixed" ? lab.qaFixed : it.change === "broke" ? lab.qaBroke : "";
  const before = `Before: ${answerWords(it.before.answer)}, ${it.before.correct ? "right" : "wrong"}`;
  const after = `After: ${answerWords(it.after.answer)}, ${it.after.correct ? "right" : "wrong"}`;
  return (
    <div className={`${lab.qa} ${tone}`}>
      <div className={lab.qaHead}>
        <span className={lab.qaTitle}>{it.question}</span>
        {scored ? (
          <span key="scored" className={`${lab.change} ${lab.flipIn} ${it.change === "fixed" ? lab.changeFixed : it.change === "broke" ? lab.changeBroke : ""}`}>{CHANGE_LABEL[it.change]}</span>
        ) : (
          <span className={lab.change}>Not scored yet</span>
        )}
      </div>
      {it.expected ? <p className={lab.qaRight}>Right answer: {answerWords(it.expected)}</p> : null}
      <div className={lab.pills}>
        <span className={`${lab.pill} ${it.before.correct ? lab.pillRight : lab.pillWrong}`} title={before}>
          {before}
        </span>
        {scored ? (
          <span key="after" className={`${lab.pill} ${lab.flipIn} ${it.after.correct ? lab.pillRight : lab.pillWrong}`} title={after}>
            {after}
          </span>
        ) : (
          <span className={lab.pill}>After: not scored yet</span>
        )}
      </div>
      <details className={lab.raw}>
        <summary>raw</summary>
        <pre className={lab.rawText}>{`Right answer:\n${it.expected ?? ""}\n\nBefore:\n${it.before.answer}\n\nAfter:\n${it.after.answer}`}</pre>
      </details>
    </div>
  );
}

/* ---------- 4. the gate ---------- */

function Gate({ story, on, onBox, realRun, onPickRun }: { story: LabStory; on: boolean; onBox: boolean; realRun?: { id: string; label: string } | null; onPickRun?: (id: string) => void }) {
  const ev = story.eval;
  if (storyIsStandIn(story)) return <StandIn realRun={realRun} onPickRun={onPickRun} />;
  if (!ev.verdict) return <p className={lab.waiting}>{ev.verdict_detail[0] ?? "The model has not been tested yet."}</p>;
  if (!on) return <p className={lab.waiting}>The gate decides once every held-back question has been scored.</p>;
  // The gate's own call: a model it cleared is good news whatever the box did with it afterwards.
  const good = ev.decision === "deploy";
  const alpha = ev.alpha ?? 0.05;
  // "..., but the box went back to the old model" is stale once the box serves this model again
  const verdict = onBox ? ev.verdict.replace(/,\s*but the box went back[^.]*\./i, ".") : ev.verdict;
  const p = isNum(ev.p_value) ? ev.p_value.value : null;
  const pText = p !== null ? String(Number(p.toPrecision(2))) : null;
  const sc = ev.scores;
  const sentence =
    isNum(sc.current) && isNum(sc.candidate)
      ? `The score went from ${pct(sc.current.value)} to ${pct(sc.candidate.value)} on questions the model never trained on${p !== null && p > 0 ? `; a difference this large would show up by chance about 1 time in ${Math.round(1 / p).toLocaleString()}` : ""}.`
      : (ev.verdict_detail[0] ?? null);
  const noise = isNum(ev.noise) ? ev.noise.value : null;
  const groups = ev.groups ?? [];
  const better = groups.filter((g) => g.diff > 0).length;
  const worse = groups.filter((g) => g.diff < 0).length;
  return (
    <div className={lab.gateBox}>
      <Headline
        big={
          <span className={good ? lab.good : lab.bad}>
            {good ? "Cleared by the gate" : "Not cleared by the gate"}
            {pText ? <small> · p = {pText}</small> : null}
          </span>
        }
        why={good && p !== null && p < alpha ? "Proven better than chance, so it may ship." : null}
        line={good ? null : "The model on the box did not change."}
      />
      <GateChart ev={ev} notes={false} />
      <div className={`${lab.stats} ${lab.gateTiles}`}>
        <Stat n={ev.n_items} label={isNum(ev.n_groups) ? `questions, in ${ev.n_groups.value} groups` : "questions scored"} value={isNum(ev.n_items) ? String(ev.n_items.value) : undefined} />
        <Stat n={ev.noise} label="run-to-run noise" value={isNum(ev.noise) ? `±${pct(ev.noise.value, 1)}` : undefined} />
        <Stat n={ev.p_value} label={`p-value (needs to be below ${alpha})`} value={pText ?? undefined} tone={p !== null ? (p < alpha ? "good" : "bad") : undefined} />
      </div>
      {sentence ? <p className={lab.gateSentence}>{sentence}</p> : null}
      <Missing nums={[ev.n_items, ev.noise, ev.p_value]} />
      <More title="How the test works">
        <p>{verdict}</p>
        <ul className={lab.why}>
          {ev.verdict_detail.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <p>
          {noise !== null
            ? `The grey band on the slider is run-to-run noise: scoring the same model again moves it by about ${pct(noise, 1)}.`
            : "Run-to-run noise was not measured for this run, so no noise band is drawn."}
        </p>
        {groups.length ? (
          <p>
            Each dot is one group of questions (different wordings of one case), placed by how much its score moved: {better} of {groups.length} got better, {worse} got worse,{" "}
            {groups.length - better - worse} did not move. The test asks how often a pattern this lopsided would appear if the change were only chance.
          </p>
        ) : null}
        {ev.test || ev.decode ? (
          <p>
            {ev.test ? `Test: ${ev.test}. ` : ""}
            {ev.decode ? `All three models were scored the same way: ${ev.decode}.` : ""}
          </p>
        ) : null}
      </More>
    </div>
  );
}

/* ---------- 5. on the box ---------- */

const SWAP_WHY = "One button swaps models; rollback is automatic.";

function OnTheBox({ story, frac, gated, boxNow }: { story: LabStory; frac: number; gated: boolean; boxNow: string | null }) {
  const { deploy, box, package: pkg } = story;
  const onBox = !!boxNow && !!deploy.model_version && boxNow === deploy.model_version;
  const headline = onBox ? (
    <Headline big="On the box now" why={SWAP_WHY} line={`The box serves ${deploy.model_version} now.`} />
  ) : deploy.deployed ? (
    <Headline big={<small>Not on the box now</small>} why={SWAP_WHY} line={boxNow ? `The box serves ${boxNow} now.` : null} />
  ) : null;
  if (!gated && story.eval.verdict) return <p className={lab.waiting}>The swap happens only after the gate clears the model.</p>;
  const sizeNote = (
    <More title="About the swap">
      <p>The new model is the same size as the old one, so speed and energy per answer should barely move. What changes is how often it is right.</p>
    </More>
  );
  const packageLine = pkg.gguf ? (
    <div className={lab.stats}>
      <Stat n={pkg.bytes} label={`model file${pkg.quantization ? `, ${pkg.quantization}` : ""}`} value={isNum(pkg.bytes) ? fileSize(pkg.bytes.value) : undefined} />
      <div className={lab.stat}>
        <span className={`${lab.code} ${ui.breakWords}`}>{pkg.gguf}</span>
        <span className={lab.statLabel}>{pkg.sha256 ? `sha256 ${pkg.sha256.slice(0, 12)}…` : "no checksum in the log"}</span>
      </div>
    </div>
  ) : null;
  const boxRows = box.before && box.after
    ? ([
        ["Energy per answer", box.before.j_per_answer, box.after.j_per_answer],
        ["Tokens per second", box.before.tok_s, box.after.tok_s],
      ] as [string, LabNum, LabNum][])
    : [];
  const shownRows = boxRows.filter(([, a, b]) => isNum(a) || isNum(b));
  const missing = (
    <Missing
      nums={[pkg.gguf ? pkg.bytes : null, deploy.deployed ? deploy.swap_s : null]}
      more={[...(pkg.gguf ? [] : ["the model package"]), ...boxRows.filter(([, a, b]) => !isNum(a) && !isNum(b)).map(([name]) => `${name.toLowerCase()} on the box`)]}
    />
  );
  if (!deploy.deployed) {
    // a stand-in gate decided nothing about the model, so its refusal is not repeated here
    const standIn = storyIsStandIn(story);
    return (
      <>
        <p className={`${lab.verdict} ${story.status === "not deployed" && !standIn ? lab.verdictBad : ""}`}>
          {story.status === "not deployed" ? "Nothing changed on the box." : "Not on the box yet."}
          {standIn ? null : <span className={lab.verdictSub}>{deploy.deployed_reason}</span>}
        </p>
        <Why text={SWAP_WHY} />
        {packageLine}
        {missing}
      </>
    );
  }
  if (frac <= 0) return <p className={lab.waiting}>The swap happens only after the gate clears the model.</p>;
  const done = frac >= 0.5;
  return (
    <>
      {headline}
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
        <p className={lab.verdict}>
          The box went back to {deploy.rolled_back.to ?? "the old model"} at {clock(deploy.rolled_back.ts)}.
          {onBox ? <span className={lab.verdictSub}>{deploy.model_version} is on the box again now.</span> : null}
          {deploy.rolled_back.reason ? <span className={lab.verdictSub}>{deploy.rolled_back.reason}</span> : null}
        </p>
      ) : null}
      {isNum(deploy.swap_s) ? (
        <div className={lab.stats}>
          <Stat n={deploy.swap_s} label="hot-swap time (no answers meanwhile)" value={`${deploy.swap_s.value.toFixed(1)} s`} />
        </div>
      ) : null}
      {packageLine}
      {missing}
      {box.before && box.after && shownRows.length ? (
        <table className={ui.table}>
          <thead>
            <tr>
              <th>On the box</th>
              <th className={ui.num}>Before</th>
              <th className={ui.num}>After</th>
            </tr>
          </thead>
          <tbody>
            {shownRows.map(([name, a, b]) => (
              <tr key={name}>
                <td>{name}</td>
                <BoxCell n={a} />
                <BoxCell n={b} show={done} />
              </tr>
            ))}
            <tr>
              <td>Answers measured</td>
              <td className={ui.num}>{box.before.answers}</td>
              <td className={ui.num}>{done ? box.after.answers : "…"}</td>
            </tr>
          </tbody>
        </table>
      ) : null}
      {sizeNote}
    </>
  );
}

function BoxCell({ n, show: visible = true }: { n: LabNum; show?: boolean }) {
  if (!visible) return <td className={ui.num}>…</td>;
  return (
    <td className={ui.num} title={isNum(n) ? `Source: ${n.source?.ref ?? ""}` : (n.reason ?? "")}>
      {isNum(n) ? `${n.value} ${n.unit}` : <span className={lab.note}>none posted</span>}
      {isNum(n) && lowConfidence(n) ? (
        <>
          <br />
          <span className={lab.low}>low confidence</span>
        </>
      ) : null}
    </td>
  );
}

/* ---------- drawers ---------- */

const GATE_REFS = /learning\.run\.evaluated|learning\.eval\.reported|eval report/;

function Sources({ story }: { story: LabStory }) {
  const standIn = storyIsStandIn(story);
  const known = story.sources.filter((n) => n.value !== null && !(standIn && GATE_REFS.test(`${n.source?.ref ?? ""} ${(n.source?.ids ?? []).join(" ")}`)));
  return (
    <>
      <p className={ui.muted}>Every number on this screen, and the event or file it was read from. Nothing is filled in.</p>
      {standIn ? <p className={lab.note}>This run&apos;s gate was a stand-in, so its scores and p-value are left out.</p> : null}
      <div className={lab.sourceList}>
        {known.map((n, i) => (
          <div key={`${n.label}-${i}`} className={lab.sourceRow}>
            <div className={lab.sourceTop}>
              <span>{n.label}</span>
              <b>
                {show(n)}
                {lowConfidence(n) ? <span className={lab.low}>low confidence</span> : null}
              </b>
            </div>
            <span className={`${lab.note} ${ui.breakWords}`}>
              {n.source?.kind === "derived" ? "Worked out: " : n.source?.kind === "file" ? "File: " : "Event: "}
              {n.source?.ref}
              {n.source?.kind === "derived" && n.source.events ? ` (${n.source.events} events)` : ""}
            </span>
          </div>
        ))}
        {story.measured.filter(onOurHardware).map((m) => (
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
  "One stage at a time: press a stage on the bar, use the arrows beside the card, or the left and right keys. Replay walks through all five.",
  "Loss falling means the model is getting its training examples right. It does not prove the model got better.",
  "The proof is the held-back questions: ones it never trained on, answered before and after. Regressions are listed first.",
  "The gate only lets a model through when the gain is larger than run-to-run noise and unlikely to be chance (p below 0.05).",
  "A figure the relay did not post is named in one muted line under its card. Open Sources to see where every other number came from.",
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

export function LabRun({
  story,
  initial,
  forceReduced,
  picker,
  onBoxChanged,
  status,
  boxNow = null,
  realRun = null,
  onPickRun,
  initialStage = null,
}: {
  story: LabStory;
  initial: number | null;
  forceReduced: boolean;
  picker?: React.ReactNode;
  /** a swap started from the On the box card ended: the page reads the run again (L35) */
  onBoxChanged?: () => void;
  /** the run's status as the picker says it (lib/lab.ts runStatus); the run's own word when absent */
  status?: { label: string; tone: RunTone } | null;
  /** the version the box serves now (GET /api/learning/lab/deploy current) */
  boxNow?: string | null;
  /** the newest run with a real gate result, for a stand-in gate's one line */
  realRun?: { id: string; label: string } | null;
  onPickRun?: (id: string) => void;
  /** ?stage=1..5; without it a live run opens on Training and a run with a real gate on Before / after */
  initialStage?: number | null;
}) {
  const firstStage =
    initialStage && initialStage >= 1 && initialStage <= CHAPTERS.length ? initialStage : isLive(story) ? 2 : story.eval.decision && !storyIsStandIn(story) ? 3 : 1;
  const [u, setU] = useState(initial ?? (firstStage === CHAPTERS.length ? 1 : CHAPTERS[firstStage - 1]!.to - 1e-4));
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
      setU((v) => (v >= 1 ? CHAPTERS[0]!.to - 1e-4 : (CHAPTERS.find((c) => c.to > v + 1e-6) ?? CHAPTERS[CHAPTERS.length - 1]!).to));
      return;
    }
    setLive(true);
    setPlaying((p) => {
      if (!p) setU(0); // Replay always walks all five stages from the start
      return !p;
    });
  }, [reduced]);

  const closeDrawer = useCallback(() => setDrawer(null), []);
  const chapter = chapterAt(u);
  const stageN = chapter.n;
  // One stage's card at a time: picking one stops the replay and shows that card whole.
  const go = useCallback((n: number) => {
    const c = CHAPTERS[Math.min(CHAPTERS.length, Math.max(1, n)) - 1]!;
    setPlaying(false);
    setLive(true);
    setU(c.n === CHAPTERS.length ? 1 : c.to - 1e-4);
  }, []);
  // ?stage= follows the card on show
  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("stage") === String(stageN)) return;
    url.searchParams.set("stage", String(stageN));
    window.history.replaceState(null, "", url);
  }, [stageN]);
  // left and right keys step through the stages (never while typing, or with a sheet open)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (drawer || e.altKey || e.ctrlKey || e.metaKey || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      e.preventDefault();
      go(stageN + (e.key === "ArrowRight" ? 1 : -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawer, go, stageN]);
  const touch = useRef<{ x: number; y: number } | null>(null);
  // while it replays, each card fills in as the playhead crosses it; otherwise the card is shown whole
  const frac = (k: ChapterKey) => (playing ? within(u, k) : 1);
  const trainFrac = frac("training");
  const good = status ? status.tone === "good" : story.status === "deployed";
  const bad = status ? status.tone === "bad" : story.status === "not deployed" || story.status === "rolled back" || story.status === "failed";
  const running = isLive(story);
  const when: Record<ChapterKey, string | null> = {
    taught: story.marks.exported ?? story.marks.taught,
    training: story.marks.training_start,
    answers: story.marks.evaluated,
    gate: story.marks.evaluated,
    box: story.marks.deployed,
  };
  const nowLabel =
    chapter.key === "training" && playing && story.marks.training_start && isNum(story.training.wall_s)
      ? clock(new Date(new Date(story.marks.training_start).getTime() + within(u, "training") * story.training.wall_s.value * 1000).toISOString(), true)
      : clock(when[chapter.key]);
  const card: Record<ChapterKey, React.ReactNode> = {
    taught: <TaughtFrom story={story} />,
    training: <Training story={story} frac={trainFrac} measured={story.measured} />,
    answers: <BeforeAfter story={story} frac={frac("answers")} realRun={realRun} onPickRun={onPickRun} />,
    gate: <Gate story={story} on onBox={!!boxNow && boxNow === story.eval.model_version} realRun={realRun} onPickRun={onPickRun} />,
    box: (
      <>
        <OnTheBox story={story} frac={frac("box")} gated boxNow={boxNow} />
        <DeployButton onBoxChanged={onBoxChanged} />
      </>
    ),
  };
  const arrow = (dir: -1 | 1) => (
    <button
      type="button"
      className={`${lab.arrow} ${dir < 0 ? lab.arrowPrev : lab.arrowNext}`}
      aria-label={dir < 0 ? "Previous stage" : "Next stage"}
      disabled={dir < 0 ? stageN <= 1 : stageN >= CHAPTERS.length}
      onClick={() => go(stageN + dir)}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d={dir < 0 ? "M15 5l-7 7 7 7" : "M9 5l7 7-7 7"} />
      </svg>
    </button>
  );

  return (
    <div className={`${lab.page} ${live ? lab.live : ""}`}>
      <div className={ui.pageHead}>
        <div className={lab.headTop}>
          <h1>What Atlas learned</h1>
          <span className={`${lab.status} ${good ? lab.statusGood : bad ? lab.statusBad : ""} ${running ? lab.statusLive : ""}`}>
            {running ? <Running story={story} /> : (status?.label ?? STATUS_LABEL[story.status])}
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
        <div className={lab.player}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${lab.play}`} onClick={toggle}>
            <svg viewBox="0 0 24 24" aria-hidden="true">{playing ? <path d="M8 5v14M16 5v14" fill="none" /> : <path d="M8 5l11 7-11 7z" />}</svg>
            {reduced ? "Next" : playing ? "Pause" : "Replay"}
          </button>
          <div className={lab.scrub}>
            <div className={lab.track} role="tablist" aria-label="Stages of the run">
              {CHAPTERS.map((c) => (
                <button
                  key={c.key}
                  type="button"
                  role="tab"
                  aria-selected={c.n === stageN}
                  className={`${lab.seg} ${lab.segBtn} ${c.n === stageN ? lab.segOn : ""}`}
                  style={{ flex: 1 }}
                  onClick={() => go(c.n)}
                >
                  <div className={lab.segFill} style={{ width: `${(playing ? within(u, c.key) : c.n <= stageN ? 1 : 0) * 100}%` }} />
                  <span className={lab.segLabel}>
                    {c.n}
                    <span className={lab.segTitle}> {c.title}</span>
                  </span>
                </button>
              ))}
            </div>
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

      <div
        className={lab.stageWrap}
        onTouchStart={(e) => {
          const t = e.touches[0];
          touch.current = t ? { x: t.clientX, y: t.clientY } : null;
        }}
        onTouchEnd={(e) => {
          const t = e.changedTouches[0];
          const from = touch.current;
          touch.current = null;
          if (!t || !from) return;
          const dx = t.clientX - from.x;
          if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(t.clientY - from.y)) go(stageN + (dx < 0 ? 1 : -1));
        }}
      >
        {arrow(-1)}
        <Section key={chapter.key} n={chapter.n} title={chapter.title} now className={lab.stageCard}>
          {card[chapter.key]}
        </Section>
        {arrow(1)}
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
