"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import teach from "@/components/teach/teach.module.css";
import ui from "@/components/ui.module.css";
import { ApiError, api, keyframeUrl, type LearningStage, type LearningTask } from "@/lib/api";
import { useAtlas, useAtlasEvents } from "@/lib/atlas";
import { cameraName, dayTime } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { ZONES } from "@/lib/zones.generated";

/**
 * Teach: show Atlas a few pictures, say what you want it to learn, and watch it get better.
 *
 *  1. Describe the task and mark a few recent pictures thumbs up (like this) or thumbs down (not this):
 *     POST /api/teach. That raises an approval card: nothing is exported until you approve it on Approvals.
 *  2. Progress comes from GET /api/learning (rebuilt from the learning.* events) and refreshes as they arrive:
 *     labeling, training, eval, packaging, deploying. After eval it shows the before and after scores.
 */

const STEPS: LearningStage[] = ["waiting for approval", "exported", "labeling", "training", "eval", "packaging", "deploying", "done"];
const STEP_LABEL: Record<LearningStage, string> = {
  "waiting for approval": "Waiting for you",
  exported: "Pictures sent",
  labeling: "Labeling",
  training: "Training",
  eval: "Testing",
  packaging: "Packaging",
  deploying: "Installing",
  done: "Done",
  failed: "Failed",
  denied: "Not approved",
  expired: "Timed out",
};

function slug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function percent(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function TaskCard({ task }: { task: LearningTask }) {
  const at = Math.max(STEPS.indexOf(task.stage), 0);
  const failed = task.stage === "failed" || task.stage === "denied" || task.stage === "expired";
  const scores = task.scores;
  return (
    <div className={`${teach.card} ${teach.task}`}>
      <div className={teach.taskHead}>
        <span className={teach.taskName}>{task.task}</span>
        <span className={`${teach.stage} ${failed ? teach.stageBad : ""}`}>{STEP_LABEL[task.stage]}</span>
      </div>
      <p className={teach.help}>{task.instruction}</p>
      <div className={teach.steps} aria-label="Steps">
        {STEPS.map((s, i) => (
          <span key={s} className={i <= at && !failed ? teach.stepOn : ""}>
            {STEP_LABEL[s]}
          </span>
        ))}
      </div>
      <div className={teach.bar} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(task.progress * 100)}>
        <div className={teach.barFill} style={{ width: `${Math.round(task.progress * 100)}%` }} />
      </div>
      <p className={teach.help}>
        {task.message ?? ""}
        {task.loss != null ? ` Loss ${task.loss.toFixed(2)}.` : ""}
        {task.examples ? ` ${task.examples} pictures.` : ""} Updated {dayTime(task.updated)}.
      </p>
      {scores ? (
        <div className={teach.scores}>
          {scores.current != null ? (
            <div className={teach.score}>
              <b>{percent(scores.current)}</b>
              <span>before</span>
            </div>
          ) : null}
          {scores.candidate != null ? (
            <div className={teach.score}>
              <b>{percent(scores.candidate)}</b>
              <span>after</span>
            </div>
          ) : null}
          {scores.teacher != null ? (
            <div className={teach.score}>
              <b>{percent(scores.teacher)}</b>
              <span>teacher</span>
            </div>
          ) : null}
          <div className={teach.score}>
            <b>{task.deployed ? "Installed" : task.decision === "reject" ? "Not installed" : "Pending"}</b>
            <span>{task.model_version ?? ""}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Form({ onStarted }: { onStarted: () => void }) {
  const [name, setName] = useState("");
  const [instruction, setInstruction] = useState("");
  const [camera, setCamera] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [verdicts, setVerdicts] = useState<Record<string, "up" | "down">>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [started, setStarted] = useState<string | null>(null);
  const [missing, setMissing] = useState<Set<string>>(new Set()); // pictures the box no longer has (demo fixtures have none)

  const episodes = useFetch((signal) => api.episodes({}, signal));
  const pictures = useMemo(() => {
    const refs: string[] = [];
    for (const e of [...(episodes.data ?? [])].reverse()) {
      const ref = e.keyframe_refs[0];
      if (ref && (!camera || e.camera_id === camera)) refs.push(ref);
      if (refs.length >= 12) break;
    }
    return refs;
  }, [episodes.data, camera]);

  const id = slug(name);
  const ready = id.length > 0 && instruction.trim().length > 3 && !busy;

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const out = await api.teach({
        task: id,
        instruction: instruction.trim(),
        ...(camera ? { camera } : {}),
        ...(label.trim() ? { label: label.trim() } : {}),
        examples: Object.entries(verdicts).map(([frame_ref, verdict]) => ({ frame_ref, verdict })),
      });
      setStarted(out.task);
      onStarted();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  }

  const mark = (ref: string, v: "up" | "down") =>
    setVerdicts((cur) => {
      const next = { ...cur };
      if (next[ref] === v) delete next[ref];
      else next[ref] = v;
      return next;
    });

  return (
    <div className={teach.card}>
      <label className={teach.field}>
        <span className={teach.label}>What should it be called?</span>
        <input className={teach.text} value={name} onChange={(e) => setName(e.target.value)} placeholder="Desk objects" maxLength={40} />
      </label>
      <label className={teach.field}>
        <span className={teach.label}>What should Atlas do with each picture?</span>
        <textarea className={teach.text} value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="Name the small things on the desk, like keys, a wallet or a charger." />
      </label>
      <div className={teach.field}>
        <span className={teach.label}>Which camera?</span>
        <div className={ui.chips} role="radiogroup" aria-label="Camera">
          <button type="button" role="radio" aria-checked={camera === null} className={`${ui.chip} ${camera === null ? ui.chipOn : ""}`} onClick={() => setCamera(null)}>
            Any
          </button>
          {Object.keys(ZONES).map((c) => (
            <button key={c} type="button" role="radio" aria-checked={camera === c} className={`${ui.chip} ${camera === c ? ui.chipOn : ""}`} onClick={() => setCamera(c)}>
              {cameraName(c)}
            </button>
          ))}
        </div>
      </div>
      <label className={teach.field}>
        <span className={teach.label}>Only moments with a… (optional)</span>
        <input className={teach.text} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="person" maxLength={30} />
        <p className={teach.help}>One of the things the camera already recognizes, such as person or laptop.</p>
      </label>
      <div className={teach.field}>
        <span className={teach.label}>Show it a few examples</span>
        <p className={teach.help}>Thumbs up: like this. Thumbs down: not this. Optional, but a handful helps a lot.</p>
        {episodes.loading && !episodes.data ? <Loading what="recent pictures" /> : null}
        {episodes.error ? <ErrorBox what="recent pictures" error={episodes.error} onRetry={episodes.reload} /> : null}
        {!episodes.loading && !episodes.error && pictures.length === 0 ? <Empty title="No pictures yet">Pictures show up here once the cameras have seen something.</Empty> : null}
        <div className={teach.picks}>
          {pictures.map((ref) => {
            const v = verdicts[ref];
            return (
              <div key={ref} className={`${teach.pick} ${v === "up" ? teach.pickUp : v === "down" ? teach.pickDown : ""}`}>
                {missing.has(ref) ? (
                  <div className={teach.gone}>Picture not stored</div>
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={keyframeUrl(ref)} alt="A recent camera picture" loading="lazy" onError={() => setMissing((m) => new Set(m).add(ref))} />
                )}
                <div className={teach.thumbs}>
                  <button type="button" className={`${teach.thumb} ${v === "up" ? teach.thumbOn : ""}`} aria-pressed={v === "up"} aria-label="Thumbs up: like this" onClick={() => mark(ref, "up")}>
                    👍
                  </button>
                  <button type="button" className={`${teach.thumb} ${v === "down" ? teach.thumbOnDown : ""}`} aria-pressed={v === "down"} aria-label="Thumbs down: not this" onClick={() => mark(ref, "down")}>
                    👎
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {error ? <WriteError error={error} /> : null}
      <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!ready} onClick={start}>
        Start teaching
      </button>
      {started ? (
        <div className={teach.ok} role="status">
          <strong>{started}</strong> is waiting for you. Nothing leaves the box until you approve it: open <Link href="/approvals">Approvals</Link>.
        </div>
      ) : null}
    </div>
  );
}

export default function TeachPage() {
  const board = useFetch((signal) => api.learning(signal));
  const { model } = useAtlas();
  const { reload } = board;
  useAtlasEvents(["learning."], reload);

  const tasks = board.data?.tasks ?? [];
  const latest = board.data?.deployments[0];
  return (
    <>
      <div className={ui.pageHead}>
        <h1>Teach</h1>
        <p className={ui.muted}>
          Atlas runs {model ?? "its model"} on the box.
          {latest ? ` Last update: ${latest.model_version}, installed ${dayTime(latest.ts)}.` : ""}
        </p>
      </div>
      <section className={teach.section}>
        <h2>Teach it something new</h2>
        <Form onStarted={reload} />
      </section>
      <section className={teach.section}>
        <h2>Progress</h2>
        <p className={ui.muted}>
          See a training run step by step, and why a model was or was not installed, in the <Link href="/lab">Learning lab</Link>.
        </p>
        {board.loading && !board.data ? <Loading what="what Atlas is learning" /> : null}
        {board.error && !board.data ? <ErrorBox what="what Atlas is learning" error={board.error} onRetry={reload} /> : null}
        {board.data && tasks.length === 0 ? <Empty title="Nothing taught yet">Describe a task above and approve the export; progress shows here.</Empty> : null}
        {tasks.map((t) => (
          <TaskCard key={t.task} task={t} />
        ))}
      </section>
    </>
  );
}
