"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { duration } from "@/lib/lab";
import { DEPLOY_STAGES, type DeployAnswer, type DeployRun, type DeployStageState, type DeployView, callText, energyLine, shortVersion } from "@/lib/labDeploy";
import dep from "./deploy.module.css";
import lab from "./lab.module.css";

/**
 * The On the box card's live part (L35): what the box serves now and what it served before, the two buttons that
 * swap models on the box itself ("Put v2 on the box", "Back to base"), the five stages of a running swap lit from
 * GET /api/learning/lab/deploy every 2 s, and the before/after question asked straight to the model server.
 * Shown whatever the run's timeline says, so the version on the box is always on the card.
 */

const RUNNING_POLL_MS = 2000;
const IDLE_POLL_MS = 15000;
const DEFAULT_QUESTION = "Have you seen my water bottle?";

const STAGE_CLASS: Record<DeployStageState, string> = {
  waiting: "",
  skipped: "",
  running: lab.stage_now ?? "",
  done: lab.stage_done ?? "",
  failed: lab.stage_failed ?? "",
};
const STAGE_WORD: Record<DeployStageState, string> = { waiting: "waiting", skipped: "skipped", running: "now", done: "done", failed: "failed" };

function problem(e: unknown): string {
  if (e instanceof ApiError) return e.needsToken ? "This box asks for a token first (Settings, top right)." : e.message;
  return String(e);
}

function StageStrip({ run }: { run: DeployRun }) {
  return (
    <ol className={lab.stages} aria-label="Where the swap is">
      {DEPLOY_STAGES.map((s) => {
        const stage = run.stages.find((x) => x.name === s.key);
        const state = stage?.state ?? "waiting";
        return (
          <li key={s.key} className={`${lab.stage} ${STAGE_CLASS[state]}`}>
            <span className={lab.stageDot} aria-hidden="true" />
            {s.title}
            {stage?.seconds !== null && stage?.seconds !== undefined && state !== "waiting" ? <span className={dep.secs}>{stage.seconds.toFixed(0)} s</span> : null}
            <span className={lab.srOnly}>: {STAGE_WORD[state]}</span>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * A swap the record calls failed while the box serves its very version and the health route says ok: a restart cut the
 * swap short after the switch (the gateway was restarted mid-check). Said plainly, not as a failure.
 */
export function swapInterrupted(run: DeployRun, current: string | null, health: { ok: boolean; model?: string | null } | null): boolean {
  return run.state === "failed" && !!run.version && current === run.version && !!health?.ok && (!health.model || health.model === run.version);
}

function Outcome({ run, interrupted }: { run: DeployRun; interrupted: boolean }) {
  const target = run.version ?? "the model";
  if (interrupted) {
    return (
      <p className={dep.line}>
        The last swap was interrupted by a restart; the box is serving {target} and healthy.
      </p>
    );
  }
  const took = run.seconds !== null ? duration(run.seconds) : null;
  if (run.state === "running") {
    return (
      <p className={dep.line}>
        Swapping to <b>{target}</b>
        {took ? ` · ${took} so far` : ""}. The model gives no answers from the restart until the check.
      </p>
    );
  }
  if (run.state === "done") {
    const told = run.stages.find((s) => s.name === "publish")?.state;
    return (
      <p className={`${dep.line} ${dep.lineGood}`}>
        {target} is serving. The swap took {took ?? "an unknown time"}.
        {told === "failed" ? <span className={dep.sub}>The app was not told, so the Teach screen does not list it yet.</span> : null}
      </p>
    );
  }
  if (run.state === "failed") {
    const back = run.result?.rolled_back_to;
    return (
      <p className={`${dep.line} ${dep.lineBad}`}>
        {back ? `The swap to ${target} failed and rolled back to ${back}` : `The swap to ${target} failed`}
        {took ? ` after ${took}` : ""}.
        {run.result?.message ? <span className={dep.sub}>{run.result.message}</span> : null}
      </p>
    );
  }
  return null;
}

function Ask({ disabled }: { disabled: boolean }) {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [answers, setAnswers] = useState<DeployAnswer[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = () => {
    if (!question.trim() || busy) return;
    setBusy(true);
    setError(null);
    api
      .labDeployAsk(question.trim())
      .then((a) => setAnswers((list) => [a, ...list].slice(0, 4)))
      .catch((e: unknown) => setError(problem(e)))
      .finally(() => setBusy(false));
  };

  return (
    <div className={dep.ask}>
      <h3 className={dep.askHead}>Ask the model itself</h3>
      <p className={lab.note}>
        Straight to the model server with the eval&apos;s prompt, so the answer shows which tool the serving model picks. The app&apos;s own Ask answers
        this kind of question in code first, so it reads the same on both models.
      </p>
      <form
        className={dep.askRow}
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <input className={ui.input} value={question} onChange={(e) => setQuestion(e.target.value)} aria-label="Question for the model" maxLength={300} />
        <button type="submit" className={ui.btn} disabled={busy || disabled || !question.trim()}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </form>
      {disabled ? <p className={lab.note}>The model is being swapped; ask again when the check is done.</p> : null}
      {error ? <p className={`${dep.line} ${dep.lineBad}`}>{error}</p> : null}
      {answers.length ? (
        <ul className={dep.answers} aria-live="polite">
          {answers.map((a, i) => {
            const picked = a.calls.length ? a.calls.map(callText).join(", ") : null;
            const right = a.expected && a.calls.length ? callText(a.calls[0]!) === callText(a.expected[0]!) : null;
            return (
              <li key={`${a.serving}-${a.seconds}-${i}`} className={dep.answer}>
                <span className={lab.statLabel}>
                  {a.serving ?? "the model"} · {a.seconds.toFixed(1)} s
                </span>
                <span className={`${lab.code} ${ui.breakWords}`}>{picked ?? (a.text || "no answer")}</span>
                {a.expected ? (
                  <span className={`${dep.verdict} ${right ? dep.good : dep.bad}`}>
                    {right ? "the call the eval expects" : `the eval expects ${a.expected.map(callText).join(", ")}`}
                  </span>
                ) : null}
                <Cost a={a} />
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

/** Beat 6: what the answer cost on the box, from the Jetson's power sampler over the answer (gross, idle beside it). */
function Cost({ a }: { a: DeployAnswer }) {
  const line = energyLine(a.energy);
  if (line === null) return null;
  const e = a.energy;
  return (
    <span className={dep.cost}>
      <b className={e ? dep.costOn : dep.costOff}>{line}</b>
      {e ? (
        <span className={dep.sub}>
          {e.seconds.toFixed(1)} s on the {e.sampler} sampler, gross{e.idle_w !== null ? `; the box idles at ${e.idle_w} W` : ""}
          {e.low_confidence ? `; ${e.samples ?? "few"} samples, a rough figure` : ""}
        </span>
      ) : null}
    </span>
  );
}

export function DeployButton({ onBoxChanged }: { onBoxChanged?: () => void }) {
  const [view, setView] = useState<DeployView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [health, setHealth] = useState<{ ok: boolean; model?: string | null } | null>(null);
  const lastState = useRef<string | null>(null);
  const changed = useRef(onBoxChanged);
  changed.current = onBoxChanged;

  const load = useCallback((signal?: AbortSignal) => {
    // the health route says whether the box answers, and with which model (read only for a failed swap's wording)
    api
      .health(signal)
      .then((h) => setHealth({ ok: !!h.ok, model: h.model ?? null }))
      .catch(() => setHealth(null));
    return api
      .labDeploy(signal)
      .then((v) => {
        setView(v);
        setError(null);
        const was = lastState.current;
        lastState.current = v.deploy.state;
        if (was === "running" && v.deploy.state !== "running") changed.current?.();
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(problem(e));
      });
  }, []);

  const running = view?.deploy.state === "running";
  useEffect(() => {
    const ctl = new AbortController();
    void load(ctl.signal);
    const id = window.setInterval(() => void load(ctl.signal), running ? RUNNING_POLL_MS : IDLE_POLL_MS);
    return () => {
      ctl.abort();
      window.clearInterval(id);
    };
  }, [load, running]);

  const start = (call: () => Promise<DeployView>) => {
    setSending(true);
    setError(null);
    call()
      .then((v) => {
        setView(v);
        lastState.current = v.deploy.state;
      })
      .catch((e: unknown) => setError(problem(e)))
      .finally(() => setSending(false));
  };

  if (!view) {
    return error ? <p className={`${dep.line} ${dep.lineBad}`}>Could not read the box&apos;s models ({error}).</p> : <p className={lab.waiting}>Reading the box&apos;s models…</p>;
  }

  const { current, previous, base, featured, deploy } = view;
  const busy = running || sending;
  const next = featured;
  const stageDone = (name: string) => deploy.stages.some((s) => s.name === name && s.state === "done");
  // While a swap runs the link moves before the model server does: say what actually answers.
  const restarted = running && stageDone("restart");
  const nowName = running ? (restarted ? deploy.version : deploy.from) : current;
  const nowLabel = running ? (restarted ? "starting up, not answering yet" : "serving until the restart") : "on the box now";
  // Before the restart the right tile is the model coming in; after it, the one that went out.
  const beforeName = running ? (restarted ? deploy.from : deploy.version) : previous;
  const beforeLabel = running && !restarted ? "next, after the restart" : "before it";
  return (
    <div className={dep.panel}>
      <div className={`${lab.swap} ${dep.links}`}>
        <div className={`${lab.model} ${lab.modelOn}`}>
          <span className={lab.statLabel}>{nowLabel}</span>
          <span className={lab.modelName}>{nowName ?? "no current link"}</span>
        </div>
        <svg className={lab.swapArrow} viewBox="0 0 44 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M40 12H6M14 5l-8 7 8 7" />
        </svg>
        <div className={`${lab.model} ${lab.modelOff}`}>
          <span className={lab.statLabel}>{beforeLabel}</span>
          <span className={lab.modelName}>{beforeName ?? "nothing recorded"}</span>
        </div>
      </div>

      <div className={dep.buttons}>
        <button
          type="button"
          className={`${ui.btn} ${ui.btnPrimary}`}
          disabled={busy || !next || next === current}
          onClick={() => next && start(() => api.labDeployStart(next))}
        >
          {!next ? "No trained model on the box" : next === current && !running ? `${shortVersion(next)} is on the box` : `Put ${shortVersion(next)} on the box`}
        </button>
        <button
          type="button"
          className={ui.btn}
          disabled={busy || !base || current === base}
          onClick={() => base && start(() => api.labDeployRollback(base))}
        >
          Back to base
        </button>
      </div>
      {next && next !== current ? (
        <p className={lab.note}>
          {next} is on the box&apos;s disk already: nothing is downloaded, and its files are checked against the manifest before the switch.
        </p>
      ) : null}

      {deploy.state !== "idle" ? (
        <div aria-live="polite">
          {swapInterrupted(deploy, current, health) ? null : <StageStrip run={deploy} />}
          <Outcome run={deploy} interrupted={swapInterrupted(deploy, current, health)} />
        </div>
      ) : null}
      {error ? <p className={`${dep.line} ${dep.lineBad}`}>{error}</p> : null}

      <Ask disabled={running} />
    </div>
  );
}
