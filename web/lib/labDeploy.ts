/**
 * The On the box card's deploy button (L35): types for GET/POST /api/learning/lab/deploy, /deploy/rollback and
 * /deploy/ask (contexts/learning/lab_deploy.py). The swap itself runs on the box (jetson/atlas_swap.py).
 */

export type DeployState = "idle" | "running" | "done" | "failed";
export type DeployStageState = "waiting" | "running" | "done" | "failed" | "skipped";
export type DeployStageName = "verify" | "switch" | "restart" | "check" | "publish";

export interface DeployVersion {
  version: string;
  base: string;
  task: string | null;
  created: string;
  current: boolean;
  previous: boolean;
  /** manifest.json reads, every file is there, and no stale .verified.json says otherwise */
  verified: boolean;
  record: "fresh" | "stale" | "none";
}

export interface DeployStage {
  name: DeployStageName;
  state: DeployStageState;
  seconds: number | null;
}

export interface DeployResult {
  ok: boolean;
  returncode: number | null;
  message: string | null;
  rolled_back_to: string | null;
}

export interface DeployRun {
  state: DeployState;
  action: "deploy" | "rollback" | null;
  version: string | null;
  from: string | null;
  started: string | null;
  ended: string | null;
  seconds: number | null;
  stages: DeployStage[];
  log_tail: string[];
  result: DeployResult | null;
}

export interface DeployView {
  versions: DeployVersion[];
  current: string | null;
  previous: string | null;
  base: string | null;
  /** the newest verified version that is not current (and not the base) */
  suggested: string | null;
  deploy: DeployRun;
}

export interface DeployAnswer {
  question: string;
  serving: string | null;
  text: string;
  calls: { name: string; arguments: Record<string, unknown> }[];
  expected: { name: string; arguments: Record<string, unknown> }[] | null;
  seconds: number;
  prompt: "eval" | "eval template";
}

export const DEPLOY_STAGES: { key: DeployStageName; title: string }[] = [
  { key: "verify", title: "verify" },
  { key: "switch", title: "switch" },
  { key: "restart", title: "restart" },
  { key: "check", title: "check" },
  { key: "publish", title: "published" },
];

/** "v2" for atlas-tool-calling-v2-a100-f2r4, else the version itself. */
export function shortVersion(version: string): string {
  const m = /(?:^|-)(v\d+)(?:-|$)/.exec(version);
  return m ? m[1]! : version;
}

/** last_seen(label="water bottle") */
export function callText(call: { name: string; arguments: Record<string, unknown> }): string {
  const args = Object.entries(call.arguments ?? {})
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
  return `${call.name}(${args})`;
}
