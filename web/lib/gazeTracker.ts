/**
 * The camera half of gaze: opt-in, in this browser only.
 *
 *  - The face model (MediaPipe Face Landmarker, Apache 2.0, with iris points and the face transform) and its wasm
 *    runtime are served by this app from /gaze/ (web/public/gaze/, see its README). Nothing is fetched from another origin.
 *  - The camera is asked for only when the person switches "Use my eyes to guide me" on. The track is stopped when the
 *    switch goes off, when the tab is hidden, and when the page goes away.
 *  - A frame is handed to the model and dropped. No frame, landmark or gaze point is uploaded, drawn to a canvas for
 *    export, logged or stored. localStorage keeps two things under "atlas.gaze": the on/off choice and the calibration
 *    (its weights, the head pose it was made at and its measured error).
 *
 * One tracker for the whole app (`gaze`), so every screen's guide shares the camera and the calibration.
 */

import { Fixation, OneEuro, drifted, errorRadius, features, predict, quantise, type Calibration, type Features, type Landmark } from "./gaze";

export type GazeStatus =
  | "off"
  | "starting"
  | "on"
  | "insecure" // plain http: the browser will not open a camera
  | "unsupported"
  | "no-model" // web/public/gaze/ is missing its files
  | "denied"
  | "error";

export interface GazeState {
  status: GazeStatus;
  /** The person's remembered choice. */
  wanted: boolean;
  calibration: Calibration | null;
  /** "ok" one face; "none" nobody; "many" more than one (then gaze is not used: whose eyes?). */
  face: "ok" | "none" | "many";
  /** The head has moved far from where it was calibrated, or the calibration is old: a re-check is worth it. */
  drift: "moved" | "old" | null;
  detail: string | null;
}

export interface GazeEstimate {
  /** The filtered estimate, as fractions of the viewport (may be a little outside 0..1). */
  x: number;
  y: number;
  /** The unfiltered estimate of the latest look, for the debug overlay. */
  raw: { x: number; y: number };
  /** The cell of the measured grid it falls in. */
  col: number;
  row: number;
  cols: number;
  rows: number;
  /** The measured mean miss, in pixels of this viewport: the radius the true gaze point is expected within. */
  radius: number;
  /** How long the eyes have rested where they are (ms), and where that is (fractions of the viewport). */
  rest: { ms: number; x: number; y: number };
  at: number;
}

const KEY = "atlas.gaze";
const VERSION = 2;
const MODEL_URL = "/gaze/face_landmarker.task";
const BUNDLE_URL = "/gaze/vision_bundle.mjs";
type VisionModule = {
  FilesetResolver: { forVisionTasks(wasm: string): Promise<unknown> };
  FaceLandmarker: { createFromOptions(files: unknown, options: Record<string, unknown>): Promise<unknown> };
};
const WASM_URL = "/gaze/wasm";
const INTERVAL_MS = 40; // up to 25 looks a second
const STALE_MS = 600;

interface Landmarker {
  detectForVideo(video: HTMLVideoElement, t: number): { faceLandmarks: Landmark[][]; facialTransformationMatrixes?: { data: ArrayLike<number> }[] };
  close(): void;
}

function load(): { wanted: boolean; calibration: Calibration | null } {
  try {
    const raw = JSON.parse(window.localStorage.getItem(KEY) ?? "{}") as { v?: number; on?: boolean; calibration?: Calibration };
    const c = raw.calibration;
    const ok = raw.v === VERSION && c && Array.isArray(c.wx) && Array.isArray(c.wy) && Array.isArray(c.mean) && Array.isArray(c.scale) && Array.isArray(c.pose) && (c.model === "linear" || c.model === "poly");
    return { wanted: raw.on === true, calibration: ok ? c : null };
  } catch {
    return { wanted: false, calibration: null };
  }
}

class Tracker {
  private state: GazeState = { status: "off", wanted: false, calibration: null, face: "none", drift: null, detail: null };
  private listeners = new Set<() => void>();
  private featureListeners = new Set<(f: Features | null) => void>();
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private landmarker: Landmarker | null = null;
  private timer: number | undefined;
  private filter = new OneEuro();
  private fixation = new Fixation();
  private last: GazeEstimate | null = null;
  private sim: GazeEstimate | null = null;
  private simOn = false;
  private booted = false;
  private run = 0;

  /** Read the remembered choice once, in the browser; start again if the person had it on. */
  boot(): void {
    if (this.booted || typeof window === "undefined") return;
    this.booted = true;
    const saved = load();
    this.set({ wanted: saved.wanted, calibration: saved.calibration });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.release();
      else if (this.state.wanted && this.state.status === "off") void this.start();
    });
    window.addEventListener("pagehide", () => this.release());
    if (saved.wanted) void this.start();
  }

  get = (): GazeState => this.state;

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  /** Raw features of each look, for the calibration. null: no single face in this look, or a blink. */
  onFeatures(fn: (f: Features | null) => void): () => void {
    this.featureListeners.add(fn);
    return () => this.featureListeners.delete(fn);
  }

  /** Why the switch cannot be turned on here, or null when it can. */
  blocked(): string | null {
    if (typeof window === "undefined") return null;
    if (!window.isSecureContext) return "The camera needs a secure address. Open this box over https (ATLAS_HTTPS=1 on the gateway) and the switch works.";
    if (!navigator.mediaDevices?.getUserMedia) return "This browser cannot open a camera.";
    return null;
  }

  private set(patch: Partial<GazeState>): void {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn();
  }

  private save(): void {
    try {
      window.localStorage.setItem(KEY, JSON.stringify({ v: VERSION, on: this.state.wanted, calibration: this.state.calibration }));
    } catch {
      /* private mode: the choice lasts for this visit */
    }
  }

  async enable(): Promise<void> {
    this.set({ wanted: true });
    this.save();
    await this.start();
  }

  disable(): void {
    this.release();
    this.set({ wanted: false, status: "off", face: "none", drift: null, detail: null });
    this.save();
  }

  setCalibration(c: Calibration | null): void {
    this.filter.clear();
    this.fixation = new Fixation(c?.validation ? Math.max(0.06, Math.min(0.3, 2.5 * c.validation.ofWidth)) : 0.12);
    this.last = null;
    this.set({ calibration: c, drift: null });
    this.save();
  }

  /** Is gaze good enough to use right now? Off, hidden, no face, two faces, unchecked or poor calibration, or stale: no. */
  usable(): boolean {
    const s = this.state;
    return s.status === "on" && s.face === "ok" && s.calibration !== null && s.calibration.validation !== null && s.calibration.quality !== "poor";
  }

  /**
   * A scripted stand-in for the camera (?gazeSim=...): the page feeds estimates itself so the attention check can be shown
   * and tested without a webcam. It goes through the same snap and fixation rules; the screen says it is simulated.
   */
  simulate(e: GazeEstimate | null, on = true): void {
    this.simOn = on;
    this.sim = e;
  }

  simulated(): boolean {
    return this.simOn;
  }

  /** The current estimate, or null when gaze should not be used now. */
  estimate(): GazeEstimate | null {
    if (this.simOn) return this.sim;
    if (!this.usable() || !this.last) return null;
    if (document.visibilityState !== "visible" || performance.now() - this.last.at > STALE_MS) return null;
    return this.last;
  }

  private async start(): Promise<void> {
    if (this.stream || this.state.status === "starting") return;
    const why = this.blocked();
    if (why) {
      this.set({ status: window.isSecureContext ? "unsupported" : "insecure", detail: why });
      return;
    }
    const mine = ++this.run;
    this.set({ status: "starting", detail: null });
    try {
      // the model first: without it there is no reason to open the camera
      const head = await fetch(MODEL_URL, { method: "HEAD" });
      if (!head.ok || (head.headers.get("content-type") ?? "").includes("text/html")) {
        this.set({ status: "no-model", detail: "The gaze model is not installed on this box (web/public/gaze/). The pointer still works without it." });
        return;
      }
      if (!this.landmarker) {
        // Loaded at run time from this app's own files (the prebuild step copies the bundle to public/gaze/), not bundled:
        // a checkout that has not installed the package still builds, and the switch then says the runtime is missing.
        const bundle = await fetch(BUNDLE_URL, { method: "HEAD" });
        if (!bundle.ok || (bundle.headers.get("content-type") ?? "").includes("text/html")) {
          this.set({ status: "no-model", detail: "The eye-tracking runtime is not installed on this box (run npm install, then build). The pointer still works without it." });
          return;
        }
        const vision = (await import(/* webpackIgnore: true */ BUNDLE_URL)) as VisionModule;
        const files = await vision.FilesetResolver.forVisionTasks(WASM_URL);
        this.landmarker = (await vision.FaceLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
          runningMode: "VIDEO",
          numFaces: 2,
          outputFacialTransformationMatrixes: true,
        })) as unknown as Landmarker;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      if (mine !== this.run || !this.state.wanted || document.visibilityState !== "visible") {
        stream.getTracks().forEach((t) => t.stop());
        if ((this.state.status as GazeStatus) === "starting") this.set({ status: "off" });
        return;
      }
      const video = document.createElement("video"); // never added to the page
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      this.stream = stream;
      this.video = video;
      this.set({ status: "on", face: "none", detail: null });
      this.timer = window.setInterval(() => this.look(), INTERVAL_MS);
    } catch (err) {
      const name = (err as { name?: string })?.name ?? "";
      const denied = name === "NotAllowedError" || name === "SecurityError";
      this.release();
      this.set({
        status: denied ? "denied" : name === "NotFoundError" || name === "OverconstrainedError" ? "unsupported" : "error",
        detail: denied ? "The camera was not allowed. Allow it for this address in the browser, then switch this on again." : name === "NotFoundError" ? "No camera was found." : "Eye tracking could not start on this device.",
      });
    }
  }

  private look(): void {
    const video = this.video;
    if (!video || !this.landmarker || video.readyState < 2) return;
    let faces: Landmark[][];
    let matrix: ArrayLike<number> | null = null;
    const now = performance.now();
    try {
      const out = this.landmarker.detectForVideo(video, now);
      faces = out.faceLandmarks ?? [];
      matrix = out.facialTransformationMatrixes?.[0]?.data ?? null;
    } catch {
      return;
    }
    const face = faces.length === 1 ? "ok" : faces.length === 0 ? "none" : "many";
    const f = face === "ok" ? features(faces[0]!, matrix) : null;
    for (const fn of this.featureListeners) fn(f);
    const c = this.state.calibration;
    let drift = this.state.drift;
    if (f && c && c.validation && c.quality !== "poor") {
      const raw = predict(c, f);
      const p = this.filter.push(raw, now);
      const rest = this.fixation.push(p, now);
      this.last = { ...quantise(p, c.cols, c.rows), x: p.x, y: p.y, raw, cols: c.cols, rows: c.rows, radius: errorRadius(c, window.innerWidth), rest, at: now };
      drift = drifted(c, f, Date.now());
    } else if (face !== "ok") {
      // a blink keeps the last estimate for a moment (it goes stale by itself); a lost face drops it
      this.filter.clear();
      this.fixation.clear();
      this.last = null;
    }
    if (face !== this.state.face || drift !== this.state.drift) this.set({ face, drift });
  }

  /** Stop the camera. The choice stays as it was. */
  private release(): void {
    this.run++;
    window.clearInterval(this.timer);
    this.timer = undefined;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.video = null;
    this.filter.clear();
    this.fixation.clear();
    this.last = null;
    if (this.state.status === "on" || this.state.status === "starting") this.set({ status: "off", face: "none" });
  }
}

export const gaze = new Tracker();
