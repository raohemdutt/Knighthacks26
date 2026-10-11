import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { type DeviceModel, type DeviceTextures, type Materials, buildDevice, makeMaterials, makeTextures } from "@/components/home/deviceModel";
import type { LoopKey, LoopStage, LoopState } from "@/lib/labWords";

/**
 * The learning loop in three dimensions: the Atlas box on the left, the lab's two pods on the right, and the path a
 * fine-tune takes between them. Examples go out to the teacher's pod, the teacher labels them for the trainer's pod,
 * the trainer trains, the test decides, and the model comes back to the box.
 *
 * The box is Xander's device, built by his own builder with his materials (components/home/deviceModel.ts, imported,
 * not copied or edited); the renderer, the room-shaped environment and the lights follow his deviceScene.ts. Only the
 * pods, the paths and the test's ring are drawn here.
 *
 * Nothing here knows about a run. `setStages` is given the five stages as lib/labWords.ts loopStages worked them out
 * from the run's real events, and each path and lamp takes its colour from its stage's state: done, now, waiting,
 * refused or failed, or "none" when the run reported nothing for it. Motion is short and has a reason: the stages
 * light in order once when a run is shown, and beads travel only along a stage that is happening now. With `calm`
 * (reduced motion) nothing moves: one frame is drawn whenever something changes. Loaded on demand (LoopView.tsx), so
 * no other screen and no first paint of this one pays for three.js.
 */

export interface LoopOptions {
  calm: boolean;
  dark: boolean;
  /** The class of a stage's tag in the overlay (story.module.css loopTag). */
  tagClass: string;
}

const ORDER: LoopKey[] = ["out", "label", "train", "test", "box"];
const LIGHT_IN_MS = 420; // each stage lights this long after the one before, once

const P = {
  box: new THREE.Vector3(-3.1, 0, 0.2),
  teacher: new THREE.Vector3(2.5, 0, -1.25),
  trainer: new THREE.Vector3(2.9, 0, 1.35),
  gate: new THREE.Vector3(-0.2, 0.62, 1.95),
};

function arc(points: [number, number, number][]): THREE.CatmullRomCurve3 {
  return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, "catmullrom", 0.5);
}

const CURVES: Partial<Record<LoopKey, THREE.CatmullRomCurve3>> = {
  out: arc([[-2.0, 0.62, -0.1], [-0.6, 1.5, -1.2], [1.0, 1.25, -1.4], [1.72, 0.62, -1.25]]),
  label: arc([[2.5, 0.95, -0.7], [3.3, 1.35, 0.05], [2.9, 0.95, 0.8]]),
  test: arc([[2.1, 0.62, 1.4], [1.1, 0.95, 1.85], [0.25, 0.62, 1.95]]),
  box: arc([[-0.65, 0.62, 1.95], [-1.4, 0.85, 1.6], [-2.05, 0.62, 0.75]]),
};

/** Where each stage's tag sits, in the scene. */
const TAG_AT: Record<LoopKey, THREE.Vector3> = {
  out: new THREE.Vector3(-0.6, 1.9, -1.2), // over the top of the arc out
  label: new THREE.Vector3(2.7, 1.75, -1.6), // above the teacher's pod
  train: new THREE.Vector3(3.4, -0.15, 2.3), // in front of the trainer's pod, under it
  test: new THREE.Vector3(-0.2, -0.2, 2.5), // under the ring
  box: new THREE.Vector3(-3.3, 1.75, -0.2), // above the box
};

interface Lit {
  material: THREE.MeshStandardMaterial;
  state: LoopState;
  shown: number; // 0..1: how far this stage has lit in
}

export class LoopScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.1, 80);
  private key = new THREE.DirectionalLight(0xfff3e4, 2.4);
  private hemi = new THREE.HemisphereLight(0xe8f0ff, 0x8a8f99, 0.95);
  private rim = new THREE.DirectionalLight(0x7fa6ff, 0.5);
  private envMap: THREE.Texture | null = null;
  private textures: DeviceTextures;
  private materials: Materials;
  private device: DeviceModel;
  private own: { dispose(): void }[] = []; // geometry and materials made here
  private lit = {} as Record<LoopKey, Lit>;
  private beads: THREE.Mesh[] = [];
  private beadMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private tags = {} as Record<LoopKey, HTMLElement>;
  private stages: LoopStage[] = [];
  private colors: Record<LoopState, THREE.Color> = { done: new THREE.Color(), now: new THREE.Color(), waiting: new THREE.Color(), failed: new THREE.Color(), refused: new THREE.Color(), none: new THREE.Color() };
  private watcher: ResizeObserver;
  private frame = 0;
  private active = true;
  private gone = false;
  private dark: boolean;
  private start = 0; // when the stages began to light in
  private v = new THREE.Vector3();

  constructor(
    private host: HTMLElement,
    private overlay: HTMLElement,
    private opts: LoopOptions,
  ) {
    this.dark = opts.dark;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const canvas = this.renderer.domElement;
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.setAttribute("aria-hidden", "true"); // the strip of stages under the scene says the same in words
    host.insertBefore(canvas, host.firstChild);

    try {
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      const room = new RoomEnvironment();
      this.envMap = pmrem.fromScene(room, 0.04).texture;
      room.dispose();
      pmrem.dispose();
      this.scene.environment = this.envMap;
    } catch {
      this.envMap = null;
    }

    this.textures = makeTextures();
    this.materials = makeMaterials(this.textures);
    this.device = buildDevice(this.materials, this.textures.blob);
    // the device as its builder left it, set on the floor at the box's place and turned a little towards the lab
    const bounds = new THREE.Box3().setFromObject(this.device.root);
    this.device.root.position.set(P.box.x - (bounds.min.x + bounds.max.x) / 2, -bounds.min.y, P.box.z - (bounds.min.z + bounds.max.z) / 2);
    const turned = new THREE.Group();
    turned.add(this.device.root);
    turned.rotation.y = 0.35;
    this.scene.add(turned);

    this.buildPods();
    this.buildPaths();
    this.buildTags();

    this.key.position.set(-2.6, 6.2, 4.2);
    this.rim.position.set(0.6, 2.4, -5);
    this.scene.add(this.key, this.key.target, this.hemi, this.rim, this.rim.target);
    this.applyTheme();

    this.watcher = new ResizeObserver(() => this.resize());
    this.watcher.observe(host);
    this.resize();
  }

  // ---------- the public surface ----------

  /** The five stages of the run on show. `replay` lights them in order once (never with reduced motion). */
  setStages(stages: LoopStage[], replay: boolean): void {
    this.stages = stages;
    for (const s of stages) {
      const lit = this.lit[s.key];
      if (!lit) continue;
      lit.state = s.state;
      if (replay && !this.opts.calm) lit.shown = 0;
      else lit.shown = 1;
      const tag = this.tags[s.key];
      tag.replaceChildren(document.createTextNode(s.title));
      if (s.value) {
        const value = document.createElement("span");
        value.textContent = s.value;
        tag.appendChild(value);
      }
      tag.dataset.state = s.state;
    }
    if (replay && !this.opts.calm) this.start = performance.now();
    this.kick();
  }

  setTheme(dark: boolean): void {
    if (dark === this.dark) return;
    this.dark = dark;
    this.applyTheme();
    this.kick();
  }

  setActive(on: boolean): void {
    if (on === this.active) return;
    this.active = on;
    if (on) this.kick();
    else if (this.frame) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
    }
  }

  dispose(): void {
    this.gone = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.watcher.disconnect();
    for (const tag of Object.values(this.tags)) tag.remove();
    this.scene.environment = null;
    this.envMap?.dispose();
    this.device.dispose();
    for (const m of Object.values(this.materials)) m.dispose();
    for (const t of Object.values(this.textures)) t.dispose();
    for (const o of this.own) o.dispose();
    this.beadMat.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---------- building ----------

  private keep<T extends { dispose(): void }>(x: T): T {
    this.own.push(x);
    return x;
  }

  private lamp(key: LoopKey): THREE.MeshStandardMaterial {
    const material = this.keep(new THREE.MeshStandardMaterial({ color: 0x9aa3b5, emissive: 0x9aa3b5, emissiveIntensity: 0.2, roughness: 0.45, metalness: 0, transparent: true }));
    this.lit[key] = { material, state: "waiting", shown: 1 };
    return material;
  }

  /** A pod: a low dark machine with a light bar across its front, in the device's own black and steel. */
  private pod(at: THREE.Vector3, turn: number, bar: THREE.Material): void {
    const g = new THREE.Group();
    const body = new THREE.Mesh(this.keep(new THREE.BoxGeometry(1.5, 0.82, 1.05)), this.materials.black);
    body.position.y = 0.47;
    const top = new THREE.Mesh(this.keep(new THREE.BoxGeometry(1.54, 0.05, 1.09)), this.materials.steel);
    top.position.y = 0.9;
    const foot = new THREE.Mesh(this.keep(new THREE.BoxGeometry(1.4, 0.06, 0.95)), this.materials.fin);
    foot.position.y = 0.03;
    const light = new THREE.Mesh(this.keep(new THREE.BoxGeometry(1.1, 0.07, 0.03)), bar);
    light.position.set(0, 0.6, 0.535);
    const shadow = new THREE.Mesh(this.keep(new THREE.PlaneGeometry(2.6, 2.1)), this.keep(new THREE.MeshBasicMaterial({ map: this.textures.blob, transparent: true, opacity: 0.55, depthWrite: false })));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.004;
    g.add(body, top, foot, light, shadow);
    g.position.copy(at);
    g.rotation.y = turn;
    this.scene.add(g);
  }

  private buildPods(): void {
    // the teacher's pod lights with "label"; the trainer's with "train" (training has no path: it happens in the pod)
    this.pod(P.teacher, -0.5, this.lamp("label"));
    this.pod(P.trainer, -0.75, this.lamp("train"));
  }

  private buildPaths(): void {
    for (const key of ["out", "label", "test", "box"] as LoopKey[]) {
      // the teacher's bar already carries "label": its path shares that lamp
      const material = key === "label" ? this.lit.label.material : this.lamp(key);
      const tube = new THREE.Mesh(this.keep(new THREE.TubeGeometry(CURVES[key]!, 48, 0.035, 8, false)), material);
      this.scene.add(tube);
    }
    // the test: a ring the model has to pass through on its way back
    const ring = new THREE.Mesh(this.keep(new THREE.TorusGeometry(0.42, 0.05, 12, 48)), this.lit.test.material);
    ring.position.copy(P.gate);
    ring.rotation.y = Math.PI / 2 - 0.2;
    const post = new THREE.Mesh(this.keep(new THREE.CylinderGeometry(0.03, 0.03, 0.2, 10)), this.materials.steel);
    post.position.set(P.gate.x, 0.1, P.gate.z);
    this.scene.add(ring, post);
    const sphere = this.keep(new THREE.SphereGeometry(0.07, 12, 10));
    for (let i = 0; i < 3; i++) {
      const bead = new THREE.Mesh(sphere, this.beadMat);
      bead.visible = false;
      this.beads.push(bead);
      this.scene.add(bead);
    }
  }

  private buildTags(): void {
    for (const key of ORDER) {
      const tag = document.createElement("div");
      tag.className = this.opts.tagClass;
      this.overlay.appendChild(tag);
      this.tags[key] = tag;
    }
  }

  // ---------- light and colour ----------

  private applyTheme(): void {
    const css = getComputedStyle(document.documentElement);
    const read = (name: string, fallback: string) => new THREE.Color((css.getPropertyValue(name) || fallback).trim() || fallback);
    this.colors.done.copy(read("--ok", "#0f7b5f"));
    this.colors.now.copy(read("--accent", "#5b8cff"));
    this.colors.failed.copy(read("--danger", "#c0332b"));
    this.colors.refused.copy(this.colors.failed);
    this.colors.waiting.copy(read("--unknown", "#8a95ab"));
    this.colors.none.copy(this.colors.waiting);
    if (this.dark) {
      this.key.color.setHex(0xdfe8ff);
      this.key.intensity = 1.4;
      this.hemi.color.setHex(0x5e74a8);
      this.hemi.groundColor.setHex(0x12151c);
      this.hemi.intensity = 0.5;
      this.rim.intensity = 2.4;
      this.scene.environmentIntensity = 0.5;
    } else {
      this.key.color.setHex(0xfff3e4);
      this.key.intensity = 2.4;
      this.hemi.color.setHex(0xe8f0ff);
      this.hemi.groundColor.setHex(0x8a8f99);
      this.hemi.intensity = 0.95;
      this.rim.intensity = 0.5;
      this.scene.environmentIntensity = 1.0;
    }
  }

  // ---------- the frame ----------

  private resize(): void {
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // the whole loop is about 8.5 wide: stand far enough back that it fits the width, however narrow the phone
    const half = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fitWidth = 5.7 / (Math.tan(half) * this.camera.aspect);
    const dist = Math.max(9.2, fitWidth);
    this.camera.position.set(0.1, dist * 0.42, dist * 0.91);
    this.camera.lookAt(-0.1, 0.55, 0.35);
    this.camera.updateProjectionMatrix();
    this.kick();
  }

  /** Draw now, and keep drawing only while something is still moving. */
  private kick(): void {
    if (this.gone || !this.active || this.frame) return;
    this.frame = requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    this.frame = 0;
    if (this.gone) return;
    let moving = false;
    const since = now - this.start;
    ORDER.forEach((key, i) => {
      const lit = this.lit[key];
      if (lit.shown < 1) {
        lit.shown = Math.min(1, Math.max(0, (since - i * LIGHT_IN_MS) / LIGHT_IN_MS));
        if (lit.shown < 1) moving = true;
      }
      const base = this.colors.waiting;
      const to = this.colors[lit.state];
      const k = lit.shown;
      const pulse = lit.state === "now" && !this.opts.calm ? 0.75 + 0.25 * Math.sin(now / 320) : 1;
      lit.material.color.copy(base).lerp(to, k);
      lit.material.emissive.copy(lit.material.color);
      const strength = lit.state === "waiting" || lit.state === "none" ? 0.12 : lit.state === "now" ? 1.5 : 0.85;
      lit.material.emissiveIntensity = (0.12 + (strength - 0.12) * k) * pulse;
      lit.material.opacity = lit.state === "none" ? 0.35 : 1;
    });
    // beads travel only along a stage that is happening now
    const live = this.stages.find((s) => s.state === "now" && CURVES[s.key]);
    this.beads.forEach((bead, i) => {
      bead.visible = !!live && !this.opts.calm;
      if (bead.visible) {
        CURVES[live!.key]!.getPointAt((now / 1800 + i / this.beads.length) % 1, bead.position);
        this.beadMat.color.copy(this.colors.now);
      }
    });
    if (this.stages.some((s) => s.state === "now") && !this.opts.calm) moving = true;
    this.renderer.render(this.scene, this.camera);
    this.placeTags();
    if (moving && this.active) this.frame = requestAnimationFrame(this.tick);
  };

  private placeTags(): void {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    for (const key of ORDER) {
      this.v.copy(TAG_AT[key]).project(this.camera);
      const tag = this.tags[key];
      const half = tag.offsetWidth / 2 + 6; // a tag is centred on its point: keep the whole of it inside the stage
      const x = Math.min(w - half, Math.max(half, (this.v.x * 0.5 + 0.5) * w));
      const y = Math.max(tag.offsetHeight + 6, (-this.v.y * 0.5 + 0.5) * h);
      tag.style.left = `${x.toFixed(1)}px`;
      tag.style.top = `${y.toFixed(1)}px`;
      tag.style.opacity = String(0.35 + 0.65 * this.lit[key].shown);
    }
  }
}
