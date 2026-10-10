import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { FloorTrack, RoomFurniture, RoomPlan } from "@/lib/api";
import { humanize } from "@/lib/format";
import { createFsrPipeline, type FsrInfo, type FsrPipeline } from "@/lib/fsr";
import {
  C,
  WALLS,
  type Wall,
  type WallParts,
  binColor,
  binDecal,
  blobTexture,
  buildBin,
  buildFurniture,
  buildPerson,
  buildThing,
  buildWall,
  chevronTexture,
  contactTexture,
  glowTexture,
  sunTexture,
  posterTexture,
  rugTexture,
  wallPoint,
  wallTurn,
  woodTexture,
} from "./roomModels";

/**
 * The room twin's three.js scene, with no React in it: RoomTwin.tsx owns one of these and tells it what changed.
 *
 *   setPlan   the room itself: floor, walls with door and window, furniture, bins, zones on the floor
 *   setTracks people and things, moved smoothly between the once-a-second observations
 *   setPins   memory pins ("your keys are here")
 *   setRoutes Sort It: glowing routes on the floor, the right bin marked, the wrong one dimmed
 *
 * Cost, sized for a phone: all static geometry is merged into about fifteen draw calls, one 1024 px shadow map for
 * the sun through the window (redrawn only when something moved), contact shading painted once onto the floor, and
 * frames are drawn only while something is moving. Everything renders through the FSR 1 pipeline (lib/fsr.ts).
 * Names are DOM labels over the canvas, laid out every drawn frame so that none overlaps another.
 */

export type TwinView = "iso" | "top";

export interface SceneRoute {
  player: number;
  track: string | null;
  points: [number, number][];
}

export interface ScenePin {
  id: string;
  text: string; // what the chip says
  x: number;
  y: number;
  z: number;
}

export interface SceneClasses {
  label: string;
  person: string;
  thing: string;
  bin: string;
  zone: string;
  pin: string;
  pinOn: string;
  target: string;
  dot: string;
}

export interface SceneOptions {
  scale: number;
  simulate: "webgl1" | "badshader" | null;
  compare: boolean;
  split: number;
  calm: boolean;
  view: TwinView;
  classes: SceneClasses;
  onInfo: (info: FsrInfo) => void;
  onPick: (id: string) => void;
}

export const ROUTE_COLORS = [0x5ce1c6, 0xffa94d, 0xb794f6];
const MEMBER_COLORS = [0xe8724a, 0x8e6fd8, 0x3fa7d6, 0xd6a13f];
const PERSON = 0x7d8ea3;
const ZONE_COLORS: Record<string, number> = { door: 0xf0b85a, desk: 0x5ce1c6, recycling: 0x4f9df0, trash: 0xaab3bd };
const MOVE_MS = 1000; // observations arrive once a second: a figure takes that long to reach the new spot
const FOV = 30;

interface Label {
  el: HTMLElement;
  anchor: THREE.Vector3;
  priority: number;
  must: boolean;
  w: number;
  h: number;
  off?: boolean; // not shown at all (a thing in a hand, or one whose pin already names it)
}

interface Figure {
  key: string;
  person: boolean;
  group: THREE.Group;
  body?: THREE.Group;
  shirt?: THREE.MeshStandardMaterial;
  ring?: THREE.Mesh;
  shade?: THREE.Mesh;
  from: THREE.Vector3;
  to: THREE.Vector3;
  t0: number;
  born: number;
  heading: number;
  wantHeading: number;
  seated: boolean;
  heldBy: string | null;
  label: Label;
  top: number; // how far above the figure its name floats
}

function hash(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function disposeTree(root: THREE.Object3D) {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    for (const m of mats) {
      (m as THREE.MeshBasicMaterial).map?.dispose?.();
      m.dispose();
    }
  });
}

/** Round the corners of a polyline (Chaikin), keeping both ends where they are. */
function rounded(points: [number, number][], rounds = 2): [number, number][] {
  let pts = points;
  for (let r = 0; r < rounds && pts.length > 2; r++) {
    const next: [number, number][] = [pts[0] as [number, number]];
    for (let i = 0; i < pts.length - 1; i++) {
      const [ax, ay] = pts[i] as [number, number];
      const [bx, by] = pts[i + 1] as [number, number];
      if (i > 0) next.push([ax * 0.75 + bx * 0.25, ay * 0.75 + by * 0.25]);
      if (i < pts.length - 2) next.push([ax * 0.25 + bx * 0.75, ay * 0.25 + by * 0.75]);
    }
    next.push(pts[pts.length - 1] as [number, number]);
    pts = next;
  }
  return pts;
}

/** A flat strip along floor points (plan metres), `half` wide each side; u runs along it in metres. */
function ribbon(points: [number, number][], half: number, y: number): THREE.BufferGeometry {
  const pos: number[] = [];
  const uv: number[] = [];
  const index: number[] = [];
  let run = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i] as [number, number];
    const a = points[Math.max(0, i - 1)] as [number, number];
    const b = points[Math.min(points.length - 1, i + 1)] as [number, number];
    let tx = b[0] - a[0];
    let ty = b[1] - a[1];
    const len = Math.hypot(tx, ty) || 1;
    tx /= len;
    ty /= len;
    if (i > 0) run += Math.hypot(p[0] - a[0], p[1] - a[1]);
    pos.push(p[0] - ty * half, y, p[1] + tx * half, p[0] + ty * half, y, p[1] - tx * half);
    uv.push(run, 0, run, 1);
    if (i > 0) {
      const k = (i - 1) * 2;
      index.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(index);
  return g;
}

export class RoomScene {
  readonly pipeline: FsrPipeline;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 80);
  private controls: OrbitControls;
  private sun = new THREE.DirectionalLight(0xfff0dc, 2.5);
  private watcher: ResizeObserver;
  private frame = 0;
  private dirty = true;
  private shadowDirty = true;
  private still = 0;
  private gone = false;
  private glide: { from: THREE.Vector3; to: THREE.Vector3; target: THREE.Vector3; aim: THREE.Vector3; start: number } | null = null;
  private view: TwinView;
  private plan: RoomPlan | null = null;
  private room = new THREE.Group(); // rebuilt with the plan
  private zones = new THREE.Group();
  private things = new THREE.Group(); // figures
  private pinsGroup = new THREE.Group();
  private routeGroup = new THREE.Group();
  private walls = new Map<Wall, WallParts & { hidden: boolean }>();
  private bins = new Map<string, { f: RoomFurniture; group: THREE.Group; body: THREE.MeshStandardMaterial; face: THREE.MeshBasicMaterial; label: Label }>();
  private figures = new Map<string, Figure>();
  private labels = new Set<Label>();
  private staticLabels: Label[] = [];
  private pinLabels: Label[] = [];
  private routeLabels: Label[] = [];
  private routes: SceneRoute[] = [];
  private bob: THREE.Object3D[] = [];
  private chevrons: THREE.Texture[] = [];
  private materials: { matte: THREE.MeshStandardMaterial; gloss: THREE.MeshStandardMaterial; glow: THREE.MeshBasicMaterial };
  private blob = blobTexture();
  private size = { w: 1, h: 1 };

  constructor(
    private host: HTMLElement,
    private overlay: HTMLElement,
    private opts: SceneOptions,
  ) {
    this.view = opts.view;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    const backdrop = new THREE.Color(0x1c2229);
    try {
      backdrop.setStyle(getComputedStyle(host).backgroundColor);
    } catch {
      // keep the default
    }
    this.renderer.setClearColor(backdrop, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    host.insertBefore(this.renderer.domElement, host.firstChild);
    this.renderer.domElement.setAttribute("role", "img");
    this.renderer.domElement.setAttribute("aria-label", "The room in 3D, with the people and things the cameras see. Drag to rotate, pinch to zoom.");

    this.pipeline = createFsrPipeline(this.renderer, { scale: opts.scale, simulate: opts.simulate });
    this.pipeline.setCompare(opts.compare, opts.split);

    this.materials = {
      matte: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.86, metalness: 0 }),
      gloss: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.32, metalness: 0.18 }),
      glow: new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }),
    };

    // Soft sky and floor bounce everywhere, a shadowless fill from the viewer's side, and the sun through the window.
    this.scene.add(new THREE.HemisphereLight(0xe6eeff, 0xa08a6e, 1.7));
    const fill = new THREE.DirectionalLight(0xfff4e6, 0.7);
    fill.position.set(6, 9, 8);
    this.scene.add(fill);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    this.sun.shadow.bias = -0.0006;
    this.sun.shadow.normalBias = 0.025;
    this.sun.shadow.radius = 3;
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(this.room, this.zones, this.routeGroup, this.things, this.pinsGroup);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.12; // never under the floor
    this.controls.addEventListener("change", this.draw);
    this.controls.addEventListener("start", () => {
      this.glide = null;
    });

    this.watcher = new ResizeObserver(() => this.resize());
    this.watcher.observe(host);
    this.resize();
  }

  // ---------- drawing ----------

  draw = () => {
    this.dirty = true;
    this.still = 0;
    if (!this.frame && !this.gone) this.frame = requestAnimationFrame(this.tick);
  };

  private tick = (now: number) => {
    this.frame = 0;
    if (this.gone) return;
    let moving = false;
    if (this.glide) {
      const t = Math.min(1, (now - this.glide.start) / 700);
      const k = 1 - Math.pow(1 - t, 3);
      this.camera.position.lerpVectors(this.glide.from, this.glide.to, k);
      this.controls.target.lerpVectors(this.glide.target, this.glide.aim, k);
      if (t >= 1) this.glide = null;
      moving = true;
    }
    if (this.stepFigures(now)) {
      moving = true;
      this.shadowDirty = true;
    }
    if ((this.routes.length > 0 || this.bob.length > 0) && !this.opts.calm) {
      for (const t of this.chevrons) t.offset.x = -((now / 1000) * 1.1) % 1;
      for (const o of this.bob) o.position.y = (o.userData.base as number) + Math.sin(now / 260) * 0.05;
      moving = true;
    }
    const turned = this.controls.update();
    if (this.cutWalls()) this.shadowDirty = true;
    if (moving || turned || this.dirty) {
      this.paint();
      this.still = 0;
    } else this.still++;
    if (!this.frame && (moving || this.still < 12)) this.frame = requestAnimationFrame(this.tick);
  };

  private paint() {
    this.dirty = false;
    if (this.shadowDirty) {
      this.renderer.shadowMap.needsUpdate = true;
      this.shadowDirty = false;
    }
    this.pipeline.render(this.scene, this.camera);
    this.layoutLabels();
  }

  private resize() {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    const first = this.size.w === 1 && this.size.h === 1;
    const reshaped = Math.abs(w / h - this.size.w / this.size.h) > 0.01;
    this.size = { w, h };
    this.pipeline.setSize(w, h, window.devicePixelRatio || 1);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.opts.onInfo(this.pipeline.info());
    if (first || reshaped) this.home(false);
    this.shadowDirty = true;
    this.paint();
    this.draw();
  }

  info(): FsrInfo {
    return this.pipeline.info();
  }

  setRender(scale: number, compare: boolean) {
    this.pipeline.setScale(scale);
    this.pipeline.setCompare(compare);
    this.opts.onInfo(this.pipeline.info());
    this.draw();
  }

  /** Average milliseconds a frame over `frames` frames at each scale, each one read back so the GPU has finished it. */
  async bench(modes: { name: string; scale: number; direct?: boolean }[], frames: number, restore: { scale: number; compare: boolean }) {
    const gl = this.renderer.getContext();
    const pixel = new Uint8Array(4);
    const run = (n: number) => {
      for (let i = 0; i < n; i++) {
        this.renderer.shadowMap.needsUpdate = true; // as when a figure is walking
        this.pipeline.render(this.scene, this.camera);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      }
    };
    const pause = () => new Promise<void>((resolve) => window.setTimeout(resolve, 30));
    const rows: { name: string; ms: number; internal: [number, number]; output: [number, number] }[] = [];
    this.pipeline.setCompare(false);
    for (const m of modes) {
      if (this.gone) return rows;
      this.pipeline.setBypass(m.direct === true);
      this.pipeline.setScale(m.scale);
      run(8);
      const t0 = performance.now();
      run(frames);
      const now = this.pipeline.info();
      rows.push({ name: m.name, ms: (performance.now() - t0) / frames, internal: now.internal, output: now.output });
      await pause();
    }
    this.pipeline.setBypass(false);
    this.setRender(restore.scale, restore.compare);
    return rows;
  }

  drawCalls(): number {
    return this.renderer.info.render.calls;
  }

  // ---------- camera ----------

  private pose(view: TwinView): { to: THREE.Vector3; aim: THREE.Vector3 } {
    const [w, d] = this.plan?.size ?? [4, 4];
    const high = this.plan?.wall_height ?? 2.5;
    const tanV = Math.tan((FOV * Math.PI) / 360);
    const tanH = tanV * this.camera.aspect;
    if (view === "top") {
      const chips = Math.min(0.2, 58 / this.size.h); // the share of the stage's height the chips along the top take
      // 0.3 m of margin, and more at the near edge: tall furniture leans outwards in a view from above
      const dist = Math.max((d / 2 + 0.55) / (tanV * (1 - chips)), (w / 2 + 0.45) / tanH);
      const aim = new THREE.Vector3(w / 2, 0, d / 2 + 0.25 - chips * dist * tanV);
      return { aim, to: new THREE.Vector3(aim.x, dist, aim.z + dist * 0.012) };
    }
    const aim = new THREE.Vector3(w / 2, high * 0.34, d / 2);
    const dir = new THREE.Vector3(0.6, 0.8, 0.95).normalize();
    // Nearest distance at which every corner of the room (to wall height at the back, kerb height in front) is in frame.
    const corners: THREE.Vector3[] = [];
    for (const x of [-0.2, w + 0.2]) for (const z of [-0.2, d + 0.2]) corners.push(new THREE.Vector3(x, -0.1, z), new THREE.Vector3(x, x < 0 || z < 0 ? high : 0.3, z));
    const probe = this.camera.clone();
    const v = new THREE.Vector3();
    let lo = 2;
    let hi = 60;
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) / 2;
      probe.position.copy(aim).addScaledVector(dir, mid);
      probe.lookAt(aim);
      probe.updateMatrixWorld();
      const fits = corners.every((c) => {
        v.copy(c).project(probe);
        return Math.abs(v.x) < 0.94 && v.y < 0.8 && v.y > -0.9; // a little room at the top for the chips
      });
      if (fits) hi = mid;
      else lo = mid;
    }
    // Centre the room up and down in what is left under the chips.
    probe.position.copy(aim).addScaledVector(dir, hi);
    probe.lookAt(aim);
    probe.updateMatrixWorld();
    let top = -1;
    let bottom = 1;
    for (const c of corners) {
      v.copy(c).project(probe);
      top = Math.max(top, v.y);
      bottom = Math.min(bottom, v.y);
    }
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(probe.quaternion);
    aim.addScaledVector(up, ((top + bottom) / 2 + 0.08) * hi * tanV);
    return { aim, to: aim.clone().addScaledVector(dir, hi) };
  }

  home(ease: boolean) {
    const { to, aim } = this.pose(this.view);
    this.controls.minDistance = 2;
    this.controls.maxDistance = to.distanceTo(aim) * 1.6;
    if (ease && !this.opts.calm) {
      this.glide = { from: this.camera.position.clone(), to, target: this.controls.target.clone(), aim, start: performance.now() };
    } else {
      this.glide = null;
      this.controls.target.copy(aim);
      this.camera.position.copy(to);
      this.controls.update();
    }
    this.draw();
  }

  setView(view: TwinView) {
    this.view = view;
    this.home(true);
  }

  /** Walls between the viewer and the room are cut down to a kerb; from straight above, all four are. */
  private cutWalls(): boolean {
    if (!this.plan) return false;
    const [w, d] = this.plan.size;
    const p = this.camera.position;
    const above = p.y > 3 && Math.hypot(p.x - this.controls.target.x, p.z - this.controls.target.z) < p.y * 0.3; // looking down: a floor plan
    const outside: Record<Wall, boolean> = { north: above || p.z < 0.25, south: above || p.z > d - 0.25, west: above || p.x < 0.25, east: above || p.x > w - 0.25 };
    let changed = false;
    for (const [wall, parts] of this.walls) {
      const hide = outside[wall];
      if (hide === parts.hidden) continue;
      parts.hidden = hide;
      changed = true;
      parts.full.visible = !hide;
      parts.curb.visible = hide;
    }
    return changed;
  }

  // ---------- labels ----------

  private makeLabel(text: string, cls: string, priority: number, must = false, button = false, dot?: string): Label {
    const el = document.createElement(button ? "button" : "div");
    el.className = `${this.opts.classes.label} ${cls}`;
    if (button) (el as HTMLButtonElement).type = "button";
    if (dot) {
      const d = document.createElement("span");
      d.className = this.opts.classes.dot;
      d.style.background = dot;
      el.appendChild(d);
    }
    el.appendChild(document.createTextNode(text));
    el.style.visibility = "hidden";
    this.overlay.appendChild(el);
    const label: Label = { el, anchor: new THREE.Vector3(), priority, must, w: el.offsetWidth, h: el.offsetHeight };
    this.labels.add(label);
    return label;
  }

  private dropLabel(label: Label) {
    label.el.remove();
    this.labels.delete(label);
  }

  private retext(label: Label, text: string) {
    const node = label.el.lastChild;
    if (node && node.nodeValue !== text) {
      node.nodeValue = text;
      label.w = label.el.offsetWidth;
    }
  }

  /** Place every name next to its thing; a name that would cover a more important one moves, or is left out. */
  private layoutLabels() {
    const { w, h } = this.size;
    const placed: [number, number, number, number][] = [];
    const frame = this.host.getBoundingClientRect();
    for (const el of this.host.querySelectorAll("[data-keepclear]")) {
      const r = el.getBoundingClientRect(); // the chips and cards drawn over the stage
      if (r.width > 0) placed.push([r.left - frame.left, r.top - frame.top, r.width, r.height]);
    }
    const v = new THREE.Vector3();
    const order = [...this.labels].sort((a, b) => b.priority - a.priority);
    for (const l of order) {
      v.copy(l.anchor).project(this.camera);
      if (l.off || v.z > 1 || v.z < -1) {
        l.el.style.visibility = "hidden";
        continue;
      }
      const sx = (v.x * 0.5 + 0.5) * w;
      const sy = (-v.y * 0.5 + 0.5) * h;
      if (!l.w) l.w = l.el.offsetWidth;
      if (!l.h) l.h = l.el.offsetHeight;
      const tries: [number, number][] = [[sx - l.w / 2, sy - l.h - 6]];
      const rows = l.must ? 5 : 2; // how far a name may move from its thing before it is left out
      for (let k = 0; k <= rows; k++) {
        const up = sy - l.h - 6 - k * (l.h + 5);
        if (k > 0) tries.push([sx - l.w / 2, up]);
        tries.push([sx + 8, up + l.h / 2], [sx - l.w - 8, up + l.h / 2]);
        if (l.must) tries.push([sx + l.w * 0.6, up], [sx - l.w * 1.6, up]);
      }
      tries.push([sx - l.w / 2, sy + 10]);
      let spot: [number, number] | null = null;
      for (const [tx, ty] of tries) {
        const x = Math.min(Math.max(tx, 4), w - l.w - 4);
        const y = Math.min(Math.max(ty, 4), h - l.h - 4);
        if (!placed.some(([px, py, pw, ph]) => x < px + pw + 4 && px < x + l.w + 4 && y < py + ph + 3 && py < y + l.h + 3)) {
          spot = [x, y];
          break;
        }
      }
      if (!spot || sx < -20 || sx > w + 20 || sy < -20 || sy > h + 40) {
        l.el.style.visibility = "hidden";
        continue;
      }
      placed.push([spot[0], spot[1], l.w, l.h]);
      l.el.style.transform = `translate(${Math.round(spot[0])}px, ${Math.round(spot[1])}px)`;
      l.el.style.visibility = "visible";
    }
  }

  // ---------- the room ----------

  setPlan(plan: RoomPlan) {
    const firstPlan = this.plan === null;
    const reshaped = !this.plan || this.plan.size[0] !== plan.size[0] || this.plan.size[1] !== plan.size[1] || this.plan.name !== plan.name;
    this.plan = plan;
    for (const l of this.staticLabels) this.dropLabel(l);
    this.staticLabels = [];
    this.scene.remove(this.room);
    disposeRoom(this.room, this.materials);
    this.room = new THREE.Group();
    this.walls.clear();
    this.bins.clear();
    const [w, d] = plan.size;
    const high = plan.wall_height;

    // the ground the model stands on, and the floor
    const base = new THREE.Mesh(new THREE.BoxGeometry(w + 0.5, 0.16, d + 0.5), new THREE.MeshStandardMaterial({ color: C.base, roughness: 0.9 }));
    base.position.set(w / 2, -0.081, d / 2);
    base.receiveShadow = true;
    this.room.add(base);
    const wood = woodTexture();
    wood.repeat.set(w / 2, d / 2);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ map: wood, roughness: 0.78 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(w / 2, 0, d / 2);
    floor.receiveShadow = true;
    this.room.add(floor);
    const contact = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ map: contactTexture(w, d, plan.furniture), transparent: true, depthWrite: false, opacity: 0.85 }));
    contact.rotation.x = -Math.PI / 2;
    contact.position.set(w / 2, 0.004, d / 2);
    contact.renderOrder = 1;
    this.room.add(contact);

    for (const wall of WALLS) {
      const parts = buildWall(wall, w, d, high, plan.doors, plan.windows, this.materials);
      for (const p of plan.posters.filter((x) => x.wall === wall)) {
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(p.width, p.height ?? 0.6), new THREE.MeshStandardMaterial({ map: posterTexture(p.art), roughness: 0.9 }));
        const [x, z] = wallPoint(wall, p.at, -0.008, w, d);
        mesh.position.set(x, (p.sill ?? 1.3) + (p.height ?? 0.6) / 2, z);
        mesh.rotation.y = wallTurn(wall);
        parts.full.add(mesh);
        parts.extras.push(mesh);
      }
      parts.curb.visible = false;
      parts.full.traverse((o) => {
        o.castShadow = false; // a wall's shadow would black out half the room; the window's light is painted on the floor
      });
      this.room.add(parts.full, parts.curb);
      this.walls.set(wall, { ...parts, hidden: false });
    }

    for (const f of plan.furniture.filter((x) => x.type === "rug")) {
      const rug = new THREE.Mesh(new THREE.PlaneGeometry(f.size[0], f.size[1]), new THREE.MeshStandardMaterial({ map: rugTexture(), roughness: 0.95 }));
      rug.rotation.set(-Math.PI / 2, 0, (f.rot * Math.PI) / 180);
      rug.position.set(f.at[0], 0.012, f.at[1]);
      rug.receiveShadow = true;
      rug.renderOrder = 2;
      this.room.add(rug);
    }
    const plainChairs = plan.furniture.some((f) => f.type === "table");
    this.room.add(buildFurniture(plan.furniture, plainChairs, this.materials));

    for (const f of plan.furniture.filter((x) => x.type === "bin")) {
      const made = buildBin(f, binDecal(f.bin ?? "trash"));
      made.group.traverse((o) => {
        o.castShadow = true;
        o.receiveShadow = true;
      });
      this.room.add(made.group);
      const label = this.makeLabel(humanize(f.bin ?? "bin"), this.opts.classes.bin, 40, true, false, `#${binColor(f.color).toString(16).padStart(6, "0")}`);
      label.anchor.set(f.at[0], f.height + 0.12, f.at[1]);
      this.staticLabels.push(label);
      this.bins.set(f.id, { f, ...made, label });
    }
    for (const door of plan.doors) {
      const [x, z] = wallPoint(door.wall, door.at, -0.15, w, d);
      const label = this.makeLabel("Door", this.opts.classes.zone, 12);
      label.anchor.set(x, 2.15, z);
      this.staticLabels.push(label);
    }

    // The sun comes through the first window, slanting down and a little along the wall.
    const win = plan.windows[0];
    const inward: Record<Wall, [number, number]> = { north: [0, 1], south: [0, -1], west: [1, 0], east: [-1, 0] };
    const [ix, iz] = win ? inward[win.wall] : [-0.5, -0.6];
    const dir = new THREE.Vector3(ix * 0.5 + iz * 0.16, -1, iz * 0.5 + ix * 0.16).normalize();
    if (win) {
      // Where the sun through the window lands: the opening slid along a lower beam onto the floor.
      const sill = win.sill ?? 0.95;
      const top = sill + (win.height ?? 1.1);
      const beam = new THREE.Vector3(ix * 0.9 + iz * 0.3, -0.72, iz * 0.9 + ix * 0.3);
      const hit = (u: number, v: number) => {
        const [x, z] = wallPoint(win.wall, u, 0, w, d);
        const k = v / -beam.y;
        return [Math.min(Math.max(x + beam.x * k, 0.02), w - 0.02), Math.min(Math.max(z + beam.z * k, 0.02), d - 0.02)] as [number, number];
      };
      const a = hit(win.at - win.width / 2 + 0.06, sill + 0.05);
      const b = hit(win.at + win.width / 2 - 0.06, sill + 0.05);
      const c = hit(win.at + win.width / 2 - 0.06, top - 0.05);
      const e = hit(win.at - win.width / 2 + 0.06, top - 0.05);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute([a[0], 0.02, a[1], b[0], 0.02, b[1], c[0], 0.02, c[1], e[0], 0.02, e[1]], 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
      g.setIndex([0, 1, 2, 0, 2, 3]);
      const patch = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: sunTexture(), color: 0xffe2b0, transparent: true, opacity: 0.34, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      patch.renderOrder = 3;
      this.room.add(patch);
    }
    this.sun.target.position.set(w / 2, 0, d / 2);
    this.sun.position.copy(this.sun.target.position).addScaledVector(dir, -12);
    const reach = Math.hypot(w, d) / 2 + 1.2;
    const cam = this.sun.shadow.camera;
    cam.left = -reach;
    cam.right = reach;
    cam.top = reach;
    cam.bottom = -reach;
    cam.near = 1;
    cam.far = 26;
    cam.updateProjectionMatrix();

    this.scene.add(this.room);
    this.setZones(plan);
    this.shadowDirty = true;
    if (firstPlan || reshaped) this.home(false);
    this.cutWalls();
    this.applyRoutes();
    this.draw();
  }

  private setZones(plan: RoomPlan) {
    this.scene.remove(this.zones);
    disposeTree(this.zones);
    this.zones = new THREE.Group();
    for (const zones of Object.values(plan.zones)) {
      for (const [id, poly] of Object.entries(zones)) {
        if (poly.length < 3) continue;
        const color = ZONE_COLORS[id] ?? 0xb794f6;
        const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
        const geometry = new THREE.ShapeGeometry(shape);
        geometry.rotateX(Math.PI / 2);
        const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.17, depthWrite: false, side: THREE.DoubleSide }));
        fill.position.y = 0.016;
        fill.renderOrder = 3;
        this.zones.add(fill);
        const edge = new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(poly.map(([x, y]) => new THREE.Vector3(x, 0.02, y))),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 }),
        );
        edge.renderOrder = 3;
        this.zones.add(edge);
      }
    }
    this.scene.add(this.zones);
  }

  // ---------- people and things ----------

  setTracks(tracks: FloorTrack[], players: Map<string, number>, quiet: Set<string> = new Set()) {
    const now = performance.now();
    const seen = new Set<string>();
    for (const t of tracks) {
      seen.add(t.key);
      const person = t.kind === "person";
      const to = new THREE.Vector3(t.x, person ? 0 : t.z, t.y);
      let fig = this.figures.get(t.key);
      const player = players.get(t.key);
      const name = person ? (player ? `Player ${player}` : t.member_id ? humanize(t.member_id.replace(/^m-/, "")) : "Person") : humanize(t.label);
      if (!fig) {
        const group = new THREE.Group();
        let made: ReturnType<typeof buildPerson> | null = null;
        if (person) {
          made = buildPerson(PERSON);
          group.add(made.group);
        } else group.add(buildThing(t.label));
        const shade = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: this.blob, transparent: true, depthWrite: false, opacity: person ? 0.9 : 0.7 }));
        shade.rotation.x = -Math.PI / 2;
        shade.scale.setScalar(person ? 0.75 : 0.3);
        shade.position.y = 0.012;
        shade.renderOrder = 4;
        group.add(shade);
        let ring: THREE.Mesh | undefined;
        if (person) {
          ring = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.37, 40), new THREE.MeshBasicMaterial({ color: ROUTE_COLORS[0], transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide }));
          ring.rotation.x = -Math.PI / 2;
          ring.position.y = 0.03;
          ring.renderOrder = 6;
          ring.visible = false;
          group.add(ring);
        }
        group.position.copy(to);
        this.things.add(group);
        const label = this.makeLabel(name, person ? this.opts.classes.person : this.opts.classes.thing, person ? 60 : 50, person);
        fig = { key: t.key, person, group, body: made?.body, shirt: made?.shirt, ring, shade, from: to.clone(), to, t0: now, born: now, heading: 0, wantHeading: 0, seated: false, heldBy: null, label, top: person ? 1.8 : 0.34 };
        this.figures.set(t.key, fig);
      } else {
        fig.from.copy(fig.group.position);
        fig.to.copy(to);
        fig.t0 = now;
        this.retext(fig.label, name);
      }
      fig.heldBy = t.held_by;
      fig.label.off = !person && (t.held_by !== null || quiet.has(t.label));
      if (fig.shade) fig.shade.visible = !t.held_by && !t.in_bin;
      if (person) {
        const moved = Math.hypot(fig.to.x - fig.from.x, fig.to.z - fig.from.z);
        const seat = t.seated && t.on ? this.plan?.furniture.find((f) => f.id === t.on) : undefined;
        if (seat && seat.type === "chair") fig.wantHeading = (-seat.rot * Math.PI) / 180;
        else if (moved > 0.08) fig.wantHeading = Math.atan2(fig.to.x - fig.from.x, fig.to.z - fig.from.z);
        fig.seated = t.seated;
        fig.top = t.seated ? 1.5 : 1.8;
        const color = player ? ROUTE_COLORS[(player - 1) % ROUTE_COLORS.length] : t.member_id ? MEMBER_COLORS[hash(t.member_id) % MEMBER_COLORS.length] : PERSON;
        fig.shirt?.color.setHex(color as number);
        if (fig.ring) {
          fig.ring.visible = player !== undefined;
          (fig.ring.material as THREE.MeshBasicMaterial).color.setHex(ROUTE_COLORS[((player ?? 1) - 1) % ROUTE_COLORS.length] as number);
        }
      }
    }
    for (const [key, fig] of this.figures) {
      if (seen.has(key)) continue;
      this.things.remove(fig.group);
      disposeFigure(fig.group, this.blob);
      this.dropLabel(fig.label);
      this.figures.delete(key);
    }
    this.shadowDirty = true;
    this.draw();
  }

  /** Move every figure towards where it was last seen. True while anything is still on its way. */
  private stepFigures(now: number): boolean {
    let moving = false;
    for (const fig of this.figures.values()) {
      const t = this.opts.calm ? 1 : Math.min(1, (now - fig.t0) / MOVE_MS);
      if (t < 1) moving = true;
      const holder = fig.heldBy ? this.figures.get(fig.heldBy) : undefined;
      if (holder) {
        // in the holder's right hand, a little in front
        const a = holder.heading;
        fig.group.position.set(holder.group.position.x + Math.sin(a) * 0.2 - Math.cos(a) * 0.24, 0.98, holder.group.position.z + Math.cos(a) * 0.2 + Math.sin(a) * 0.24);
        fig.group.rotation.y = a;
      } else fig.group.position.lerpVectors(fig.from, fig.to, t);
      if (fig.person) {
        let turn = fig.wantHeading - fig.heading;
        turn = Math.atan2(Math.sin(turn), Math.cos(turn));
        if (Math.abs(turn) > 0.01) {
          fig.heading += this.opts.calm ? turn : turn * 0.18;
          moving = true;
        }
        fig.group.rotation.y = fig.heading;
        if (fig.body) {
          const want = fig.seated ? 0.8 : 1;
          const sy = fig.body.scale.y + (want - fig.body.scale.y) * (this.opts.calm ? 1 : 0.2);
          if (Math.abs(sy - fig.body.scale.y) > 0.002) moving = true;
          fig.body.scale.y = sy;
          if (!fig.seated && t < 1 && fig.from.distanceToSquared(fig.to) > 0.01) fig.body.position.y = Math.abs(Math.sin(t * Math.PI * 2)) * 0.03; // a step
          else fig.body.position.y = 0;
        }
      }
      const grow = this.opts.calm ? 1 : Math.min(1, (now - fig.born) / 260);
      if (grow < 1) moving = true;
      fig.group.scale.setScalar(0.4 + 0.6 * grow);
      fig.label.anchor.set(fig.group.position.x, fig.group.position.y + fig.top, fig.group.position.z);
    }
    return moving;
  }

  // ---------- memory pins ----------

  setPins(pins: ScenePin[], selected: string | null) {
    for (const l of this.pinLabels) this.dropLabel(l);
    this.pinLabels = [];
    this.scene.remove(this.pinsGroup);
    disposeTree(this.pinsGroup);
    this.pinsGroup = new THREE.Group();
    for (const pin of pins) {
      const on = pin.id === selected;
      const color = on ? 0x5ce1c6 : 0xf3f5f7;
      const g = new THREE.Group();
      const tall = 0.62;
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, tall, 8), new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
      stem.position.y = tall / 2;
      const tip = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.16, 16), new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
      tip.rotation.x = Math.PI;
      tip.position.y = tall - 0.03;
      const head = new THREE.Mesh(new THREE.SphereGeometry(on ? 0.11 : 0.095, 18, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.35, emissive: color, emissiveIntensity: on ? 0.35 : 0.12 }));
      head.position.y = tall + 0.08;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.135, 28), new THREE.MeshBasicMaterial({ color: 0x5ce1c6, transparent: true, opacity: on ? 0.95 : 0.6, depthWrite: false, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.012;
      ring.renderOrder = 5;
      g.add(stem, tip, head, ring);
      g.position.set(pin.x, pin.z, pin.y);
      this.pinsGroup.add(g);
      const label = this.makeLabel(pin.text, `${this.opts.classes.pin} ${on ? this.opts.classes.pinOn : ""}`, on ? 95 : 70, true, true);
      label.el.setAttribute("aria-label", `${pin.text}: where it was last seen`);
      label.el.setAttribute("aria-pressed", on ? "true" : "false");
      label.el.addEventListener("click", () => this.opts.onPick(pin.id));
      label.anchor.set(pin.x, pin.z + tall + 0.2, pin.y);
      this.pinLabels.push(label);
    }
    this.scene.add(this.pinsGroup);
    this.draw();
  }

  // ---------- Sort It ----------

  private target: string | null = null;
  private targetText = "";

  /** The routes to draw, the bin they lead to (a furniture id), and what to write on it. */
  setRoutes(routes: SceneRoute[], targetBin: string | null, targetText: string) {
    this.routes = routes;
    this.target = targetBin;
    this.targetText = targetText;
    this.applyRoutes();
    this.draw();
  }

  private applyRoutes() {
    for (const l of this.routeLabels) this.dropLabel(l);
    this.routeLabels = [];
    this.scene.remove(this.routeGroup);
    disposeTree(this.routeGroup);
    this.routeGroup = new THREE.Group();
    this.bob = [];
    this.chevrons = [];
    const glow = glowTexture();
    for (const route of this.routes) {
      if (route.points.length < 2) continue;
      const color = ROUTE_COLORS[(route.player - 1) % ROUTE_COLORS.length] as number;
      const line = rounded(route.points);
      const halo = new THREE.Mesh(ribbon(line, 0.3, 0.022), new THREE.MeshBasicMaterial({ color, map: glow, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      halo.renderOrder = 5;
      const tex = chevronTexture();
      tex.repeat.set(1 / 0.34, 1); // an arrow head every 34 cm
      this.chevrons.push(tex);
      const strip = new THREE.Mesh(ribbon(line, 0.085, 0.028), new THREE.MeshBasicMaterial({ color, map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      strip.renderOrder = 6;
      const [sx, sy] = line[0] as [number, number];
      const start = new THREE.Mesh(new THREE.CircleGeometry(0.1, 24), new THREE.MeshBasicMaterial({ color, depthWrite: false, side: THREE.DoubleSide }));
      start.rotation.x = -Math.PI / 2;
      start.position.set(sx, 0.03, sy);
      start.renderOrder = 6;
      this.routeGroup.add(halo, strip, start);
    }
    for (const [id, bin] of this.bins) {
      const dim = this.target !== null && id !== this.target;
      bin.body.transparent = dim;
      bin.body.opacity = dim ? 0.22 : 1;
      bin.body.depthWrite = !dim;
      bin.face.opacity = dim ? 0.18 : 1;
      bin.label.el.style.opacity = dim ? "0.45" : "1";
      bin.label.priority = id === this.target ? 20 : 40; // the "goes here" label speaks for the target
      if (id !== this.target) continue;
      const color = ROUTE_COLORS[0] as number;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.36, 0.43, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(bin.f.at[0], 0.03, bin.f.at[1]);
      ring.renderOrder = 6;
      const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.24, 20), new THREE.MeshBasicMaterial({ color }));
      arrow.rotation.x = Math.PI;
      arrow.position.set(bin.f.at[0], bin.f.height + 0.42, bin.f.at[1]);
      arrow.userData.base = arrow.position.y;
      this.bob.push(arrow);
      this.routeGroup.add(ring, arrow);
      if (this.targetText) {
        const label = this.makeLabel(this.targetText, this.opts.classes.target, 90, true);
        label.anchor.set(bin.f.at[0], bin.f.height + 0.62, bin.f.at[1]);
        this.routeLabels.push(label);
      }
    }
    this.scene.add(this.routeGroup);
  }

  dispose() {
    this.gone = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.watcher.disconnect();
    this.controls.dispose();
    for (const l of [...this.labels]) this.dropLabel(l);
    disposeRoom(this.room, this.materials);
    for (const fig of this.figures.values()) disposeFigure(fig.group, this.blob);
    for (const g of [this.zones, this.pinsGroup, this.routeGroup]) disposeTree(g);
    this.blob.dispose();
    for (const m of Object.values(this.materials)) m.dispose();
    this.pipeline.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

/** Free a room's geometry and its own textures, but not the three materials every room shares. */
function disposeRoom(root: THREE.Object3D, shared: Record<string, THREE.Material>) {
  const keep = new Set<THREE.Material>(Object.values(shared));
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const own = mesh.material;
    const mats = Array.isArray(own) ? own : own ? [own] : [];
    for (const m of mats) {
      if (keep.has(m)) continue;
      (m as THREE.MeshBasicMaterial).map?.dispose?.();
      m.dispose();
    }
  });
}

function disposeFigure(root: THREE.Object3D, sharedMap: THREE.Texture) {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    for (const m of mats) {
      const map = (m as THREE.MeshBasicMaterial).map;
      if (map && map !== sharedMap) map.dispose();
      m.dispose();
    }
  });
}
