import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import {
  ANCHORS,
  DEVICE_PARTS,
  type DeviceModel,
  type DevicePart,
  type DeviceTextures,
  type Materials,
  type PartId,
  TRAVEL,
  buildDevice,
  makeMaterials,
  makeTextures,
} from "./deviceModel";

export { DEVICE_PARTS };
export type { DevicePart };

/**
 * The Atlas device on the Home page, with no React in it: Device.tsx owns one of these and drives it.
 *
 *   setExplode(t)   0 is the sealed box, 1 is the end of the stage's scroll; the page maps scroll onto it
 *   setPointer      a gentle tilt towards the pointer
 *   setTheme        relight for Night or Day
 *   setActive       stop drawing while the stage is off screen
 *
 * The model itself is in deviceModel.ts. This file owns the renderer, the lights, the camera, the explode
 * choreography (each part has its own stretch of the factor, so the pull-apart reads as a sequence rather than one
 * jump), the fan and the breathing glows, and the callouts: numbered DOM chips in the overlay, laid out beside the
 * device and tied to their part by a thin line. Only the part named most recently shows its full text; the rest
 * stay as a name, so eight of them never bury the device.
 *
 * The canvas is transparent so the page's aurora shows through, and it covers the whole stage. The camera is fitted
 * twice: the sealed box into the room the hero copy leaves, and the whole exploded stack into the room the chip
 * columns leave. Scrolling eases between the two, so the box glides out from beside the copy to the middle as it
 * opens, while the view rises a little to show the layers from above.
 */

export interface DeviceOptions {
  /** prefers-reduced-motion: no idle sway, no fan motion, no breathing. */
  calm: boolean;
  dark: boolean;
  classes: { callout: string; calloutOn: string; calloutActive: string; line: string };
  /** The hero copy over the stage: the sealed box is framed clear of it (beside it when wide, above it when not). */
  hero?: HTMLElement | null;
  /** The bar along the bottom of the stage: nothing is framed under it. */
  footer?: HTMLElement | null;
  /** The part named most recently changed: its place in the order (-1 while the box is sealed), the count, its name. */
  onStep?: (index: number, total: number, name: string) => void;
}

const FOV = 28;
const VIEW_SEALED = new THREE.Vector3(0.62, 0.36, 1).normalize(); // front right, a little above: the box as a product shot
const VIEW_APART = new THREE.Vector3(0.42, 0.52, 1).normalize(); // higher and more frontal, so the layers read
const EXPLODE_FROM = 0.06; // the share of the scroll before anything moves, so the hero reads first
const EXPLODE_TO = 0.9; // after this the stack holds, fully apart, until the stage scrolls away
const CHIP_GAP = 48; // pixels between a chip column and the device
const MIN_DELTA = 0.5; // a chip or a line is restyled only when it moved by more than this many pixels
const NARROW = 700; // below this many pixels wide, one callout at a time: two columns would cover the device

interface Callout {
  id: PartId;
  index: number;
  name: string;
  el: HTMLElement;
  line: HTMLElement;
  group: THREE.Group;
  local: THREE.Vector3;
  threshold: number;
  side: "left" | "right";
  w: number;
  h: number;
  on: boolean;
  x: number; // where the chip is now (smoothed towards where it should be)
  y: number;
  placed: boolean;
  lastChip: string;
  lastLine: string;
}

/** Pixels of the host kept clear on each side. */
interface Reserve {
  l: number;
  r: number;
  t: number;
  b: number;
}

interface Fit {
  aim: THREE.Vector3;
  dist: number;
  dir: THREE.Vector3;
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function smoothstep(a: number, b: number, x: number): number {
  const k = clamp01((x - a) / (b - a));
  return k * k * (3 - 2 * k);
}

export class DeviceScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 60);
  private pivot = new THREE.Group(); // turns for the idle sway and the pointer tilt
  private model: DeviceModel;
  private textures: DeviceTextures;
  private materials: Materials;
  private envMap: THREE.Texture | null = null;
  private key = new THREE.DirectionalLight(0xfff3e4, 2.4);
  private hemi = new THREE.HemisphereLight(0xe8f0ff, 0x8a8f99, 0.95);
  private rim = new THREE.DirectionalLight(0x7fa6ff, 0.5);
  private accent = new THREE.PointLight(0xffb070, 0, 7, 2);
  private dark: boolean;
  private watcher: ResizeObserver;
  private size = { w: 1, h: 1 }; // the host, where the canvas is
  private area = { w: 1, h: 1 }; // the overlay, where the chips go
  private shift = { x: 0, y: 0 }; // the host's top left in the overlay
  private frame = 0;
  private active = true;
  private gone = false;
  private last = 0;
  private te = 0; // the explode factor: the scroll with the lead-in and the hold taken off
  private applied = -1; // the factor the parts were last moved to
  private pointer = { x: 0, y: 0 };
  private tilt = { yaw: 0, pitch: 0 };
  private spin = 0;
  private box0 = new THREE.Box3(); // sealed
  private box1 = new THREE.Box3(); // fully apart
  private boxNow = new THREE.Box3(); // as the parts stand now, in the model's frame
  private fit0: Fit = { aim: new THREE.Vector3(), dist: 6, dir: VIEW_SEALED.clone() };
  private fit1: Fit = { aim: new THREE.Vector3(), dist: 9, dir: VIEW_APART.clone() };
  private callouts: Callout[] = [];
  private latest: Callout | null = null;
  private glow = { bar: 1.2, logo: 0.15 };
  private v = new THREE.Vector3();
  private dir = new THREE.Vector3();
  private part = new THREE.Box3();
  private corners: THREE.Vector3[] = Array.from({ length: 8 }, () => new THREE.Vector3());

  constructor(
    private host: HTMLElement,
    private overlay: HTMLElement,
    private opts: DeviceOptions,
  ) {
    this.dark = opts.dark;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const canvas = this.renderer.domElement;
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "The Atlas device: a small white box with a webcam on its top, shown pulled apart into its parts.");
    host.insertBefore(canvas, host.firstChild);

    // A room-shaped environment lights the metal and the glass from all around.
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
    this.model = buildDevice(this.materials, this.textures.blob);
    this.pivot.add(this.model.root);
    this.scene.add(this.pivot);

    // One soft key from the upper left with the only shadow map, a hemisphere fill, and for the night a cool rim
    // from behind and a warm accent low in front (always in the scene, at zero when it is day).
    this.key.position.set(-2.6, 5.2, 3.2);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(1024, 1024);
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.02;
    this.key.shadow.radius = 3;
    this.scene.add(this.key, this.key.target, this.hemi);
    this.rim.position.set(0.6, 2.4, -4);
    this.scene.add(this.rim, this.rim.target);
    this.accent.position.set(1.4, -0.1, 2.8);
    this.scene.add(this.accent);

    this.measureBounds();
    this.applyTheme();
    this.makeCallouts();
    this.applyExplode(0);

    this.watcher = new ResizeObserver(() => this.resize());
    this.watcher.observe(host);
    if (overlay !== host) this.watcher.observe(overlay);
    if (opts.hero) this.watcher.observe(opts.hero);
    this.resize();
    (host as HTMLElement & { deviceScene?: DeviceScene }).deviceScene = this; // for checking draw calls from the console
    this.frame = requestAnimationFrame(this.tick);
  }

  // ---------- the public surface ----------

  setExplode(t: number): void {
    const k = Number.isFinite(t) ? t : 0;
    // Scroll is already under the reader's hand, so it maps straight on; each part eases over its own stretch.
    this.te = clamp01((k - EXPLODE_FROM) / (EXPLODE_TO - EXPLODE_FROM));
  }

  setPointer(x: number, y: number): void {
    this.pointer.x = Math.min(1, Math.max(-1, x || 0));
    this.pointer.y = Math.min(1, Math.max(-1, y || 0));
  }

  setTheme(dark: boolean): void {
    if (dark === this.dark) return;
    this.dark = dark;
    this.applyTheme();
  }

  setActive(on: boolean): void {
    if (on === this.active) return;
    this.active = on;
    if (on && !this.frame && !this.gone) {
      this.last = 0;
      this.frame = requestAnimationFrame(this.tick);
    }
    if (!on && this.frame) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
    }
  }

  /** Draw calls of one frame, the shadow map included. */
  drawCalls(): number {
    const info = this.renderer.info;
    info.autoReset = false;
    info.reset();
    this.renderer.render(this.scene, this.camera);
    const calls = info.render.calls;
    info.autoReset = true;
    return calls;
  }

  dispose(): void {
    this.gone = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.watcher.disconnect();
    for (const c of this.callouts) {
      c.el.remove();
      c.line.remove();
    }
    this.callouts = [];
    this.scene.environment = null;
    this.envMap?.dispose();
    this.model.dispose();
    for (const m of Object.values(this.materials)) m.dispose();
    for (const t of Object.values(this.textures)) t.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    delete (this.host as HTMLElement & { deviceScene?: DeviceScene }).deviceScene;
  }

  // ---------- light ----------

  private applyTheme(): void {
    const dark = this.dark;
    if (dark) {
      this.key.color.setHex(0xdfe8ff);
      this.key.intensity = 1.4;
      this.hemi.color.setHex(0x5e74a8);
      this.hemi.groundColor.setHex(0x12151c);
      this.hemi.intensity = 0.5;
      this.rim.color.setHex(0x7fa6ff);
      this.rim.intensity = 2.8;
      this.accent.intensity = 6;
      this.scene.environmentIntensity = 0.5;
      this.renderer.toneMappingExposure = 1.0;
      this.glow = { bar: 1.25, logo: 0.3 };
    } else {
      this.key.color.setHex(0xfff3e4);
      this.key.intensity = 2.4;
      this.hemi.color.setHex(0xe8f0ff);
      this.hemi.groundColor.setHex(0x8a8f99);
      this.hemi.intensity = 0.95;
      this.rim.color.setHex(0xbfd4ff);
      this.rim.intensity = 0.5;
      this.accent.intensity = 0;
      this.scene.environmentIntensity = 1.0;
      this.renderer.toneMappingExposure = 1.05;
      this.glow = { bar: 0.75, logo: 0.08 };
    }
    (this.model.blob.material as THREE.MeshBasicMaterial).opacity = dark ? 0.45 : 0.6;
    this.breathe(0);
  }

  /** The light bar and the mark breathe together; `phase` is in radians. */
  private breathe(phase: number): void {
    const k = 0.84 + 0.16 * Math.sin(phase);
    (this.materials.bar as THREE.MeshStandardMaterial).emissiveIntensity = this.glow.bar * k;
    (this.materials.logo as THREE.MeshPhysicalMaterial).emissiveIntensity = this.glow.logo * (0.8 + 0.2 * Math.sin(phase + 0.6));
  }

  // ---------- the pull-apart ----------

  /** Move every part to where it stands at factor `te`, and note the bounds of the whole as it now stands. */
  private applyExplode(te: number): void {
    this.applied = te;
    const groups = this.model.groups;
    for (const id of Object.keys(TRAVEL) as PartId[]) {
      const spec = TRAVEL[id];
      const p = smoothstep(spec.from, spec.to, te);
      const g = groups[id];
      g.position.copy(spec.move).multiplyScalar(p);
      if (id === "lid") g.position.add(this.model.lidPivot);
      if (spec.tilt) g.rotation.set(spec.tilt.x * p, spec.tilt.y * p, spec.tilt.z * p);
    }
    this.measure(this.boxNow, false);
  }

  /**
   * The bounds of the parts as they stand, in the model's own frame (the sway is taken off while measuring). From
   * the parts only: the shadow's plane under the base is far wider than the box and must not count.
   */
  private measure(box: THREE.Box3, precise: boolean): void {
    const { base, lid, board } = this.model.groups;
    const turn = this.pivot.rotation.clone();
    this.pivot.rotation.set(0, 0, 0);
    this.pivot.updateMatrixWorld(true);
    box.makeEmpty();
    for (const g of [base, lid, board]) box.union(this.part.setFromObject(g, precise));
    this.pivot.rotation.copy(turn);
    this.pivot.updateMatrixWorld(true);
  }

  /** The device's bounds sealed and apart, found by standing the parts at each end once. */
  private measureBounds(): void {
    this.applyExplode(1);
    this.measure(this.box1, true);
    this.applyExplode(0);
    this.measure(this.box0, true);
    // The shadow camera covers the whole exploded stack.
    const centre = this.box1.getCenter(new THREE.Vector3());
    const radius = this.box1.getSize(new THREE.Vector3()).length() / 2;
    this.key.target.position.copy(centre);
    this.rim.target.position.copy(centre);
    const cam = this.key.shadow.camera;
    cam.left = -radius;
    cam.right = radius;
    cam.top = radius;
    cam.bottom = -radius;
    cam.near = 0.5;
    cam.far = 20;
    cam.updateProjectionMatrix();
  }

  // ---------- camera ----------

  /**
   * The nearest camera along `dir` from which every corner of `box` stays inside the given share of the window
   * that `keep` leaves clear. The model is fitted into that window and the aim is slid so it sits in its middle.
   */
  private fit(box: THREE.Box3, dir: THREE.Vector3, marginX: number, marginY: number, keep: Reserve): Fit {
    const aim = box.getCenter(new THREE.Vector3());
    const corners: THREE.Vector3[] = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
    const xLo = -1 + (2 * keep.l) / this.size.w;
    const xHi = 1 - (2 * keep.r) / this.size.w;
    const yLo = -1 + (2 * keep.b) / this.size.h;
    const yHi = 1 - (2 * keep.t) / this.size.h;
    const halfX = Math.max(0.2, ((xHi - xLo) / 2) * marginX);
    const halfY = Math.max(0.2, ((yHi - yLo) / 2) * marginY);
    const cx = (xLo + xHi) / 2;
    const cy = (yLo + yHi) / 2;
    const probe = this.camera.clone();
    const v = new THREE.Vector3();
    let lo = 0.5;
    let hi = 60;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      probe.position.copy(aim).addScaledVector(dir, mid);
      probe.lookAt(aim);
      probe.updateMatrixWorld();
      const fits = corners.every((c) => {
        v.copy(c).project(probe);
        return Math.abs(v.x) < halfX && Math.abs(v.y) < halfY;
      });
      if (fits) hi = mid;
      else lo = mid;
    }
    // Moving the aim one way moves the picture the other way, by the frame's half size at that distance.
    const forward = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, forward).normalize();
    const halfH = hi * Math.tan((FOV * Math.PI) / 360);
    aim.addScaledVector(right, -cx * halfH * this.camera.aspect).addScaledVector(up, -cy * halfH);
    return { aim, dist: hi, dir: dir.clone() };
  }

  private resize(): void {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.size = { w, h };
    const hostRect = this.host.getBoundingClientRect();
    const overRect = this.overlay.getBoundingClientRect();
    this.area = { w: this.overlay.clientWidth || w, h: this.overlay.clientHeight || h };
    this.shift = { x: hostRect.left - overRect.left, y: hostRect.top - overRect.top };
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    const footer = this.opts.footer?.getBoundingClientRect();
    const foot = footer && footer.height ? Math.max(0, hostRect.bottom - footer.top) : 0;
    // Sealed: clear of the hero copy, beside it on a wide stage and above it on a narrow one.
    const sealed: Reserve = { l: 0, r: 0, t: 0, b: foot };
    const hero = this.opts.hero?.getBoundingClientRect();
    if (hero && hero.width && hero.height) {
      if (hero.width < w * 0.62) sealed.l = Math.min(w * 0.6, Math.max(0, hero.right - hostRect.left + 16));
      else sealed.b = Math.min(h * 0.6, Math.max(foot, hostRect.bottom - hero.top + 8));
    }
    // Apart: a column of chips each side on a wide stage; a narrow one shows one chip at a time over the device.
    let chipW = 0;
    for (const c of this.callouts) {
      c.w = 0;
      c.h = 0;
      chipW = Math.max(chipW, c.el.offsetWidth);
    }
    const side = this.area.w < NARROW ? 0 : chipW + CHIP_GAP;
    const apart: Reserve = { l: side, r: side, t: 8, b: foot };
    this.fit0 = this.fit(this.box0, VIEW_SEALED, 0.74, 0.6, sealed);
    this.fit1 = this.fit(this.box1, VIEW_APART, 0.94, 0.9, apart);
    if (!this.frame && !this.gone && this.active) this.frame = requestAnimationFrame(this.tick);
  }

  /** How far the camera has gone from the sealed frame to the apart one. It follows the lid, the tallest move. */
  private framing(): number {
    return smoothstep(0, 0.6, this.te);
  }

  private pose(): void {
    const k = this.framing();
    this.v.lerpVectors(this.fit0.aim, this.fit1.aim, k);
    const dist = this.fit0.dist + (this.fit1.dist - this.fit0.dist) * k;
    this.dir.lerpVectors(this.fit0.dir, this.fit1.dir, k).normalize();
    this.camera.position.copy(this.v).addScaledVector(this.dir, dist);
    this.camera.lookAt(this.v);
  }

  // ---------- the loop ----------

  private tick = (now: number) => {
    this.frame = 0;
    if (this.gone || !this.active) return;
    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 1 / 60;
    this.last = now;
    const calm = this.opts.calm;

    // The idle sway and the pointer tilt, both eased so nothing snaps. The sway settles as the device opens, so
    // the chips' lines hold still while they are being read.
    if (!calm) this.spin = Math.sin((now / 10000) * Math.PI * 2) * 0.07 * (1 - 0.6 * this.framing());
    const wantYaw = this.spin + this.pointer.x * 0.12;
    const wantPitch = -this.pointer.y * 0.06;
    const ease = 1 - Math.exp(-dt * 4);
    this.tilt.yaw += (wantYaw - this.tilt.yaw) * ease;
    this.tilt.pitch += (wantPitch - this.tilt.pitch) * ease;
    this.pivot.rotation.set(this.tilt.pitch, this.tilt.yaw, 0);

    if (!calm) {
      this.breathe(now / 1400);
      // The fan turns slowly all the time and faster once the device is open.
      this.model.rotor.rotation.y += dt * (0.9 + this.te * 7);
    }
    if (this.applied !== this.te) this.applyExplode(this.te);

    this.pose();
    this.renderer.render(this.scene, this.camera);
    this.layoutCallouts();
    this.frame = requestAnimationFrame(this.tick);
  };

  // ---------- callouts ----------

  private makeCallouts(): void {
    const { callout, line } = this.opts.classes;
    DEVICE_PARTS.forEach((part, index) => {
      const anchor = ANCHORS.find((a) => a.part === part.id);
      if (!anchor) return;
      const el = document.createElement("div");
      el.className = callout;
      const num = document.createElement("em");
      num.textContent = String(index + 1).padStart(2, "0");
      const text = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = part.name;
      const blurb = document.createElement("span");
      blurb.textContent = part.blurb;
      text.append(name, blurb);
      el.append(num, text);
      el.style.visibility = "hidden";
      const lead = document.createElement("i");
      lead.className = line;
      lead.style.visibility = "hidden";
      this.overlay.append(el, lead); // the line right after its chip, so the stylesheet can pair them
      this.callouts.push({ id: part.id, index, name: part.name, el, line: lead, group: this.model.groups[anchor.group], local: anchor.local, threshold: anchor.threshold, side: anchor.side, w: 0, h: 0, on: false, x: 0, y: 0, placed: false, lastChip: "", lastLine: "" });
    });
  }

  /**
   * Lay the chips out and tie each to its part with a line whose far end carries the dot. The part revealed most
   * recently is the active one: its chip opens to show the blurb. On a wide overlay every chip whose threshold the
   * factor has passed sits beside the device on its part's side, stacked so none overlaps another. On a narrow one
   * (a phone) two columns would cover the device, so only the active chip shows, as a tooltip under or over its
   * anchor. Styles are written only when something moved.
   */
  private layoutCallouts(): void {
    const { w, h } = this.area;
    const hostW = this.size.w;
    const hostH = this.size.h;
    const { x: ox, y: oy } = this.shift;
    const narrow = w < NARROW;
    let latest: Callout | null = null;
    for (const c of this.callouts) if (this.te >= c.threshold && (!latest || c.threshold > latest.threshold)) latest = c;
    if (latest !== this.latest) {
      const active = this.opts.classes.calloutActive;
      for (const c of [this.latest, latest]) {
        if (!c) continue;
        c.el.classList.toggle(active, c === latest);
        c.w = 0; // its size changes with the blurb shown or hidden
        c.h = 0;
      }
      this.latest = latest;
      this.opts.onStep?.(latest ? latest.index : -1, this.callouts.length, latest ? latest.name : "");
    }
    // The device's extent on screen, from the corners of its current bounds, so the columns sit just outside it.
    let left = w;
    let right = 0;
    const { min, max } = this.boxNow;
    let i = 0;
    for (const x of [min.x, max.x]) {
      for (const y of [min.y, max.y]) {
        for (const z of [min.z, max.z]) {
          const c = this.corners[i++] as THREE.Vector3;
          c.set(x, y, z).applyMatrix4(this.pivot.matrixWorld).project(this.camera);
          const sx = (c.x * 0.5 + 0.5) * hostW + ox;
          left = Math.min(left, sx);
          right = Math.max(right, sx);
        }
      }
    }
    const columns: Record<"left" | "right", Callout[]> = { left: [], right: [] };
    const want = new Map<Callout, [number, number]>(); // the anchor on screen
    for (const c of this.callouts) {
      const on = narrow ? c === latest : this.te >= c.threshold;
      if (on !== c.on) {
        c.on = on;
        c.el.classList.toggle(this.opts.classes.calloutOn, on);
        if (on) {
          c.el.style.visibility = "visible";
          c.line.style.visibility = "visible";
        }
      }
      if (!on) continue; // a chip fading out stays where it was, so the stylesheet's transition can finish
      if (!c.w || !c.h) {
        c.w = c.el.offsetWidth || 220;
        c.h = c.el.offsetHeight || 44;
      }
      this.v.copy(c.local);
      c.group.localToWorld(this.v).project(this.camera);
      const sx = (this.v.x * 0.5 + 0.5) * hostW + ox;
      const sy = (-this.v.y * 0.5 + 0.5) * hostH + oy;
      want.set(c, [sx, sy]);
      columns[c.side].push(c);
      // A column never covers the point it names.
      if (c.side === "left") left = Math.min(left, sx - 12);
      else right = Math.max(right, sx + 12);
    }
    if (narrow) {
      if (!latest || !want.has(latest)) return;
      const [sx, sy] = want.get(latest) as [number, number];
      const below = sy + 28 + latest.h < h - 8;
      const tx = Math.min(Math.max(8, sx - latest.w / 2), w - latest.w - 8);
      const ty = below ? sy + 28 : Math.max(8, sy - 28 - latest.h);
      this.placeChip(latest, tx, ty, sx, sy, (c) => [c.x + c.w / 2, below ? c.y : c.y + c.h]);
      return;
    }
    const gap = 8;
    for (const side of ["left", "right"] as const) {
      const list = columns[side];
      if (list.length === 0) continue;
      list.sort((a, b) => (want.get(a) as [number, number])[1] - (want.get(b) as [number, number])[1]);
      // Each chip wants to sit level with its part; it moves down only as far as the chip above it needs.
      const targets: number[] = [];
      let bottom = 8;
      for (const c of list) {
        const [, sy] = want.get(c) as [number, number];
        const y = Math.max(bottom, sy - c.h / 2);
        targets.push(y);
        bottom = y + c.h + gap;
      }
      // If the column runs off the bottom, lift the whole column.
      const over = bottom - gap - (h - 8);
      if (over > 0) {
        for (let j = 0; j < targets.length; j++) targets[j] = (targets[j] as number) - over;
        let top = 8;
        for (let j = 0; j < targets.length; j++) {
          const c = list[j] as Callout;
          targets[j] = Math.max(top, targets[j] as number);
          top = (targets[j] as number) + c.h + gap;
        }
      }
      list.forEach((c, j) => {
        const [sx, sy] = want.get(c) as [number, number];
        const tx = side === "left" ? Math.max(8, Math.min(left - 28 - c.w, w - c.w - 8)) : Math.min(w - 8 - c.w, Math.max(right + 28, 8));
        this.placeChip(c, tx, targets[j] as number, sx, sy, (cc) => [side === "left" ? cc.x + cc.w : cc.x, cc.y + cc.h / 2]);
      });
    }
  }

  /** Ease a chip towards (tx, ty) and draw its line from `edge` (the chip's near edge) to the anchor (sx, sy). */
  private placeChip(c: Callout, tx: number, ty: number, sx: number, sy: number, edge: (c: Callout) => [number, number]): void {
    if (!c.placed) {
      c.x = tx;
      c.y = ty;
      c.placed = true;
    } else {
      c.x += (tx - c.x) * 0.3;
      c.y += (ty - c.y) * 0.3;
      if (Math.abs(tx - c.x) < 0.3) c.x = tx;
      if (Math.abs(ty - c.y) < 0.3) c.y = ty;
    }
    const chip = `${Math.round(c.x)}|${Math.round(c.y)}`;
    if (chip !== c.lastChip) {
      c.lastChip = chip;
      c.el.style.left = `${Math.round(c.x)}px`;
      c.el.style.top = `${Math.round(c.y)}px`;
    }
    const [x0, y0] = edge(c);
    const dx = sx - x0;
    const dy = sy - y0;
    const len = Math.max(0, Math.hypot(dx, dy) - 4);
    const ang = Math.atan2(dy, dx);
    const key = `${Math.round(x0 / MIN_DELTA)}|${Math.round(y0 / MIN_DELTA)}|${Math.round(len / MIN_DELTA)}|${Math.round(ang * 200)}`;
    if (key !== c.lastLine) {
      c.lastLine = key;
      c.line.style.width = `${len.toFixed(1)}px`;
      c.line.style.transform = `translate(${x0.toFixed(1)}px, ${y0.toFixed(1)}px) rotate(${ang.toFixed(4)}rad)`;
    }
  }
}
