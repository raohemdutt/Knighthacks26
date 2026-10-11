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
 *   setExplode(t)   0 is the sealed box, 1 is everything pulled apart; the page maps scroll onto it
 *   setPointer      a gentle tilt towards the pointer
 *   setTheme        relight for Night or Day
 *   setActive       stop drawing while the stage is off screen
 *
 * The model itself is in deviceModel.ts. This file owns the renderer, the lights, the camera, the explode
 * choreography (each part has its own stretch of the eased factor, so the pull-apart reads as a sequence rather
 * than one jump), the fan and the breathing glows, and the callouts: DOM chips in the overlay, laid out beside the
 * device and tied to their part by a thin line, drawn only when the layout changed.
 *
 * The canvas is transparent so the page's aurora shows through, and the camera is fitted to the whole exploded
 * stack at any aspect: it eases between a close frame of the sealed box and a wide one of the stack.
 */

export interface DeviceOptions {
  /** prefers-reduced-motion: no idle sway, no fan motion, no breathing. */
  calm: boolean;
  dark: boolean;
  classes: { callout: string; calloutOn: string; line: string };
}

const FOV = 28;
const VIEW_DIR = new THREE.Vector3(0.55, 0.42, 1).normalize(); // from the front, a little above and to the right
const HIDE_BELOW = 0.15; // no callouts until the lid has clearly started to lift
const MIN_DELTA = 0.5; // a chip or a line is restyled only when it moved by more than this many pixels
const NARROW = 700; // below this many pixels wide, one callout at a time: two columns would cover the device

interface Callout {
  id: PartId;
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

interface Fit {
  aim: THREE.Vector3;
  dist: number;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function smoothstep(a: number, b: number, x: number): number {
  const k = Math.min(1, Math.max(0, (x - a) / (b - a)));
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
  private area = { w: 1, h: 1 }; // the overlay, where the chips go; it may be wider than the host
  private shift = { x: 0, y: 0 }; // the host's top left in the overlay
  private frame = 0;
  private active = true;
  private gone = false;
  private last = 0;
  private t = 0; // as set
  private te = 0; // eased
  private applied = -1; // the eased factor the parts were last moved to
  private pointer = { x: 0, y: 0 };
  private tilt = { yaw: 0, pitch: 0 };
  private spin = 0;
  private box0 = new THREE.Box3();
  private box1 = new THREE.Box3();
  private fit0: Fit = { aim: new THREE.Vector3(), dist: 6 };
  private fit1: Fit = { aim: new THREE.Vector3(), dist: 9 };
  private callouts: Callout[] = [];
  private glow = { bar: 1.2, logo: 0.15 };
  private v = new THREE.Vector3();
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
    canvas.setAttribute("aria-label", "The Atlas device: a small white box with a webcam set into its top, shown pulled apart into its parts.");
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
    this.key.position.set(-2.6, 4.6, 3.2);
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
    this.resize();
    (host as HTMLElement & { deviceScene?: DeviceScene }).deviceScene = this; // for checking draw calls from the console
    this.frame = requestAnimationFrame(this.tick);
  }

  // ---------- the public surface ----------

  setExplode(t: number): void {
    this.t = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
    this.te = easeInOut(this.t);
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

  /** Move every part to where it stands at eased factor `te`; also the ribbon and the shadow under the base. */
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
    this.model.setRibbon(groups.webcam.position);
    // The base drops away from the desk, so its shadow spreads and fades.
    const drop = smoothstep(TRAVEL.base.from, TRAVEL.base.to, te);
    this.model.blob.scale.setScalar(1 + drop * 0.4);
    const blobMat = this.model.blob.material as THREE.MeshBasicMaterial;
    blobMat.opacity = (this.dark ? 0.45 : 0.6) * (1 - drop * 0.6);
  }

  /** The device's bounds sealed and apart, found by standing the parts at each end once. */
  private measureBounds(): void {
    // From the parts only: the shadow's plane under the base is far wider than the box and must not count.
    const { base, lid, board } = this.model.groups;
    const measure = (box: THREE.Box3) => {
      this.model.root.updateMatrixWorld(true);
      box.makeEmpty();
      for (const g of [base, lid, board]) box.union(new THREE.Box3().setFromObject(g, true));
    };
    this.applyExplode(0);
    measure(this.box0);
    this.applyExplode(1);
    measure(this.box1);
    this.applyExplode(0);
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
   * The nearest camera along VIEW_DIR from which every corner of `box` stays inside the given share of the frame.
   * `reserveL` and `reserveR` are pixels of the host kept clear on each side for a column of chips: the model is
   * fitted into the window that is left and the aim is slid sideways so it sits in the middle of that window.
   */
  private fit(box: THREE.Box3, marginX: number, marginY: number, reserveL = 0, reserveR = 0): Fit {
    const aim = box.getCenter(new THREE.Vector3());
    const corners: THREE.Vector3[] = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
    const xLo = -1 + (2 * reserveL) / this.size.w;
    const xHi = 1 - (2 * reserveR) / this.size.w;
    const half = Math.max(0.2, ((xHi - xLo) / 2) * marginX);
    const centre = (xLo + xHi) / 2;
    const probe = this.camera.clone();
    const v = new THREE.Vector3();
    let lo = 0.5;
    let hi = 60;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      probe.position.copy(aim).addScaledVector(VIEW_DIR, mid);
      probe.lookAt(aim);
      probe.updateMatrixWorld();
      const fits = corners.every((c) => {
        v.copy(c).project(probe);
        return Math.abs(v.x) < half && Math.abs(v.y) < marginY;
      });
      if (fits) hi = mid;
      else lo = mid;
    }
    if (centre !== 0) {
      // Moving the aim one way moves the picture the other way, by the frame's half width at that distance.
      const right = new THREE.Vector3().crossVectors(VIEW_DIR.clone().negate(), new THREE.Vector3(0, 1, 0)).normalize();
      const halfWidth = hi * Math.tan((FOV * Math.PI) / 360) * this.camera.aspect;
      aim.addScaledVector(right, -centre * halfWidth);
    }
    return { aim, dist: hi };
  }

  private resize(): void {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.size = { w, h };
    // The chips live in the overlay, which need not be the host's box (the page may keep the device to one side).
    const hostRect = this.host.getBoundingClientRect();
    const overRect = this.overlay.getBoundingClientRect();
    this.area = { w: this.overlay.clientWidth || w, h: this.overlay.clientHeight || h };
    this.shift = { x: hostRect.left - overRect.left, y: hostRect.top - overRect.top };
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // Where the overlay has no room beside the host for a column of chips, the exploded stack leaves that room
    // itself (a phone-width overlay shows one chip at a time instead, so it reserves nothing).
    let chipW = 0;
    for (const c of this.callouts) {
      c.w = 0;
      c.h = 0;
      chipW = Math.max(chipW, c.el.offsetWidth);
    }
    const need = this.area.w < NARROW ? 0 : chipW + 40;
    const reserveL = Math.min(w * 0.35, Math.max(0, need - this.shift.x));
    const reserveR = Math.min(w * 0.35, Math.max(0, need - (this.area.w - this.shift.x - w)));
    // The sealed box sits with air around it; the exploded stack fills the frame but for a margin.
    this.fit0 = this.fit(this.box0, 0.72, 0.62);
    this.fit1 = this.fit(this.box1, 0.9, 0.9, reserveL, reserveR);
    if (!this.frame && !this.gone && this.active) this.frame = requestAnimationFrame(this.tick);
  }

  private pose(): void {
    const k = this.te;
    this.v.lerpVectors(this.fit0.aim, this.fit1.aim, k);
    const dist = this.fit0.dist + (this.fit1.dist - this.fit0.dist) * k;
    this.camera.position.copy(this.v).addScaledVector(VIEW_DIR, dist);
    this.camera.lookAt(this.v);
  }

  // ---------- the loop ----------

  private tick = (now: number) => {
    this.frame = 0;
    if (this.gone || !this.active) return;
    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 1 / 60;
    this.last = now;
    const calm = this.opts.calm;

    // The idle sway and the pointer tilt, both eased so nothing snaps.
    if (!calm) this.spin = Math.sin((now / 10000) * Math.PI * 2) * 0.07;
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
    for (const part of DEVICE_PARTS) {
      const anchor = ANCHORS.find((a) => a.part === part.id);
      if (!anchor) continue;
      const el = document.createElement("div");
      el.className = callout;
      const name = document.createElement("strong");
      name.textContent = part.name;
      const blurb = document.createElement("span");
      blurb.textContent = part.blurb;
      el.append(name, blurb);
      el.style.visibility = "hidden";
      const lead = document.createElement("i");
      lead.className = line;
      lead.style.visibility = "hidden";
      this.overlay.append(el, lead); // the line right after its chip, so the stylesheet can pair them
      this.callouts.push({ id: part.id, el, line: lead, group: this.model.groups[anchor.group], local: anchor.local, threshold: anchor.threshold, side: anchor.side, w: 0, h: 0, on: false, x: 0, y: 0, placed: false, lastChip: "", lastLine: "" });
    }
  }

  /**
   * Lay the chips out and tie each to its part with a line whose far end carries the dot. On a wide overlay every
   * chip whose threshold the explode factor has passed sits beside the device on its part's side, stacked so none
   * overlaps another. On a narrow one (a phone) two columns would cover the device, so only the part revealed
   * most recently is named, as a tooltip under or over its anchor. Styles are written only when something moved.
   */
  private layoutCallouts(): void {
    const { w, h } = this.area;
    const hostW = this.size.w;
    const hostH = this.size.h;
    const { x: ox, y: oy } = this.shift;
    const showAny = this.t >= HIDE_BELOW;
    const narrow = w < NARROW;
    // On a narrow overlay, the one chip to show: the highest threshold the factor has passed.
    let latest: Callout | null = null;
    if (narrow && showAny) for (const c of this.callouts) if (this.te >= c.threshold && (!latest || c.threshold > latest.threshold)) latest = c;
    // The device's extent on screen, from the corners of its current bounds, so the columns sit just outside it.
    let left = w;
    let right = 0;
    const k = this.te;
    const min = this.box0.min.clone().lerp(this.box1.min, k);
    const max = this.box0.max.clone().lerp(this.box1.max, k);
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
      const on = narrow ? c === latest : showAny && this.te >= c.threshold;
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
        c.h = c.el.offsetHeight || 56;
      }
      this.v.copy(c.local);
      c.group.localToWorld(this.v).project(this.camera);
      const sx = (this.v.x * 0.5 + 0.5) * hostW + ox;
      const sy = (-this.v.y * 0.5 + 0.5) * hostH + oy;
      want.set(c, [sx, sy]);
      columns[c.side].push(c);
      // A column never covers the point it names: a part eased to its full travel may be past the lerped bounds.
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
    const gap = 10;
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
