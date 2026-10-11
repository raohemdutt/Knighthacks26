import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import type { RoomFurniture, RoomOpening } from "@/lib/api";

/**
 * Low-poly furniture for the room twin, modelled here in code (no downloaded assets: the box is often offline).
 *
 * Everything static is baked into a handful of merged meshes with per-vertex colour, so a whole furnished room is
 * about a dozen draw calls. A piece is built in its own frame (metres, origin on the floor at its centre, width along
 * x, its front towards +z) and placed by the plan's centre and rotation. Plan (x, y) is world (x, z); y is up.
 *
 * Soft things (mattresses, pillows, seats) and slabs with visible edges are rounded boxes with a few segments, which
 * catch the light along their edges and read as furniture at phone size. Lamps register where their bulb is, so the
 * scene can put a warm light there at night.
 */

export type Wall = "north" | "east" | "south" | "west";
export const WALLS: Wall[] = ["north", "east", "south", "west"];
export const WALL_T = 0.1;
export const DOOR_H = 2.04;

type Layer = "matte" | "gloss" | "glow";

/** A small repeatable random source, so a room looks the same every time it is built. */
export function seeded(text: string): () => number {
  let a = 2166136261;
  for (let i = 0; i < text.length; i++) a = Math.imul(a ^ text.charCodeAt(i), 16777619);
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const scratch = new THREE.Color();

interface Place {
  layer?: Layer;
  ry?: number;
  rx?: number;
  rz?: number;
  vary?: number;
}

/** Collects coloured geometry, already transformed, and merges each layer into one mesh. */
export class Mesher {
  private layers: Record<Layer, THREE.BufferGeometry[]> = { matte: [], gloss: [], glow: [] };
  private stack: THREE.Matrix4[] = [new THREE.Matrix4()];
  private rand: () => number;
  /** World positions of every lamp bulb built so far: where the scene puts its lights at night. */
  readonly lamps: THREE.Vector3[] = [];

  constructor(seed: string) {
    this.rand = seeded(seed);
  }

  /** A number in -1..1, the same on every build. */
  jitter(): number {
    return this.rand() * 2 - 1;
  }

  pick<T>(items: T[]): T {
    return items[Math.floor(this.rand() * items.length) % items.length] as T;
  }

  /** Run `build` with everything moved to (x, y, z) and turned by `ry` about the vertical. */
  at(x: number, y: number, z: number, ry: number, build: () => void): void {
    const top = this.stack[this.stack.length - 1] as THREE.Matrix4;
    const m = new THREE.Matrix4().makeRotationY(ry).setPosition(x, y, z);
    this.stack.push(top.clone().multiply(m));
    build();
    this.stack.pop();
  }

  /** Note a lamp's bulb at (x, y, z) in the current frame. */
  lamp(x: number, y: number, z: number): void {
    this.lamps.push(new THREE.Vector3(x, y, z).applyMatrix4(this.stack[this.stack.length - 1] as THREE.Matrix4));
  }

  add(geometry: THREE.BufferGeometry, color: number, layer: Layer = "matte", vary = 0.035, local?: THREE.Matrix4): void {
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    if (g !== geometry) geometry.dispose();
    if (local) g.applyMatrix4(local);
    g.applyMatrix4(this.stack[this.stack.length - 1] as THREE.Matrix4);
    scratch.setHex(color, THREE.SRGBColorSpace);
    if (vary > 0) scratch.offsetHSL(this.jitter() * vary * 0.25, this.jitter() * vary, this.jitter() * vary);
    const n = g.getAttribute("position").count;
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      colors[i * 3] = scratch.r;
      colors[i * 3 + 1] = scratch.g;
      colors[i * 3 + 2] = scratch.b;
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.deleteAttribute("uv");
    this.layers[layer].push(g);
  }

  private frame(x: number, y: number, z: number, o: Place): THREE.Matrix4 {
    return new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(o.rx ?? 0, o.ry ?? 0, o.rz ?? 0)).setPosition(x, y, z);
  }

  /** A box by its size and its centre. `ry`/`rx`/`rz` turn it about its own centre. */
  box(w: number, h: number, d: number, x: number, y: number, z: number, color: number, o: Place = {}): void {
    this.add(new THREE.BoxGeometry(w, h, d), color, o.layer, o.vary, this.frame(x, y, z, o));
  }

  /** A box with rounded edges, for soft things and slabs whose edges should catch the light. `r` is the radius. */
  rbox(w: number, h: number, d: number, x: number, y: number, z: number, color: number, o: Place & { r?: number; seg?: number } = {}): void {
    const r = Math.min(o.r ?? 0.03, w / 2, h / 2, d / 2);
    this.add(new RoundedBoxGeometry(w, h, d, o.seg ?? 2, r), color, o.layer, o.vary, this.frame(x, y, z, o));
  }

  /** An upright cylinder (or cone) by the centre of its axis. */
  cyl(rTop: number, rBottom: number, h: number, x: number, y: number, z: number, color: number, o: Place & { seg?: number; open?: boolean } = {}): void {
    this.add(new THREE.CylinderGeometry(rTop, rBottom, h, o.seg ?? 14, 1, o.open ?? false), color, o.layer, o.vary, this.frame(x, y, z, o));
  }

  ball(r: number, x: number, y: number, z: number, color: number, o: { layer?: Layer; sy?: number; sx?: number; sz?: number; vary?: number } = {}): void {
    const m = new THREE.Matrix4().makeScale(o.sx ?? 1, o.sy ?? 1, o.sz ?? 1).setPosition(x, y, z);
    this.add(new THREE.SphereGeometry(r, 12, 9), color, o.layer, o.vary, m);
  }

  /** A ring lying flat (or turned by `rx`), by its centre. */
  torus(r: number, tube: number, x: number, y: number, z: number, color: number, o: Place & { seg?: number } = {}): void {
    const m = this.frame(x, y, z, { ...o, rx: (o.rx ?? 0) + Math.PI / 2 });
    this.add(new THREE.TorusGeometry(r, tube, 6, o.seg ?? 16), color, o.layer, o.vary, m);
  }

  /** A flat panel facing +z (for screens and glass), by its centre. */
  panel(w: number, h: number, x: number, y: number, z: number, color: number, o: { layer?: Layer; rx?: number; ry?: number } = {}): void {
    const m = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(o.rx ?? 0, o.ry ?? 0, 0)).setPosition(x, y, z);
    this.add(new THREE.PlaneGeometry(w, h), color, o.layer ?? "glow", 0, m);
  }

  /** A thin cable from a to b, sagging a little in the middle. */
  cable(a: [number, number, number], b: [number, number, number], color = 0x2a2d33, r = 0.006): void {
    const sag = 0.05 + Math.hypot(b[0] - a[0], b[2] - a[2]) * 0.12;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(...a),
      new THREE.Vector3((a[0] + b[0]) / 2, Math.min(a[1], b[1]) + Math.abs(a[1] - b[1]) * 0.25 - sag, (a[2] + b[2]) / 2),
      new THREE.Vector3(...b),
    ]);
    this.add(new THREE.TubeGeometry(curve, 6, r, 5, false), color, "gloss", 0);
  }

  build(materials: Record<Layer, THREE.Material>): THREE.Group {
    const group = new THREE.Group();
    for (const layer of ["matte", "gloss", "glow"] as Layer[]) {
      const parts = this.layers[layer];
      if (parts.length === 0) continue;
      const merged = mergeGeometries(parts, false);
      for (const p of parts) p.dispose();
      if (!merged) continue;
      const mesh = new THREE.Mesh(merged, materials[layer]);
      mesh.castShadow = layer !== "glow";
      mesh.receiveShadow = layer !== "glow";
      group.add(mesh);
    }
    this.layers = { matte: [], gloss: [], glow: [] };
    return group;
  }
}

// ---------- palette ----------

export const C = {
  wall: 0xe9e4da,
  wallCut: 0xcfc8bb,
  trim: 0xf6f3ee,
  base: 0x23282f,
  oak: 0xc9a373,
  oakDark: 0x9a7448,
  walnut: 0x6e4f35,
  white: 0xf0eeea,
  linen: 0xe6dfd2,
  steel: 0x8b929b,
  brass: 0xd1a95a,
  charcoal: 0x3a3f47,
  black: 0x1d2025,
  duvet: 0x5f7fa3,
  duvetFold: 0xdfe6ee,
  mustard: 0xd2a23c,
  sage: 0x8fa58a,
  plant: 0x4f8a57,
  plantDark: 0x376a40,
  terracotta: 0xb86b4b,
  cardboard: 0xc49a6c,
  screen: 0xa8d6ff,
  screenWarm: 0xffe1b0,
  lamp: 0xffd9a0,
  sky: 0xd6ebff,
  binBlue: 0x2f7ed8,
  binGrey: 0x6c747e,
  binGreen: 0x3f9b5a,
  skin: 0xe2b48f,
  fabric: 0x8795a8,
};

const BOOKS = [0xb8473d, 0x2f5d8a, 0xe0b23f, 0x3f7f5f, 0x7a4f8f, 0xe8e2d2, 0x2c2f36, 0xd77a3a, 0x4aa3a8];

export function binColor(name: string | null | undefined): number {
  if (name === "blue") return C.binBlue;
  if (name === "green") return C.binGreen;
  return C.binGrey;
}

// ---------- furniture ----------

function laptop(m: Mesher, x: number, y: number, z: number, ry: number, glow = C.screen): void {
  m.at(x, y, z, ry, () => {
    m.rbox(0.32, 0.016, 0.22, 0, 0.008, 0, 0xb9bec6, { layer: "gloss", vary: 0.01, r: 0.006, seg: 1 });
    m.box(0.26, 0.002, 0.1, 0, 0.0175, 0.02, 0x4a4f57, { vary: 0 });
    m.box(0.1, 0.001, 0.06, 0, 0.0175, 0.085, 0x5b6068, { vary: 0 }); // the trackpad
    m.rbox(0.32, 0.21, 0.012, 0, 0.115, -0.125, 0xb9bec6, { layer: "gloss", rx: -0.26, vary: 0.01, r: 0.006, seg: 1 });
    m.panel(0.29, 0.18, 0, 0.116, -0.1175, glow, { rx: -0.26 });
  });
}

function mug(m: Mesher, x: number, y: number, z: number, color: number): void {
  m.cyl(0.04, 0.035, 0.09, x, y + 0.045, z, color, { seg: 12 });
  m.cyl(0.033, 0.033, 0.004, x, y + 0.088, z, 0x4a2e1c, { seg: 12, vary: 0 });
  m.torus(0.022, 0.006, x + 0.045, y + 0.05, z, color, { rx: Math.PI / 2, rz: Math.PI / 2 });
}

/** A pot plant: a tapered pot, soil, and a fan of leaves. `s` scales the whole thing (1 is about 45 cm tall). */
function plant(m: Mesher, x: number, y: number, z: number, s = 1, pot = C.terracotta): void {
  m.at(x, y, z, m.jitter() * Math.PI, () => {
    m.cyl(0.08 * s, 0.06 * s, 0.14 * s, 0, 0.07 * s, 0, pot);
    m.cyl(0.085 * s, 0.085 * s, 0.02 * s, 0, 0.13 * s, 0, pot);
    m.cyl(0.07 * s, 0.07 * s, 0.01 * s, 0, 0.14 * s, 0, 0x3b2d22, { vary: 0 });
    const leaves = 7;
    for (let i = 0; i < leaves; i++) {
      const a = (i / leaves) * Math.PI * 2 + m.jitter() * 0.3;
      const len = (0.17 + Math.abs(m.jitter()) * 0.09) * s;
      const tilt = 0.55 + Math.abs(m.jitter()) * 0.5;
      m.at(0, 0.14 * s, 0, a, () => {
        // a stem leaning outwards, and a leaf hanging off its end
        m.cyl(0.004 * s, 0.006 * s, len, 0, (len / 2) * Math.cos(tilt), (len / 2) * Math.sin(tilt), C.plantDark, { seg: 5, rx: tilt });
        m.rbox(0.07 * s, 0.01 * s, 0.14 * s, 0, len * Math.cos(tilt) + 0.01, len * Math.sin(tilt) + 0.05 * s, i % 2 ? C.plant : C.plantDark, { rx: tilt - 0.4, r: 0.02 * s, seg: 1, vary: 0.05 });
      });
    }
  });
}

function bed(m: Mesher, w: number, d: number, h: number): void {
  const top = Math.max(h, 0.4);
  m.rbox(w, 0.2, d, 0, 0.2, 0, C.oakDark, { r: 0.015, seg: 1 }); // the frame
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) m.box(0.06, 0.1, 0.06, sx * (w / 2 - 0.05), 0.05, sz * (d / 2 - 0.05), C.walnut);
  m.rbox(w, 0.95, 0.06, 0, 0.475, -d / 2 + 0.03, C.oakDark, { r: 0.012, seg: 1 }); // headboard
  for (const y of [0.62, 0.78]) m.box(w - 0.1, 0.06, 0.012, 0, y, -d / 2 + 0.066, C.oak, { vary: 0.01 }); // its rails
  m.rbox(w - 0.06, top - 0.3, d - 0.1, 0, 0.3 + (top - 0.3) / 2, 0.02, C.white, { r: 0.05, vary: 0.01 }); // mattress
  m.box(w - 0.08, 0.008, d - 0.12, 0, 0.3 + (top - 0.3) / 2, 0.02, C.linen, { vary: 0.01 }); // its piping line
  m.rbox(w * 0.62, 0.12, 0.4, -0.03, top + 0.06, -d / 2 + 0.33, C.white, { ry: 0.06, vary: 0.01, r: 0.045 }); // pillow
  m.rbox(w * 0.46, 0.09, 0.3, 0.14, top + 0.14, -d / 2 + 0.3, C.mustard, { ry: -0.25, r: 0.035 }); // a second, smaller pillow
  m.rbox(w + 0.04, 0.11, d * 0.6, 0, top + 0.05, d / 2 - d * 0.3 - 0.02, C.duvet, { r: 0.045 }); // duvet
  m.rbox(w + 0.045, 0.1, 0.17, 0, top + 0.06, d / 2 - d * 0.6 - 0.02, C.duvetFold, { vary: 0.01, r: 0.04 }); // its turned-back edge
  m.rbox(w + 0.06, 0.04, 0.34, 0, top + 0.12, d / 2 - 0.3, C.terracotta, { ry: 0.03, r: 0.015, seg: 1 }); // a throw at the foot
  m.rbox(0.3, 0.025, 0.2, w / 2 - 0.25, top + 0.15, d / 2 - 0.55, 0x2c2f36, { ry: 0.5, r: 0.008, seg: 1 }); // a book left on the bed
}

function monitor(m: Mesher, x: number, y: number, z: number, glow = C.screen): void {
  m.at(x, y, z, 0, () => {
    m.cyl(0.1, 0.11, 0.012, 0, 0.006, 0.02, C.black, { layer: "gloss" });
    m.box(0.03, 0.14, 0.02, 0, 0.08, 0.01, C.black, { layer: "gloss" });
    m.rbox(0.6, 0.36, 0.028, 0, 0.32, 0.02, C.black, { layer: "gloss", r: 0.008, seg: 1 });
    m.panel(0.565, 0.32, 0, 0.325, 0.0345, glow);
    m.box(0.56, 0.01, 0.004, 0, 0.15, 0.036, 0x30353c, { vary: 0 }); // the bezel's lower lip
  });
}

function desk(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w, 0.04, d, 0, h - 0.02, 0, C.oak, { r: 0.01, seg: 1 });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) m.box(0.045, h - 0.04, 0.045, sx * (w / 2 - 0.05), (h - 0.04) / 2, sz * (d / 2 - 0.05), C.charcoal);
  m.rbox(0.38, 0.5, d - 0.1, w / 2 - 0.24, h - 0.04 - 0.25, 0, C.white, { r: 0.008, seg: 1 }); // drawers
  for (const y of [0.13, 0.3]) m.box(0.34, 0.006, 0.01, w / 2 - 0.24, h - 0.04 - y, d / 2 - 0.045, C.steel, { layer: "gloss" });
  for (const y of [0.08, 0.25, 0.42]) m.box(0.1, 0.012, 0.02, w / 2 - 0.24, h - 0.04 - y, d / 2 - 0.04, C.brass, { layer: "gloss" }); // handles
  m.box(w - 0.12, 0.3, 0.02, 0, h - 0.2, -d / 2 + 0.05, C.oakDark); // back panel
  monitor(m, -0.12, h, -d / 2 + 0.14);
  m.cable([-0.12, h + 0.02, -d / 2 + 0.17], [-0.1, h - 0.05, -d / 2 + 0.02]); // the monitor's lead over the back edge
  m.cable([-0.1, h - 0.05, -d / 2 + 0.02], [0.05, 0.02, -d / 2 + 0.06]);
  // keyboard and mouse, on a mat
  m.rbox(0.62, 0.006, 0.3, -0.08, h + 0.003, 0.08, 0x2b3038, { r: 0.004, seg: 1, vary: 0.01 });
  m.rbox(0.38, 0.016, 0.13, -0.1, h + 0.012, 0.1, 0xd9dce0, { r: 0.005, seg: 1, vary: 0.01 });
  m.box(0.35, 0.006, 0.1, -0.1, h + 0.021, 0.1, 0x8b929b, { vary: 0 });
  m.rbox(0.058, 0.03, 0.095, 0.15, h + 0.015, 0.11, 0xe8eaed, { r: 0.015, ry: 0.2 });
  laptop(m, 0.28, h, 0.07, -0.2);
  // lamp
  m.cyl(0.07, 0.08, 0.02, -w / 2 + 0.14, h + 0.01, -d / 2 + 0.16, C.charcoal);
  m.cyl(0.011, 0.011, 0.36, -w / 2 + 0.17, h + 0.19, -d / 2 + 0.16, C.charcoal, { rz: -0.18 });
  m.cyl(0.05, 0.1, 0.11, -w / 2 + 0.22, h + 0.39, -d / 2 + 0.17, C.charcoal, { rz: -0.5, open: true });
  m.cyl(0.045, 0.085, 0.09, -w / 2 + 0.225, h + 0.385, -d / 2 + 0.17, C.lamp, { layer: "glow", rz: -0.5 });
  m.lamp(-w / 2 + 0.26, h + 0.34, -d / 2 + 0.2);
  mug(m, -w / 2 + 0.2, h, 0.14, C.terracotta);
  m.rbox(0.21, 0.02, 0.28, w / 2 - 0.2, h + 0.01, 0.14, 0x39424e, { ry: 0.2, r: 0.006, seg: 1 }); // a notebook on the drawers side
  m.rbox(0.2, 0.012, 0.27, w / 2 - 0.21, h + 0.026, 0.13, C.mustard, { ry: -0.1, r: 0.004, seg: 1 });
  m.cyl(0.035, 0.03, 0.1, w / 2 - 0.09, h + 0.05, -d / 2 + 0.12, C.sage); // pen pot
  for (let i = 0; i < 4; i++) m.cyl(0.004, 0.004, 0.15, w / 2 - 0.09 + Math.sin(i * 1.7) * 0.018, h + 0.13, -d / 2 + 0.12 + Math.cos(i * 1.7) * 0.018, m.pick(BOOKS), { seg: 5, rz: (i - 1.5) * 0.08 });
  plant(m, w / 2 - 0.3, h, -d / 2 + 0.12, 0.55);
}

function chair(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w * 0.92, 0.08, d * 0.9, 0, h, 0.01, C.charcoal, { r: 0.03 });
  m.rbox(w * 0.8, 0.03, d * 0.7, 0, h + 0.05, 0.03, 0x4a505a, { r: 0.012, seg: 1 }); // the seat cushion's dip
  m.rbox(w * 0.86, 0.48, 0.06, 0, h + 0.33, -d / 2 + 0.04, C.charcoal, { rx: -0.08, r: 0.025 });
  m.rbox(w * 0.6, 0.1, 0.02, 0, h + 0.5, -d / 2 + 0.072, 0x4a505a, { rx: -0.08, r: 0.008, seg: 1 }); // a lighter headrest pad
  m.box(0.05, 0.14, 0.03, 0, h + 0.07, -d / 2 + 0.05, C.black);
  for (const sx of [-1, 1]) {
    m.box(0.025, 0.2, 0.025, sx * (w / 2 - 0.02), h + 0.1, 0.02, C.black); // armrest posts
    m.rbox(0.05, 0.025, 0.26, sx * (w / 2 - 0.02), h + 0.21, 0, C.black, { r: 0.01, seg: 1 });
  }
  m.cyl(0.028, 0.028, h - 0.1, 0, (h - 0.1) / 2 + 0.06, 0, C.steel, { layer: "gloss" });
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    m.box(0.045, 0.03, 0.27, Math.sin(a) * 0.13, 0.05, Math.cos(a) * 0.13, C.black, { ry: a });
    m.cyl(0.025, 0.025, 0.03, Math.sin(a) * 0.26, 0.02, Math.cos(a) * 0.26, C.black, { seg: 8 });
  }
}

function stool(m: Mesher, w: number, d: number, h: number): void {
  // a plain four-legged chair, for the table
  m.rbox(w * 0.9, 0.045, d * 0.86, 0, h, 0, C.oak, { r: 0.012, seg: 1 });
  m.rbox(w * 0.9, 0.34, 0.035, 0, h + 0.3, -d / 2 + 0.04, C.oak, { rx: -0.07, r: 0.012, seg: 1 });
  m.rbox(w * 0.76, 0.03, d * 0.7, 0, h + 0.035, 0.02, C.fabric, { r: 0.012, seg: 1 }); // a thin cushion
  for (const sx of [-1, 1]) {
    m.box(0.035, h, 0.035, sx * (w / 2 - 0.05), h / 2, d / 2 - 0.06, C.charcoal);
    m.box(0.035, h + 0.46, 0.035, sx * (w / 2 - 0.05), (h + 0.46) / 2, -d / 2 + 0.05, C.charcoal);
  }
}

function wardrobe(m: Mesher, w: number, d: number, h: number): void {
  m.box(w, 0.08, d - 0.04, 0, 0.04, -0.02, C.walnut);
  m.rbox(w, h - 0.08, d - 0.02, 0, 0.08 + (h - 0.08) / 2, -0.01, C.oak, { r: 0.01, seg: 1 });
  for (const sx of [-1, 1]) {
    m.rbox(w / 2 - 0.012, h - 0.14, 0.02, sx * (w / 4), 0.1 + (h - 0.14) / 2, d / 2 - 0.01, C.white, { vary: 0.012, r: 0.006, seg: 1 });
    m.box(w / 2 - 0.1, h - 0.3, 0.004, sx * (w / 4), 0.1 + (h - 0.14) / 2, d / 2 + 0.001, 0xe4e0da, { vary: 0.005 }); // the door's inset panel
    m.box(0.016, 0.26, 0.022, sx * 0.045, h * 0.52, d / 2 + 0.008, C.brass, { layer: "gloss" });
  }
  m.rbox(0.5, 0.26, 0.36, -0.18, h + 0.13, -0.04, C.cardboard, { ry: 0.08, r: 0.008, seg: 1 }); // a storage box on top
  m.rbox(0.38, 0.16, 0.3, 0.3, h + 0.08, -0.02, 0x7b8fa6, { ry: -0.1, r: 0.02 });
}

function shelf(m: Mesher, w: number, d: number, h: number): void {
  const t = 0.025;
  for (const sx of [-1, 1]) m.box(t, h, d, sx * (w / 2 - t / 2), h / 2, 0, C.oak);
  m.box(w, 0.012, d, 0, h - 0.006, 0, C.oak);
  m.box(w - 2 * t, h, 0.012, 0, h / 2, -d / 2 + 0.006, C.oakDark);
  const levels = Math.max(2, Math.round(h / 0.36));
  for (let i = 0; i < levels; i++) {
    const y = 0.06 + (i * (h - 0.1)) / levels;
    m.box(w - 2 * t, t, d - 0.02, 0, y, 0.005, C.oak);
    if (i === 0) {
      m.rbox(0.3, 0.2, d - 0.08, -0.14, y + 0.11, 0.01, 0x8b9bb0, { r: 0.015, seg: 1 }); // a storage bin on the bottom shelf
      m.rbox(0.22, 0.14, d - 0.1, 0.2, y + 0.085, 0.01, C.linen, { r: 0.02, seg: 1 });
      continue;
    }
    let x = -w / 2 + t + 0.02;
    const end = w / 2 - t - 0.02 - (i % 2 ? 0.2 : 0.06);
    while (x < end) {
      const bw = 0.028 + Math.abs(m.jitter()) * 0.03;
      const bh = 0.19 + Math.abs(m.jitter()) * 0.08;
      if (x + bw > end) break;
      m.box(bw, bh, d * 0.62, x + bw / 2, y + t / 2 + bh / 2, 0.02, m.pick(BOOKS), { vary: 0.06 });
      x += bw + 0.003;
    }
    if (i % 2) plant(m, w / 2 - t - 0.11, y + t / 2, 0.02, 0.45, C.white);
  }
  m.rbox(0.24, 0.03, 0.18, -0.1, h + 0.015, 0, 0x2f5d8a, { ry: 0.3, r: 0.006, seg: 1 }); // books lying on top
  m.rbox(0.22, 0.025, 0.17, -0.1, h + 0.043, 0, 0xe0b23f, { ry: 0.1, r: 0.006, seg: 1 });
  plant(m, 0.2, h, 0, 0.7, C.white);
}

function fridge(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w, h, d - 0.03, 0, h / 2, -0.015, C.white, { layer: "gloss", vary: 0.01, r: 0.02 });
  m.rbox(w - 0.01, h * 0.3 - 0.01, 0.03, 0, h - h * 0.15, d / 2 - 0.015, 0xe4e6e9, { layer: "gloss", vary: 0.005, r: 0.008, seg: 1 });
  m.rbox(w - 0.01, h * 0.7 - 0.02, 0.03, 0, h * 0.35, d / 2 - 0.015, 0xe4e6e9, { layer: "gloss", vary: 0.005, r: 0.008, seg: 1 });
  for (const y of [h - h * 0.15, h * 0.52]) m.rbox(0.02, 0.16, 0.025, w / 2 - 0.06, y, d / 2 + 0.012, C.steel, { layer: "gloss", r: 0.008, seg: 1 });
  m.box(0.07, 0.09, 0.004, -w * 0.2, h * 0.5, d / 2 + 0.004, 0xd7553f, { vary: 0 }); // a magnet and a photo on the door
  m.box(0.1, 0.07, 0.004, w * 0.12, h * 0.42, d / 2 + 0.004, 0xe8e2d2, { vary: 0, ry: 0 });
  // a microwave on top, and a cereal box
  m.rbox(w * 0.92, 0.26, d * 0.7, 0, h + 0.13, -0.03, C.charcoal, { layer: "gloss", r: 0.012, seg: 1 });
  m.box(w * 0.55, 0.17, 0.01, -w * 0.13, h + 0.13, d * 0.35 - 0.025, C.black, { layer: "gloss", vary: 0 });
  m.box(0.02, 0.09, 0.012, w * 0.33, h + 0.13, d * 0.35 - 0.02, C.steel, { layer: "gloss" }); // its handle
  m.box(0.07, 0.26, 0.19, w * 0.3, h + 0.26 + 0.13, -0.05, C.mustard, { ry: 0.3 });
}

function nightstand(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w, h - 0.06, d, 0, 0.06 + (h - 0.06) / 2, 0, C.oak, { r: 0.01, seg: 1 });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) m.box(0.035, 0.06, 0.035, sx * (w / 2 - 0.04), 0.03, sz * (d / 2 - 0.04), C.walnut);
  m.box(w - 0.05, 0.17, 0.015, 0, h - 0.13, d / 2 + 0.004, C.oakDark);
  m.ball(0.014, 0, h - 0.13, d / 2 + 0.02, C.brass, { layer: "gloss" });
  m.cyl(0.05, 0.06, 0.03, -0.08, h + 0.015, -0.03, C.brass, { layer: "gloss" });
  m.cyl(0.012, 0.012, 0.2, -0.08, h + 0.12, -0.03, C.brass, { layer: "gloss" });
  m.cyl(0.075, 0.1, 0.14, -0.08, h + 0.28, -0.03, C.linen, { open: true });
  m.cyl(0.07, 0.094, 0.13, -0.08, h + 0.28, -0.03, C.lamp, { layer: "glow" });
  m.lamp(-0.08, h + 0.24, -0.03);
  m.rbox(0.14, 0.022, 0.19, 0.1, h + 0.011, 0.04, 0x7a4f8f, { ry: 0.3, r: 0.006, seg: 1 });
  m.rbox(0.07, 0.008, 0.14, 0.12, h + 0.026, 0.02, C.black, { ry: 0.4, r: 0.004, seg: 1, layer: "gloss" }); // a phone on the book
}

function table(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w, 0.04, d, 0, h - 0.02, 0, C.linen, { r: 0.012, seg: 1 });
  m.box(w - 0.16, 0.07, d - 0.16, 0, h - 0.075, 0, C.steel);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) m.box(0.045, h - 0.04, 0.045, sx * (w / 2 - 0.08), (h - 0.04) / 2, sz * (d / 2 - 0.08), C.charcoal);
  // the front-left corner is kept clear: that is where the demo's keys land
  laptop(m, 0.42, h, 0.16, 0.06);
  laptop(m, -0.45, h, -0.16, Math.PI + 0.08, C.screenWarm);
  laptop(m, 0.45, h, -0.17, Math.PI - 0.1);
  mug(m, 0.02, h, 0.22, C.terracotta);
  mug(m, -0.1, h, -0.24, C.duvet);
  m.cyl(0.033, 0.033, 0.12, 0.78, h + 0.06, 0.02, 0xc8ccd2, { layer: "gloss" }); // an energy drink
  m.rbox(0.34, 0.03, 0.06, 0, h + 0.015, 0, C.white, { ry: 0.04, r: 0.008, seg: 1 }); // power strip
  m.cable([0.16, h + 0.02, 0], [0.3, h + 0.01, 0.1]);
  m.cable([-0.16, h + 0.02, 0.01], [-0.4, h + 0.01, -0.1]);
  m.cable([0, h + 0.01, -0.03], [0.1, 0.02, -d / 2 - 0.05]); // down to the floor
  m.rbox(0.2, 0.014, 0.27, -0.72, h + 0.007, -0.2, 0x39424e, { ry: -0.3, r: 0.005, seg: 1 });
  m.rbox(0.3, 0.1, 0.22, 0.02, h + 0.05, -0.02, C.cardboard, { ry: 0.5, r: 0.006, seg: 1 }); // a box of hardware
  m.box(0.31, 0.004, 0.05, 0.02, h + 0.102, -0.02, 0xe9dcc2, { ry: 0.5, vary: 0 }); // its tape
}

/** A tall floor lamp: a weighted base, a thin pole and a drum shade that glows. */
function floorLamp(m: Mesher): void {
  m.cyl(0.14, 0.15, 0.025, 0, 0.0125, 0, C.charcoal);
  m.cyl(0.012, 0.012, 1.35, 0, 0.7, 0, C.brass, { layer: "gloss" });
  m.cyl(0.16, 0.17, 0.3, 0, 1.47, 0, C.linen, { open: true, seg: 18 });
  m.cyl(0.15, 0.16, 0.29, 0, 1.47, 0, C.lamp, { layer: "glow", seg: 18 });
  m.lamp(0, 1.4, 0);
}

/** A laundry basket, woven. */
function basket(m: Mesher): void {
  m.cyl(0.2, 0.17, 0.5, 0, 0.25, 0, 0xc9b08a, { seg: 12 });
  for (let i = 0; i < 4; i++) m.cyl(0.203 + i * 0.001, 0.2 + i * 0.001, 0.012, 0, 0.09 + i * 0.12, 0, 0xa88f68, { seg: 12, vary: 0.02 });
  m.rbox(0.26, 0.08, 0.2, 0, 0.52, 0, C.fabric, { ry: 0.6, r: 0.03 }); // a towel over the rim
}

function generic(m: Mesher, w: number, d: number, h: number): void {
  m.rbox(w, h, d, 0, h / 2, 0, C.linen, { r: 0.015, seg: 1 });
}

const BUILDERS: Record<string, (m: Mesher, w: number, d: number, h: number) => void> = { bed, desk, chair, wardrobe, shelf, fridge, nightstand, table };

/** True when a circle of radius `r` at (x, z) is clear of every piece of furniture (grown by a little). */
function clear(furniture: RoomFurniture[], x: number, z: number, r: number): boolean {
  for (const f of furniture) {
    if (f.type === "rug") continue;
    const t = (f.rot * Math.PI) / 180;
    const dx = x - f.at[0];
    const dz = z - f.at[1];
    const lx = dx * Math.cos(t) + dz * Math.sin(t);
    const lz = -dx * Math.sin(t) + dz * Math.cos(t);
    if (Math.abs(lx) <= f.size[0] / 2 + r + 0.08 && Math.abs(lz) <= f.size[1] / 2 + r + 0.08) return false;
  }
  return true;
}

/**
 * Every piece of furniture except rugs and bins (those are their own meshes), merged. With the room's size, a few
 * decorations are added in whichever corners are free (a plant, a floor lamp, a laundry basket), and `lamps` is
 * filled with where the lamp bulbs are.
 */
export function buildFurniture(furniture: RoomFurniture[], plainChairs: boolean, materials: Record<Layer, THREE.Material>, room?: [number, number], lamps?: THREE.Vector3[]): THREE.Group {
  const m = new Mesher("furniture");
  for (const f of furniture) {
    if (f.type === "rug" || f.type === "bin") continue;
    const build = f.type === "chair" && plainChairs ? stool : (BUILDERS[f.type] ?? generic);
    m.at(f.at[0], 0, f.at[1], (-f.rot * Math.PI) / 180, () => build(m, f.size[0], f.size[1], f.height));
  }
  if (room) {
    const [w, d] = room;
    const inset = 0.3;
    const corners: [number, number, number][] = [
      [inset, inset, 0.3],
      [w - inset, inset, -0.7],
      [w - inset, d - inset, -2.4],
      [inset, d - inset, 2.2],
    ];
    const decor = [(x: number, z: number, a: number) => m.at(x, 0, z, a, () => floorLamp(m)), (x: number, z: number, a: number) => plant(m, x, 0, z, 1.6, 0xd9d4ca), (x: number, z: number, a: number) => m.at(x, 0, z, a, () => basket(m))];
    let i = 0;
    for (const [x, z, a] of corners) {
      if (i >= decor.length) break;
      if (!clear(furniture, x, z, 0.22)) continue;
      decor[i]?.(x, z, a);
      i++;
    }
  }
  const group = m.build(materials);
  if (lamps) lamps.push(...m.lamps);
  return group;
}

/** One bin: its own group, so it can be lit up or dimmed during a round. */
export function buildBin(f: RoomFurniture, decal: THREE.Texture): { group: THREE.Group; body: THREE.MeshStandardMaterial; face: THREE.MeshBasicMaterial } {
  const [w, d] = f.size;
  const h = f.height;
  const m = new Mesher(f.id);
  const color = binColor(f.color);
  const wall = 0.018;
  for (const sx of [-1, 1]) m.box(wall, h, d, sx * (w / 2 - wall / 2), h / 2, 0, color, { vary: 0.015 });
  for (const sz of [-1, 1]) m.box(w - 2 * wall, h, wall, 0, h / 2, sz * (d / 2 - wall / 2), color, { vary: 0.015 });
  m.box(w - 2 * wall, 0.02, d - 2 * wall, 0, 0.01, 0, 0x15181c, { vary: 0 });
  m.box(w - 2 * wall, 0.01, d - 2 * wall, 0, h * 0.62, 0, 0x15181c, { vary: 0 }); // the dark inside
  m.rbox(w + 0.03, 0.035, d + 0.03, 0, h, 0, color, { vary: 0.03, r: 0.012, seg: 1 }); // the rim
  m.box(w - 2 * wall - 0.01, 0.012, d - 2 * wall - 0.01, 0, h, 0, 0x15181c, { vary: 0 });
  const body = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.05 });
  const group = m.build({ matte: body, gloss: body, glow: body });
  group.position.set(f.at[0], 0, f.at[1]);
  group.rotation.y = (-f.rot * Math.PI) / 180;
  const face = new THREE.MeshBasicMaterial({ map: decal, transparent: true, depthWrite: false });
  const front = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.72, w * 0.72), face);
  front.position.set(0, h * 0.5, d / 2 + 0.004);
  group.add(front);
  return { group, body, face };
}

/** A pendant light for the middle of the ceiling: a cord, a shade and the glowing bulb under it. */
export function buildFixture(width: number, depth: number, height: number, materials: Record<Layer, THREE.Material>): THREE.Group {
  const m = new Mesher("fixture");
  const drop = 0.42;
  m.at(width / 2, height, depth / 2, 0, () => {
    m.cyl(0.05, 0.05, 0.02, 0, -0.01, 0, C.trim, { vary: 0 });
    m.cyl(0.006, 0.006, drop, 0, -drop / 2, 0, C.charcoal, { seg: 6 });
    m.cyl(0.2, 0.09, 0.16, 0, -drop - 0.06, 0, C.charcoal, { open: true, seg: 20 });
    m.cyl(0.19, 0.085, 0.15, 0, -drop - 0.06, 0, 0xffe6c2, { layer: "glow", seg: 20 });
    m.ball(0.035, 0, -drop - 0.11, 0, 0xfff3dc, { layer: "glow" });
  });
  return m.build(materials);
}

// ---------- walls ----------

interface Hole {
  from: number;
  to: number;
  bottom: number;
  top: number;
}

/** Where a wall runs. `u` is metres along it from its west or north end; `out` is metres outwards from the room. */
export function wallPoint(wall: Wall, u: number, out: number, width: number, depth: number): [number, number] {
  if (wall === "north") return [u, -out];
  if (wall === "south") return [u, depth + out];
  if (wall === "west") return [-out, u];
  return [width + out, u];
}

export function wallLength(wall: Wall, width: number, depth: number): number {
  return wall === "north" || wall === "south" ? width : depth;
}

/** Turn that makes a thing built facing +z look into the room from this wall. */
export function wallTurn(wall: Wall): number {
  return wall === "north" ? 0 : wall === "south" ? Math.PI : wall === "west" ? Math.PI / 2 : -Math.PI / 2;
}

export interface WallParts {
  full: THREE.Group; // the wall with its openings, door, window and trim
  curb: THREE.Group; // what is left of it when it is cut away
  extras: THREE.Object3D[]; // glass, posters: hidden with the wall
}

/** `glass`, when given, is the material of the window panes (so the scene can darken the sky at night). */
export function buildWall(
  wall: Wall,
  width: number,
  depth: number,
  height: number,
  doors: RoomOpening[],
  windows: RoomOpening[],
  materials: Record<Layer, THREE.Material>,
  glassMaterial?: THREE.Material,
): WallParts {
  const length = wallLength(wall, width, depth);
  const along = wall === "north" || wall === "south";
  const m = new Mesher(`wall-${wall}`);
  // A slab of wall from u0..u1 and v0..v1 (height), `out` metres outside the room's face.
  const slab = (u0: number, u1: number, v0: number, v1: number, t: number, out: number, color: number, layer: Layer = "matte", vary = 0.012) => {
    if (u1 - u0 < 1e-4 || v1 - v0 < 1e-4) return;
    const [x, z] = wallPoint(wall, (u0 + u1) / 2, out, width, depth);
    m.box(along ? u1 - u0 : t, v1 - v0, along ? t : u1 - u0, x, (v0 + v1) / 2, z, color, { layer, vary });
  };
  const holes: Hole[] = [
    ...doors.filter((d) => d.wall === wall).map((d) => ({ from: d.at - d.width / 2, to: d.at + d.width / 2, bottom: 0, top: Math.min(DOOR_H, height - 0.1) })),
    ...windows.filter((w) => w.wall === wall).map((w) => ({ from: w.at - w.width / 2, to: w.at + w.width / 2, bottom: w.sill ?? 0.9, top: Math.min((w.sill ?? 0.9) + (w.height ?? 1.1), height - 0.1) })),
  ].sort((a, b) => a.from - b.from);

  let u = -WALL_T; // the corners are filled by the walls that run east-west
  const end = length + WALL_T;
  const lo = along ? u : 0;
  const hi = along ? end : length;
  u = lo;
  for (const hole of holes) {
    slab(u, hole.from, 0, height, WALL_T, WALL_T / 2, C.wall);
    slab(hole.from, hole.to, 0, hole.bottom, WALL_T, WALL_T / 2, C.wall);
    slab(hole.from, hole.to, hole.top, height, WALL_T, WALL_T / 2, C.wall);
    u = hole.to;
  }
  slab(u, hi, 0, height, WALL_T, WALL_T / 2, C.wall);

  // skirting board with a small cap, broken at the doors, and a thin picture rail higher up
  u = 0;
  for (const hole of holes.filter((h) => h.bottom === 0)) {
    slab(u, hole.from, 0, 0.1, 0.018, -0.009, C.trim, "matte", 0.005);
    slab(u, hole.from, 0.1, 0.115, 0.026, -0.013, C.trim, "matte", 0.005);
    u = hole.to;
  }
  slab(u, length, 0, 0.1, 0.018, -0.009, C.trim, "matte", 0.005);
  slab(u, length, 0.1, 0.115, 0.026, -0.013, C.trim, "matte", 0.005);
  slab(lo, hi, height - 0.07, height, 0.022, -0.011, C.trim, "matte", 0.005); // cornice

  const extras: THREE.Object3D[] = [];
  const glass = new Mesher(`glass-${wall}`);
  const turn = wallTurn(wall);
  for (const d of doors.filter((x) => x.wall === wall)) {
    const top = Math.min(DOOR_H, height - 0.1);
    const [x, z] = wallPoint(wall, d.at, WALL_T / 2, width, depth);
    m.at(x, 0, z, turn, () => {
      for (const sx of [-1, 1]) m.box(0.06, top + 0.03, WALL_T + 0.03, sx * (d.width / 2 + 0.01), (top + 0.03) / 2, 0, C.trim, { vary: 0.005 });
      m.box(d.width + 0.14, 0.06, WALL_T + 0.03, 0, top + 0.03, 0, C.trim, { vary: 0.005 });
      m.box(d.width + 0.02, 0.02, WALL_T + 0.02, 0, 0.01, 0, C.oakDark, { vary: 0.005 }); // the threshold
      // the leaf, a little ajar, hinged on the left as seen from inside
      m.at(-d.width / 2 + 0.02, 0, 0.03, -0.3, () => {
        const w = d.width - 0.05;
        m.rbox(w, top - 0.02, 0.04, w / 2, top / 2, 0, C.oak, { vary: 0.01, r: 0.006, seg: 1 });
        for (const y of [top * 0.72, top * 0.28]) m.box(w - 0.22, top * 0.36, 0.008, w / 2, y, 0.022, C.oakDark, { vary: 0.01 });
        m.rbox(0.12, 0.022, 0.05, w - 0.1, top * 0.48, 0.045, C.brass, { layer: "gloss", r: 0.008, seg: 1 });
        m.cyl(0.026, 0.026, 0.012, w - 0.06, top * 0.48, 0.026, C.brass, { layer: "gloss", rx: Math.PI / 2 });
        m.cyl(0.014, 0.014, 0.03, w - 0.06, top * 0.48 - 0.06, 0.026, C.steel, { layer: "gloss", rx: Math.PI / 2, seg: 8 }); // the lock
        for (const y of [top * 0.15, top * 0.85]) m.box(0.012, 0.09, 0.01, 0.01, y, -0.024, C.steel, { layer: "gloss" }); // hinges
      });
    });
  }
  for (const w of windows.filter((x) => x.wall === wall)) {
    const sill = w.sill ?? 0.9;
    const h = Math.min(w.height ?? 1.1, height - 0.1 - sill);
    const [x, z] = wallPoint(wall, w.at, WALL_T / 2, width, depth);
    const frame = () => {
      for (const sx of [-1, 1]) m.box(0.05, h, WALL_T + 0.02, sx * (w.width / 2 - 0.025), sill + h / 2, 0, C.trim, { vary: 0.005 });
      for (const y of [sill + 0.025, sill + h - 0.025]) m.box(w.width, 0.05, WALL_T + 0.02, 0, y, 0, C.trim, { vary: 0.005 });
      m.box(0.035, h, 0.04, 0, sill + h / 2, 0, C.trim, { vary: 0.005 }); // mullions
      m.box(w.width, 0.03, 0.04, 0, sill + h * 0.55, 0, C.trim, { vary: 0.005 });
      for (const sx of [-1, 1]) m.box(0.012, h * 0.55, 0.02, sx * w.width * 0.25, sill + h * 0.275, 0, C.trim, { vary: 0.005 }); // the lower glazing bars
      m.rbox(w.width + 0.12, 0.035, 0.16, 0, sill - 0.015, WALL_T / 2 + 0.05, C.trim, { vary: 0.005, r: 0.006, seg: 1 }); // the sill
      m.box(0.03, 0.05, 0.02, w.width * 0.25, sill + h * 0.3, WALL_T / 2 + 0.025, C.brass, { layer: "gloss" }); // the latch
      plant(m, -w.width * 0.3, sill + 0.003, WALL_T / 2 + 0.06, 0.4, C.white); // a plant on the sill
      // curtains, drawn back, on a rod
      m.cyl(0.012, 0.012, w.width + 0.5, 0, sill + h + 0.14, WALL_T / 2 + 0.06, C.brass, { rz: Math.PI / 2, layer: "gloss" });
      for (const sx of [-1, 1]) {
        m.ball(0.022, sx * (w.width / 2 + 0.25), sill + h + 0.14, WALL_T / 2 + 0.06, C.brass, { layer: "gloss" }); // finials
        for (let i = 0; i < 4; i++) m.rbox(0.07, h + 0.3, 0.05, sx * (w.width / 2 + 0.2 - i * 0.065), sill + h / 2 - 0.02, WALL_T / 2 + 0.05 + (i % 2) * 0.018, C.sage, { vary: 0.03, r: 0.02, seg: 1 });
      }
    };
    m.at(x, 0, z, turn, frame);
    glass.at(x, 0, z, turn, () => glass.panel(w.width - 0.08, h - 0.08, 0, sill + h / 2, -0.01, C.sky));
  }
  const full = m.build(materials);
  const pane = glass.build({ ...materials, glow: glassMaterial ?? materials.glow });
  full.add(pane);
  extras.push(pane);

  const c = new Mesher(`curb-${wall}`);
  const [cx, cz] = wallPoint(wall, length / 2, WALL_T / 2, width, depth);
  const span = along ? length + 2 * WALL_T : length;
  c.box(along ? span : WALL_T, 0.1, along ? WALL_T : span, cx, 0.05, cz, C.wallCut, { vary: 0.01 });
  const curb = c.build(materials);
  return { full, curb, extras };
}

// ---------- painted textures ----------

function canvasOf(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D | null] {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  return [canvas, canvas.getContext("2d")];
}

function textureOf(canvas: HTMLCanvasElement, repeat = false): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

/** Two metres square of wood floor: planks with staggered ends and a little grain. */
export function woodTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(512, 512);
  const rand = seeded("floor");
  if (ctx) {
    const planks = 14;
    const pw = 512 / planks;
    for (let i = 0; i < planks; i++) {
      let y = -rand() * 300;
      while (y < 512) {
        const len = 170 + rand() * 200;
        const l = 58 + rand() * 9;
        ctx.fillStyle = `hsl(${28 + rand() * 6}, ${34 + rand() * 8}%, ${l}%)`;
        ctx.fillRect(i * pw, y, pw, len);
        ctx.globalAlpha = 0.06;
        for (let g = 0; g < 5; g++) {
          ctx.fillStyle = rand() > 0.5 ? "#000" : "#fff";
          ctx.fillRect(i * pw + rand() * pw, y, 1 + rand() * 1.5, len);
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = "rgba(60,38,20,0.5)";
        ctx.fillRect(i * pw, y + len - 1.2, pw, 1.2);
        y += len;
      }
      ctx.fillStyle = "rgba(60,38,20,0.45)";
      ctx.fillRect(i * pw, 0, 1.2, 512);
    }
  }
  return textureOf(canvas, true);
}

export function rugTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(256, 384);
  const rand = seeded("rug");
  if (ctx) {
    ctx.fillStyle = "#e7dcc8";
    ctx.fillRect(0, 0, 256, 384);
    ctx.strokeStyle = "#b5624a";
    ctx.lineWidth = 12;
    ctx.strokeRect(12, 12, 232, 360);
    ctx.strokeStyle = "#3d5a73";
    ctx.lineWidth = 3;
    ctx.strokeRect(28, 28, 200, 328);
    ctx.strokeStyle = "#b5624a";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(36, 36, 184, 312);
    for (let y = 56; y < 340; y += 44) {
      ctx.fillStyle = y % 88 === 56 ? "#3d5a73" : "#b5624a";
      for (let x = 48; x < 216; x += 32) {
        ctx.beginPath();
        ctx.moveTo(x + 14, y);
        ctx.lineTo(x + 28, y + 14);
        ctx.lineTo(x + 14, y + 28);
        ctx.lineTo(x, y + 14);
        ctx.closePath();
        ctx.fill();
      }
    }
    // the fringe at both short ends
    ctx.strokeStyle = "#d9cdb5";
    ctx.lineWidth = 2;
    for (let x = 14; x < 246; x += 5) {
      for (const [y0, y1] of [
        [12, 2],
        [372, 382],
      ] as [number, number][]) {
        ctx.beginPath();
        ctx.moveTo(x, y0);
        ctx.lineTo(x + rand() * 2 - 1, y1);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 0.05;
    for (let i = 0; i < 900; i++) {
      ctx.fillStyle = rand() > 0.5 ? "#000" : "#fff";
      ctx.fillRect(rand() * 256, rand() * 384, 2, 1);
    }
  }
  return textureOf(canvas);
}

export function posterTexture(art: string | undefined): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(256, 256);
  if (ctx) {
    if (art === "pennant") {
      ctx.fillStyle = "#1d2a3a";
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = "#e0b23f";
      ctx.beginPath();
      ctx.moveTo(10, 30);
      ctx.lineTo(246, 128);
      ctx.lineTo(10, 226);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "#1d2a3a";
      ctx.font = "800 56px system-ui, sans-serif";
      ctx.fillText("GO", 28, 148);
    } else if (art === "grid") {
      ctx.fillStyle = "#f3efe6";
      ctx.fillRect(0, 0, 256, 256);
      const colors = ["#d7553f", "#2f5d8a", "#e0b23f", "#2c2f36", "#8fa58a"];
      for (let i = 0; i < 9; i++) {
        ctx.fillStyle = colors[(i * 2) % colors.length] as string;
        const x = 28 + (i % 3) * 70;
        const y = 28 + Math.floor(i / 3) * 70;
        if (i % 2) ctx.fillRect(x, y, 58, 58);
        else {
          ctx.beginPath();
          ctx.arc(x + 29, y + 29, 29, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    } else {
      const sky = ctx.createLinearGradient(0, 0, 0, 256);
      sky.addColorStop(0, "#2b3a67");
      sky.addColorStop(0.55, "#e8825a");
      sky.addColorStop(1, "#f3c56b");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = "#fbe3a4";
      ctx.beginPath();
      ctx.arc(128, 150, 44, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#1f2540";
      ctx.beginPath();
      ctx.moveTo(0, 256);
      ctx.lineTo(0, 190);
      ctx.lineTo(70, 140);
      ctx.lineTo(120, 185);
      ctx.lineTo(180, 128);
      ctx.lineTo(256, 196);
      ctx.lineTo(256, 256);
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, 248, 248);
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, 254, 254);
  }
  return textureOf(canvas);
}

/** The mark on the front of a bin: three chasing arrows for recycling, a plain lidded bin for the rest. */
export function binDecal(kind: string): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(128, 128);
  if (ctx) {
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (kind === "recycling") {
      ctx.lineWidth = 11;
      for (let i = 0; i < 3; i++) {
        const a0 = (i * 2 * Math.PI) / 3 - Math.PI / 2 + 0.38;
        const a1 = a0 + (2 * Math.PI) / 3 - 0.76;
        ctx.beginPath();
        ctx.arc(64, 66, 36, a0, a1);
        ctx.stroke();
        const hx = 64 + 36 * Math.cos(a1);
        const hy = 66 + 36 * Math.sin(a1);
        const dir = a1 + Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(hx + 17 * Math.cos(dir), hy + 17 * Math.sin(dir));
        ctx.lineTo(hx + 12 * Math.cos(dir + 2.3), hy + 12 * Math.sin(dir + 2.3));
        ctx.lineTo(hx + 12 * Math.cos(dir - 2.3), hy + 12 * Math.sin(dir - 2.3));
        ctx.closePath();
        ctx.fill();
      }
    } else {
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.moveTo(30, 42);
      ctx.lineTo(98, 42);
      ctx.moveTo(52, 42);
      ctx.lineTo(54, 30);
      ctx.lineTo(74, 30);
      ctx.lineTo(76, 42);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(38, 52);
      ctx.lineTo(44, 102);
      ctx.lineTo(84, 102);
      ctx.lineTo(90, 52);
      ctx.stroke();
      ctx.lineWidth = 6;
      for (const x of [54, 64, 74]) {
        ctx.beginPath();
        ctx.moveTo(x, 60);
        ctx.lineTo(x, 92);
        ctx.stroke();
      }
    }
  }
  return textureOf(canvas);
}

/** A soft round shadow. */
export function blobTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(64, 64);
  if (ctx) {
    const fade = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    fade.addColorStop(0, "rgba(0,0,0,0.55)");
    fade.addColorStop(0.45, "rgba(0,0,0,0.28)");
    fade.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 64, 64);
  }
  return new THREE.CanvasTexture(canvas);
}

/** A soft ring of light, bright at the ring and fading in and out: the halo under a pin. */
export function haloTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(128, 128);
  if (ctx) {
    const fade = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    fade.addColorStop(0, "rgba(255,255,255,0.35)");
    fade.addColorStop(0.42, "rgba(255,255,255,0.15)");
    fade.addColorStop(0.55, "rgba(255,255,255,1)");
    fade.addColorStop(0.66, "rgba(255,255,255,0.35)");
    fade.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 128, 128);
  }
  return new THREE.CanvasTexture(canvas);
}

/**
 * Contact shading for the whole floor, painted once: darker at the foot of every piece of furniture and wall, and a
 * soft vignette towards the corners. With `warmth` above zero, a warm pool of light is painted under each lamp (for
 * the night, when the lamps are what lights the floor).
 */
export function contactTexture(width: number, depth: number, furniture: RoomFurniture[], lamps: { x: number; z: number }[] = [], warmth = 0): THREE.CanvasTexture {
  const px = 72; // pixels a metre
  const [canvas, ctx] = canvasOf(Math.round(width * px), Math.round(depth * px));
  if (ctx) {
    const vignette = ctx.createRadialGradient(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) * 0.3, canvas.width / 2, canvas.height / 2, Math.hypot(canvas.width, canvas.height) * 0.55);
    vignette.addColorStop(0, "rgba(0,0,0,0)");
    vignette.addColorStop(1, "rgba(0,0,0,0.28)");
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000";
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 16;
    for (const f of furniture) {
      if (f.type === "rug") continue;
      ctx.save();
      ctx.translate(f.at[0] * px, f.at[1] * px);
      ctx.rotate((f.rot * Math.PI) / 180);
      ctx.globalAlpha = f.type === "chair" ? 0.35 : 0.5;
      ctx.fillRect((-f.size[0] / 2) * px, (-f.size[1] / 2) * px, f.size[0] * px, f.size[1] * px);
      ctx.restore();
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    const edge = (x0: number, y0: number, x1: number, y1: number) => {
      const g = ctx.createLinearGradient(x0, y0, x1, y1);
      g.addColorStop(0, "rgba(0,0,0,0.3)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };
    edge(0, 0, 0, 22);
    edge(0, canvas.height, 0, canvas.height - 22);
    edge(0, 0, 22, 0);
    edge(canvas.width, 0, canvas.width - 22, 0);
    if (warmth > 0) {
      ctx.globalCompositeOperation = "destination-out"; // a lamp lifts the shading under it
      for (const l of lamps) {
        const r = 1.1 * px;
        const g = ctx.createRadialGradient(l.x * px, l.z * px, 0, l.x * px, l.z * px, r);
        g.addColorStop(0, `rgba(0,0,0,${warmth})`);
        g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.fillRect(l.x * px - r, l.z * px - r, r * 2, r * 2);
      }
      ctx.globalCompositeOperation = "source-over";
    }
  }
  return new THREE.CanvasTexture(canvas);
}

/** The whole floor at night: a warm pool of light under each lamp, nothing elsewhere. Drawn additively. */
export function poolTexture(width: number, depth: number, lamps: { x: number; y: number; z: number }[]): THREE.CanvasTexture {
  const px = 48;
  const [canvas, ctx] = canvasOf(Math.round(width * px), Math.round(depth * px));
  if (ctx) {
    for (const l of lamps) {
      const r = (0.9 + l.y * 0.5) * px; // a tall lamp lights a wider patch
      const g = ctx.createRadialGradient(l.x * px, l.z * px, 0, l.x * px, l.z * px, r);
      g.addColorStop(0, "rgba(255,170,90,0.75)");
      g.addColorStop(0.3, "rgba(255,160,80,0.38)");
      g.addColorStop(1, "rgba(255,150,70,0)");
      ctx.fillStyle = g;
      ctx.fillRect(l.x * px - r, l.z * px - r, r * 2, r * 2);
    }
  }
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Arrow heads marching along a route. Tiles along u. */
export function chevronTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(64, 32);
  if (ctx) {
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillRect(0, 0, 64, 32);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(14, 2);
    ctx.lineTo(36, 16);
    ctx.lineTo(14, 30);
    ctx.lineTo(26, 16);
    ctx.closePath();
    ctx.fill();
  }
  const t = new THREE.CanvasTexture(canvas);
  t.wrapS = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Bright in the middle, fading to nothing at both edges (across v): the glow under a route. */
export function glowTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(4, 32);
  if (ctx) {
    const g = ctx.createLinearGradient(0, 0, 0, 32);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(0.5, "rgba(255,255,255,1)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 4, 32);
  }
  return new THREE.CanvasTexture(canvas);
}

/** The window's patch of sun on the floor: soft-edged, with the shadow of the frame's cross. */
export function sunTexture(): THREE.CanvasTexture {
  const [canvas, ctx] = canvasOf(128, 128);
  if (ctx) {
    ctx.filter = "blur(5px)";
    ctx.fillStyle = "#fff";
    ctx.fillRect(12, 12, 104, 104);
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(60, 0, 8, 128);
    ctx.fillRect(0, 66, 128, 7);
  }
  return new THREE.CanvasTexture(canvas);
}

// ---------- people and things ----------

function standard(color: number, roughness = 0.7, metalness = 0): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

/** Collects the pieces of a figure or a prop and merges them into one mesh a material, so each is a few draw calls. */
class Props {
  private byMaterial = new Map<THREE.Material, THREE.BufferGeometry[]>();

  add(g: THREE.BufferGeometry, material: THREE.Material, x: number, y: number, z: number, o: { rx?: number; ry?: number; rz?: number } = {}): void {
    const n = g.index ? g.toNonIndexed() : g;
    if (n !== g) g.dispose();
    if (o.rx) n.rotateX(o.rx);
    if (o.ry) n.rotateY(o.ry);
    if (o.rz) n.rotateZ(o.rz);
    n.translate(x, y, z);
    n.deleteAttribute("uv");
    let list = this.byMaterial.get(material);
    if (!list) this.byMaterial.set(material, (list = []));
    list.push(n);
  }

  into(parent: THREE.Object3D, shadows = true): void {
    for (const [material, parts] of this.byMaterial) {
      const first = parts[0] as THREE.BufferGeometry;
      const all = parts.length === 1 ? first : mergeGeometries(parts, false);
      if (all !== first) for (const p of parts) p.dispose();
      if (!all) continue;
      const mesh = new THREE.Mesh(all, material);
      mesh.castShadow = shadows;
      parent.add(mesh);
    }
    this.byMaterial.clear();
  }
}

/**
 * A figure 1.7 m tall, facing +z: two legs with shoes, a torso with shoulders and a neck, a two-tone shirt (the
 * sleeves are a shade darker than the body, and `sleeve` follows `shirt` when it is recoloured). Six meshes, one a
 * material. `seated` shortens it to chair height.
 */
export function buildPerson(color: number): { group: THREE.Group; shirt: THREE.MeshStandardMaterial; sleeve: THREE.MeshStandardMaterial; body: THREE.Group } {
  const group = new THREE.Group();
  const body = new THREE.Group();
  const shirt = standard(color, 0.72);
  const sleeve = standard(color, 0.78);
  sleeve.color.offsetHSL(0, 0.02, -0.09);
  const legs = standard(0x39424e, 0.85);
  const shoe = standard(0x23262b, 0.6);
  const skin = standard(C.skin, 0.8);
  const hair = standard(0x3b2a20, 0.9);
  const p = new Props();
  for (const sx of [-1, 1]) {
    p.add(new THREE.CylinderGeometry(0.075, 0.062, 0.72, 10), legs, sx * 0.085, 0.4, 0);
    p.add(new RoundedBoxGeometry(0.11, 0.07, 0.24, 1, 0.03), shoe, sx * 0.085, 0.035, 0.03);
    p.add(new THREE.CapsuleGeometry(0.052, 0.4, 3, 8), sleeve, sx * 0.235, 1.04, 0.02, { rz: sx * 0.1 });
    p.add(new THREE.SphereGeometry(0.05, 10, 8), skin, sx * 0.26, 0.78, 0.03); // hands
  }
  p.add(new THREE.CapsuleGeometry(0.18, 0.36, 4, 12), shirt, 0, 1.08, 0);
  p.add(new RoundedBoxGeometry(0.46, 0.13, 0.22, 2, 0.06), sleeve, 0, 1.36, 0); // shoulders
  p.add(new THREE.CylinderGeometry(0.05, 0.055, 0.1, 10), skin, 0, 1.44, 0); // neck
  p.add(new THREE.SphereGeometry(0.125, 14, 10), skin, 0, 1.57, 0);
  p.add(new THREE.BoxGeometry(0.03, 0.03, 0.03), skin, 0, 1.56, 0.125); // a nose: which way they face
  p.add(new THREE.SphereGeometry(0.13, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.5), hair, 0, 1.585, -0.012);
  p.into(body);
  group.add(body);
  return { group, shirt, sleeve, body };
}

const SMALL = 1.9; // small things are drawn larger than life so a phone can show them

/** A recognisable prop for a tracked label, standing on its base at the origin. A few meshes, one a material. */
export function buildThing(label: string): THREE.Group {
  const g = new THREE.Group();
  const name = label.toLowerCase();
  const p = new Props();
  if (name.includes("package") || name.includes("box") || name === "cardboard") {
    const tape = standard(0xe9dcc2, 0.6);
    const ink = standard(0x2c2f36, 0.8);
    p.add(new RoundedBoxGeometry(0.42, 0.28, 0.32, 1, 0.012), standard(C.cardboard, 0.9), 0, 0.14, 0);
    p.add(new THREE.BoxGeometry(0.43, 0.005, 0.07), tape, 0, 0.283, 0);
    p.add(new THREE.BoxGeometry(0.07, 0.29, 0.005), tape, 0, 0.14, 0.163); // the tape down the front
    p.add(new THREE.BoxGeometry(0.16, 0.1, 0.005), standard(0xf4f1ea, 0.8), 0.1, 0.17, 0.163); // the label
    p.add(new THREE.BoxGeometry(0.12, 0.012, 0.004), ink, 0.1, 0.19, 0.166);
    p.add(new THREE.BoxGeometry(0.08, 0.012, 0.004), ink, 0.08, 0.16, 0.166);
    p.into(g);
    return g;
  }
  if (name.includes("laptop")) {
    const shell = standard(0xb9bec6, 0.35, 0.55);
    p.add(new RoundedBoxGeometry(0.32, 0.018, 0.22, 1, 0.006), shell, 0, 0.009, 0);
    p.add(new THREE.BoxGeometry(0.26, 0.003, 0.1), standard(0x4a4f57, 0.6), 0, 0.019, 0.02);
    p.add(new RoundedBoxGeometry(0.32, 0.21, 0.012, 1, 0.006), shell, 0, 0.115, -0.125, { rx: -0.26 });
    p.into(g);
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.29, 0.18), new THREE.MeshBasicMaterial({ color: C.screen }));
    screen.position.set(0, 0.116, -0.1175);
    screen.rotation.x = -0.26;
    g.add(screen);
    return g;
  }
  const s = new THREE.Group();
  s.scale.setScalar(SMALL);
  g.add(s);
  if (name.includes("bottle")) {
    const plastic = new THREE.MeshStandardMaterial({ color: 0x8fd0f0, roughness: 0.2, metalness: 0.05, transparent: true, opacity: 0.78 });
    const water = new THREE.MeshStandardMaterial({ color: 0x6fb8e0, roughness: 0.15, transparent: true, opacity: 0.6 });
    p.add(new THREE.CylinderGeometry(0.034, 0.034, 0.15, 14), plastic, 0, 0.075, 0);
    p.add(new THREE.CylinderGeometry(0.014, 0.034, 0.04, 14), plastic, 0, 0.17, 0);
    p.add(new THREE.CylinderGeometry(0.03, 0.03, 0.09, 12), water, 0, 0.046, 0);
    p.add(new THREE.CylinderGeometry(0.016, 0.016, 0.022, 12), standard(0x2f7ed8, 0.4), 0, 0.2, 0);
    p.add(new THREE.BoxGeometry(0.07, 0.05, 0.07), standard(0xf0f4f8, 0.6), 0, 0.09, 0); // the label
  } else if (name.includes("can")) {
    const tin = standard(0xd0d4d9, 0.3, 0.85);
    p.add(new THREE.CylinderGeometry(0.033, 0.033, 0.115, 16), standard(0xd8433a, 0.35, 0.5), 0, 0.0575, 0);
    p.add(new THREE.CylinderGeometry(0.03, 0.033, 0.008, 16), tin, 0, 0.119, 0);
    p.add(new THREE.CylinderGeometry(0.033, 0.03, 0.008, 16), tin, 0, 0.004, 0);
    p.add(new THREE.BoxGeometry(0.018, 0.002, 0.01), tin, 0, 0.124, 0); // the tab
    p.add(new THREE.BoxGeometry(0.068, 0.03, 0.068), standard(0xf4f1ea, 0.4, 0.3), 0, 0.06, 0); // a band of label
  } else if (name.includes("cup") || name.includes("glass")) {
    const lid = standard(0x3a3f47, 0.5);
    p.add(new THREE.CylinderGeometry(0.043, 0.03, 0.11, 16), standard(0xf4f1ea, 0.55), 0, 0.055, 0);
    p.add(new THREE.CylinderGeometry(0.046, 0.046, 0.012, 16), lid, 0, 0.116, 0);
    p.add(new THREE.CylinderGeometry(0.012, 0.012, 0.006, 10), lid, 0, 0.125, 0); // the sip hole's rim
    p.add(new THREE.CylinderGeometry(0.039, 0.035, 0.035, 16), standard(0x8a5a3c, 0.8), 0, 0.055, 0); // the sleeve
  } else if (name.includes("phone")) {
    const black = standard(0x1d2025, 0.3, 0.5);
    p.add(new RoundedBoxGeometry(0.075, 0.01, 0.155, 1, 0.006), black, 0, 0.005, 0);
    p.add(new RoundedBoxGeometry(0.022, 0.003, 0.03, 1, 0.004), standard(0x3a3f47, 0.4, 0.5), 0.022, -0.0005, -0.055); // the camera bump
    p.into(s);
    const screen = new THREE.Mesh(new THREE.BoxGeometry(0.066, 0.002, 0.145), new THREE.MeshBasicMaterial({ color: 0x6fb3ff }));
    screen.position.set(0, 0.011, 0);
    s.add(screen);
  } else if (name.includes("key")) {
    const brass = standard(0xd9b04a, 0.3, 0.8);
    p.add(new THREE.TorusGeometry(0.026, 0.005, 6, 16), brass, 0, 0.006, 0, { rx: Math.PI / 2 });
    for (const [a, len] of [
      [0.5, 0.07],
      [1.5, 0.06],
      [-0.4, 0.05],
    ] as [number, number][]) {
      p.add(new THREE.BoxGeometry(0.014, 0.004, len), brass, Math.sin(a) * (0.026 + len / 2), 0.004, Math.cos(a) * (0.026 + len / 2), { ry: a });
      p.add(new THREE.CylinderGeometry(0.014, 0.014, 0.004, 10), brass, Math.sin(a) * 0.03, 0.004, Math.cos(a) * 0.03);
      for (let i = 0; i < 3; i++) p.add(new THREE.BoxGeometry(0.006, 0.004, 0.006), brass, Math.sin(a) * (0.03 + len * (0.5 + i * 0.15)) + Math.cos(a) * 0.009, 0.004, Math.cos(a) * (0.03 + len * (0.5 + i * 0.15)) - Math.sin(a) * 0.009); // teeth
    }
    p.add(new RoundedBoxGeometry(0.03, 0.01, 0.045, 1, 0.005), standard(0xd8433a, 0.55), -0.045, 0.006, -0.01); // a fob
    p.add(new THREE.BoxGeometry(0.012, 0.003, 0.012), standard(0x1d2025, 0.4), -0.045, 0.012, -0.005); // its button
  } else if (name.includes("wallet")) {
    p.add(new RoundedBoxGeometry(0.11, 0.022, 0.09, 1, 0.006), standard(0x5a3d2b, 0.75), 0, 0.011, 0);
    p.add(new THREE.BoxGeometry(0.1, 0.002, 0.08), standard(0x7a5a44, 0.7), 0, 0.023, 0); // the stitched flap
    p.add(new THREE.BoxGeometry(0.05, 0.004, 0.03), standard(0x2f7ed8, 0.3, 0.2), 0.02, 0.011, 0.047); // a card peeking out
  } else if (name.includes("charger")) {
    const white = standard(0xf0eeea, 0.45);
    const metal = standard(0xb9bec6, 0.3, 0.8);
    p.add(new RoundedBoxGeometry(0.05, 0.03, 0.05, 1, 0.008), white, 0, 0.015, 0);
    for (const dx of [-0.008, 0.008]) p.add(new THREE.BoxGeometry(0.004, 0.012, 0.002), metal, dx, 0.015, 0.026); // the prongs
    const lead = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0.02, -0.025), new THREE.Vector3(0.05, 0.006, -0.06), new THREE.Vector3(0.11, 0.004, -0.02), new THREE.Vector3(0.14, 0.004, 0.03)]);
    p.add(new THREE.TubeGeometry(lead, 10, 0.004, 5, false), white, 0, 0, 0);
    p.add(new RoundedBoxGeometry(0.012, 0.006, 0.02, 1, 0.002), metal, 0.14, 0.004, 0.04); // the plug
  } else if (name.includes("backpack") || name.includes("bag")) {
    s.scale.setScalar(1);
    const cloth = standard(0x39506b, 0.85);
    const trim = standard(0x2c3d52, 0.85);
    p.add(new THREE.CapsuleGeometry(0.15, 0.2, 4, 10), cloth, 0, 0.25, 0);
    p.add(new RoundedBoxGeometry(0.2, 0.14, 0.07, 1, 0.025), trim, 0, 0.18, 0.15);
    p.add(new THREE.TorusGeometry(0.04, 0.008, 5, 10, Math.PI), trim, 0, 0.44, -0.02); // the carry loop
    for (const sx of [-1, 1]) p.add(new RoundedBoxGeometry(0.05, 0.3, 0.02, 1, 0.008), trim, sx * 0.07, 0.24, -0.15); // straps
    p.add(new THREE.BoxGeometry(0.18, 0.01, 0.02), standard(0xd0d4d9, 0.5), 0, 0.3, 0.19); // the zip
  } else {
    p.add(new THREE.SphereGeometry(0.045, 12, 9), standard(0xd7dbe0, 0.6), 0, 0.045, 0);
  }
  p.into(s);
  return g;
}
