// GENERATED from web/components/home/deviceModel.ts on main (the website's Atlas device, by Ilan and Xander), compiled
// to JavaScript with esbuild and its three/examples imports pointed at three/addons. Do not edit; re-copy it if the
// website model changes. The film adapts it in device.js.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
const DEVICE_PARTS = [
  { id: "lid", name: "The lid", blurb: "Lift it and the whole machine is there. The mark is pressed into the white, and a thin line of light along the front edge says Atlas is listening." },
  { id: "webcam", name: "One eye on the desk", blurb: "A small webcam on the top. It watches the desk, not the room, and nothing it sees leaves the box." },
  { id: "base", name: "The base", blurb: "One white block, 18 by 11 centimetres, no screen and nothing to tap. It sits on the desk and listens." },
  { id: "board", name: "Carrier board", blurb: "Power, four USB ports, Ethernet and HDMI, for the day the box needs a screen. It never has so far." },
  { id: "speaker", name: "Speaker", blurb: "Atlas talks back. One full-range driver behind the grille, loud enough for a room and quiet enough for a desk." },
  { id: "module", name: "Jetson Orin Nano", blurb: "Runs the eyes and the model, on the desk, with nothing leaving the box." },
  { id: "ssd", name: "A year of memory", blurb: "A terabyte of solid state, stored where you are and nowhere else. Ask what happened; it is all here." },
  { id: "heatsink", name: "Heatsink and fan", blurb: "A black finned block and a fan that turns slowly in the day and barely at all at night. You will not hear it think." }
];
const W = 2;
const D = 1.222;
const H = 1;
const LID_H = 0.2;
const TRAY_H = H - LID_H;
const WALL = 0.06;
const FLOOR = 0.04;
const FOOT = 0.03;
const CORNER = 0.022;
const BEVEL = 0.012;
const BOARD_Y = 0.11;
const MODULE = { x: -0.2, z: -0.12, w: 0.87, d: 0.56 };
const SSD = { x: 0.62, z: -0.05, w: 0.26, l: 0.9 };
const WEBCAM = { x: 0.35, z: 0.3, w: 0.67, h: 0.11, d: 0.22 };
const CSI = new THREE.Vector3(WEBCAM.x, BOARD_Y + 0.05, 0.42);
const SPEAKER = { y: 0.42, r: 0.23 };
const GRILLE = { w: 1.5, h: 0.52 };
const LID_PIVOT = new THREE.Vector3(0, TRAY_H + LID_H / 2, 0);
const ANCHORS = [
  { part: "lid", group: "lid", local: new THREE.Vector3(-0.98, 0, 0.3), threshold: 0.12, side: "left" },
  { part: "webcam", group: "webcam", local: new THREE.Vector3(WEBCAM.x + WEBCAM.w / 2, H + WEBCAM.h / 2, WEBCAM.z).sub(LID_PIVOT), threshold: 0.3, side: "right" },
  { part: "base", group: "base", local: new THREE.Vector3(-W / 2 + 0.02, 0.3, 0.3), threshold: 0.36, side: "left" },
  { part: "board", group: "board", local: new THREE.Vector3(-0.86, BOARD_Y, 0.25), threshold: 0.45, side: "left" },
  { part: "speaker", group: "speaker", local: new THREE.Vector3(SPEAKER.r, SPEAKER.y, D / 2 - 0.05), threshold: 0.55, side: "right" },
  { part: "module", group: "module", local: new THREE.Vector3(MODULE.x - MODULE.w / 2, 0.2, MODULE.z + 0.1), threshold: 0.62, side: "left" },
  { part: "ssd", group: "ssd", local: new THREE.Vector3(SSD.x + SSD.w / 2, 0.17, SSD.z + 0.1), threshold: 0.7, side: "right" },
  { part: "heatsink", group: "heatsink", local: new THREE.Vector3(MODULE.x - 0.42, 0.37, MODULE.z), threshold: 0.8, side: "left" }
];
const TRAVEL = {
  base: { from: 0, to: 1, move: new THREE.Vector3(0, 0, 0) },
  lid: { from: 0, to: 0.42, move: new THREE.Vector3(0, 2.35, 0) },
  webcam: { from: 0.18, to: 0.5, move: new THREE.Vector3(0, 0.34, 0) },
  // off the lid, which is already rising
  board: { from: 0.14, to: 0.56, move: new THREE.Vector3(0, 1, 0) },
  speaker: { from: 0.32, to: 0.72, move: new THREE.Vector3(0, 0, 0.85) },
  // forward out of the front of the base
  module: { from: 0.3, to: 0.68, move: new THREE.Vector3(0, 0.4, 0) },
  ssd: { from: 0.44, to: 0.82, move: new THREE.Vector3(0.62, 0, 0) },
  heatsink: { from: 0.42, to: 0.8, move: new THREE.Vector3(0, 0.34, 0) },
  fan: { from: 0.54, to: 0.92, move: new THREE.Vector3(0, 0.3, 0) }
};
const UNLIT_KEYS = /* @__PURE__ */ new Set(["glass", "spot", "bar", "logo", "lens"]);
function makeTextures() {
  const traces = new THREE.CanvasTexture(tracesCanvasOf());
  traces.colorSpace = THREE.SRGBColorSpace;
  traces.anisotropy = 8;
  const holes = grilleCanvasOf();
  const grille = new THREE.CanvasTexture(holes);
  grille.colorSpace = THREE.SRGBColorSpace;
  grille.anisotropy = 8;
  const grilleBump = new THREE.CanvasTexture(holes);
  grilleBump.anisotropy = 8;
  return { traces, grille, grilleBump, blob: blobTexture() };
}
function makeMaterials(tex) {
  return {
    // A matte white polymer: soft, barely reflective, with a touch of clearcoat so the edges catch the light.
    white: new THREE.MeshPhysicalMaterial({ color: 15987439, roughness: 0.55, metalness: 0.02, clearcoat: 0.15, clearcoatRoughness: 0.5, envMapIntensity: 0.9 }),
    seam: new THREE.MeshStandardMaterial({ color: 9080210, roughness: 0.6, metalness: 0.05 }),
    // The grille: the same white with a field of dark holes painted on, and the holes dimpled into it.
    grille: new THREE.MeshStandardMaterial({ map: tex.grille, bumpMap: tex.grilleBump, bumpScale: 4e-3, roughness: 0.55, metalness: 0.02 }),
    fin: new THREE.MeshStandardMaterial({ color: 1908515, metalness: 0.6, roughness: 0.5, envMapIntensity: 0.8 }),
    black: new THREE.MeshStandardMaterial({ color: 1711136, roughness: 0.62, metalness: 0.15 }),
    pcb: new THREE.MeshStandardMaterial({ map: tex.traces, roughness: 0.7, metalness: 0.1 }),
    chip: new THREE.MeshStandardMaterial({ color: 790034, roughness: 0.32, metalness: 0.25, envMapIntensity: 0.9 }),
    steel: new THREE.MeshStandardMaterial({ color: 12830668, metalness: 0.95, roughness: 0.28, envMapIntensity: 1.2 }),
    gold: new THREE.MeshStandardMaterial({ color: 14265679, metalness: 1, roughness: 0.3 }),
    rubber: new THREE.MeshStandardMaterial({ color: 12106429, roughness: 0.92, metalness: 0 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 3820128, metalness: 0, roughness: 0.05, transparent: true, opacity: 0.72, envMapIntensity: 2.6, clearcoat: 1, clearcoatRoughness: 0.05, depthWrite: false }),
    lens: new THREE.MeshStandardMaterial({ color: 461328, metalness: 0.5, roughness: 0.12, envMapIntensity: 1.6 }),
    bar: new THREE.MeshStandardMaterial({ color: 10206450, emissive: 10206450, emissiveIntensity: 1.2, roughness: 0.4, metalness: 0 }),
    logo: new THREE.MeshPhysicalMaterial({ color: 15987439, roughness: 0.55, metalness: 0.02, clearcoat: 0.15, clearcoatRoughness: 0.5, emissive: 10206450, emissiveIntensity: 0.15 }),
    spot: new THREE.MeshBasicMaterial({ color: 16777215 })
  };
}
function frame(x, y, z, o = {}) {
  return new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(o.rx ?? 0, o.ry ?? 0, o.rz ?? 0)).setPosition(x, y, z);
}
class Collector {
  bins = /* @__PURE__ */ new Map();
  add(geometry, key, local) {
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    if (g !== geometry) geometry.dispose();
    if (local) g.applyMatrix4(local);
    if (!g.getAttribute("uv")) g.setAttribute("uv", new THREE.Float32BufferAttribute(new Float32Array(g.getAttribute("position").count * 2), 2));
    const bin = this.bins.get(key);
    if (bin) bin.push(g);
    else this.bins.set(key, [g]);
  }
  box(w, h, d, x, y, z, key, o = {}) {
    this.add(new THREE.BoxGeometry(w, h, d), key, frame(x, y, z, o));
  }
  rbox(w, h, d, x, y, z, key, r, seg = 3, o = {}) {
    this.add(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, w / 2, h / 2, d / 2)), key, frame(x, y, z, o));
  }
  /** A cylinder along y (or along z or x), by the centre of its axis. */
  cyl(rTop, rBottom, h, x, y, z, key, o = {}) {
    const turn = { ...o };
    if (o.axis === "z") turn.rx = (o.rx ?? 0) + Math.PI / 2;
    if (o.axis === "x") turn.rz = (o.rz ?? 0) + Math.PI / 2;
    this.add(new THREE.CylinderGeometry(rTop, rBottom, h, o.seg ?? 24, 1, o.open ?? false), key, frame(x, y, z, turn));
  }
  /** A ring whose axis is y (or z), by its centre. */
  torus(r, tube, x, y, z, key, o = {}) {
    const turn = o.axis === "z" ? {} : { rx: Math.PI / 2 };
    this.add(new THREE.TorusGeometry(r, tube, 8, o.seg ?? 40), key, frame(x, y, z, turn));
  }
  ball(r, x, y, z, key, seg = 16) {
    this.add(new THREE.SphereGeometry(r, seg, Math.max(8, seg >> 1)), key, frame(x, y, z));
  }
  /** The part's meshes, one per material, in a group. Transparent and glowing things cast no shadow. */
  build(materials) {
    const group = new THREE.Group();
    for (const [key, parts] of this.bins) {
      const merged = mergeGeometries(parts, false);
      for (const p of parts) p.dispose();
      if (!merged) continue;
      const mesh = new THREE.Mesh(merged, materials[key]);
      const lit = !UNLIT_KEYS.has(key);
      mesh.castShadow = lit;
      mesh.receiveShadow = lit;
      if (key === "glass") mesh.renderOrder = 10;
      group.add(mesh);
    }
    this.bins.clear();
    return group;
  }
}
function roundedRect(w, h, r, cx = 0, cy = 0) {
  const s = new THREE.Shape();
  const x = w / 2;
  const y = h / 2;
  s.moveTo(cx - x + r, cy - y);
  s.lineTo(cx + x - r, cy - y);
  s.absarc(cx + x - r, cy - y + r, r, -Math.PI / 2, 0, false);
  s.lineTo(cx + x, cy + y - r);
  s.absarc(cx + x - r, cy + y - r, r, 0, Math.PI / 2, false);
  s.lineTo(cx - x + r, cy + y);
  s.absarc(cx - x + r, cy + y - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(cx - x, cy - y + r);
  s.absarc(cx - x + r, cy - y + r, r, Math.PI, Math.PI * 1.5, false);
  return s;
}
function slab(c, key, w, d, height, holes = [], y = 0) {
  const outline = roundedRect(w - 2 * BEVEL, d - 2 * BEVEL, CORNER - BEVEL);
  outline.holes.push(...holes);
  const g = new THREE.ExtrudeGeometry(outline, { depth: height - 2 * BEVEL, bevelEnabled: true, bevelThickness: BEVEL, bevelSize: BEVEL, bevelSegments: 2, curveSegments: 6 });
  c.add(g, key, new THREE.Matrix4().makeRotationX(-Math.PI / 2).premultiply(new THREE.Matrix4().makeTranslation(0, y + BEVEL, 0)));
}
function meet(a, da, b, db) {
  const den = da.x * db.y - da.y * db.x;
  const t = ((b.x - a.x) * db.y - (b.y - a.y) * db.x) / (den || 1e-9);
  return new THREE.Vector2(a.x + da.x * t, a.y + da.y * t);
}
function logoShapes() {
  const ring = new THREE.Shape();
  ring.absarc(50, 50, 47, 0, Math.PI * 2, false);
  const hole = new THREE.Path();
  hole.absarc(50, 50, 41, 0, Math.PI * 2, true);
  ring.holes.push(hole);
  const dot = new THREE.Shape();
  dot.absarc(50, 59, 5.5, 0, Math.PI * 2, false);
  const h = 3.5;
  const L = new THREE.Vector2(30, 73);
  const P = new THREE.Vector2(50, 27);
  const R = new THREE.Vector2(70, 73);
  const d1 = P.clone().sub(L).normalize();
  const d2 = R.clone().sub(P).normalize();
  const out1 = new THREE.Vector2(d1.y, -d1.x);
  const out2 = new THREE.Vector2(d2.y, -d2.x);
  const ang = (v) => Math.atan2(v.y, v.x);
  const a = new THREE.Shape();
  const p0 = L.clone().addScaledVector(out1, h);
  a.moveTo(p0.x, p0.y);
  const p1 = P.clone().addScaledVector(out1, h);
  a.lineTo(p1.x, p1.y);
  a.absarc(P.x, P.y, h, ang(out1), ang(out2), false);
  const p2 = R.clone().addScaledVector(out2, h);
  a.lineTo(p2.x, p2.y);
  a.absarc(R.x, R.y, h, ang(out2), ang(out2) + Math.PI, false);
  const inner2 = R.clone().addScaledVector(out2, -h);
  const inner1 = L.clone().addScaledVector(out1, -h);
  const q = meet(inner1, d1, inner2, d2);
  a.lineTo(q.x, q.y);
  a.lineTo(inner1.x, inner1.y);
  a.absarc(L.x, L.y, h, ang(out1) - Math.PI, ang(out1), false);
  return [ring, a, dot];
}
function logoRelief(c, cx, top, cz, size) {
  const depth = 5e-3;
  const bevel = 15e-4;
  const scale = size / 100;
  const geometries = logoShapes().map((shape) => new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: 12e-4, bevelSegments: 2, curveSegments: 40 }));
  const m = new THREE.Matrix4().makeRotationX(Math.PI / 2).premultiply(new THREE.Matrix4().makeTranslation(cx, top + depth + bevel, cz));
  const centre = new THREE.Matrix4().makeTranslation(-50, -50, 0).premultiply(new THREE.Matrix4().makeScale(scale, scale, 1));
  for (const g of geometries) c.add(g, "logo", m.clone().multiply(centre));
}
function buildBase(c) {
  const cavity = roundedRect(W - 2 * WALL + 2 * BEVEL, D - 2 * WALL + 2 * BEVEL, CORNER + BEVEL);
  slab(c, "white", W, D, TRAY_H, [cavity]);
  c.box(W - 2 * WALL + 2 * BEVEL, FLOOR, D - 2 * WALL + 2 * BEVEL, 0, FLOOR / 2, 0, "seam");
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) c.cyl(0.08, 0.075, FOOT, sx * (W / 2 - 0.26), -FOOT / 2, sz * (D / 2 - 0.24), "rubber", { seg: 20 });
  const seam = roundedRect(W - 0.012, D - 0.012, CORNER);
  seam.holes.push(roundedRect(W - 2 * WALL + 0.02, D - 2 * WALL + 0.02, CORNER));
  c.add(new THREE.ExtrudeGeometry(seam, { depth: 0.016, bevelEnabled: false, curveSegments: 6 }), "seam", new THREE.Matrix4().makeRotationX(-Math.PI / 2).premultiply(new THREE.Matrix4().makeTranslation(0, TRAY_H - 8e-3, 0)));
  c.box(GRILLE.w + 0.05, GRILLE.h + 0.05, 4e-3, 0, SPEAKER.y, D / 2 + 1e-3, "seam");
  c.box(GRILLE.w, GRILLE.h, 6e-3, 0, SPEAKER.y, D / 2 + 3e-3, "grille");
  c.box(GRILLE.w - 0.02, GRILLE.h - 0.02, 0.01, 0, SPEAKER.y, D / 2 - WALL - 0.01, "black");
}
function buildSpeaker(c) {
  const y = SPEAKER.y;
  const r = SPEAKER.r;
  const front = D / 2 - WALL - 0.02;
  c.cyl(r + 0.01, r + 0.01, 0.03, 0, y, front - 0.015, "black", { axis: "z", seg: 48 });
  c.torus(r - 0.03, 0.022, 0, y, front - 0.02, "black", { axis: "z", seg: 48 });
  c.cyl(r - 0.05, 0.07, 0.11, 0, y, front - 0.075, "black", { axis: "z", seg: 48, open: true });
  c.ball(0.065, 0, y, front - 0.12, "black", 20);
  for (let i = 0; i < 6; i++) {
    const a = i / 6 * Math.PI * 2;
    c.box(0.025, 0.02, 0.16, Math.cos(a) * (r - 0.06), y + Math.sin(a) * (r - 0.06), front - 0.1, "black", { rz: a });
  }
  c.cyl(0.14, 0.14, 0.12, 0, y, front - 0.2, "steel", { axis: "z", seg: 36 });
  c.cyl(0.07, 0.07, 0.04, 0, y, front - 0.28, "steel", { axis: "z", seg: 24 });
}
function buildLid(c) {
  const slot = roundedRect(0.24 + 2 * BEVEL, 0.06 + 2 * BEVEL, 0.02 + BEVEL, WEBCAM.x, -WEBCAM.z);
  slab(c, "white", W, D, LID_H, [slot], TRAY_H);
  logoRelief(c, -0.5, H, 0.02, 0.42);
  c.box(1.6, 0.011, 8e-3, 0, TRAY_H + LID_H / 2, D / 2 + 4e-3, "bar");
}
function buildWebcam(c) {
  const x = WEBCAM.x;
  const z = WEBCAM.z;
  const top = H + WEBCAM.h;
  c.rbox(WEBCAM.w, WEBCAM.h, WEBCAM.d, x, H + WEBCAM.h / 2, z, "black", 0.045, 3);
  c.cyl(0.075, 0.075, 0.03, x, top + 5e-3, z, "black", { seg: 40 });
  c.cyl(0.06, 0.06, 8e-3, x, top + 0.02, z, "lens", { seg: 40 });
  c.ball(0.028, x, top + 0.022, z, "glass", 24);
  c.add(new THREE.SphereGeometry(0.062, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2.3), "glass", new THREE.Matrix4().makeTranslation(x, top + 0.016, z));
  c.ball(5e-3, x - 0.02, top + 0.068, z - 0.02, "spot", 8);
  c.ball(7e-3, x + 0.22, top + 1e-3, z, "spot", 10);
  for (const sx of [-1, 1]) c.cyl(0.01, 0.01, 4e-3, x + sx * 0.29, top + 1e-3, z, "lens", { seg: 12 });
  c.box(0.2, 0.04, 0.04, x, H - 0.02, z, "black");
}
function buildBoard(c) {
  const bz = -0.02;
  const top = BOARD_Y + 0.01;
  const back = bz - 0.5;
  c.box(1.72, 0.02, 1, 0, BOARD_Y, bz, "pcb");
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      c.cyl(0.025, 0.025, 0.06, sx * 0.8, FLOOR + 0.03, bz + sz * 0.44, "gold", { seg: 12 });
      c.cyl(0.03, 0.03, 8e-3, sx * 0.8, top + 4e-3, bz + sz * 0.44, "steel", { seg: 16 });
    }
  }
  c.cyl(0.045, 0.045, 0.1, -0.72, top + 0.05, back - 0.02, "black", { axis: "z", seg: 20 });
  c.cyl(0.03, 0.03, 6e-3, -0.72, top + 0.05, back - 0.072, "chip", { axis: "z", seg: 16 });
  c.box(0.17, 0.06, 0.1, -0.48, top + 0.03, back - 0.02, "steel");
  c.box(0.14, 0.03, 4e-3, -0.48, top + 0.03, back - 0.071, "chip");
  for (const x of [-0.2, 0.02]) {
    c.box(0.15, 0.17, 0.12, x, top + 0.085, back - 0.02, "steel");
    for (const y of [0.05, 0.125]) {
      c.box(0.12, 0.052, 4e-3, x, top + y, back - 0.081, "chip");
      c.box(0.1, 8e-3, 4e-3, x, top + y + 0.01, back - 0.082, "steel");
    }
  }
  c.box(0.17, 0.14, 0.14, 0.32, top + 0.07, back - 0.03, "steel");
  c.box(0.13, 0.1, 4e-3, 0.32, top + 0.06, back - 0.101, "chip");
  for (const dx of [-0.055, 0.055]) c.ball(6e-3, 0.32 + dx, top + 0.125, back - 0.101, "steel", 8);
  c.box(0.06, 0.03, 0.8, 0.8, top + 0.015, bz - 0.05, "black");
  for (let i = 0; i < 20; i++) for (const dx of [-0.013, 0.013]) c.cyl(5e-3, 5e-3, 0.06, 0.8 + dx, top + 0.045, bz - 0.43 + i * (0.76 / 19), "gold", { seg: 6 });
  c.box(0.9, 0.06, 0.07, MODULE.x, top + 0.03, MODULE.z + MODULE.d / 2 + 5e-3, "black");
  for (const sx of [-1, 1]) c.cyl(0.02, 0.02, 0.06, MODULE.x + sx * (MODULE.w / 2 - 0.04), top + 0.03, MODULE.z - MODULE.d / 2 + 0.04, "gold", { seg: 10 });
  c.box(0.3, 0.05, 0.05, SSD.x, top + 0.025, SSD.z - SSD.l / 2 + 0.05, "black");
  c.cyl(0.02, 0.02, 0.04, SSD.x, top + 0.02, SSD.z + SSD.l / 2 - 0.02, "gold", { seg: 10 });
  c.box(0.2, 0.035, 0.05, CSI.x, top + 0.0175, CSI.z, "steel");
  c.box(0.05, 0.035, 0.035, 0.3, top + 0.0175, 0.25, "black");
  for (const x of [-0.72, -0.62]) {
    c.box(0.05, 0.03, 0.05, x, top + 0.015, 0.4, "black");
    c.cyl(0.012, 0.012, 0.012, x, top + 0.036, 0.4, "chip", { seg: 10 });
  }
  const chips = [
    [-0.68, 0.1, 0.16, 0.035, 0.16],
    [-0.5, 0.3, 0.12, 0.02, 0.12],
    [0.35, -0.3, 0.2, 0.025, 0.2],
    [0.33, 0.12, 0.1, 0.02, 0.16],
    [0.4, -0.05, 0.08, 0.015, 0.08],
    [-0.4, 0.38, 0.14, 0.02, 0.1],
    [-0.1, 0.4, 0.12, 0.018, 0.08]
  ];
  for (const [x, z, w, h, d] of chips) c.box(w, h, d, x, top + h / 2, z, "chip");
  for (const [x, z] of [
    [-0.76, -0.3],
    [-0.76, -0.18],
    [-0.64, -0.3],
    [-0.64, -0.18],
    [-0.3, 0.4]
  ]) {
    c.cyl(0.028, 0.028, 0.07, x, top + 0.035, z, "black", { seg: 14 });
    c.cyl(0.027, 0.027, 2e-3, x, top + 0.071, z, "steel", { seg: 14 });
  }
  c.box(0.1, 0.06, 0.1, -0.74, top + 0.03, -0.42, "black");
  let seed = 7;
  const rand = () => {
    seed = seed * 16807 % 2147483647;
    return seed / 2147483647;
  };
  for (let i = 0; i < 70; i++) {
    const x = -0.8 + rand() * 1.55;
    const z = bz - 0.45 + rand() * 0.9;
    const inModule = x > MODULE.x - MODULE.w / 2 - 0.03 && x < MODULE.x + MODULE.w / 2 + 0.03 && z > MODULE.z - MODULE.d / 2 - 0.03 && z < MODULE.z + MODULE.d / 2 + 0.06;
    const inSsd = x > SSD.x - 0.17 && x < SSD.x + 0.17 && z > SSD.z - SSD.l / 2 - 0.1 && z < SSD.z + SSD.l / 2;
    const inPorts = z < back + 0.16 || x > 0.74 || Math.abs(x - CSI.x) < 0.14 && z > 0.34;
    if (inModule || inSsd || inPorts) continue;
    const along = rand() > 0.5;
    c.box(along ? 0.028 : 0.014, 8e-3, along ? 0.014 : 0.028, x, top + 4e-3, z, rand() > 0.6 ? "steel" : "chip");
  }
}
function buildModule(c) {
  const y = BOARD_Y + 0.01 + 0.06 + 75e-4;
  c.box(MODULE.w, 0.015, MODULE.d, MODULE.x, y, MODULE.z, "pcb");
  c.box(0.8, 0.016, 0.03, MODULE.x, y, MODULE.z + MODULE.d / 2 - 5e-3, "gold");
  const top = y + 75e-4;
  c.box(0.3, 0.03, 0.3, MODULE.x, top + 0.015, MODULE.z - 0.02, "chip");
  for (const [dx, dz] of [
    [-0.3, -0.12],
    [-0.3, 0.1],
    [0.3, -0.12],
    [0.3, 0.1]
  ]) {
    c.box(0.14, 0.018, 0.1, MODULE.x + dx, top + 9e-3, MODULE.z + dz, "chip");
  }
  for (const dx of [-0.12, 0.12]) c.box(0.06, 0.012, 0.04, MODULE.x + dx, top + 6e-3, MODULE.z + 0.2, "chip");
  for (const sx of [-1, 1]) c.cyl(0.028, 0.028, 4e-3, MODULE.x + sx * (MODULE.w / 2 - 0.04), top + 2e-3, MODULE.z - MODULE.d / 2 + 0.04, "gold", { seg: 12 });
}
const PLATE_Y = 0.245;
const FIN_H = 0.22;
const FAN_Y = PLATE_Y + 0.015 + FIN_H + 0.025;
function buildHeatsink(c) {
  c.box(0.84, 0.03, 0.52, MODULE.x, PLATE_Y, MODULE.z, "fin");
  const fins = 15;
  for (let i = 0; i < fins; i++) {
    const x = MODULE.x - 0.4 + i * 0.8 / (fins - 1);
    c.box(0.02, FIN_H, 0.48, x, PLATE_Y + 0.015 + FIN_H / 2, MODULE.z, "fin");
  }
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      c.cyl(0.018, 0.018, 0.03, MODULE.x + sx * 0.39, PLATE_Y - 0.03, MODULE.z + sz * 0.23, "black", { seg: 10 });
      c.cyl(0.02, 0.02, 6e-3, MODULE.x + sx * 0.39, PLATE_Y + 0.018, MODULE.z + sz * 0.23, "black", { seg: 12 });
    }
  }
}
function buildFan(c, rotor) {
  const y = FAN_Y;
  const cx = MODULE.x;
  const cz = MODULE.z;
  const span = 0.5;
  for (const s of [-1, 1]) {
    c.box(span, 0.05, 0.05, cx, y, cz + s * (span / 2 - 0.025), "black");
    c.box(0.05, 0.05, span - 0.1, cx + s * (span / 2 - 0.025), y, cz, "black");
  }
  c.cyl(0.225, 0.225, 0.05, cx, y, cz, "black", { seg: 48, open: true });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) c.cyl(0.014, 0.014, 4e-3, cx + sx * 0.225, y + 0.027, cz + sz * 0.225, "steel", { seg: 10 });
  rotor.cyl(0.085, 0.08, 0.04, 0, 0, 0, "black", { seg: 32 });
  const blades = 7;
  for (let i = 0; i < blades; i++) {
    const a = i / blades * Math.PI * 2;
    const m = new THREE.Matrix4().makeRotationY(a).multiply(new THREE.Matrix4().makeTranslation(0.145, 0, 0)).multiply(new THREE.Matrix4().makeRotationX(0.6));
    rotor.add(new THREE.BoxGeometry(0.13, 3e-3, 0.06), "black", m);
  }
  rotor.torus(0.21, 4e-3, 0, 0, 0, "black", { seg: 48 });
  return new THREE.Vector3(cx, y, cz);
}
function buildSsd(c) {
  const y = BOARD_Y + 0.01 + 0.045;
  c.box(SSD.w, 0.012, SSD.l, SSD.x, y, SSD.z, "pcb");
  c.box(0.22, 0.013, 0.05, SSD.x, y, SSD.z - SSD.l / 2 + 0.02, "gold");
  const top = y + 6e-3;
  for (const dz of [-0.26, -0.04]) c.box(0.2, 0.014, 0.18, SSD.x, top + 7e-3, SSD.z + dz, "chip");
  c.box(0.15, 0.012, 0.15, SSD.x, top + 6e-3, SSD.z + 0.2, "chip");
  c.box(0.08, 0.01, 0.06, SSD.x - 0.07, top + 5e-3, SSD.z + 0.34, "chip");
  c.cyl(0.024, 0.024, 8e-3, SSD.x, top + 4e-3, SSD.z + SSD.l / 2 - 0.02, "gold", { seg: 14 });
}
function buildDevice(materials, blobMap) {
  const root = new THREE.Group();
  const build = (fn) => {
    const c = new Collector();
    fn(c);
    return c.build(materials);
  };
  const base = build(buildBase);
  const speaker = build(buildSpeaker);
  const lid = build(buildLid);
  const webcam = build(buildWebcam);
  const board = build(buildBoard);
  const module = build(buildModule);
  const heatsink = build(buildHeatsink);
  const ssd = build(buildSsd);
  const fanParts = new Collector();
  const rotorParts = new Collector();
  const axis = buildFan(fanParts, rotorParts);
  const fan = fanParts.build(materials);
  const rotor = rotorParts.build(materials);
  rotor.position.copy(axis);
  fan.add(rotor);
  const lidPivot = LID_PIVOT.clone();
  for (const mesh of [...lid.children, ...webcam.children]) mesh.position.sub(lidPivot);
  lid.position.copy(lidPivot);
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 2.6), new THREE.MeshBasicMaterial({ map: blobMap, transparent: true, depthWrite: false, opacity: 0.55 }));
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = -FOOT - 4e-3;
  blob.renderOrder = -1;
  base.add(speaker);
  lid.add(webcam);
  board.add(module, ssd);
  module.add(heatsink);
  heatsink.add(fan);
  root.add(base, lid, board, blob);
  const groups = { base, lid, webcam, speaker, board, module, heatsink, fan, ssd };
  const dispose = () => {
    root.traverse((o) => {
      const mesh = o;
      mesh.geometry?.dispose?.();
    });
    blob.material.dispose();
  };
  return { root, groups, rotor, blob, lidPivot, dispose };
}
function canvasOf(w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  return [canvas, canvas.getContext("2d")];
}
function grilleCanvasOf() {
  const w = 768;
  const h = Math.round(w * GRILLE.h / GRILLE.w);
  const [canvas, ctx] = canvasOf(w, h);
  if (ctx) {
    ctx.fillStyle = "#efeeeb";
    ctx.fillRect(0, 0, w, h);
    const pitch = 15;
    const r = 4.4;
    const coneR = SPEAKER.r / GRILLE.w * w;
    for (let row = 0, y = 10; y < h - 6; row++, y += pitch * 0.866) {
      for (let x = row % 2 ? 10 + pitch / 2 : 10; x < w - 6; x += pitch) {
        const d = Math.hypot(x - w / 2, y - h / 2);
        const overCone = d < coneR;
        ctx.fillStyle = overCone ? "#131519" : "#2a2d32";
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,0.5)";
        ctx.beginPath();
        ctx.arc(x, y + 1.2, r + 0.6, Math.PI * 0.15, Math.PI * 0.85, false);
        ctx.arc(x, y + 1.2, r - 0.4, Math.PI * 0.85, Math.PI * 0.15, true);
        ctx.fill();
      }
    }
    const shade = (x0, y0, x1, y1, strength) => {
      const g = ctx.createLinearGradient(x0, y0, x1, y1);
      g.addColorStop(0, `rgba(0,0,0,${strength})`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    };
    shade(0, 0, 0, 18, 0.28);
    shade(0, 0, 14, 0, 0.14);
    shade(w, 0, w - 14, 0, 0.14);
    shade(0, h, 0, h - 8, 0.08);
  }
  return canvas;
}
function tracesCanvasOf() {
  const [canvas, ctx] = canvasOf(512, 512);
  if (ctx) {
    ctx.fillStyle = "#14161a";
    ctx.fillRect(0, 0, 512, 512);
    let seed = 11;
    const rand = () => {
      seed = seed * 16807 % 2147483647;
      return seed / 2147483647;
    };
    ctx.globalAlpha = 0.5;
    for (let i = 0; i < 60; i++) {
      ctx.fillStyle = rand() > 0.5 ? "#181a1f" : "#101216";
      ctx.fillRect(rand() * 512, rand() * 512, 20 + rand() * 120, 20 + rand() * 120);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#2a2e35";
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    for (let i = 0; i < 150; i++) {
      const x0 = rand() * 512;
      const y0 = rand() * 512;
      const run = 20 + rand() * 120;
      const drop = (rand() - 0.5) * 100;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      if (rand() > 0.5) {
        ctx.lineTo(x0 + run, y0);
        ctx.lineTo(x0 + run + Math.abs(drop) * 0.4, y0 + drop);
      } else {
        ctx.lineTo(x0, y0 + run);
        ctx.lineTo(x0 + drop, y0 + run + Math.abs(drop) * 0.4);
      }
      ctx.stroke();
      ctx.fillStyle = "#3a3f47";
      ctx.beginPath();
      ctx.arc(x0, y0, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 0; i < 160; i++) {
      ctx.fillStyle = rand() > 0.6 ? "#b9bcc2" : "#4a4f57";
      ctx.beginPath();
      ctx.arc(rand() * 512, rand() * 512, 1.3 + rand() * 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(230,232,236,0.4)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 70; i++) {
      const w = 8 + rand() * 30;
      const h = 6 + rand() * 14;
      ctx.strokeRect(rand() * 500, rand() * 500, rand() > 0.5 ? w : h, rand() > 0.5 ? h : w);
    }
    ctx.fillStyle = "rgba(230,232,236,0.38)";
    for (let i = 0; i < 90; i++) ctx.fillRect(rand() * 500, rand() * 510, 3 + rand() * 9, 1.5);
  }
  return canvas;
}
function blobTexture() {
  const [canvas, ctx] = canvasOf(128, 128);
  if (ctx) {
    const fade = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    fade.addColorStop(0, "rgba(0,0,0,0.7)");
    fade.addColorStop(0.35, "rgba(0,0,0,0.5)");
    fade.addColorStop(0.7, "rgba(0,0,0,0.16)");
    fade.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 128, 128);
  }
  return new THREE.CanvasTexture(canvas);
}
export {
  ANCHORS,
  Collector,
  D,
  DEVICE_PARTS,
  H,
  TRAVEL,
  W,
  blobTexture,
  buildDevice,
  makeMaterials,
  makeTextures
};
