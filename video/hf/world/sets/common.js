// Shared palette, helpers and materials for every set in the Atlas world. See WORLD.md.
// Pure helpers only: no clocks, no Math.random (use hash).
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

export const SURFACE = 0xe4e8ee, SHADE = 0xb7c0cd, LIGHT = 0xffffff, INK = 0x2b313b, ACCENT = 0xd9774a, MUTED = 0x6f7886;
// ACES pulls the cool cast out of a lit surface, so the 3D base colour sits a touch bluer than the CSS token.
export const SURFACE_3D = 0xdfe6f0;
export const ENV_INTENSITY = 0.35;
export const DESK_Y = -1.0;
/** The room and the kitchen share one floor: a 75 cm desk under the device (1 unit = 100 mm). */
export const FLOOR_Y = -8.5;

/** Deterministic 0..1 hash of two integers (same every frame, every render). */
export function hash(i, j = 0) {
  let n = (i * 374761393 + j * 668265263) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}
export const clamp01 = (v) => Math.min(1, Math.max(0, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (a, b, v) => { const x = clamp01((v - a) / (b - a)); return x * x * (3 - 2 * x); };
const bump = (x, w) => Math.exp(-(x * x) / (w * w));
/** A slow heartbeat, period 2 s: a strong bump and a softer echo. Pure function of t. */
export const heartbeat = (t) => { const p = (t % 2.0) / 2.0; return bump(p - 0.12, 0.07) + 0.55 * bump(p - 0.36, 0.09); };

// ---- The enclosure's coordinate system (ENCLOSURE.md) ----
// CAD origin = lower-front-right outside corner. +X device LEFT, +Y BACK, +Z UP, millimetres.
// Three.js: 1 unit = 100 mm, centred on x/z, front facing +z, sitting on the desk at y = -1.0.
/** CAD millimetres -> world units (the enclosure's own frame, before it moves). */
export function mm(X, Y, Z) { return new THREE.Vector3((X - 62.5) / 100, DESK_Y + Z / 100, (99 - Y) / 100); }
/** A CAD box: size [sx, sy, sz] mm (X x Y x Z) and min corner [X, Y, Z] mm -> { center, size } in world units. */
export function mmBox(size, min) {
  const [sx, sy, sz] = size, [X, Y, Z] = min;
  const a = mm(X, Y, Z), b = mm(X + sx, Y + sy, Z + sz);
  return { center: a.clone().add(b).multiplyScalar(0.5), size: new THREE.Vector3(sx / 100, sz / 100, sy / 100) };
}
/** A mesh for a CAD box. radius in mm (rounded) or 0 (sharp). */
export function boxMesh(size, min, material, radius = 0, seg = 2) {
  const { center, size: s } = mmBox(size, min);
  const r = Math.min(radius / 100, Math.min(s.x, s.y, s.z) / 2 - 1e-4);
  const geo = r > 0 ? new RoundedBoxGeometry(s.x, s.y, s.z, seg, r) : new THREE.BoxGeometry(s.x, s.y, s.z);
  const m = new THREE.Mesh(geo, material); m.position.copy(center); m.castShadow = true; m.receiveShadow = true;
  return m;
}

export function softTexture(draw, w = 512, h = 512) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}
export const FONT = `Inter, "Segoe UI", system-ui, sans-serif`;

// ---- Materials ----
/** Exact sRGB colours: labels, maps, plates, accent lines. Not tone mapped. */
export const flat = (extra = {}) => new THREE.MeshBasicMaterial({ toneMapped: false, ...extra });
/** The soft matte surface every object is made of. Emissive is the accent (for highlights), off by default. */
export const matte = (extra = {}) => new THREE.MeshStandardMaterial({ color: SURFACE_3D, roughness: 0.82, metalness: 0, envMapIntensity: ENV_INTENSITY, emissive: ACCENT, emissiveIntensity: 0, ...extra });
/** A matte tint of the palette (muted, never saturated). */
export const tint = (color, extra = {}) => matte({ color, ...extra });

/** Place a capsule (or any Y-aligned mesh) between two points. */
const _up = new THREE.Vector3(0, 1, 0);
export function between(mesh, a, b) {
  const d = b.clone().sub(a); mesh.position.copy(a).addScaledVector(d, 0.5);
  mesh.quaternion.setFromUnitVectors(_up, d.normalize()); return mesh;
}

/**
 * A fading set: every material registered here fades with the set (opacity), and every top-level piece rises in with
 * a small stagger. `update(v)` with v 0..1 drives both. Pieces are added through `add(obj)`.
 */
export function fadingSet(root) {
  const mats = new Set(), pieces = [];
  const reg = (m) => { if (!m || mats.has(m)) return; m.transparent = true; m.userData.baseOpacity = m.opacity; m.userData.baseDepth = m.depthWrite; mats.add(m); };
  function add(obj, parent = root) {
    parent.add(obj);
    obj.traverse((o) => { if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(reg); });
    pieces.push({ obj, y: obj.position.y, k: pieces.length });
    return obj;
  }
  function update(v) {
    root.visible = v > 0.001;
    if (!root.visible) return;
    for (const p of pieces) {
      const lag = hash(p.k, 31) * 0.35, u = clamp01((v - lag) / (1 - lag)), e = 1 - Math.pow(1 - u, 3);
      p.obj.position.y = p.y - (1 - e) * 2.2;
      p.obj.visible = u > 0.001;
    }
    for (const m of mats) { m.opacity = m.userData.baseOpacity * clamp01(v * 1.4); m.depthWrite = m.userData.baseDepth && v > 0.6; }
  }
  return { add, update, reg };
}
export { RoundedBoxGeometry };
