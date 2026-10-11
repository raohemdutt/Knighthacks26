// The Atlas device as the website shows it (web/components/home/deviceModel.ts, compiled into deviceModel.js): a
// white block with the pressed mark on its lid, a speaker grille across the front, a small black webcam on top and,
// inside, the carrier board, the Jetson module, its heatsink and fan, and the SSD. This adapter gives the film the
// same interface the cardboard enclosure had (enclosure.js), so every scene keeps working:
//
//   const a = buildAtlas();  scene.add(a.group);  a.update(state, t);  a.P.webcam;  a.at("jetson", 1)
//
// state.explode lifts the parts out the way the website does (TRAVEL, each part eased over its own stretch);
// state.xray makes the white shell glassy; state.hl.{webcam, jetson, speaker, lid, front} tints a part toward the
// accent; state.pulse breathes the light bar and the module; state.eyes is accepted and ignored (the camera stays black); state.spin turns it.
// state.drop is accepted and ignored (the intro no longer drops parts in). Units: the website's 1 unit = 9 cm; the
// film's 1 unit = 10 cm, so the model is scaled 0.9 and stands on the desk (y = DESK_Y), facing +z.
import * as THREE from "three";
import { buildDevice, makeMaterials, makeTextures, TRAVEL, W, D, H } from "./deviceModel.js";
import { DESK_Y, ACCENT, clamp01, heartbeat } from "./common.js";

const S = 0.9;
const LIFT = 0.3; // the whole diagram floats up a little as it opens (as the enclosure did)
const smooth = (a, b, x) => { const u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); };
// Which part answers each of the film's highlight names.
const HL = { webcam: ["webcam"], jetson: ["module", "heatsink", "fan"], speaker: ["speaker"], lid: ["lid"], front: ["base"] };
// The chain of parents each part rides with (its own travel plus theirs).
const CHAIN = { base: ["base"], lid: ["lid"], webcam: ["lid", "webcam"], board: ["board"], speaker: ["base", "speaker"],
  module: ["board", "module"], ssd: ["board", "ssd"], heatsink: ["board", "module", "heatsink"], fan: ["board", "module", "heatsink", "fan"] };

function make() {
  const tex = makeTextures(), mats = makeMaterials(tex);
  const dev = buildDevice(mats, tex.blob);
  dev.blob.visible = false; // the film's world draws its own contact shadow
  // Each part gets its own copies of its materials, so a highlight or the x-ray touches only that part.
  const own = {};
  for (const [id, g] of Object.entries(dev.groups)) {
    own[id] = [];
    for (const m of g.children) if (m.isMesh) {
      m.material = m.material.clone();
      m.material.userData.base = m.material.color ? m.material.color.clone() : null;
      m.material.userData.baseOpacity = m.material.opacity;
      m.material.userData.baseEmissive = m.material.emissiveIntensity ?? 0;
      own[id].push(m.material);
    }
  }
  // The webcam faces forward, not up (Dom): the website lays it flat on the lid with the lens up; here it stands on
  // the lid, turned a quarter about its own centre so the lens looks out of the front (+z).
  {
    const g = dev.groups.webcam, c = new THREE.Vector3(0.35, 1.0 + 0.11 / 2 - 0.9 + 0.06, 0.3); // centre in the lid's frame, lifted to sit on top
    for (const m of g.children) m.position.sub(c);
    g.position.add(c); g.rotation.x = Math.PI / 2;
  }
  // The lens dome is a tinted blue glass on the website; in the film the camera reads black everywhere (Dom).
  for (const m of own.webcam) if (m.transparent && m.color) { m.color.set(0x0d0f12); m.userData.base = m.color.clone(); m.envMapIntensity = 0.8; }
  const rest = {}; for (const [id, g] of Object.entries(dev.groups)) rest[id] = g.position.clone();
  // The front panel on the base (the grille, its frame and the dark plate behind it): the x-ray clears it completely,
  // with the walls, so nothing hangs in front of the Jetson (Dom).
  const front = [], bb = new THREE.Box3();
  // The grille's grey frame is merged into the base's "seam" mesh (with the floor and the rim): split its triangles
  // (everything on the front face, z >= D/2 - 0.01) into a mesh of their own so the x-ray can clear it.
  for (const m of [...dev.groups.base.children]) if (m.isMesh && !m.geometry.index) {
    const pos = m.geometry.attributes.position, keep = [], cut = [];
    for (let i = 0; i < pos.count; i += 3) (pos.getZ(i) >= D / 2 - 0.01 && pos.getZ(i + 1) >= D / 2 - 0.01 && pos.getZ(i + 2) >= D / 2 - 0.01 ? cut : keep).push(i);
    if (!cut.length || !keep.length) continue;
    const pick = (tris) => {
      const g = new THREE.BufferGeometry();
      for (const [name, attr] of Object.entries(m.geometry.attributes)) {
        const out = new Float32Array(tris.length * 3 * attr.itemSize);
        tris.forEach((i, k) => { for (let j = 0; j < 3 * attr.itemSize; j++) out[k * 3 * attr.itemSize + j] = attr.array[i * attr.itemSize + j]; });
        g.setAttribute(name, new THREE.BufferAttribute(out, attr.itemSize));
      }
      return g;
    };
    const panel = new THREE.Mesh(pick(cut), m.material.clone());
    panel.material.userData = { ...m.material.userData };
    panel.position.copy(m.position); panel.rotation.copy(m.rotation); panel.scale.copy(m.scale);
    m.geometry = pick(keep); dev.groups.base.add(panel); own.base.push(panel.material);
  }
  for (const m of dev.groups.base.children) if (m.isMesh) {
    m.updateMatrix(); m.geometry.computeBoundingBox(); bb.copy(m.geometry.boundingBox).applyMatrix4(m.matrix);
    if (bb.min.z > D / 2 - 0.12 && bb.max.y - bb.min.y < H * 0.8) { m.material.transparent = true; m.material.needsUpdate = true; front.push(m); }
  }
  return { dev, own, rest, front };
}

/** The hero device: explode, x-ray, highlights, pulse, eyes, anchors. */
export function buildAtlas() {
  const { dev, own, rest, front } = make();
  const group = new THREE.Group(); group.name = "atlas";
  const inner = new THREE.Group(); inner.scale.setScalar(S); inner.position.y = DESK_Y; inner.add(dev.root); group.add(inner);
  const accentC = new THREE.Color(ACCENT);
  const parts = Object.fromEntries(Object.entries(dev.groups).map(([id, g]) => [id, { g, name: id, rest: rest[id], mats: own[id] }]));
  const lit = (id, name) => HL[name].includes(id);
  const isShell = (m) => m.transparent !== undefined && ["white", "grille", "seam", "logo"].some((k) => m.name === k) || false;

  // Anchors in device space (the website's units), turned into film space below.
  const DEV = {
    box: new THREE.Vector3(0, H / 2, 0),
    webcam: new THREE.Vector3(0.35, H + 0.17, 0.42), // the lens, facing forward
    jetson: new THREE.Vector3(-0.2, 0.22, -0.12), // the module
    speaker: new THREE.Vector3(0, 0.42, D / 2), // behind the grille
    lid: new THREE.Vector3(0, H, 0),
    front: new THREE.Vector3(0, 0.42, D / 2),
    ethernet: new THREE.Vector3(-0.55, 0.14, -D / 2), // the carrier board's ports along the back
  };
  const PART_OF = { webcam: "webcam", jetson: "module", speaker: "speaker", lid: "lid", front: "base", ethernet: "board", box: null };
  const toFilm = (v) => new THREE.Vector3(v.x * S, DESK_Y + v.y * S, v.z * S);
  const travel = (id, e) => {
    const out = new THREE.Vector3();
    for (const p of CHAIN[id] || []) { const tr = TRAVEL[p]; out.addScaledVector(tr.move, smooth(tr.from, tr.to, e)); }
    return out;
  };
  const P = Object.fromEntries(Object.entries(DEV).map(([k, v]) => [k, toFilm(v).toArray()]));
  /** World position of an anchor at explode e (default: assembled). Ignores spin. */
  function at(name, e = 0) {
    const v = (DEV[name] || DEV.box).clone(), pid = PART_OF[name];
    if (pid) v.add(travel(pid, e));
    const f = toFilm(v); if (name !== "box") f.y += LIFT * e;
    return f.toArray();
  }

  // Keep the original material keys (Collector names meshes by nothing, so remember which materials were shell white).
  const shellMats = [];
  for (const id of Object.keys(own)) for (const m of own[id]) {
    const c = m.userData.base;
    const white = c && c.r > 0.8 && c.g > 0.8 && c.b > 0.8 && !m.map; // the matte white polymer and the mark (linear colour: 0xf3f2ef is about 0.9)
    if (white || (m.map && id === "base")) { m.transparent = true; m.needsUpdate = true; shellMats.push(m); }
  }

  function update(state, t) {
    const ex = clamp01(Math.max(state.explode || 0, state.open || 0));
    const xr = clamp01(state.xray || 0), pulse = state.pulse || 0, beat = heartbeat(t), eyes = clamp01(state.eyes || 0);
    const hl = state.hl || {};
    group.rotation.y = (state.spin || 0) * Math.PI * 2;
    group.position.y = LIFT * ex;
    // Explode: each part moves along its own path over its own stretch, as on the website.
    for (const [id, g] of Object.entries(dev.groups)) {
      const tr = TRAVEL[id], p = smooth(tr.from, tr.to, ex);
      g.position.copy(rest[id]).addScaledVector(tr.move, p);
      if (tr.tilt) g.rotation.set(tr.tilt.x * p, tr.tilt.y * p, tr.tilt.z * p);
    }
    dev.rotor.rotation.y = -t * (1.5 + ex * 6);
    // Highlights: tint toward the accent with a little glow.
    for (const [id, mats] of Object.entries(own)) {
      if (id === "webcam") continue; // the camera stays black all film (Dom)
      let level = 0; for (const name of Object.keys(HL)) if (lit(id, name)) level = Math.max(level, clamp01(hl[name] || 0));
      for (const m of mats) {
        if (m.userData.base && m.color) m.color.copy(m.userData.base).lerp(accentC, level * 0.35);
        if (m.emissive && level > 0) { m.emissive.copy(accentC); m.emissiveIntensity = m.userData.baseEmissive + level * 0.22; }
        else if (m.emissive) m.emissiveIntensity = m.userData.baseEmissive;
      }
    }
    // X-ray: the white shell goes glassy so the board and the module show through.
    for (const m of shellMats) { m.opacity = (m.userData.baseOpacity ?? 1) * (1 - xr * 0.82); m.depthWrite = xr < 0.05; }
    for (const m of front) { m.material.opacity = (m.material.userData.baseOpacity ?? 1) * (1 - xr); m.material.depthWrite = xr < 0.05; m.visible = xr < 0.99; }
    // Pulse: the light bar along the lid's front edge breathes; the module warms.
    for (const m of own.lid) if (m.emissive && m.userData.baseEmissive >= 1) m.emissiveIntensity = 0.6 + pulse * (0.6 + 0.6 * beat);
    for (const m of own.module) if (m.emissive) m.emissiveIntensity += pulse * beat * 0.25;
  }

  return { group, parts, P, at, update, LIFT };
}

/** A copy for the friends' boxes, the kitchen and the control room: no explode. update(glow, t, eyes). */
export function buildAtlasLite() {
  const { dev, own } = make();
  const group = new THREE.Group(); group.name = "atlas-lite";
  const inner = new THREE.Group(); inner.scale.setScalar(S); inner.position.y = DESK_Y; inner.add(dev.root); group.add(inner);
  const accentC = new THREE.Color(ACCENT);
  const all = Object.values(own).flat(), webcam = new Set(own.webcam);
  function update(v, t, eyes = 0) {
    const g = clamp01(v), beat = heartbeat(t);
    for (const m of all) {
      if (webcam.has(m)) continue; // the camera stays black all film (Dom)
      if (m.userData.base && m.color) m.color.copy(m.userData.base).lerp(accentC, g * 0.2);
      if (m.emissive && m.userData.baseEmissive >= 1) m.emissiveIntensity = 0.6 + g * (0.6 + 0.5 * beat); // the light bar
    }
    dev.rotor.rotation.y = -t * 1.5;
  }
  return { group, update };
}
