// The person: Quaternius' Universal Animation Library mannequin (CC0, assets/people/UAL1_Standard.glb, 43 clips on a
// humanoid rig: Sitting_Exit, Walk_Loop, Sitting_Enter, Sitting_Idle_Loop, Idle_Loop, Interact, ...). Faceless, so the
// person reads as anyone and nobody, which is the point of a privacy film. Recoloured to the film's palette.
//
// Deterministic: nothing plays on its own. Every frame the caller says which clips are on, at which clip time and
// weight (`sample`), or hands over a schedule of acts on the film's clock (`act` + `at(t)`), so the pose is a pure
// function of t and seeks cleanly. Units: 1 = 100 mm (the glb is in metres, scaled 10x). Faces +z before rotation.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
import { matte, clamp01, lerp } from "./common.js";

let cached = null;
const HAIRS = ["Hair_SimpleParted", "Hair_Buzzed", "Hair_Long", "Eyebrows_Regular"];
/** Load the glb and the hair (Quaternius Universal Base Characters, CC0, "origin at 0" meshes) once. */
export async function loadHuman() {
  if (!cached) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync("assets/people/UAL1_Standard.glb");
    gltf.hair = {};
    for (const h of HAIRS) gltf.hair[h] = (await loader.loadAsync(`assets/people/hair/${h}.gltf`)).scene;
    cached = gltf;
  }
  return cached;
}

// Who wears what: per-person palettes (clothes by body region, painted per vertex from the skin weights) and hair.
export const LOOKS = {
  hemdutt: { shirt: 0x5b7fb8, pants: 0x3d4656, shoes: 0xf1f3f6, skin: 0xb98a6a, hair: "Hair_SimpleParted", hairColor: 0x2a211c },
  ilan: { shirt: 0x6f9a7a, pants: 0x55606f, shoes: 0x2f343b, skin: 0xe2bc9c, hair: "Hair_Long", hairColor: 0x4a3426 },
  dominick: { shirt: 0xc98b5e, pants: 0x2f3a4a, shoes: 0xe9edf2, skin: 0xd8ad8a, hair: "Hair_Buzzed", hairColor: 0x3a2a20 },
  xander: { shirt: 0xe5d54a, pants: 0x3a4250, shoes: 0x4a3a2a, skin: 0xe7c3a5, hair: "Hair_Buzzed", hairColor: 0x7a5a3a }, // hi-vis
  trainee: { shirt: 0xd9e05a, pants: 0x4a4f5a, shoes: 0x3a3a3a, skin: 0x9a6a4a, hair: "Hair_SimpleParted", hairColor: 0x1e1a18 },
};
const REGION = (name) => /Head|neck/.test(name) ? "skin" : /hand|index|middle|pinky|ring|thumb|lowerarm/.test(name) ? "skin"
  : /foot|ball/.test(name) ? "shoes" : /thigh|calf|pelvis/.test(name) ? "pants" : "shirt";
const smooth = (u) => { const x = clamp01(u); return x * x * (3 - 2 * x); };
const angLerp = (a, b, u) => a + (((b - a + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI) * u;

export function makeHuman(gltf, outfit = LOOKS.hemdutt) {
  const model = SkeletonUtils.clone(gltf.scene);
  model.scale.setScalar(10);
  model.rotation.y = Math.PI; // the UE-rigged mannequin faces -z in its own space; turn it so the person faces +z
  const root = new THREE.Group(); root.add(model);
  const mats = [];
  // One material, coloured per vertex by the bone that moves it most: shirt, trousers, shoes, skin (short sleeves).
  const main = matte({ color: 0xffffff, roughness: 0.8, vertexColors: true });
  mats.push(main);
  const bones = {}; model.traverse((o) => { if (o.isBone) bones[o.name] = o; });
  const C = { shirt: new THREE.Color(outfit.shirt), pants: new THREE.Color(outfit.pants), shoes: new THREE.Color(outfit.shoes), skin: new THREE.Color(outfit.skin) };
  let skinned = null;
  model.traverse((o) => {
    if (!o.isMesh) return;
    o.material = main; o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
    if (o.isSkinnedMesh) {
      skinned = o;
      const g = o.geometry = o.geometry.clone(), si = g.attributes.skinIndex, sw = g.attributes.skinWeight, n = si.count;
      const col = new Float32Array(n * 3), names = o.skeleton.bones.map((b) => b.name);
      for (let v = 0; v < n; v++) {
        let best = 0, bw = -1; for (let k = 0; k < 4; k++) { const w = sw.getComponent(v, k); if (w > bw) { bw = w; best = si.getComponent(v, k); } }
        const c = C[REGION(names[best])]; col[v * 3] = c.r; col[v * 3 + 1] = c.g; col[v * 3 + 2] = c.b;
      }
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    }
  });

  // A soft face on the head bone: eyes with lids, a mouth that smiles, flattens or talks. The head's front and centre
  // are found from the vertices the head bone moves (rest pose), so the face sits on the surface.
  const headBox = new THREE.Box3(), vtx = new THREE.Vector3();
  if (skinned) {
    const g = skinned.geometry, pos = g.attributes.position, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const hi = skinned.skeleton.bones.indexOf(bones.Head);
    for (let v = 0; v < pos.count; v++) {
      let best = 0, bw = -1; for (let k = 0; k < 4; k++) { const w = sw.getComponent(v, k); if (w > bw) { bw = w; best = si.getComponent(v, k); } }
      if (best === hi) headBox.expandByPoint(vtx.fromBufferAttribute(pos, v));
    }
  }
  const hc = headBox.getCenter(new THREE.Vector3()), hs = headBox.getSize(new THREE.Vector3());
  // The mannequin's front is -z in its own (mesh) space, so the face sits on the head box's min z, turned to face out.
  const face = new THREE.Group();
  face.position.set(hc.x, hc.y - hs.y * 0.06, headBox.min.z + 0.004); face.rotation.y = Math.PI;
  const ink = new THREE.MeshBasicMaterial({ color: 0x1b2433, toneMapped: false, transparent: true });
  const lip = new THREE.MeshBasicMaterial({ color: 0x7a4a42, toneMapped: false, transparent: true });
  const hairMat = matte({ color: outfit.hairColor, roughness: 0.7 });
  const hairInk = new THREE.MeshBasicMaterial({ color: outfit.hairColor, toneMapped: false, transparent: true });
  mats.push(ink, lip, hairMat, hairInk);
  const eyes = [-1, 1].map((sd) => { const e = new THREE.Mesh(new THREE.SphereGeometry(0.0105, 14, 10), ink); e.scale.set(1, 1.25, 0.45); e.position.set(sd * 0.032, 0.012, -0.004); face.add(e); return e; });
  const brows = [-1, 1].map((sd) => { const b = new THREE.Mesh(new THREE.CapsuleGeometry(0.0035, 0.018, 4, 8), hairInk); b.rotation.z = Math.PI / 2; b.position.set(sd * 0.032, 0.034, -0.002); face.add(b); return b; });
  const smile = new THREE.Mesh(new THREE.TorusGeometry(0.02, 0.0032, 6, 20, Math.PI * 0.8), lip); smile.rotation.z = Math.PI + Math.PI * 0.1; smile.position.set(0, -0.03, -0.006); face.add(smile);
  const flatM = new THREE.Mesh(new THREE.CapsuleGeometry(0.003, 0.026, 4, 8), lip); flatM.rotation.z = Math.PI / 2; flatM.position.set(0, -0.036, -0.004); face.add(flatM);
  const open = new THREE.Mesh(new THREE.SphereGeometry(0.011, 14, 10), lip); open.scale.set(1.2, 0.2, 0.35); open.position.set(0, -0.036, -0.004); face.add(open);
  model.add(face); face.updateMatrixWorld(true); bones.Head.attach(face);

  // Hair: a soft cap fitted to the head (top and back, a little fringe at the front), in the person's hair colour.
  // Styles: "short" (a close cap), "long" (down to the neck at the back), "buzzed" (thin). Rest pose, mesh space.
  const style = outfit.hair === "Hair_Long" ? "long" : outfit.hair === "Hair_Buzzed" ? "buzzed" : "short";
  const cap = new THREE.Group(); cap.position.copy(hc);
  const r = Math.max(hs.x, hs.z) / 2, grow = style === "buzzed" ? 1.03 : 1.08;
  const dome = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20, 0, Math.PI * 2, 0, style === "long" ? Math.PI * 0.62 : Math.PI * 0.5), hairMat);
  dome.scale.set(hs.x / 2 * grow, hs.y / 2 * (style === "buzzed" ? 1.02 : 1.1), hs.z / 2 * grow); dome.position.y = hs.y * 0.06;
  dome.rotation.x = 0.28; // tilt back: more hair over the back of the head, the forehead stays clear
  cap.add(dome);
  if (style !== "buzzed") {
    const fringe = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.35), hairMat);
    fringe.scale.set(hs.x / 2 * 1.04, hs.y * 0.22, hs.z / 2 * 0.9); fringe.position.set(0, hs.y * 0.3, -r * 0.18); fringe.rotation.x = -0.5;
    cap.add(fringe);
  }
  if (style === "long") { const back = new THREE.Mesh(new THREE.CapsuleGeometry(r * 0.75, hs.y * 0.45, 6, 16), hairMat); back.position.set(0, -hs.y * 0.18, r * 0.55); cap.add(back); }
  cap.traverse((o) => { if (o.isMesh) { o.castShadow = true; } });
  model.add(cap); cap.updateMatrixWorld(true); bones.Head.attach(cap);
  /** Expressions, each 0..1, set every frame by the caller (pure function of t): smile, sleepy, strain, talk. */
  const expr = { smile: 0, sleepy: 0, strain: 0, talk: 0, t: 0 };
  function updateFace() {
    const lid = 1 - 0.75 * Math.max(expr.sleepy, expr.strain * 0.45);
    const blink = Math.max(0, 1 - Math.abs(((expr.t + hashSeed) % 4.1) - 0.08) / 0.08); // a blink every ~4 s
    for (const e of eyes) e.scale.y = 1.25 * Math.max(0.12, lid * (1 - blink * 0.9));
    brows.forEach((b, i) => { const sd = i ? 1 : -1; b.position.y = 0.034 - expr.strain * 0.006 + expr.smile * 0.003 - expr.sleepy * 0.004; b.rotation.z = Math.PI / 2 + sd * (expr.strain * 0.35 - expr.smile * 0.1); });
    const talkOpen = expr.talk * (0.5 + 0.5 * Math.abs(Math.sin(expr.t * 11.0) * Math.sin(expr.t * 4.3 + 1.1)));
    smile.visible = expr.smile > 0.35 && talkOpen < 0.25;
    flatM.visible = !smile.visible && talkOpen < 0.25;
    flatM.rotation.z = Math.PI / 2 + (expr.strain - expr.smile) * 0.15;
    open.visible = talkOpen >= 0.25; open.scale.y = 0.2 + talkOpen * 0.9;
  }
  const hashSeed = (outfit.shirt % 97) / 31;
  const clips = Object.fromEntries(gltf.animations.map((c) => [c.name, c]));
  const mixer = new THREE.AnimationMixer(model);
  const actions = {};
  const action = (name) => {
    if (!actions[name]) { const a = mixer.clipAction(clips[name]); a.play(); a.setEffectiveWeight(0); actions[name] = a; }
    return actions[name];
  };
  const duration = (name) => clips[name].duration;

  /** Pose from clips: [[name, time, weight], ...]. Weights should add up to 1. */
  function sample(list) {
    for (const a of Object.values(actions)) a.setEffectiveWeight(0);
    for (const [name, time, w] of list) { const a = action(name); a.time = Math.max(0, Math.min(time, clips[name].duration - 1e-4)); a.setEffectiveWeight(w); }
    mixer.update(0);
  }

  // Turning the head on top of the clip: yaw about world up, pitch about the body's right, split neck 40 / head 60.
  const _q = new THREE.Quaternion(), _pq = new THREE.Quaternion(), _ax = new THREE.Vector3();
  function turn(bone, axisWorld, ang) {
    if (!bone || !ang) return;
    bone.parent.getWorldQuaternion(_pq).invert();
    _ax.copy(axisWorld).applyQuaternion(_pq).normalize();
    bone.quaternion.premultiply(_q.setFromAxisAngle(_ax, ang));
  }
  const UP = new THREE.Vector3(0, 1, 0), RIGHT = new THREE.Vector3();
  /** yaw > 0 turns the face to the person's left; pitch > 0 looks down. Call after sample()/at(). */
  function look(yaw = 0, pitch = 0) {
    model.updateMatrixWorld(true);
    RIGHT.set(-1, 0, 0).applyQuaternion(root.getWorldQuaternion(_pq)); // the model faces +z, so its right is -x
    turn(bones.neck_01, UP, yaw * 0.4); turn(bones.Head, UP, yaw * 0.6);
    turn(bones.neck_01, RIGHT, pitch * 0.4); turn(bones.Head, RIGHT, pitch * 0.6);
  }

  // A schedule of acts on the film's clock. Each act: { t0, t1, clip, loop?, speed?, from?: [x,y,z], to?: [x,y,z],
  // yaw?: [a, b] (radians about +y; 0 faces +z), fade? (crossfade from the previous act, s) }. Position and yaw ease
  // from `from`/`yaw[0]` to `to`/`yaw[1]` over the act (linear for walks so the feet keep pace: set ease: "none").
  const acts = [];
  function act(list) { acts.push(...list); acts.sort((a, b) => a.t0 - b.t0); }
  const clipTime = (a, t) => {
    const local = (Math.min(Math.max(t, a.t0), a.t1) - a.t0) * (a.speed || 1) + (a.offset || 0);
    return a.loop ? local % clips[a.clip].duration : Math.min(local, clips[a.clip].duration - 1e-4);
  };
  const place = (a, t) => {
    const u = a.t1 > a.t0 ? clamp01((t - a.t0) / (a.t1 - a.t0)) : 1, e = a.ease === "none" ? u : smooth(u);
    const f = a.from || a.to, to = a.to || a.from;
    const p = f ? [lerp(f[0], to[0], e), lerp(f[1], to[1], e), lerp(f[2], to[2], e)] : null;
    const y = a.yaw ? angLerp(a.yaw[0], a.yaw[1] ?? a.yaw[0], e) : null;
    return { p, y };
  };
  /** Pose and place the person for film time t from the schedule. */
  function at(t) {
    if (!acts.length) return;
    let i = 0; while (i + 1 < acts.length && acts[i + 1].t0 <= t) i++;
    const cur = acts[i], prev = acts[i - 1];
    const w = prev && t >= cur.t0 ? smooth((t - cur.t0) / (cur.fade ?? 0.3)) : 1;
    if (prev && w < 1 && prev.clip !== cur.clip) sample([[cur.clip, clipTime(cur, t), w], [prev.clip, clipTime(prev, t), 1 - w]]);
    else sample([[cur.clip, clipTime(cur, t), 1]]);
    const c = place(cur, t), pv = prev && w < 1 ? place(prev, t) : null;
    if (c.p) root.position.set(...(pv?.p ? c.p.map((v, k) => lerp(pv.p[k], v, w)) : c.p));
    if (c.y !== null) root.rotation.y = pv && pv.y !== null ? angLerp(pv.y, c.y, w) : c.y;
  }

  const _hv = new THREE.Vector3();
  /** World position of a hand ("R" or "L"). */
  function handPos(side = "R", out = _hv) { model.updateMatrixWorld(true); return bones[side === "R" ? "hand_r" : "hand_l"].getWorldPosition(out); }
  function headPos(out = new THREE.Vector3()) { model.updateMatrixWorld(true); return bones.Head.getWorldPosition(out); }

  sample([["Idle_Loop", 0, 1]]);
  /** Set the expression for time t (call each frame after posing). */
  function express(e = {}, t = 0) { Object.assign(expr, { smile: 0, sleepy: 0, strain: 0, talk: 0 }, e, { t }); updateFace(); }
  express({}, 0);
  return { root, model, mats, bones, sample, look, act, at, handPos, headPos, duration, express };
}
