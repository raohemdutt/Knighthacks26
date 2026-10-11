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
/** Load the glb once (await before building the sets). */
export async function loadHuman() {
  if (!cached) cached = await new GLTFLoader().loadAsync("assets/people/UAL1_Standard.glb");
  return cached;
}

const BODY = 0x9aa8bd, JOINTS = 0x5d6878; // a soft blue-grey "clothes" tone, darker joints (the site's muted ink)
const smooth = (u) => { const x = clamp01(u); return x * x * (3 - 2 * x); };
const angLerp = (a, b, u) => a + (((b - a + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI) * u;

export function makeHuman(gltf, { body = BODY, joints = JOINTS } = {}) {
  const model = SkeletonUtils.clone(gltf.scene);
  model.scale.setScalar(10);
  const root = new THREE.Group(); root.add(model);
  const mats = [];
  const main = matte({ color: body, roughness: 0.78 }), jm = matte({ color: joints, roughness: 0.6 });
  mats.push(main, jm);
  model.traverse((o) => {
    if (o.isMesh) {
      o.material = /joint/i.test(o.material?.name || "") ? jm : main;
      o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
    }
  });
  const bones = {}; model.traverse((o) => { if (o.isBone) bones[o.name] = o; });
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
  return { root, model, mats, bones, sample, look, act, at, handPos, headPos, duration };
}
