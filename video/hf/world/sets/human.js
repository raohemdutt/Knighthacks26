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
  dominick: { shirt: 0xc98b5e, pants: 0x2f3a4a, shoes: 0xe9edf2, skin: 0xd8ad8a, hair: "Hair_SimpleParted", hairColor: 0x3a2a20 },
  xander: { shirt: 0xe5d54a, pants: 0x3a4250, shoes: 0x4a3a2a, skin: 0xe7c3a5, hair: "Hair_SimpleParted", hairColor: 0x7a5a3a }, // hi-vis
  trainee: { shirt: 0xd9e05a, pants: 0x4a4f5a, shoes: 0x3a3a3a, skin: 0x9a6a4a, hair: "Hair_SimpleParted", hairColor: 0x1e1a18 },
};
const REGION = (name) => /Head|neck/.test(name) ? "skin" : /hand|index|middle|pinky|ring|thumb|lowerarm/.test(name) ? "skin"
  : /foot|ball/.test(name) ? "shoes" : /thigh|calf|pelvis/.test(name) ? "pants" : "shirt";
const smooth = (u) => { const x = clamp01(u); return x * x * (3 - 2 * x); };
const angLerp = (a, b, u) => a + (((b - a + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI) * u;

export function makeHuman(gltf, outfit = LOOKS.hemdutt) {
  const model = SkeletonUtils.clone(gltf.scene);
  model.scale.setScalar(10);
  // The mesh faces +z in its own space (the toes reach z = +0.21 m, the heels -0.16 m), so no turn is needed.
  const root = new THREE.Group(); root.rotation.order = "YXZ"; root.add(model);
  const mats = [];
  // One material, coloured by region: each vertex carries a one-hot weight (skin, shirt, pants, shoes) from the bone
  // that moves it most, and the fragment shader takes the strongest region, so the edges (the collar, the sleeves,
  // the shoes) are crisp instead of a colour fading across the triangles that straddle them.
  const main = matte({ color: 0xffffff, roughness: 0.8 });
  const RC = { uSkin: { value: new THREE.Color(outfit.skin) }, uShirt: { value: new THREE.Color(outfit.shirt) }, uPants: { value: new THREE.Color(outfit.pants) }, uShoes: { value: new THREE.Color(outfit.shoes) } };
  main.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, RC);
    sh.vertexShader = `attribute vec4 region;
varying vec4 vRegion;
` + sh.vertexShader.replace("#include <begin_vertex>", `#include <begin_vertex>
  vRegion = region;`);
    sh.fragmentShader = `varying vec4 vRegion;
uniform vec3 uSkin; uniform vec3 uShirt; uniform vec3 uPants; uniform vec3 uShoes;
` + sh.fragmentShader.replace("#include <color_fragment>", `#include <color_fragment>
  { vec3 rc = uShirt; float m = vRegion.y;
    if (vRegion.x > m) { m = vRegion.x; rc = uSkin; }
    if (vRegion.z > m) { m = vRegion.z; rc = uPants; }
    if (vRegion.w > m) { m = vRegion.w; rc = uShoes; }
    diffuseColor.rgb *= rc; }`);
  };
  main.customProgramCacheKey = () => "atlas-person-regions";
  mats.push(main);
  const bones = {}; model.traverse((o) => { if (o.isBone) bones[o.name] = o; });
  const C = { shirt: new THREE.Color(outfit.shirt), pants: new THREE.Color(outfit.pants), shoes: new THREE.Color(outfit.shoes), skin: new THREE.Color(outfit.skin) };
  let skinned = null; const skins = [];
  model.traverse((o) => {
    if (!o.isMesh) return;
    o.material = main; o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
    if (o.isSkinnedMesh) {
      skinned = o; skins.push(o);
      const g = o.geometry = o.geometry.clone(), si = g.attributes.skinIndex, sw = g.attributes.skinWeight, n = si.count;
      const reg = new Float32Array(n * 4), names = o.skeleton.bones.map((b) => b.name), SLOT = { skin: 0, shirt: 1, pants: 2, shoes: 3 };
      for (let v = 0; v < n; v++) {
        let best = 0, bw = -1; for (let k = 0; k < 4; k++) { const w = sw.getComponent(v, k); if (w > bw) { bw = w; best = si.getComponent(v, k); } }
        reg[v * 4 + SLOT[REGION(names[best])]] = 1;
      }
      g.setAttribute("region", new THREE.BufferAttribute(reg, 4));
    }
  });

  // A soft face on the head bone: eyes with lids, a mouth that smiles, flattens or talks. Placed in BIND space (the
  // mesh's own coordinates, front +z): the skeleton's rest pose is not the bind pose, so measuring the posed head puts
  // things in the wrong place. The head is every vertex above the head bone's bind position. A head frame built in
  // bind space hangs from the head bone through the bone's inverse-bind matrix, exactly as the skin follows it, so the
  // face and the hair move with the head in every clip.
  const hi = skinned ? skinned.skeleton.bones.indexOf(bones.Head) : -1;
  const bindHead = hi >= 0 ? skinned.skeleton.boneInverses[hi].clone().invert() : new THREE.Matrix4();
  const headY = new THREE.Vector3().setFromMatrixPosition(bindHead).y;
  const headBox = new THREE.Box3(), vtx = new THREE.Vector3();
  for (const sk of skins) { // the mannequin is several skinned meshes (one per primitive): measure the head across all
    const pos = sk.geometry.attributes.position;
    for (let v = 0; v < pos.count; v++) { vtx.fromBufferAttribute(pos, v); if (vtx.y > headY) headBox.expandByPoint(vtx); }
  }
  const hc = headBox.getCenter(new THREE.Vector3()), hs = headBox.getSize(new THREE.Vector3()), hcy = 0;
  const headFrame = new THREE.Group(); // bind space: origin at the head's centre, +z forward, +y up
  // In the head frame the face is the box's front (max z); its features are built along local -z, so it turns round.
  const face = new THREE.Group();
  face.position.set(0, -hs.y * 0.06, headBox.max.z - hc.z - 0.004); face.rotation.y = Math.PI;
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
  headFrame.add(face);

  // Hair: a soft cap fitted to the head (top and back, a little fringe at the front), in the person's hair colour.
  // Styles: "short" (a close cap), "long" (down to the neck at the back), "buzzed" (thin). Rest pose, mesh space.
  const style = outfit.hair === "Hair_Long" ? "long" : outfit.hair === "Hair_Buzzed" ? "buzzed" : "short";
  const cap = new THREE.Group(); cap.position.set(0, hcy, 0);
  const r = Math.max(hs.x, hs.z) / 2, grow = style === "buzzed" ? 1.1 : 1.12; // loose enough that the head never pokes through
  const dome = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20, 0, Math.PI * 2, 0, style === "long" ? Math.PI * 0.62 : Math.PI * 0.5), hairMat);
  dome.scale.set(hs.x / 2 * grow, hs.y / 2 * (style === "buzzed" ? 1.1 : 1.14), hs.z / 2 * grow); dome.position.y = hs.y * 0.06;
  dome.rotation.x = -0.28; // tilt back (toward -z): more hair over the back of the head, the forehead stays clear
  cap.add(dome);
  if (style !== "buzzed") {
    const fringe = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.35), hairMat);
    fringe.scale.set(hs.x / 2 * 1.04, hs.y * 0.22, hs.z / 2 * 0.9); fringe.position.set(0, hs.y * 0.3, r * 0.18); fringe.rotation.x = 0.5;
    cap.add(fringe);
  }
  if (style === "long") { const back = new THREE.Mesh(new THREE.CapsuleGeometry(r * 0.75, hs.y * 0.45, 6, 16), hairMat); back.position.set(0, -hs.y * 0.18, -r * 0.55); cap.add(back); }
  cap.traverse((o) => { if (o.isMesh) { o.castShadow = true; } });
  // A natural haircut: Quaternius' Universal Base Characters hair, modelled for this same mannequin in its bind space
  // (side part by default), so it fits the head exactly; it rides in the head frame (origin at the head's centre, hence
  // the -hc offset). The soft cap above is kept only as a fallback if the hair model did not load.
  const hairSrc = gltf.hair && gltf.hair[outfit.hair];
  if (hairSrc) {
    const hair = hairSrc.clone(true);
    hairMat.side = THREE.DoubleSide;
    hair.traverse((o) => { if (o.isMesh) { o.material = hairMat; o.castShadow = true; o.receiveShadow = true; } });
    hair.position.set(-hc.x, -hc.y, -hc.z);
    // The mannequin's head is a little larger than the base character's, so the hair is scaled up about the head's
    // centre (and lifted a touch) to sit on the scalp instead of inside it.
    const fit = new THREE.Group(); fit.scale.set(1.16, 1.12, 1.18); fit.position.set(0, 0.012, -0.004); fit.add(hair);
    headFrame.add(fit);
  } else {
    headFrame.add(cap);
  }
  // Bone-local transform = inverse bind of the head bone x (head centre in bind space).
  if (hi >= 0) {
    const local = skinned.skeleton.boneInverses[hi].clone().multiply(new THREE.Matrix4().makeTranslation(hc.x, hc.y, hc.z));
    local.decompose(headFrame.position, headFrame.quaternion, headFrame.scale);
    bones.Head.add(headFrame);
  }
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
  // look() turns the neck and head ON TOP of their current rotation; a clip that does not animate them would never
  // reset them, so the turn would pile up frame after frame (the head spinning in the kitchen). Reset them first.
  const restQ = ["neck_01", "Head"].filter((n) => bones[n]).map((n) => [bones[n], bones[n].quaternion.clone()]);
  function sample(list) {
    for (const [b, q] of restQ) b.quaternion.copy(q);
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
  // yaw?: [a, b] (radians about +y; 0 faces +z), tip?: [a, b] (tipped back, radians; pi/2 = lying on the back), fade? (crossfade from the previous act, s) }. Position and yaw ease
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
    const k = a.tip ? lerp(a.tip[0], a.tip[1] ?? a.tip[0], e) : 0; // tipped back (radians): pi/2 lies on the back
    return { p, y, k };
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
    root.rotation.x = -(pv ? lerp(pv.k, c.k, w) : c.k); // about the body's own left-right axis (order YXZ)
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
