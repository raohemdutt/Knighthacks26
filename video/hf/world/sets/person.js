// A stylised soft person: rounded capsule body, arms and legs on joints, a head with a calm real face
// (eyes, brows, nose, a soft mouth, hair, ears). Muted tints of the palette. Faces -z by default; the root sits at the
// pelvis. pose(p) sets joint angles; look(yaw, pitch) turns the head. Units: 1 = 100 mm.
import * as THREE from "three";
import { tint, flat } from "./common.js";

const SKIN = 0xd9bfae, TOP = 0x8f9cb0, PANTS = 0x5d6573, SHOE = 0xe9edf2, HAIR = 0x4a3f38, FEATURE = 0x1b2433, LIP = 0xa8766a;

const cap = (r, len, mat) => {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.001, len - 2 * r), 6, 14), mat);
  m.position.y = -len / 2; m.castShadow = true; m.receiveShadow = true; return m;
};

export function makePerson() {
  const skin = tint(SKIN, { roughness: 0.75 }), top = tint(TOP, { roughness: 0.9 }), pants = tint(PANTS, { roughness: 0.9 });
  const shoe = tint(SHOE, { roughness: 0.7 }), hair = tint(HAIR, { roughness: 0.85 }), feature = flat({ color: FEATURE }), lip = tint(LIP, { roughness: 0.6 });
  const root = new THREE.Group(); root.name = "person";
  const J = {};
  const joint = (name, parent, x, y, z) => { const g = new THREE.Group(); g.position.set(x, y, z); parent.add(g); J[name] = g; return g; };

  // Torso: a soft rounded capsule, a little wider than deep, from the pelvis to the shoulders.
  const spine = joint("spine", root, 0, 0, 0);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(1.35, 2.7, 8, 20), top); torso.scale.set(1.32, 1, 0.78); torso.position.y = 2.6; torso.castShadow = true; torso.receiveShadow = true; spine.add(torso);
  const hips = new THREE.Mesh(new THREE.SphereGeometry(1.3, 20, 14), pants); hips.scale.set(1.38, 0.8, 0.95); hips.position.y = 0.2; hips.castShadow = true; spine.add(hips);
  // Neck and head.
  const neckJ = joint("neck", spine, 0, 5.15, 0.05);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.9, 16), skin); neck.position.y = 0.35; neckJ.add(neck);
  const head = joint("head", neckJ, 0, 1.55, 0);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(1, 36, 24), skin); skull.scale.set(0.9, 1.08, 0.98); skull.castShadow = true; head.add(skull);
  // Hair: a cap over the top and back, leaving the face open.
  const hairCap = new THREE.Mesh(new THREE.SphereGeometry(1.05, 36, 20, 0, Math.PI * 2, 0, Math.PI * 0.4), hair);
  hairCap.scale.set(0.93, 1.06, 1.0); hairCap.rotation.x = 0.12; hairCap.position.set(0, 0.04, 0.03); hairCap.castShadow = true; head.add(hairCap);
  // The back of the hair, down to the nape (+z is the back of the head).
  const hairBack = new THREE.Mesh(new THREE.SphereGeometry(1.045, 36, 20, Math.PI / 2 - 1.75, 3.5, Math.PI * 0.18, Math.PI * 0.6), hair);
  hairBack.scale.set(0.93, 1.06, 1.0); hairBack.position.set(0, 0.04, 0.03); hairBack.castShadow = true; head.add(hairBack);
  // Face (on -z): eyes, brows, a small nose, a soft smile, ears.
  const eyeGeo = new THREE.SphereGeometry(0.11, 16, 10);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeo, feature); eye.scale.set(1, 1.25, 0.5); eye.position.set(s * 0.33, 0.12, -0.9); head.add(eye);
    const brow = new THREE.Mesh(new THREE.CapsuleGeometry(0.035, 0.22, 4, 8), hair); brow.rotation.z = Math.PI / 2 + s * 0.12; brow.position.set(s * 0.34, 0.36, -0.87); head.add(brow);
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.2, 12, 10), skin); ear.scale.set(0.5, 1, 0.8); ear.position.set(s * 0.89, 0.02, 0.02); head.add(ear);
  }
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.13, 14, 10), skin); nose.scale.set(0.9, 1.1, 1); nose.position.set(0, -0.08, -0.98); head.add(nose);
  const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.035, 8, 24, Math.PI * 0.8), lip); mouth.rotation.z = Math.PI + Math.PI * 0.1; mouth.position.set(0, -0.34, -0.9); mouth.rotation.x = -0.15; head.add(mouth);

  // Arms: shoulder -> upper arm (3.0) -> elbow -> forearm (2.7) -> hand.
  for (const s of [-1, 1]) {
    const side = s < 0 ? "L" : "R"; // R is +x (the person faces -z)
    const sh = joint("shoulder" + side, spine, s * 1.95, 4.85, 0.05);
    const shoulderBall = new THREE.Mesh(new THREE.SphereGeometry(0.54, 16, 12), top); sh.add(shoulderBall);
    sh.add(cap(0.52, 3.0, top));
    const el = joint("elbow" + side, sh, 0, -3.0, 0);
    el.add(cap(0.44, 2.7, top));
    const wrist = joint("hand" + side, el, 0, -2.7, 0);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 12), skin); hand.scale.set(0.8, 1.05, 0.95); hand.position.y = -0.25; hand.castShadow = true; wrist.add(hand);
    // Legs: hip -> thigh (4.3) -> knee -> shin (4.2) -> foot.
    const hip = joint("hip" + side, root, s * 0.85, -0.1, 0);
    hip.add(cap(0.72, 4.3, pants));
    const kn = joint("knee" + side, hip, 0, -4.3, 0);
    kn.add(cap(0.56, 4.2, pants));
    const ft = joint("foot" + side, kn, 0, -4.2, 0);
    const foot = new THREE.Mesh(new THREE.CapsuleGeometry(0.45, 1.2, 4, 10), shoe); foot.rotation.x = Math.PI / 2; foot.scale.set(1.1, 1, 0.75); foot.position.set(0, -0.3, -0.55); foot.castShadow = true; ft.add(foot);
  }

  const base = { spineX: 0 };
  /** Joint angles (radians, x = forward pitch). Unlisted joints keep their value. */
  function pose(p) {
    for (const k in p) {
      const [name, axis] = k.split("."); if (J[name]) J[name].rotation[axis] = p[k];
    }
  }
  /** Turn the head (and a little of the neck): yaw in radians about +y (negative turns the face toward +x); pitch > 0 looks down. */
  function look(yaw, pitch = 0) { J.neck.rotation.y = yaw * 0.35; J.head.rotation.y = yaw * 0.53; J.head.rotation.x = -pitch; J.spine.rotation.y = yaw * 0.12; }
  const handPos = (side, out = new THREE.Vector3()) => { J["hand" + side].updateWorldMatrix(true, false); return J["hand" + side].localToWorld(out.set(0, -0.3, 0)); };
  return { root, J, pose, look, handPos, base, mats: [skin, top, pants, shoe, hair, feature, lip] };
}

/** Seated at a desk, hands toward the keyboard (person faces -z). */
export const SEATED = {
  "spine.x": -0.06,
  "hipL.x": Math.PI / 2 - 0.05, "hipR.x": Math.PI / 2 - 0.05, "hipL.z": -0.06, "hipR.z": 0.06,
  "kneeL.x": -Math.PI / 2 + 0.12, "kneeR.x": -Math.PI / 2 + 0.12, "footL.x": -0.12, "footR.x": -0.12,
  "shoulderL.x": 0.82, "shoulderR.x": 0.82, "shoulderL.z": 0.12, "shoulderR.z": -0.12,
  "elbowL.x": 0.95, "elbowR.x": 0.95, "elbowL.z": -0.2, "elbowR.z": 0.2,
};
/** Standing, the right hand holding something at chest height in front. */
export const STANDING = {
  "spine.x": 0, "hipL.x": 0, "hipR.x": 0, "hipL.z": -0.04, "hipR.z": 0.04, "kneeL.x": 0, "kneeR.x": 0, "footL.x": 0, "footR.x": 0,
  "shoulderL.x": 0.08, "shoulderL.z": 0.1, "elbowL.x": 0.15,
  "shoulderR.x": 0.55, "shoulderR.z": -0.12, "elbowR.x": 1.35, "elbowR.z": 0.15,
};
