// The dorm room set, around the desk the device sits on (the device stays at the origin). state.room rises/fades it in,
// state.person shows the person, state.personLook turns their head toward the device. Units: 1 = 100 mm.
import * as THREE from "three";
import { SHADE, INK, ACCENT, MUTED, FLOOR_Y, hash, clamp01, lerp, softTexture, FONT, flat, matte, tint, between, fadingSet, RoundedBoxGeometry } from "./common.js";
import { makePerson, SEATED } from "./person.js";

const WOOD = 0xe0d6c8, WALL = 0xe1e6ed, FABRIC = 0xa9b4c5, BOOKS = [0x8f9cb0, 0xc9b7a2, 0x9fb09c, 0xd2c2b0, 0x566176, 0xb4a6b8, 0xa7b8c4, 0xd9cbb9];

function rbox(w, h, d, mat, r = 0.12, seg = 3) {
  const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, Math.min(w, h, d) / 2 - 1e-3)), mat);
  m.castShadow = true; m.receiveShadow = true; return m;
}
function screenTexture() {
  return softTexture((ctx, w, h) => {
    ctx.fillStyle = "#eef1f5"; ctx.fillRect(0, 0, w, h);
    const rr = (x, y, ww, hh, r, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.roundRect(x, y, ww, hh, r); ctx.fill(); };
    rr(28, 28, 180, h - 56, 22, "#e2e7ee");
    for (let i = 0; i < 5; i++) rr(52, 70 + i * 54, 130, 18, 9, i === 1 ? "#456cb3" : "#c8d0db");
    ctx.fillStyle = "#1b2433"; ctx.font = `700 40px ${FONT}`; ctx.fillText("Today", 250, 86);
    for (let d = 0; d < 5; d++) {
      const x = 250 + d * 150; ctx.fillStyle = "#566176"; ctx.font = `600 20px ${FONT}`; ctx.fillText(["Mon", "Tue", "Wed", "Thu", "Fri"][d], x, 130);
      for (let k = 0; k < 3; k++) { if (hash(d, k + 50) < 0.3) continue; rr(x, 150 + k * 90 + hash(d, k) * 30, 130, 56 + hash(d, k + 9) * 30, 12, d === 2 && k === 1 ? "#f0c9b4" : "#dde3eb"); }
    }
  }, 1024, 576);
}
function rugTexture() {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#d3dae3"; ctx.beginPath(); ctx.roundRect(8, 8, w - 16, h - 16, 60); ctx.fill();
    ctx.strokeStyle = "#e9edf2"; ctx.lineWidth = 10; ctx.beginPath(); ctx.roundRect(46, 46, w - 92, h - 92, 40); ctx.stroke();
    ctx.strokeStyle = "#c3cbd6"; ctx.lineWidth = 4; ctx.beginPath(); ctx.roundRect(74, 74, w - 148, h - 148, 28); ctx.stroke();
  }, 512, 420);
}

export function buildRoom(scene) {
  const root = new THREE.Group(); root.name = "room"; scene.add(root);
  const set = fadingSet(root);
  const F = FLOOR_Y, TOP = -1.002; // desk top a hair under the device's base (the studio shadow plane sits at -1.0)
  const wood = matte({ color: WOOD, roughness: 0.75 }), white = matte({ roughness: 0.8 }), wall = matte({ color: WALL, roughness: 0.95 });
  const ink = matte({ color: 0x3a404a, roughness: 0.5 }), fabric = tint(FABRIC, { roughness: 1 }), soft = matte({ color: 0xd3dae3, roughness: 0.9 });
  const at = (m, x, y, z) => { m.position.set(x, y, z); return m; };

  // Floor shadows (the floor itself is the page colour, so only the shadow shows).
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 50), new THREE.ShadowMaterial({ opacity: 0.2, color: SHADE, depthWrite: false }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(0, F, 2); floor.receiveShadow = true; floor.renderOrder = -5; set.add(floor);
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(11, 9), flat({ map: rugTexture(), transparent: true, depthWrite: false }));
  rug.rotation.x = -Math.PI / 2; rug.position.set(-4.2, F + 0.01, 6.6); rug.renderOrder = -4; set.add(rug);

  // Back wall with a window on the right, a shelf with books on the left.
  const backWall = at(rbox(40, 26, 0.4, wall, 0.1, 2), -6, F + 13, -4.4); backWall.castShadow = false; set.add(backWall);
  const win = new THREE.Group(); win.position.set(-14.6, 4.4, -4.15);
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(7.6, 7.6), flat({ color: 0xf6f8fb })); glass.position.z = -0.02; win.add(glass);
  for (const [w, h, x, y] of [[8.4, 0.4, 0, 4.0], [8.4, 0.4, 0, -4.0], [0.4, 8.4, 4.0, 0], [0.4, 8.4, -4.0, 0], [7.6, 0.18, 0, 0], [0.18, 7.6, 0, 0]]) win.add(at(rbox(w, h, 0.4, white, 0.08, 2), x, y, 0.1));
  win.add(at(rbox(9.2, 0.25, 0.9, white, 0.08, 2), 0, -4.3, 0.4));
  set.add(win);
  const shelf = new THREE.Group(); shelf.position.set(1.2, 5.6, -3.4);
  shelf.add(at(rbox(8.2, 0.26, 1.7, wood, 0.08, 2), 0, 0, 0));
  for (const x of [-3.3, 3.3]) shelf.add(at(rbox(0.2, 0.9, 1.2, white, 0.05, 2), x, -0.55, -0.1));
  let bx = -3.7;
  for (let i = 0; i < 9; i++) {
    const w = 0.34 + hash(i, 61) * 0.3, h = 1.5 + hash(i, 62) * 0.9, d = 1.1 + hash(i, 63) * 0.3;
    const b = at(rbox(w, h, d, tint(BOOKS[i % BOOKS.length], { roughness: 0.85 }), 0.06, 2), bx + w / 2, 0.13 + h / 2, 0.05);
    if (i === 8) { b.rotation.z = -0.32; b.position.x += 0.35; b.position.y -= 0.12; }
    shelf.add(b); bx += w + 0.04;
  }
  const pot = at(new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.34, 0.75, 20), white), 2.4, 0.5, 0.1); pot.castShadow = true; shelf.add(pot);
  for (let k = 0; k < 5; k++) { const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.42, 14, 10), tint(0x9fb09c, { roughness: 0.9 })); leaf.scale.set(0.6, 1.2, 0.6); leaf.position.set(2.4 + (hash(k, 64) - 0.5) * 0.6, 1.25 + hash(k, 65) * 0.3, 0.1 + (hash(k, 66) - 0.5) * 0.5); leaf.rotation.z = (hash(k, 67) - 0.5) * 1.2; leaf.castShadow = true; shelf.add(leaf); }
  set.add(shelf);

  // The bed edge, right and behind.
  const bed = new THREE.Group(); bed.position.set(-15.2, F, -0.2);
  bed.add(at(rbox(10.2, 1.6, 7.8, wood, 0.15), 0, 1.6, 0));
  bed.add(at(rbox(9.9, 1.9, 7.5, white, 0.45, 4), 0, 3.3, 0));
  bed.add(at(rbox(10.1, 0.5, 5.0, fabric, 0.25, 3), 0, 4.3, 1.3));
  bed.add(at(rbox(4.0, 0.9, 1.9, soft, 0.42, 4), 1.6, 4.65, -2.6));
  bed.add(at(rbox(10.2, 4.6, 0.45, wood, 0.15), 0, 2.3, -3.85));
  set.add(bed);

  // The desk: top, two side panels.
  const desk = new THREE.Group();
  desk.add(at(rbox(14.6, 0.32, 7.2, wood, 0.1, 2), -2.6, TOP - 0.16, -0.35));
  for (const x of [-9.4, 4.2]) desk.add(at(rbox(0.36, TOP - 0.32 - F, 6.6, white, 0.08, 2), x, (TOP - 0.32 + F) / 2, -0.35));
  set.add(desk);

  // Monitor (with a webcam clipped on top), keyboard, mouse, a mug, a notebook, the lamp.
  const mon = new THREE.Group(); mon.position.set(-4.5, TOP, -2.4);
  mon.add(at(rbox(2.4, 0.12, 1.6, white, 0.05, 2), 0, 0.06, 0));
  mon.add(at(rbox(0.38, 2.4, 0.3, white, 0.1, 2), 0, 1.3, -0.15));
  mon.add(at(rbox(6.3, 3.75, 0.28, white, 0.1, 3), 0, 3.3, 0.02));
  const scr = new THREE.Mesh(new THREE.PlaneGeometry(5.95, 3.38), flat({ map: screenTexture() })); scr.position.set(0, 3.32, 0.165); mon.add(scr);
  const mcam = at(rbox(1.0, 0.32, 0.42, ink, 0.12, 3), 0, 5.35, 0.08); mon.add(mcam);
  const mlens = new THREE.Mesh(new THREE.CircleGeometry(0.07, 16), flat({ color: 0x15181d })); mlens.position.set(0, 5.35, 0.295); mon.add(mlens);
  mon.add(at(rbox(0.5, 0.36, 0.16, ink, 0.05, 2), 0, 5.12, -0.12));
  set.add(mon);
  const kb = new THREE.Group(); kb.position.set(-4.5, TOP, 1.35);
  kb.add(at(rbox(4.5, 0.16, 1.5, white, 0.06, 2), 0, 0.08, 0));
  const keys = new THREE.InstancedMesh(new RoundedBoxGeometry(0.24, 0.08, 0.24, 1, 0.04), matte({ color: 0xedf1f6, roughness: 0.7 }), 64);
  const km = new THREE.Matrix4(); let ki = 0;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 16; c++) { km.makeTranslation(-1.98 + c * 0.264, 0.2, -0.5 + r * 0.33); keys.setMatrixAt(ki++, km); }
  keys.castShadow = false; keys.receiveShadow = true; kb.add(keys);
  set.add(kb);
  const mouse = new THREE.Mesh(new THREE.SphereGeometry(0.4, 16, 12), white); mouse.scale.set(0.75, 0.35, 1.1); mouse.position.set(-1.4, TOP + 0.12, 1.5); mouse.castShadow = true; set.add(mouse);
  const mug = new THREE.Group(); mug.position.set(2.5, TOP, 1.9);
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.38, 0.95, 24), tint(0xc9b7a2, { roughness: 0.6 })); cup.position.y = 0.48; cup.castShadow = true; mug.add(cup);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.06, 8, 16), cup.material); handle.position.set(0.44, 0.5, 0); mug.add(handle);
  set.add(mug);
  const note = at(rbox(2.1, 0.14, 2.8, tint(0x8f9cb0, { roughness: 0.9 }), 0.05, 2), -8.0, TOP + 0.07, 1.2); note.rotation.y = 0.18; set.add(note);
  const lamp = new THREE.Group(); lamp.position.set(-8.3, TOP, -2.5);
  const lbase = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 0.2, 28), white); lbase.position.y = 0.1; lbase.castShadow = true; lamp.add(lbase);
  const j1 = new THREE.Vector3(0, 0.2, 0), j2 = new THREE.Vector3(0.3, 4.0, -0.4), j3 = new THREE.Vector3(-0.9, 4.7, 1.1);
  for (const [a, b] of [[j1, j2], [j2, j3]]) { const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, a.distanceTo(b), 10), white); arm.castShadow = true; lamp.add(between(arm, a, b)); }
  const joint = new THREE.Mesh(new THREE.SphereGeometry(0.17, 12, 10), white); joint.position.copy(j2); lamp.add(joint);
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.85, 1.2, 28, 1, true), matte({ roughness: 0.8, side: THREE.DoubleSide })); shade.position.copy(j3); shade.position.y -= 0.35; shade.rotation.z = 0.3; shade.castShadow = true; lamp.add(shade);
  const bulb = new THREE.Mesh(new THREE.CircleGeometry(0.62, 24), flat({ color: 0xfbf6ef })); bulb.rotation.x = Math.PI / 2; bulb.position.copy(shade.position); bulb.position.y -= 0.5; bulb.position.x -= 0.15; lamp.add(bulb);
  set.add(lamp);

  // The chair.
  const chair = new THREE.Group(); chair.position.set(-4.5, F, 6.6);
  chair.add(at(rbox(4.4, 0.6, 4.2, fabric, 0.28, 3), 0, 4.3, 0));
  const back = at(rbox(4.2, 4.6, 0.55, fabric, 0.28, 3), 0, 7.6, 2.25); back.rotation.x = 0.1; chair.add(back);
  for (const s of [-1, 1]) chair.add(at(rbox(0.25, 3.0, 0.3, ink, 0.1, 2), s * 1.4, 5.9, 2.1));
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 3.4, 12), ink); post.position.y = 2.3; post.castShadow = true; chair.add(post);
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2 + 0.3, leg = rbox(2.3, 0.22, 0.32, ink, 0.1, 2);
    leg.position.set(Math.cos(a) * 1.1, 0.62, Math.sin(a) * 1.1); leg.rotation.y = -a; chair.add(leg);
    const wheel = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 8), ink); wheel.position.set(Math.cos(a) * 2.15, 0.26, Math.sin(a) * 2.15); chair.add(wheel);
  }
  set.add(chair);

  // The person, seated, facing the monitor.
  const person = makePerson(); person.pose(SEATED);
  const pHolder = new THREE.Group(); pHolder.position.set(-4.5, -3.2, 6.15); pHolder.add(person.root); root.add(pHolder);
  for (const m of person.mats) { m.transparent = true; m.userData.baseOpacity = m.opacity; }
  const HEAD = new THREE.Vector3(-4.5, -3.2 + 6.7, 6.15);
  const toDevice = new THREE.Vector3(0, -0.5, 0).sub(HEAD), toMonitor = new THREE.Vector3(-4.5, 2.3, -2.4).sub(HEAD);
  const yawOf = (v) => -Math.atan2(v.x, -v.z), pitchOf = (v) => Math.atan2(-v.y, Math.hypot(v.x, v.z));

  const P = { desk: [-2.6, TOP, -0.35], monitor: [-4.5, 0.9, -2.2], person: HEAD.toArray(), keyboard: [-4.5, TOP + 0.2, 1.35] };
  function update(state, t) {
    set.update(state.room || 0);
    const pv = clamp01(Math.min(state.room || 0, state.person || 0));
    pHolder.visible = pv > 0.001;
    if (pHolder.visible) {
      for (const m of person.mats) { m.opacity = m.userData.baseOpacity * pv; m.depthWrite = pv > 0.6; }
      pHolder.position.y = -3.2 - (1 - pv) * 0.6;
      const l = clamp01(state.personLook || 0), e = l * l * (3 - 2 * l);
      // A breath (pure function of t), and the head turning from the monitor to the device.
      person.J.spine.position.y = 0.04 * Math.sin(t * 1.4);
      person.look(lerp(yawOf(toMonitor), yawOf(toDevice), e), lerp(pitchOf(toMonitor), pitchOf(toDevice), e));
    }
  }
  return { root, update, P, person };
}
