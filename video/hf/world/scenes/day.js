// Scene "day": Your day. The person's morning in the dorm. We start tight on Atlas on its shelf (the room rises around
// it), and pull back to the whole room while the person sits up on the edge of the bed, stands, walks round the chair
// and sits down at the desk (Quaternius clips: Sitting_Idle, Sitting_Exit, Walk_Loop, Sitting_Enter). Over the
// shoulder while Atlas briefs them, they turn to it; on "breaks" they stand up for a moment and sit back down.
// Camera + world choreography on the ROOT timeline (absolute seconds). See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.

const KEYS = ["x", "y", "z", "tx", "ty", "tz", "fov"];
const cr = (p0, p1, p2, p3, t) => { const t2 = t * t, t3 = t2 * t; return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3); };
/** One continuous camera move through several shots (Catmull-Rom through every key, one ease over the whole path). */
function camPath(tl, world, at, dur, keys, ease) {
  const cam = world.cam, p = { u: 0 }, n = keys.length - 1;
  tl.fromTo(p, { u: 0 }, {
    u: 1, duration: dur, ease, immediateRender: false,
    onUpdate: () => {
      const s = Math.min(Math.max(p.u, 0), 1) * n, i = Math.min(n - 1, Math.floor(s)), f = s - i;
      const k0 = keys[Math.max(0, i - 1)], k1 = keys[i], k2 = keys[i + 1], k3 = keys[Math.min(n, i + 2)];
      for (const k of KEYS) cam[k] = cr(k0[k], k1[k], k2[k], k3[k], f);
    },
  }, at);
}
const yawTo = (a, b) => Math.atan2(b[0] - a[0], b[2] - a[2]);

export default function choreograph(tl, world, scene) {
  const { state, SHOTS } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration; // a point inside line i, as a fraction of the spoken line
  const { person, SPOTS } = world.room, F = world.room.F;

  // ---- The person's morning, on the film clock. ----
  const D = (c) => person.duration(c);
  const exitAt = t0 + 3.0; // once the camera has pulled back far enough to see the bed
  const W1 = [SPOTS.bedFront[0], F, SPOTS.bedFront[2]], W2 = [2.8, F, 13.6], W3 = SPOTS.standDesk; // round the chair, then in front of it
  const w1 = exitAt + D("Sitting_Exit"), w2 = w1 + 1.35, w3 = w2 + 0.75, sit = w3 + D("Sitting_Enter");
  const brk = at(2, 0.04), brkUp = brk + D("Sitting_Exit"), back = tEnd - 2.4, brkDown = back - D("Sitting_Enter") + 0.6;
  person.act([
    { t0, t1: exitAt, clip: "Sitting_Idle_Loop", loop: true, speed: 0.5, from: SPOTS.bedEdge, yaw: [SPOTS.faceDesk], fade: 0 },
    { t0: exitAt, t1: w1, clip: "Sitting_Exit", from: SPOTS.bedEdge, to: SPOTS.bedFront, yaw: [SPOTS.faceDesk], fade: 0.15 },
    { t0: w1, t1: w2, clip: "Walk_Loop", loop: true, speed: 0.95, from: W1, to: W2, yaw: [SPOTS.faceDesk, yawTo(W1, W2)], ease: "none", fade: 0.3 },
    { t0: w2, t1: w3, clip: "Walk_Loop", loop: true, speed: 0.95, offset: (w2 - w1) * 0.95, from: W2, to: W3, yaw: [yawTo(W1, W2), yawTo(W2, W3)], ease: "none", fade: 0 },
    { t0: w3, t1: sit, clip: "Sitting_Enter", from: W3, to: SPOTS.seat, yaw: [yawTo(W2, W3), SPOTS.faceDesk], fade: 0.25 },
    { t0: sit, t1: brk, clip: "Sitting_Idle_Loop", loop: true, speed: 0.6, from: SPOTS.seat, yaw: [SPOTS.faceDesk], fade: 0.2 },
    // "It notices your breaks": up for a stretch, then back down.
    { t0: brk, t1: brkUp, clip: "Sitting_Exit", from: SPOTS.seat, to: SPOTS.standDesk, yaw: [SPOTS.faceDesk], fade: 0.2 },
    { t0: brkUp, t1: brkDown, clip: "Idle_Loop", loop: true, from: SPOTS.standDesk, yaw: [SPOTS.faceDesk, SPOTS.faceDesk + 0.6], fade: 0.25 },
    { t0: brkDown, t1: brkDown + D("Sitting_Enter"), clip: "Sitting_Enter", from: SPOTS.standDesk, to: SPOTS.seat, yaw: [SPOTS.faceDesk + 0.6, SPOTS.faceDesk], fade: 0.25 },
    { t0: brkDown + D("Sitting_Enter"), t1: 1e9, clip: "Sitting_Idle_Loop", loop: true, speed: 0.6, from: SPOTS.seat, yaw: [SPOTS.faceDesk], fade: 0.2 },
  ]);

  // Camera: out from the shelf past the chair to the whole room.
  const OFF_SHELF = { x: 4.5, y: 0.6, z: 11.0, tx: 0.6, ty: -1.4, tz: 2.0, fov: 34 };
  const OVER_DRIFT = { x: 14.2, y: 8.2, z: 25.6, tx: -3.5, ty: -4.0, tz: 5.2, fov: 34 };
  const BREAK = { x: 17.0, y: 3.0, z: 29.0, tx: 0.0, ty: -5.0, tz: 7.0, fov: 40 };
  const BREAK_DRIFT = { x: 16.0, y: 3.2, z: 30.0, tx: 0.0, ty: -5.0, tz: 7.0, fov: 39 };

  // ---- Arrive: from HOME into deskClose; the room and the person rise around the box in the first half second. ----
  tl.to(state, { room: 1, person: 1, duration: 0.5, ease: "power2.out" }, t0);
  world.leg(tl, t0, SHOTS.deskClose, 0.9, "power4.out");
  tl.to(state, { eyes: 1, duration: 0.8, ease: "power2.out" }, t0 + 0.6); // it starts to watch
  // Faces: sleepy on the bed, waking as they walk, a smile once seated and briefed.
  tl.set(state, { personSleepy: 1, personSmile: 0, personStrain: 0 }, t0);
  tl.to(state, { personSleepy: 0, duration: 1.2, ease: "power2.inOut" }, exitAt + 0.5);
  tl.to(state, { personSmile: 1, duration: 0.6, ease: "power2.out" }, sit + 0.4);
  tl.to(state, { personSmile: 0, duration: 0.6, ease: "power2.inOut" }, tEnd - 0.8);

  // ---- Line 0: "Point it at your week ... how the day really goes at the desk": one long pull back to the room. ----
  const pullAt = t0 + 0.9, pullEnd = t0 + 3.4;
  camPath(tl, world, pullAt, pullEnd - pullAt, [SHOTS.deskClose, OFF_SHELF, SHOTS.roomWide], "power2.inOut");

  // ---- Line 1: "Each morning it briefs you ...": over the shoulder; the person turns to the device when it speaks. ----
  // (The room holds wide until they have sat down.)
  const over = Math.max(L[1].start, sit - 0.3);
  world.leg(tl, over, SHOTS.overShoulder, 1.8, "power2.inOut");
  world.leg(tl, over + 1.8, OVER_DRIFT, L[2].start - (over + 1.8), "none");
  tl.to(state, { personLook: 0.85, duration: 0.8, ease: "power2.inOut" }, Math.max(at(1, 0.06), sit + 0.3));
  tl.to(state, { personLook: 0, duration: 0.7, ease: "power2.inOut" }, at(1, 0.5)); // back to the monitor: "Fall behind"
  tl.to(state, { personLook: 0.6, duration: 0.6, ease: "power2.inOut" }, at(1, 0.7)); // "One tap"
  tl.to(state, { personLook: 0, duration: 0.6, ease: "power2.inOut" }, at(1, 0.94));

  // ---- Line 2: "It notices your breaks ...": they stand for a stretch; the camera eases back to see it. ----
  world.leg(tl, L[2].start, BREAK, 1.6, "power2.inOut");
  world.leg(tl, L[2].start + 1.6, BREAK_DRIFT, back - (L[2].start + 1.6), "none");
  tl.to(state, { personLook: 0.5, duration: 0.8, ease: "power2.inOut" }, at(2, 0.4)); // "asks": a glance at the box

  // ---- Hand over to "remember": over the shoulder, seated again. ----
  world.leg(tl, back, SHOTS.overShoulder, 2.2, "power2.inOut");
  tl.to(state, { personLook: 0, eyes: 0, duration: 0.8, ease: "power2.inOut" }, back + 0.8);
}
