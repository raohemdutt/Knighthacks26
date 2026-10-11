// Scene "day": Your day. The person at the desk is the focus. We start tight on the device on the desk (the room and
// the person rise around it at once), pull back in one continuous move past the monitor's webcam to the whole room,
// drift over the person's shoulder while Atlas briefs them, then watch them lean back for a break.
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

export default function choreograph(tl, world, scene) {
  const { state, SHOTS } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration; // a point inside line i, as a fraction of the spoken line
  // The seated person's spine (seated pose: x = -0.06, x = forward pitch). A small lean back for the break; reset below.
  const spine = world.room.person.J.spine.rotation;

  // The webcam clipped on the monitor, on the way out.
  const PAST_WEBCAM = { x: 0.6, y: 5.0, z: 4.6, tx: -4.5, ty: 4.0, tz: -2.1, fov: 32 };
  const OVER_DRIFT = { x: -1.3, y: 5.4, z: 13.8, tx: -1.7, ty: 0.3, tz: -2.4, fov: 37 };
  // The break: round to the person's side, so the lean back and the face read; then a slow drift while the coach asks.
  const BREAK = { x: 8.5, y: 3.8, z: 9.5, tx: -3.2, ty: 1.4, tz: 3.4, fov: 36 };
  const BREAK_DRIFT = { x: 8.0, y: 3.9, z: 10.2, tx: -3.2, ty: 1.4, tz: 3.6, fov: 35.5 };

  // ---- Arrive: from HOME into deskClose; the room and the person rise around the box in the first half second. ----
  tl.to(state, { room: 1, person: 1, duration: 0.5, ease: "power2.out" }, t0);
  world.leg(tl, t0, SHOTS.deskClose, 0.9, "power4.out");
  tl.to(state, { eyes: 1, duration: 0.8, ease: "power2.out" }, t0 + 0.6); // it starts to watch

  // ---- Line 0: "Point it at your week ... how the day really goes at the desk": one long pull back and pan. ----
  const pullAt = t0 + 1.1, pullEnd = Math.min(at(0, 0.98), L[1].start - 0.1);
  camPath(tl, world, pullAt, pullEnd - pullAt, [SHOTS.deskClose, PAST_WEBCAM, SHOTS.roomWide], "power2.inOut");

  // ---- Line 1: "Each morning it briefs you ...": over the shoulder; the person turns to the device when it speaks. ----
  world.leg(tl, L[1].start, SHOTS.overShoulder, 2.0, "power2.inOut");
  world.leg(tl, L[1].start + 2.0, OVER_DRIFT, L[2].start - (L[1].start + 2.0), "none");
  tl.to(state, { personLook: 0.85, duration: 0.8, ease: "power2.inOut" }, at(1, 0.06));
  tl.to(state, { personLook: 0, duration: 0.7, ease: "power2.inOut" }, at(1, 0.5)); // back to the monitor: "Fall behind"
  tl.to(state, { personLook: 0.6, duration: 0.6, ease: "power2.inOut" }, at(1, 0.7)); // "One tap"
  tl.to(state, { personLook: 0, duration: 0.6, ease: "power2.inOut" }, at(1, 0.94));

  // ---- Line 2: "It notices your breaks ...": the person leans back, the camera eases back to see it. ----
  const back = tEnd - 2.4;
  world.leg(tl, L[2].start, BREAK, 1.6, "power2.inOut");
  world.leg(tl, L[2].start + 1.6, BREAK_DRIFT, back - (L[2].start + 1.6), "none");
  tl.to(spine, { x: -0.24, duration: 1.0, ease: "power2.inOut" }, at(2, 0.05));
  tl.to(state, { personLook: 0.35, duration: 0.8, ease: "power2.inOut" }, at(2, 0.28)); // "asks": a glance at the box

  // ---- Hand over to "remember": over the shoulder, room and person on, pose reset. ----
  world.leg(tl, back, SHOTS.overShoulder, 2.2, "power2.inOut");
  tl.to(spine, { x: -0.06, duration: 1.0, ease: "power2.inOut" }, back + 0.4);
  tl.to(state, { personLook: 0, eyes: 0, duration: 0.8, ease: "power2.inOut" }, back + 0.8);
}
