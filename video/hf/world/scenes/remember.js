// Scene "remember": It remembers. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// In the dorm room (room 1, person 1 inherited from "day"; re-asserted at t0 in case they are not). The person is the
// focus: a three-quarter from beyond the device, so when they turn from the monitor to the box we see their face.
// Line 0 "Ask where you left your keys...": the camera settles on the person and the device; they turn to it; the lens
// wakes as it answers. Line 1 "It can just talk...": a slow drift while the answer cards fall in (overlay).
// Hand-over: room 0, person 0, personLook 0, eyes 0, camera at HOME by the end (learn starts in the studio).
export default function choreograph(tl, world, scene) {
  const { state, HOME } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines;
  const word = (i, w, d = 0) => { const l = L[i], k = l.text.indexOf(w); return l.start + (k < 0 ? 0 : k / l.text.length) * l.duration + d; };

  // The person and the device in one frame, weighted to screen-left (the overlay owns the right third).
  const ASK = { x: -1.6, y: 5.4, z: 12.0, tx: -0.55, ty: 0.4, tz: 0.6, fov: 34 };
  const ASK_DRIFT = { x: -1.3, y: 5.0, z: 11.0, tx: -0.55, ty: 0.35, tz: 0.6, fov: 33 };
  const TALK = { x: 10.8, y: 7.0, z: -3.9, tx: -2.65, ty: 0.7, tz: 1.15, fov: 38 };

  tl.to(state, { room: 1, person: 1, duration: 0.6, ease: "power2.out" }, t0);
  world.leg(tl, t0, ASK, 1.6, "power2.inOut");
  world.leg(tl, t0 + 1.6, ASK_DRIFT, L[1].start - 0.3 - (t0 + 1.6), "none");

  // "Ask where you left your keys": the person turns from the monitor to the box; the lens wakes on "answers".
  tl.to(state, { personLook: 0.9, duration: 0.9, ease: "power2.inOut" }, word(0, "keys") - 0.4);
  tl.to(state, { eyes: 1, duration: 0.6, ease: "power2.out" }, word(0, "answers"));

  // "It can just talk, do your arithmetic, or look something up": a slow drift closer.
  world.leg(tl, L[1].start - 0.3, TALK, 2.2, "power2.inOut");

  // Hand-over: the room sinks away as the camera returns to the studio three-quarter.
  const OUT = L[1].start + L[1].duration + 0.4;
  tl.to(state, { personLook: 0, eyes: 0, duration: 0.8, ease: "power2.inOut" }, OUT);
  tl.to(state, { person: 0, duration: 1.0, ease: "power2.inOut" }, OUT + 0.4);
  tl.to(state, { room: 0, duration: 1.6, ease: "power2.inOut" }, OUT + 0.8);
  world.leg(tl, OUT + 0.2, { ...HOME }, END - (OUT + 0.2) - 0.15, "power2.inOut");
}
