// Scene "device": On the device. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// One line, "Pull the network cable, and it keeps seeing, remembering, and answering on the box. Only the learning
// process stops.": in through the glass (the lead), the Jetson beating; on "Pull" the camera swings round to the back
// wall's jack and the cable drops out; on "Only" it comes home, the plug goes back in and the glass turns solid.
// Hand-over: HOME, cable 0, xray 0, hl 0, pulse 0.
export default function choreograph(tl, world, scene) {
  const { state, HOME } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines, hl = state.hl;
  const word = (i, w, d = 0) => { const l = L[i], k = l.text.indexOf(w); return l.start + (k < 0 ? 0 : k / l.text.length) * l.duration + d; };

  // A high three-quarter into the box, the box weighted to screen-left (the overlay owns the right).
  const INSIDE = { x: 3.3, y: 1.85, z: 3.7, tx: 0.7, ty: -0.55, tz: -0.5, fov: 31 };
  // Behind the box, from the back-right: the jack on the back wall and the cable dropping to the desk.
  const BACK = { x: 3.1, y: 0.75, z: -4.4, tx: -0.55, ty: -0.72, tz: -1.25, fov: 30 };
  const BACK_DRIFT = { x: 2.8, y: 0.7, z: -4.1, tx: -0.5, ty: -0.74, tz: -1.25, fov: 30 };

  // ---- The lead: in through the glass; the Jetson is where everything runs. ----
  world.leg(tl, t0, INSIDE, Math.max(0.8, L[0].start - t0), "power2.inOut");
  tl.to(state, { xray: 1, duration: 0.8, ease: "sine.inOut" }, t0 + 0.2);
  tl.to(hl, { jetson: 1, duration: 0.6, ease: "sine.inOut" }, t0 + 0.4);
  tl.to(state, { pulse: 1, duration: 0.6, ease: "sine.inOut" }, t0 + 0.4);

  // ---- "Pull the network cable": swing round to the back; the plug slides out and drops. ----
  const SWING = L[0].start;
  const HOME_AT = Math.max(SWING + 2.6, word(0, "Only") - 0.3);
  world.leg(tl, SWING, BACK, 1.4, "power2.inOut");
  tl.to(state, { cable: 1, duration: 1.0, ease: "power4.out" }, SWING + 1.0);
  world.leg(tl, SWING + 1.4, BACK_DRIFT, HOME_AT - (SWING + 1.4), "none");

  // ---- "Only the learning process stops": home, plug back in, glass solid before the cut. ----
  const back = Math.max(1.2, END - HOME_AT - 0.1), fade = Math.min(1.2, back);
  world.leg(tl, HOME_AT, { ...HOME }, back, "power2.inOut");
  tl.to(state, { cable: 0, duration: fade, ease: "sine.inOut" }, HOME_AT + 0.3);
  tl.to(hl, { jetson: 0, duration: fade, ease: "sine.inOut" }, END - fade - 0.1);
  tl.to(state, { pulse: 0, xray: 0, duration: fade, ease: "sine.inOut" }, END - fade - 0.1);
}
