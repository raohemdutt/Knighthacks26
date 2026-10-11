// Scene "device": On the device. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Line 0 "Everything live runs on the Jetson...": x-ray, the Jetson lights and beats; three frames travel from the
// webcam down to the Jetson (the overlay then shows one picture blurring as it crosses "leaves the device").
// Line 1 "Pull the network cable...": the camera swings round to the back wall's jack; the cable drops out.
// Line 2 "And it keeps a ledger...": the camera comes home under the ledger card; everything resets.
// Hand-over: HOME, cable 0, xray 0, hl 0, pulse 0.
export default function choreograph(tl, world, scene) {
  const { state, HOME, hash } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines, hl = state.hl;
  const word = (i, w, d = 0) => { const l = L[i], k = l.text.indexOf(w); return l.start + (k < 0 ? 0 : k / l.text.length) * l.duration + d; };

  // A high three-quarter into the box, the box weighted to screen-left (the overlay owns the right).
  const INSIDE = { x: 3.3, y: 1.85, z: 3.7, tx: 0.7, ty: -0.55, tz: -0.5, fov: 31 };
  const INSIDE_DRIFT = { x: 3.0, y: 1.75, z: 3.45, tx: 0.65, ty: -0.55, tz: -0.45, fov: 31 };
  // Behind the box, from the back-right: the jack on the back wall and the cable dropping to the desk.
  const BACK = { x: 3.1, y: 0.75, z: -4.4, tx: -0.55, ty: -0.72, tz: -1.25, fov: 30 };
  const BACK_DRIFT = { x: 2.8, y: 0.7, z: -4.1, tx: -0.5, ty: -0.74, tz: -1.25, fov: 30 };

  // ---- Line 0: in through the glass; the Jetson is where everything live runs. ----
  world.leg(tl, t0, INSIDE, 1.8, "power2.inOut");
  world.leg(tl, t0 + 1.8, INSIDE_DRIFT, L[1].start - 0.2 - (t0 + 1.8), "none");
  tl.to(state, { xray: 1, duration: 1.0, ease: "power2.inOut" }, t0 + 0.5);
  tl.to(hl, { jetson: 1, duration: 0.45, ease: "power2.out" }, word(0, "Jetson") - 0.05);
  tl.to(state, { pulse: 1, duration: 0.6, ease: "power2.out" }, word(0, "Jetson"));

  // Three frames, one dot each, from the lens down into the Jetson. The renderer jitters dot 0 by a fixed hash offset;
  // cancel it so each dot follows its arc exactly.
  const off = [(hash(0, 2) - 0.5) * 0.5, (hash(0, 3) - 0.5) * 0.5, (hash(0, 4) - 0.5) * 0.5];
  const exact = (p) => [p[0] - off[0], p[1] - off[1], p[2] - off[2]];
  const lens = world.P.webcam, jet = world.P.jetson;
  const frames = [0, 1, 2].map((i) => {
    const f = world.flow(`device-frame-${i}`, { from: exact([lens[0] - 0.12 + i * 0.12, lens[1], lens[2] - 0.05]), to: exact([jet[0] - 0.15 + i * 0.15, jet[1] + 0.08, jet[2]]), count: 1, size: 0.75, lift: 0.35 });
    f.spread = 0;
    return f;
  });
  const F0 = word(0, "Jetson") + 0.4;
  frames.forEach((f, i) => {
    tl.fromTo(f, { progress: 0, density: 1 }, { progress: 1, duration: 0.9, ease: "power2.inOut" }, F0 + i * 0.45);
  });

  // ---- Line 1: "Pull the network cable": swing round to the back; the plug slides out and drops. ----
  const SWING = L[1].start - 0.2;
  world.leg(tl, SWING, BACK, 1.5, "power2.inOut");
  tl.to(state, { cable: 1, duration: 1.0, ease: "power4.out" }, SWING + 1.25);
  world.leg(tl, SWING + 1.5, BACK_DRIFT, L[2].start - 0.3 - (SWING + 1.5), "none");

  // ---- Line 2: the ledger. Home, plug back in, glass off. ----
  const HOME_AT = L[2].start - 0.3;
  world.leg(tl, HOME_AT, { ...HOME }, Math.min(5.2, END - HOME_AT - 0.25), "power2.inOut");
  tl.to(state, { cable: 0, duration: 1.2, ease: "power2.inOut" }, HOME_AT + 1.6);
  tl.to(hl, { jetson: 0, duration: 1.0, ease: "power2.inOut" }, HOME_AT + 2.6);
  tl.to(state, { pulse: 0, duration: 1.0, ease: "power2.inOut" }, HOME_AT + 2.6);
  tl.to(state, { xray: 0, duration: 1.2, ease: "power2.inOut" }, HOME_AT + 2.8);
}
