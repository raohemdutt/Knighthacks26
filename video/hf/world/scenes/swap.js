// Scene "swap": Box and lab. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Starts on the two-shot with lab 1 (handed over by "lab").
//   line 0  "The new model arrives as one small package. The box checks it, swaps it in, and rolls back by itself if it
//            does not answer."  → the package dot flies lab → box; the camera moves in to a close three-quarter on the
//            box; the box half-opens and the vision layer flashes as the swap; the rollback is in the overlay.
//   line 1  "We measure watts on the Jetson, watts on the GPUs, and joules per answer, so you can see what a question
//            costs."  → back out to the two-shot (Jetson left, GPUs right) under the three meters; then home.
// Hand-over to "jetson": HOME, lab 0, layers closed.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, P } = world, t0 = scene.start;
  const L = (i, f = 0) => scene.lines[i].start + f * scene.lines[i].duration;
  const END = t0 + scene.duration;

  const vis = { v: 0 };
  const lightVision = (v, at, dur, ease = "power2.out") =>
    tl.to(vis, { v, duration: dur, ease, onUpdate: () => { state.layer[1] = vis.v; } }, at);

  const pkg = world.flow("swap-pkg", { from: P.lab, to: P.box, count: 1, size: 3, lift: 2.4 });
  pkg.spread = 0;

  const TWO = { x: 7, y: 4, z: 14, tx: 7, ty: 0.5, tz: -1, fov: 40 };
  const CLOSE = { x: 3.4, y: 2.6, z: 7.0, tx: 1.1, ty: 0.45, tz: 0, fov: 36 };      // three-quarter on the box, box left of centre
  const CLOSE_DRIFT = { x: 3.1, y: 2.75, z: 7.2, tx: 1.1, ty: 0.5, tz: 0, fov: 36 };
  const TWO_DRIFT = { x: 7.2, y: 3.9, z: 13.7, tx: 7, ty: 0.5, tz: -1, fov: 40 };

  // ---- line 0: "The new model arrives as one small package": the dot leaves the lab; the camera follows it to the box. ----
  tl.to(pkg, { progress: 1, duration: 2.4, ease: "power2.inOut" }, L(0, 0.02));
  world.leg(tl, L(0, 0.08), CLOSE, 2.6, "power2.inOut");
  // "The box checks it": a short glance (the vision layer warms).
  lightVision(0.35, L(0, 0.34), 0.5);
  // "swaps it in": the box half-opens and the vision layer flashes twice: old out, new in.
  tl.to(state, { open: 0.5, duration: 0.7, ease: "power2.inOut" }, L(0, 0.46));
  lightVision(1, L(0, 0.5), 0.25);
  lightVision(0.15, L(0, 0.5) + 0.3, 0.3, "power2.inOut");
  lightVision(1, L(0, 0.5) + 0.65, 0.25);
  // "and rolls back by itself if it does not answer": the light dies, the box closes again.
  lightVision(0, L(0, 0.74), 0.5, "power2.inOut");
  tl.to(state, { open: 0, duration: 0.9, ease: "power2.inOut" }, L(0, 0.84));
  world.leg(tl, L(0, 0.08) + 2.6, CLOSE_DRIFT, L(1, 0.0) - 0.3 - (L(0, 0.08) + 2.6), "none");

  // ---- line 1: "We measure watts on the Jetson, watts on the GPUs": back out to the two-shot, Jetson left, GPUs right. ----
  world.leg(tl, L(1, 0.0) - 0.3, TWO, 2.4, "power2.inOut");
  world.leg(tl, L(1, 0.0) + 2.1, TWO_DRIFT, L(1, 0.76) - (L(1, 0.0) + 2.1), "none");
  // "so you can see what a question costs": home, the lab sinks away, everything reset before the scene ends.
  world.leg(tl, L(1, 0.76), { ...HOME }, Math.min(2.6, END - 0.2 - L(1, 0.76)), "power2.inOut");
  tl.to(state, { lab: 0, duration: 1.1, ease: "power2.inOut" }, Math.min(END - 1.3, L(1, 0.76) + 1.3));
  tl.set(pkg, { progress: 0 }, END - 0.1);
}
