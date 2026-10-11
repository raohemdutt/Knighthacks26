// Scene "learn": It learns. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// The MEMBRANE axis, first crossing. Starts at HOME with membrane 0.3 (handed over by "day").
//   line 0  "The model on the box is small … But small models have gaps."  → open the box, light the vision layer, push in.
//   line 1  "Tap teach. A handful of blurred pictures go to the lab, a big open model labels them, and the small one is
//            fine-tuned on the answers."  → the shell lights, 12 picture dots leave the box THROUGH the shell and arc to
//            the lab; the camera pans right with them; the lab rises as they land and glows on "fine-tuned".
//   line 2  "A test decides if the new model is really better. Only then does it ship back to the box. Our adapter is
//            public on Hugging Face."  → one larger dot comes back from the lab; the camera comes home.
// Hand-over to "lab": HOME, membrane 0, lab 0, layers closed, labels 0.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, P } = world, t0 = scene.start;
  const L = (i, f = 0) => scene.lines[i].start + f * scene.lines[i].duration; // a moment inside spoken line i
  const END = t0 + scene.duration;

  // state.layer is an array (GSAP would treat it as a list of targets): drive it through a proxy whose onUpdate copies
  // the value across. Seek-safe: the proxy is a pure function of time.
  const vis = { v: 0 };
  const lightVision = (v, at, dur, ease = "power2.out") =>
    tl.to(vis, { v, duration: dur, ease, onUpdate: () => { state.layer[1] = vis.v; } }, at);

  // Flows and labels, declared once (same id → same object).
  const pics = world.flow("learn-pics", { from: P.box, to: P.lab, count: 12, size: 1.5, lift: 2.6 });
  const pkg = world.flow("learn-pkg", { from: P.lab, to: P.box, count: 1, size: 3, lift: 2.4 });
  pics.spread = 0.45; pkg.spread = 0;
  const lLab = world.label("learn-lab", { text: "The lab", at: [14, 3.6, -2], size: 72, scale: 1.4 });

  // Shots.
  const CLOSE = { x: 3.0, y: 2.9, z: 8.0, tx: 1.1, ty: 0.4, tz: 0, fov: 34 };       // three-quarter on the open box, box left of centre
  const SHELL = { x: 1.6, y: 2.9, z: 9.8, tx: 0.9, ty: 0.3, tz: 0, fov: 36 };        // wide enough to see the whole shell
  const TWO = { x: 7, y: 4, z: 14, tx: 7, ty: 0.5, tz: -1, fov: 40 };               // the box-and-lab two-shot (shared with lab/swap)
  const TWO_DRIFT = { x: 6.7, y: 3.9, z: 13.7, tx: 7.1, ty: 0.55, tz: -1, fov: 40 };

  // ---- line 0: "The model on the box is small": push in, open the box, light the vision layer. ----
  world.leg(tl, L(0, 0.02), CLOSE, 2.6, "power2.inOut");
  tl.to(state, { open: 1, duration: 1.8, ease: "power2.inOut" }, L(0, 0.1));
  lightVision(1, L(0, 0.22), 0.7);
  // "But small models have gaps": the lit layer dims a little, like a doubt.
  lightVision(0.45, L(0, 0.76), 0.6, "power2.inOut");

  // ---- line 1: "Tap teach": the shell lights, the camera steps back to see it. ----
  tl.to(state, { membrane: 1, duration: 0.9, ease: "power2.out" }, L(1, 0.05));
  world.leg(tl, L(1, 0.03), SHELL, 1.5, "power2.inOut");
  // "A handful of blurred pictures go to the lab": twelve dots leave the box, cross the shell, arc right.
  tl.to(pics, { progress: 1, duration: 3.4, ease: "power1.inOut" }, L(1, 0.14));
  // The camera pans right with them and lands on the two-shot as they arrive.
  world.leg(tl, L(1, 0.2), TWO, 3.0, "power2.inOut");
  // The box closes behind them; the vision layer goes dark (it is the lab's turn).
  tl.to(state, { open: 0, duration: 1.6, ease: "power2.inOut" }, L(1, 0.3));
  lightVision(0, L(1, 0.3), 1.2, "power2.inOut");
  // The lab rises to meet the pictures.
  tl.to(state, { lab: 1, duration: 1.4, ease: "power4.out" }, L(1, 0.3));
  tl.fromTo(lLab, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, L(1, 0.42));
  // "fine-tuned on the answers": the pods glow warm.
  tl.to(state, { labGlow: 1, duration: 1.2, ease: "power2.out" }, L(1, 0.72));
  // Hold: a slow drift so the two-shot never dies.
  world.leg(tl, L(1, 0.56), TWO_DRIFT, L(2, 0.38) - L(1, 0.56), "none");

  // ---- line 2: "A test decides…": the gate is in the overlay. "Only then does it ship back to the box": one dot returns. ----
  tl.to(lLab, { opacity: 0, duration: 0.6, ease: "power2.in" }, L(2, 0.3));
  tl.to(state, { labGlow: 0, duration: 1.4, ease: "power2.inOut" }, L(2, 0.34));
  tl.to(pkg, { progress: 1, duration: 2.3, ease: "power2.inOut" }, L(2, 0.38));
  // The camera comes home with it.
  world.leg(tl, L(2, 0.38), { ...HOME }, 3.6, "power2.inOut");
  // The shell fades once the package is inside; the lab sinks out of frame. Everything is reset before the scene ends.
  tl.to(state, { membrane: 0, duration: 1.4, ease: "power2.inOut" }, L(2, 0.7));
  tl.to(state, { lab: 0, duration: 1.2, ease: "power2.inOut" }, L(2, 0.74));
  // Flows go back to rest for a clean seek (they are invisible at progress 1 anyway).
  tl.set(pics, { progress: 0 }, Math.min(END - 0.1, L(2, 0.98)));
  tl.set(pkg, { progress: 0 }, Math.min(END - 0.1, L(2, 0.98)));
  tl.set(lLab, { opacity: 0, lift: 0 }, Math.min(END - 0.1, L(2, 0.98)));
}
