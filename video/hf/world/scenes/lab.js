// Scene "lab": The learning lab. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// The SPACE axis, toward the lab. Starts at HOME with lab 0 (handed over by "learn"); the lab is off frame at HOME, so
// it can simply be up from the first frame: the camera travels there.
//   line 0  "The lab is two AMD Strix Halo pods. One serves the teacher, a twenty-seven billion parameter Qwen. The other
//            trains."  → a real flight: crane up and turn right (accelerating), then dive and land on the two pods.
//   line 1  "Labeling, training, testing and packaging run on ROCm, and every run is recorded with its loss curve and
//            its watts."  → the chips are in the overlay; the loss ribbon draws; the pods glow with the watts.
//   line 2  "The box and the lab share two files, a dataset out and a model package back. Nothing else crosses."
//            → pull back to the two-shot; two flows cross the gap in opposite directions; the shell lights and a third
//            attempted flow stops at it.
// Hand-over to "swap": stay on the two-shot, lab 1, membrane 0, curve 0, labels 0.
export default function choreograph(tl, world, scene) {
  const { cam, state, P } = world, t0 = scene.start;
  const L = (i, f = 0) => scene.lines[i].start + f * scene.lines[i].duration;
  const END = t0 + scene.duration;

  // Flows (declared once). The blocked one bunches up (small spread) and only ever reaches the shell.
  const dataset = world.flow("lab-dataset", { from: P.box, to: P.lab, count: 10, size: 1.3, lift: 2.6 });
  const pkg = world.flow("lab-pkg", { from: P.lab, to: P.box, count: 1, size: 3, lift: 2.2 });
  const blocked = world.flow("lab-blocked", { from: P.box, to: P.lab, count: 8, size: 1.2, lift: 2.6 });
  dataset.spread = 0.4; pkg.spread = 0; blocked.spread = 0.06;

  // 3D labels over the pods (compact 1.6-unit pods at x = 12.85 and 15.15, top at y ≈ 0.2, front face at z ≈ -1.2).
  // (In front of the pods' faces, so they are never occluded; the overlay's hardware card sits top right.)
  const lA = world.label("lab-podA", { text: "Pod A · teacher · Qwen 27B", at: [12.55, 1.02, -0.9], size: 64, scale: 0.72 });
  const lB = world.label("lab-podB", { text: "Pod B · trainer", at: [15.55, 0.3, -0.9], size: 64, scale: 0.58 });
  const show = (l, at) => tl.fromTo(l, { opacity: 0, lift: -0.2 }, { opacity: 1, lift: 0, duration: 0.7, ease: "power3.out" }, at);

  // Shots.
  const CRANE = { x: 3.8, y: 5.0, z: 10.4, tx: 7.5, ty: 0.2, tz: -1.5, fov: 38 };     // up and turning right, the lab ahead
  const PODS = { x: 11.2, y: 1.9, z: 3.6, tx: 14, ty: -0.1, tz: -2, fov: 38 };         // landed on the two pods, front-left
  const PODS_DRIFT = { x: 11.7, y: 1.8, z: 3.9, tx: 14, ty: -0.1, tz: -1.9, fov: 38 };
  const RIBBON = { x: 12.6, y: 2.6, z: 6.4, tx: 14, ty: 0.7, tz: -1.8, fov: 38 };     // a step back and up: the loss ribbon above the pods
  const RIBBON_DRIFT = { x: 12.3, y: 2.7, z: 6.6, tx: 14, ty: 0.72, tz: -1.8, fov: 38 };
  const TWO = { x: 7, y: 4, z: 14, tx: 7, ty: 0.5, tz: -1, fov: 40 };                 // the two-shot handed to "swap"

  // The lab is already standing when we arrive (it is out of frame at HOME).
  tl.to(state, { lab: 1, duration: 0.2, ease: "none" }, t0);

  // ---- line 0: the flight. Accelerate out of HOME while craning up and right, then a hard power4.out landing on the pods. ----
  world.leg(tl, t0, CRANE, 1.25, "power2.in");
  world.leg(tl, t0 + 1.25, PODS, 1.7, "power4.out");
  // "One serves the teacher, a twenty-seven billion parameter Qwen": Pod A. "The other trains": Pod B.
  show(lA, L(0, 0.34));
  show(lB, L(0, 0.78));
  // The hold never dies: a slow drift around the pods through the rest of the line and the first chips.
  world.leg(tl, t0 + 2.95, PODS_DRIFT, L(1, 0.58) - (t0 + 2.95), "none");

  // ---- line 1: "every run is recorded with its loss curve": step back and up, the ribbon draws. "and its watts": the pods glow. ----
  // The pod labels have done their job; the overlay's chips and watts card take over.
  tl.to(lA, { opacity: 0, duration: 0.5, ease: "power2.in" }, L(1, 0.5));
  tl.to(lB, { opacity: 0, duration: 0.5, ease: "power2.in" }, L(1, 0.54));
  world.leg(tl, L(1, 0.58), RIBBON, 1.8, "power2.inOut");
  tl.to(state, { labCurve: 1, duration: 1.8, ease: "power2.inOut" }, L(1, 0.64));
  tl.to(state, { labGlow: 1, duration: 1.2, ease: "power2.out" }, L(1, 0.82));
  world.leg(tl, L(1, 0.58) + 1.8, RIBBON_DRIFT, L(2, 0.0) - (L(1, 0.58) + 1.8), "none");

  // ---- line 2: "The box and the lab share two files": pull back to the two-shot. ----
  world.leg(tl, L(2, 0.0), TWO, 2.4, "power2.inOut");
  tl.to(state, { labGlow: 0, duration: 1.6, ease: "power2.inOut" }, L(2, 0.05));
  tl.to(state, { labCurve: 0, duration: 1.0, ease: "power2.inOut" }, L(2, 0.3));
  // "a dataset out": box → lab. "and a model package back": lab → box.
  tl.to(dataset, { progress: 1, duration: 2.0, ease: "power1.inOut" }, L(2, 0.3));
  tl.to(pkg, { progress: 1, duration: 1.9, ease: "power2.inOut" }, L(2, 0.5));
  // "Nothing else crosses": the shell lights; a third stream leaves the box and stops at the shell.
  tl.to(state, { membrane: 1, duration: 0.8, ease: "power2.out" }, L(2, 0.7));
  tl.to(blocked, { progress: 0.16, duration: 1.1, ease: "power2.out" }, L(2, 0.76));
  tl.to(blocked, { density: 0, duration: 0.5, ease: "power2.in" }, L(2, 0.76) + 1.1);
  // Hand-over: membrane back to 0 on the two-shot; lab stays 1.
  tl.to(state, { membrane: 0, duration: 0.9, ease: "power2.inOut" }, Math.min(END - 1.0, L(2, 1.0) + 0.1));
  const reset = Math.min(END - 0.1, L(2, 1.0) + 0.9);
  tl.set(dataset, { progress: 0 }, reset);
  tl.set(pkg, { progress: 0 }, reset);
  tl.set(blocked, { progress: 0, density: 1 }, reset);
  tl.set(lA, { opacity: 0, lift: 0 }, reset);
  tl.set(lB, { opacity: 0, lift: 0 }, reset);
}
