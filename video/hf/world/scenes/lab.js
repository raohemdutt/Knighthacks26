// Scene "lab": The learning lab. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Word offsets inside each line (silence gaps measured on the narration mp3s, line-local seconds):
//   line 0: "The lab is two AMD Strix Halo pods." 0.13–2.44 · "One is your trainer." 2.83–3.47 · "It fine-tunes ... examples." 3.84–6.17
//   line 1: "The other" 0.12 · "is the network's trainer." 0.72–1.68 · "With everyone's consent," 2.04–3.03
//           "it learns from everyone's examples," 3.33–4.90 · "so every Atlas gets better together." 5.06–6.78
//   line 2: "A test decides ... really better." 0.13–2.49 · "Only then does it ship back." 2.88–3.88 · "Our adapter ... Hugging Face." 4.25–6.28
// Starts on learn's TURN shot with membrane 1. Hand-over: HOME, lab 0, friends 0, labels 0, membrane 0, glow/curve 0.
// The pods: Trainer A (screen left, x 12.85) is "Your trainer", Trainer B (x 15.15) the "Network trainer".
export default function choreograph(tl, world, scene) {
  const { state, P, HOME } = world, t0 = scene.start, END = t0 + scene.duration;
  const W = (i, s) => scene.lines[i].start + s;

  const A = [12.85, 0.35, -2], B = [15.15, 0.35, -2]; // pod tops (lab 1)

  // Flows, declared once.
  const mine = world.flow("lab-mine", { from: P.box, to: A, count: 16, size: 1.3, lift: 2.6 });
  mine.spread = 0.45;
  const sources = [...P.friend, P.box];
  const toNet = sources.map((f, i) => world.flow("lab-net" + i, { from: f, to: B, count: 8, size: 1.1, lift: 3.2 }));
  toNet.forEach((f) => { f.spread = 0.4; });
  const back = sources.map((f, i) => world.flow("lab-back" + i, { from: B, to: f, count: 1, size: 2.6, lift: 3.4 }));
  back.forEach((f) => { f.spread = 0; });
  const ship = world.flow("lab-ship", { from: A, to: P.box, count: 1, size: 3, lift: 2.6 });
  ship.spread = 0;

  // Labels: close versions for the pod shot, larger ones for the wide shot (labels are sized in world units).
  const lYours = world.label("lab-yours", { text: "Your trainer", at: [12.85, 0.85, -2], size: 64, scale: 0.62 });
  const lNet = world.label("lab-network", { text: "Network trainer", at: [15.15, 0.85, -2], size: 64, scale: 0.62 });
  const lYoursW = world.label("lab-yours-w", { text: "Your trainer", at: [11.4, 1.9, -2], size: 64, scale: 2.0 });
  const lNetW = world.label("lab-network-w", { text: "Network trainer", at: [16.6, 1.9, -2], size: 64, scale: 2.0 });
  const show = (l, at, d = 0.6) => { tl.fromTo(l, { opacity: 0 }, { opacity: 1, duration: Math.max(d, 0.5), ease: "sine.inOut" }, at); tl.fromTo(l, { lift: -0.15 }, { lift: 0, duration: d, ease: "power4.out" }, at); }
  const hide = (l, at, d = 0.4) => tl.to(l, { opacity: 0, duration: d, ease: "power2.in" }, at);

  // Shots.
  const MID = { x: 8.6, y: 3.4, z: 7.6, tx: 11.2, ty: 0.1, tz: -1.8, fov: 38 };          // mid-flight, banking toward the pods
  const PODS = { x: 11.2, y: 1.9, z: 3.6, tx: 14, ty: -0.1, tz: -2, fov: 38 };            // landed on the two pods
  const RIB = { x: 12.2, y: 2.4, z: 6.8, tx: 14, ty: 0.6, tz: -1.8, fov: 38 };            // a step back and up: the loss ribbon
  const WIDE = { x: 5, y: 13, z: 20, tx: 5.5, ty: -0.5, tz: -1.2, fov: 36 };              // the friends and the lab together
  const WIDE_D = { x: 5.4, y: 12.6, z: 19.4, tx: 5.6, ty: -0.5, tz: -1.2, fov: 36 };
  const TWO = { x: 7.2, y: 5.2, z: 15.5, tx: 7.2, ty: 0.0, tz: -1.0, fov: 36 };           // box and lab, friends gone

  // ---- line 0: the flight. Accelerate off TURN, bank, land hard on the pods as "pods" is said. ----
  tl.fromTo(state, { lab: 0 }, { lab: 1, duration: 1.5, ease: "power4.out" }, t0 + 0.2);
  tl.to(state, { membrane: 0, duration: 1.2, ease: "power2.inOut" }, t0 + 0.3);
  world.leg(tl, t0, MID, 1.2, "power2.in");
  world.leg(tl, t0 + 1.2, PODS, 1.5, "power4.out");
  tl.to(mine, { progress: 1, duration: 3.8, ease: "power1.inOut" }, t0 + 0.2);
  // "One is your trainer."
  show(lYours, W(0, 2.83));
  // "It fine-tunes your own model": the pods warm, the loss ribbon draws; the camera eases back to see it.
  tl.to(state, { labGlow: 1, duration: 1.0, ease: "power2.out" }, W(0, 3.84));
  world.leg(tl, W(0, 3.5), RIB, W(1, 0.1) - W(0, 3.5), "power2.inOut");
  tl.to(state, { labCurve: 1, duration: 2.2, ease: "power2.inOut" }, W(0, 4.1));

  // ---- line 1: "The other is the network's trainer." ----
  show(lNet, W(1, 0.72));
  // Pull back: the friends' Atlases are out there too.
  world.leg(tl, W(1, 1.3), WIDE, 2.4, "power2.inOut");
  world.leg(tl, W(1, 1.3) + 2.4, WIDE_D, W(2, 0.3) - (W(1, 1.3) + 2.4), "none");
  hide(lYours, W(1, 1.4)); hide(lNet, W(1, 1.4));
  tl.fromTo(state, { friends: 0 }, { friends: 1, duration: 1.6, ease: "power4.out" }, W(1, 1.6));
  show(lYoursW, W(1, 3.2)); show(lNetW, W(1, 3.3));
  // "it learns from everyone's examples": a stream from every Atlas into the network trainer.
  toNet.forEach((f, i) => tl.to(f, { progress: 1, duration: 2.0, ease: "power1.inOut" }, W(1, 3.3) + i * 0.18));
  // "so every Atlas gets better together": one package back to each.
  back.forEach((f, i) => tl.to(f, { progress: 1, duration: 1.5, ease: "power2.inOut" }, W(1, 5.3) + i * 0.08));

  // ---- line 2: "A test decides": the friends sink away, the shot settles on box and lab. ----
  hide(lYoursW, W(2, 0.3)); hide(lNetW, W(2, 0.3));
  world.leg(tl, W(2, 0.3), TWO, 2.5, "power2.inOut");
  tl.to(state, { friends: 0, duration: 1.4, ease: "power2.inOut" }, W(2, 0.4));
  tl.to(state, { labCurve: 0, duration: 1.0, ease: "power2.inOut" }, W(2, 0.6));
  tl.to(state, { labGlow: 0.35, duration: 1.0, ease: "power2.inOut" }, W(2, 0.6));
  // "Only then does it ship back": the package rides home and the camera rides with it.
  tl.to(ship, { progress: 1, duration: 2.8, ease: "power2.inOut" }, W(2, 2.88));
  world.leg(tl, W(2, 2.88), { ...HOME }, END - 0.15 - W(2, 2.88), "power2.inOut");
  tl.to(state, { labGlow: 0, duration: 0.8, ease: "power2.inOut" }, W(2, 3.4));
  // The shell catches it as it lands, then rests.
  tl.to(state, { membrane: 0.8, duration: 0.6, ease: "power2.out" }, W(2, 4.9));
  tl.to(state, { membrane: 0, duration: 1.0, ease: "power2.inOut" }, W(2, 6.0));
  // The pods sink once they are out of frame.
  tl.to(state, { lab: 0, duration: 1.0, ease: "power2.inOut" }, W(2, 5.6));

  // Reset for a clean seek.
  [mine, ...toNet, ...back, ship].forEach((f) => tl.set(f, { progress: 0 }, END - 0.05));
  [lYours, lNet, lYoursW, lNetW].forEach((l) => tl.set(l, { opacity: 0, lift: 0 }, END - 0.05));
}
