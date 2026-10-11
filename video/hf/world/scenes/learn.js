// Scene "learn": It learns. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Word offsets inside each line (silence gaps measured on the narration mp3s, line-local seconds):
//   line 0: "...to be small." 0.12–1.82 · "It is not." 2.18–2.57 · "A highly capable Qwen runs right here." 2.95–4.95
//           "Fast," 5.39 · "accurate, and private." 5.98–6.77 · "Forty-two demo questions," 7.23–8.47 · "forty-two right." 8.64–9.26
//   line 1: "...something new," 0.13–1.61 · "Atlas learns the tool." 2.01–2.99 · "You describe the task," 3.40–4.31
//           "and the lab teaches it" 4.57–5.07 · "from your own examples." 5.21–6.51
// Starts from whatever the previous scene left (a leg, no set). Hand-over to "lab": the camera lands on TURN
// { x: 4.2, y: 2.8, z: 9.0, tx: 4.0, ty: 0, tz: -1, fov: 36 } just before the scene end; xray 0, hl 0, pulse 0,
// explode 0; membrane stays 1 (lab fades it); lab 0; flows at rest.
export default function choreograph(tl, world, scene) {
  const { state, P, SHOTS } = world, t0 = scene.start, END = t0 + scene.duration;
  const W = (i, s) => scene.lines[i].start + s; // a measured word moment inside line i

  // The tool request leaves the box, crosses the shell and heads for the lab (off frame to the right).
  const out = world.flow("learn-out", { from: P.box, to: [14, 0.6, -2], count: 16, size: 1.3, lift: 2.6 });
  out.spread = 0.45;

  const XRAY = { ...SHOTS.xray };
  const JET = { x: 1.75, y: 1.05, z: 2.25, tx: 0.05, ty: -0.78, tz: -0.3, fov: 30 };      // pushed in on the Jetson
  const JET_DRIFT = { x: 1.55, y: 1.0, z: 2.4, tx: 0.05, ty: -0.78, tz: -0.3, fov: 30 };
  const WIDE = { x: 2.9, y: 1.6, z: 6.6, tx: 0.3, ty: -0.35, tz: 0, fov: 32 };             // the closed box, room for the card
  const TURN = { x: 4.2, y: 2.8, z: 9.0, tx: 4.0, ty: 0.0, tz: -1.0, fov: 36 };           // hand-over: turned toward the lab

  // ---- line 0: "You would expect the model on the box to be small": x-ray in, the Jetson lights on "It is not." ----
  world.leg(tl, t0, XRAY, 2.0, "power2.inOut");
  tl.to(state, { xray: 1, duration: 1.2, ease: "power2.inOut" }, t0 + 0.4);
  tl.to(state.hl, { jetson: 1, duration: 0.8, ease: "power4.out" }, W(0, 2.18));
  tl.to(state, { pulse: 0.7, duration: 1.0, ease: "power2.out" }, W(0, 2.3));
  // "A highly capable Qwen runs right here": push in and land on the Jetson.
  world.leg(tl, W(0, 2.95), JET, 1.6, "power4.out");
  world.leg(tl, W(0, 2.95) + 1.6, JET_DRIFT, W(0, 9.4) - (W(0, 2.95) + 1.6), "none");

  // ---- line 1: "And when your life needs something new": pull back, the box closes up again. ----
  world.leg(tl, W(1, 0.05), WIDE, 2.0, "power2.inOut");
  tl.to(state, { xray: 0, duration: 1.4, ease: "power2.inOut" }, W(1, 0.1));
  tl.to(state.hl, { jetson: 0, duration: 1.0, ease: "power2.inOut" }, W(1, 0.1));
  tl.to(state, { pulse: 0, duration: 1.0, ease: "power2.inOut" }, W(1, 0.1));
  // "and the lab teaches it": the shell rises, the request leaves through it; the camera turns toward the lab.
  tl.to(state, { membrane: 1, duration: 1.2, ease: "power2.out" }, W(1, 3.6));
  tl.to(out, { progress: 1, duration: 4.4, ease: "power1.inOut" }, W(1, 4.5));
  const turnAt = W(1, 5.1);
  world.leg(tl, turnAt, TURN, END - 0.12 - turnAt, "power2.inOut");
  tl.set(out, { progress: 0 }, END - 0.05);
}
