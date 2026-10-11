// Scene "jetson": On the Jetson. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// The DEPTH axis: the camera dives INTO the box. Word offsets inside each line were measured on the narration
// (ffmpeg silencedetect on the two mp3s), line-local seconds:
//   line 0 (9.456 s): "Everything live runs on this Jetson." 0.14–2.25 · "Cameras in," 2.73–3.54 · "faces blurred," 3.89–4.92
//                     "the on-device vision model understanding the scene," 5.07–7.61 · "answers in seconds." 8.03–9.04
//   line 1 (6.816 s): "Pull the network cable." 0.20–1.02 · "It keeps seeing," 1.47–2.53 · "remembering," 2.76–3.42
//                     "alerting" 3.68–4.25 · "and answering." 4.42–5.00 · "Only learning pauses." 5.51–6.47
// Hand-over in: HOME (from "swap"). Hand-over out: HOME, layers closed and unlit, cable 0, pulse 0, labels 0.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, SLAB } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  // A word inside line i, measured against the narration length the offsets above were taken from, so the
  // positions scale with the line if it is ever re-spoken.
  const MEASURED = [9.456, 6.816];
  const word = (i, s) => scene.lines[i].start + (s / MEASURED[i]) * scene.lines[i].duration;

  // World-space centre height of layer k once the box is fully open (world.js: layerY(k) + open * k * SLAB.open).
  const LY = [0, 1, 2, 3].map((k) => world.layerY(k) + k * SLAB.open);
  // A camera stop beside layer k: low, front-right, close, looking a touch down at the slab.
  const stop = (k, dx = 0, dz = 0) => ({ x: 2.7 + dx, y: LY[k] + 0.62, z: 3.6 + dz, tx: -0.2, ty: LY[k] - 0.02, tz: 0, fov: 38 });

  // state.layer is an array (GSAP would read it as a list of targets), so each entry is driven through a proxy
  // whose value is a pure function of time; onUpdate only copies it across.
  const lay = LY.map(() => ({ v: 0 }));
  const lightLayer = (i, v, when, dur, ease = "power2.out") =>
    tl.to(lay[i], { v, duration: dur, ease, onUpdate: () => { state.layer[i] = lay[i].v; } }, when);

  // 3D labels beside the layers (declared once; same id returns the same object). One or two words each.
  const labels = ["Cameras", "Blur", "Vision", "Answer"].map((text, i) =>
    world.label("jetson-" + text.toLowerCase(), { text, at: [-1.95, LY[i] + 0.05, 0.95], scale: 0.42 }));
  const showLabel = (i, when) => {
    tl.fromTo(labels[i], { opacity: 0 }, { opacity: 1, duration: 0.6, ease: "sine.inOut" }, when); tl.fromTo(labels[i], { lift: -0.18 }, { lift: 0, duration: 0.6, ease: "power3.out" }, when); }

  // ---- "Everything live runs on this Jetson": rise high over the box and look down into it as it opens. ----
  world.leg(tl, t0, { x: 1.6, y: 7.4, z: 3.6, tx: 0, ty: 0.9, tz: 0, fov: 36 }, 2.6, "power2.inOut");
  tl.to(state, { open: 1, duration: 2.0, ease: "power2.inOut" }, t0 + 0.8);

  // ---- "Cameras in": plunge from above down to the bottom layer. ----
  world.leg(tl, word(0, 2.6), stop(0), 1.2, "power4.out");
  lightLayer(0, 1, word(0, 2.75), 0.5);
  showLabel(0, word(0, 2.85));

  // ---- "faces blurred": climb one layer. ----
  world.leg(tl, word(0, 3.85), stop(1), 1.0, "power4.out");
  lightLayer(1, 1, word(0, 3.95), 0.5);
  showLabel(1, word(0, 4.05));

  // ---- "the on-device vision model understanding the scene": climb again, then a slow drift while it talks. ----
  world.leg(tl, word(0, 5.05), stop(2), 1.0, "power4.out");
  lightLayer(2, 1, word(0, 5.15), 0.5);
  showLabel(2, word(0, 5.25));
  world.leg(tl, word(0, 6.05), stop(2, 0.35, 0.25), 1.9, "none");

  // ---- "answers in seconds": the top layer lights; pull up and back so the whole lit stack reads on the left
  //      and the overlay's question/answer cards take the right third. ----
  world.leg(tl, word(0, 7.95), { x: 3.4, y: 3.9, z: 5.6, tx: 0.9, ty: 0.9, tz: 0, fov: 38 }, 1.35, "power4.out");
  lightLayer(3, 1, word(0, 8.05), 0.5);
  showLabel(3, word(0, 8.15));

  // ---- "Pull the network cable": swing round the left side to behind the box, where the cable leaves it: two
  //      repositioning legs through a waypoint so the camera never crosses over the open stack. The labels have
  //      done their job. The cable snaps out on the word, sliding to the right of frame. ----
  const swingAt = scene.lines[1].start - 0.55;
  world.leg(tl, swingAt, { x: -6.8, y: 4.3, z: 0.6, tx: -1.5, ty: 0.5, tz: -0.6, fov: 38 }, 1.0, "power2.inOut");
  world.leg(tl, swingAt + 1.0, { x: -4.8, y: 4.6, z: -7.4, tx: -1.5, ty: 0.5, tz: -0.8, fov: 38 }, 1.05, "power2.inOut");
  labels.forEach((l) => tl.to(l, { opacity: 0, duration: 0.5, ease: "power2.in" }, swingAt));
  tl.to(state, { cable: 1, duration: 1.0, ease: "power4.out" }, word(1, 0.8));

  // ---- "It keeps seeing, remembering, alerting and answering": the box keeps beating; a slow drift round it. ----
  tl.to(state, { pulse: 1, duration: 1.0, ease: "power2.out" }, word(1, 1.45));
  world.leg(tl, word(1, 1.5), { x: -4.2, y: 4.9, z: -7.9, tx: -1.3, ty: 0.55, tz: -0.7, fov: 38 }, word(1, 5.4) - word(1, 1.5), "none");

  // ---- "Only learning pauses" → hand-over: home, layers closed and dark, cable back in the last second, pulse off. ----
  const homeAt = word(1, 5.5), homeDur = tEnd - 0.1 - homeAt;
  world.leg(tl, homeAt, { ...HOME }, homeDur, "power2.inOut");
  tl.to(state, { open: 0, duration: Math.min(1.8, homeDur), ease: "power2.inOut" }, homeAt + 0.2);
  LY.forEach((_, i) => lightLayer(i, 0, homeAt + 0.1 + i * 0.08, 1.2, "power2.inOut"));
  tl.to(state, { cable: 0, duration: 0.9, ease: "power2.inOut" }, tEnd - 1.0);
  tl.to(state, { pulse: 0, duration: 1.2, ease: "power2.inOut" }, tEnd - 1.4);
  labels.forEach((l) => tl.set(l, { opacity: 0, lift: 0 }, tEnd - 0.05));
}
