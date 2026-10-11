// Slot 5: On the Jetson. Camera + world choreography on the ROOT timeline; every time is t0 + seconds. See WORLD.md.
//
// The DEPTH axis: the camera dives INTO the box. Word timings (voice.json, slot-local):
//   line 0 at 1.5  "Everything you see live runs on this Jetson." (1.5–4.5) "Cameras in," (~4.6) "faces blurred," (~5.4)
//                  "the on-device vision model understanding the scene," (~6.4) "answers in a few seconds." (~9.8, ends 11.77)
//   line 1 at 12.37 "Pull the network cable." (12.4–14.0) "It keeps seeing, remembering, alerting and answering." (14.0–17.7)
//                  "Only learning pauses." (17.7–19.2)
export default function choreograph(tl, world, t0) {
  const { cam, state, HOME } = world;
  const at = (s) => t0 + s;

  // Layer centres when the box is open (world.js: y = i*0.4 - 0.6 + open*i*0.55).
  const LY = [-0.6, 0.35, 1.3, 2.25];
  // A camera stop beside layer k: low, front-right, close, looking slightly down at the slab.
  const stop = (k, dx = 0, dz = 0) => ({ x: 2.3 + dx, y: LY[k] + 0.55, z: 3.1 + dz, tx: -0.25, ty: LY[k] - 0.02, tz: 0, fov: 38 });

  // state.layer is an array; GSAP would treat it as a list of targets, so drive it through proxies (seek-safe: the
  // proxy value is a pure function of time and onUpdate only copies it across).
  const lay = LY.map(() => ({ v: 0 }));
  const lightLayer = (i, v, when, dur, ease = "power2.out") =>
    tl.to(lay[i], { v, duration: dur, ease, onUpdate: () => { state.layer[i] = lay[i].v; } }, at(when));

  // 3D labels beside the layers (declared once; same id returns the same object).
  const labels = [
    world.label("s5-cameras", { text: "Cameras", at: [-1.95, LY[0] + 0.05, 0.95], scale: 0.42 }),
    world.label("s5-blur", { text: "Faces blurred", at: [-1.95, LY[1] + 0.05, 0.95], scale: 0.42 }),
    world.label("s5-vision", { text: "Vision model", at: [-1.95, LY[2] + 0.05, 0.95], scale: 0.42 }),
    world.label("s5-answer", { text: "Answers", at: [-1.95, LY[3] + 0.05, 0.95], scale: 0.42 }),
  ];
  const showLabel = (i, when) => {
    tl.fromTo(labels[i], { opacity: 0, lift: -0.18 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, at(when));
  };

  // ---- Title card: home, calm. ----
  tl.set(cam, { ...HOME }, t0);

  // ---- "Everything you see live runs on this Jetson": rise high over the box and look down into it. ----
  world.leg(tl, at(1.5), { x: 1.6, y: 7.4, z: 3.6, tx: 0, ty: 0.9, tz: 0, fov: 36 }, 3.0, "power2.inOut");
  tl.to(state, { open: 1, duration: 2.2, ease: "power2.inOut" }, at(2.4));

  // ---- "Cameras in": plunge from above down to the bottom layer. ----
  world.leg(tl, at(4.5), stop(0), 1.3, "power4.out");
  lightLayer(0, 1, 4.6, 0.5);
  showLabel(0, 4.7);

  // ---- "faces blurred": climb one layer. ----
  world.leg(tl, at(5.35), stop(1), 1.0, "power4.out");
  lightLayer(1, 1, 5.45, 0.5);
  showLabel(1, 5.5);

  // ---- "the on-device vision model understanding the scene": climb again, then a slow drift while it talks. ----
  world.leg(tl, at(6.4), stop(2), 1.0, "power4.out");
  lightLayer(2, 1, 6.5, 0.5);
  showLabel(2, 6.55);
  world.leg(tl, at(7.4), stop(2, 0.35, 0.25), 2.4, "none");

  // ---- "answers in a few seconds": the top layer lights; pull up and back so the whole lit stack reads. ----
  world.leg(tl, at(9.8), { x: 3.4, y: 3.9, z: 5.6, tx: -0.1, ty: 0.9, tz: 0, fov: 38 }, 1.5, "power4.out");
  lightLayer(3, 1, 9.9, 0.5);
  showLabel(3, 9.95);

  // ---- "Pull the network cable": swing round to the back-left where the cable leaves the box; the cable snaps out. ----
  world.leg(tl, at(12.4), { x: -6.2, y: 2.1, z: -4.2, tx: -2.6, ty: -0.3, tz: -0.3, fov: 38 }, 1.5, "power2.inOut");
  tl.to(state, { cable: 1, duration: 1.2, ease: "power4.out" }, at(13.2));

  // ---- "It keeps seeing, remembering, alerting and answering": the box keeps beating. ----
  tl.to(state, { pulse: 1, duration: 1.2, ease: "power2.out" }, at(14.0));
  // Labels have done their job once the camera is round the back.
  labels.forEach((l) => tl.to(l, { opacity: 0, duration: 0.5, ease: "power2.in" }, at(12.4)));

  // ---- After the line: rise back out to a wide three-quarter shot, close the layers, keep the cable pulled. ----
  world.leg(tl, at(19.6), { x: -4.4, y: 3.0, z: 7.2, tx: -0.9, ty: 0.1, tz: 0, fov: 35 }, 3.6, "power2.inOut");
  tl.to(state, { open: 0, duration: 2.2, ease: "power2.inOut" }, at(20.4));
  LY.forEach((_, i) => lightLayer(i, 0, 20.6 + i * 0.1, 1.2, "power2.inOut"));
  // A slow drift while the box runs offline (the overlay keeps the chips up).
  world.leg(tl, at(23.2), { x: -3.4, y: 2.7, z: 7.7, tx: -0.7, ty: 0.15, tz: 0, fov: 35 }, 30.8, "none");

  // ---- Home. Cable back in at 56, pulse off, everything reset by 59.0. ----
  world.leg(tl, at(54.0), { ...HOME }, 4.0, "power2.inOut");
  tl.to(state, { cable: 0, duration: 1.6, ease: "power2.inOut" }, at(56.0));
  tl.to(state, { pulse: 0, duration: 2.0, ease: "power2.inOut" }, at(56.4));
  labels.forEach((l) => tl.set(l, { opacity: 0, lift: 0 }, at(58.9)));
}
