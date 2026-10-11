// Scene "intro": This is Atlas. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Beats: the parts fall into place under the wordmark (0..2.5) -> "This is Atlas": a slow push, the lens wakes ->
// line 1: a quarter orbit from the left three-quarter to the right -> line 2: the privacy membrane rises ->
// line 3: x-ray, explode, each part lights as it is named, labels on the diagram -> reassemble and land on HOME.
// Word times are character fractions of each spoken line (line.start + f * line.duration), so they follow the words.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, SHOTS } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines, hl = state.hl;
  const word = (i, w, d = 0) => { const l = L[i], k = l.text.indexOf(w); return l.start + (k < 0 ? 0 : k / l.text.length) * l.duration + d; };

  // ---- Opening: the hero three-quarter, a touch wide, while the parts fall in (0 .. 2.5). ----
  const OPEN = { x: -4.3, y: 0.35, z: 6.2, tx: 0, ty: 0.15, tz: 0, fov: 30 };
  tl.set(cam, OPEN, t0);
  tl.set(state, { drop: 1, explode: 0, xray: 0, eyes: 0, pulse: 0, membrane: 0, room: 0, person: 0 }, t0);
  tl.fromTo(state, { drop: 1 }, { drop: 0, duration: 2.1, ease: "power3.out" }, t0 + 0.15);
  world.leg(tl, t0 + 0.15, { ...SHOTS.hero, x: -3.7, y: 0.0, z: 5.3, ty: 0.0 }, L[0].start - t0 - 0.15, "power2.out");

  // ---- "This is Atlas. The personal AI computer.": a slow push in; the webcam lens glows. ----
  world.leg(tl, L[0].start, { x: -2.9, y: -0.2, z: 4.0, tx: 0, ty: -0.12, tz: 0, fov: 30 }, L[1].start - L[0].start, "none");
  tl.to(state, { eyes: 1, duration: 0.8, ease: "power2.out" }, word(0, "Atlas"));

  // ---- Line 1: a quarter orbit, left three-quarter -> right three-quarter, steady. ----
  // A proxy angle drives the camera on a circle (seek-safe: cam is a pure function of the proxy).
  const orbit = { a: 0 };
  const R0 = Math.hypot(-2.9, 4.0), A0 = Math.atan2(-2.9, 4.0); // angle measured from +z toward +x
  const A1 = A0 + Math.PI / 2, DR = 1.3;
  tl.fromTo(orbit, { a: 0 }, {
    a: 1, duration: L[1].duration + 0.3, ease: "sine.inOut",
    onUpdate: () => {
      const a = A0 + (A1 - A0) * orbit.a, r = R0 + DR * Math.sin(Math.PI * Math.min(1, orbit.a * 1.6) / 2);
      cam.x = Math.sin(a) * r; cam.z = Math.cos(a) * r; cam.y = -0.2 + 1.2 * orbit.a; cam.ty = -0.12 - 0.3 * orbit.a;
      cam.tx = 0; cam.tz = 0; cam.fov = 30;
    },
  }, L[1].start);

  // ---- Line 2: "Everything it knows about you stays on the device": the membrane rises around the box. ----
  tl.fromTo(state, { membrane: 0 }, { membrane: 1, duration: 1.4, ease: "power2.out" }, L[2].start + 0.1);
  world.leg(tl, L[2].start, { x: 3.4, y: 1.25, z: 4.4, tx: 0, ty: -0.4, tz: 0, fov: 31 }, L[2].duration - 1.2, "none");

  // ---- Line 3: the centrepiece. X-ray and explode start in line 2's tail so the diagram is open on "camera". ----
  const OPEN_AT = L[3].start - 1.2;
  tl.to(state, { membrane: 0, duration: 0.7, ease: "power2.in" }, OPEN_AT - 0.2);
  tl.to(state, { eyes: 0, duration: 0.6 }, OPEN_AT);
  tl.to(state, { xray: 1, duration: 1.0, ease: "power2.inOut" }, OPEN_AT);
  tl.to(state, { explode: 1, duration: 1.5, ease: "power4.out" }, OPEN_AT + 0.25);
  world.leg(tl, OPEN_AT, { x: 4.9, y: 2.3, z: 6.3, tx: 0.45, ty: 0.3, tz: -0.25, fov: 36 }, 1.6, "power2.inOut");
  const CLOSE = Math.min(word(3, "assistant") - 0.1, END - 1.9);
  world.leg(tl, OPEN_AT + 1.6, { x: 4.6, y: 2.15, z: 6.0, tx: 0.45, ty: 0.3, tz: -0.25, fov: 36 }, CLOSE - (OPEN_AT + 1.6), "none");

  // Labels beside each part on the exploded diagram (offset to screen-left of the part from the exploded view).
  const off = (p, dx, dy, dz) => [p[0] + dx, p[1] + dy, p[2] + dz];
  const lWeb = world.label("intro-webcam", { text: "Webcam · sees", at: off(world.at("webcam", 1), -1.05, -0.02, 0.7), size: 64, scale: 0.55 });
  const lSpk = world.label("intro-speaker", { text: "Speaker · talks", at: off(world.at("speaker", 1), -0.95, 0.12, 0.75), size: 64, scale: 0.55 });
  const lJet = world.label("intro-jetson", { text: "Jetson Orin Nano · vision, memory, agent", at: off(world.at("jetson", 1), -1.0, 0.45, 0.55), size: 44, scale: 0.8 });

  const named = [["webcam", lWeb, word(3, "camera")], ["speaker", lSpk, word(3, "speaker")], ["jetson", lJet, word(3, "Jetson")]];
  named.forEach(([k, lab, w], i) => {
    tl.to(hl, { [k]: 1, duration: 0.35, ease: "power2.out" }, w - 0.05);
    tl.fromTo(lab, { opacity: 0, lift: -0.25 }, { opacity: 1, lift: 0, duration: 0.5, ease: "power4.out" }, w);
    if (i < 2) tl.to(hl, { [k]: 0, duration: 0.4, ease: "power2.inOut" }, named[i + 1][2] - 0.05);
  });
  tl.to(state, { pulse: 1, duration: 0.6, ease: "power2.out" }, word(3, "Jetson"));

  // ---- Reassemble, membrane down, home by the end of the scene. ----
  [lWeb, lSpk, lJet].forEach((lab) => tl.to(lab, { opacity: 0, duration: 0.35, ease: "power2.in" }, CLOSE - 0.2));
  tl.to(hl, { jetson: 0, duration: 0.6, ease: "power2.inOut" }, CLOSE);
  tl.to(state, { pulse: 0, duration: 0.8, ease: "power2.inOut" }, CLOSE);
  tl.to(state, { explode: 0, duration: 1.3, ease: "power2.inOut" }, CLOSE);
  tl.to(state, { xray: 0, duration: 1.0, ease: "power2.inOut" }, CLOSE + 0.4);
  world.leg(tl, CLOSE, { ...HOME }, END - CLOSE - 0.1, "power2.inOut");
}
