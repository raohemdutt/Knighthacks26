// Scene "intro" (beat 1): what's inside. One narrator line: "Inside: a camera, a speaker, and a Jetson that runs
// the vision, the memory, and an agent that acts for you." Camera + world choreography on the ROOT timeline.
//
// Beats: the box settles from a glassy outline to solid (the lead) -> on "Inside" the box goes see-through and opens into its exploded
// diagram, already open before the first part is named -> each part lights with its label as it is said (camera,
// speaker, Jetson), the Jetson's pulse carries "the vision, the memory, and an agent" -> it closes and lands on HOME.
// Word times are character fractions of the spoken line, so they follow the words.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, SHOTS } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines, hl = state.hl, l0 = L[0];
  const word = (w, d = 0) => { const k = l0.text.indexOf(w); return l0.start + (k < 0 ? 0 : k / l0.text.length) * l0.duration + d; };

  // ---- Opening: the hero three-quarter while the parts fall in (the lead, before the first word). ----
  const OPEN = { x: -4.3, y: 0.35, z: 6.2, tx: 0, ty: 0.15, tz: 0, fov: 30 };
  tl.set(cam, OPEN, t0);
  // One calm reveal instead of parts flying in: the box appears as a glassy outline, turns a little and settles solid
  // while the camera glides in (no part moves on its own).
  tl.set(state, { drop: 0, explode: 0, xray: 1, spin: -0.08, eyes: 0, pulse: 0, membrane: 0, room: 0, person: 0 }, t0);
  tl.set(hl, { webcam: 0, jetson: 0, speaker: 0, lid: 0, front: 0 }, t0);
  const settle = Math.max(1.2, l0.start - t0 - 0.4);
  tl.to(state, { xray: 0, duration: settle, ease: "power2.inOut" }, t0 + 0.2);
  tl.to(state, { spin: 0, duration: settle + 0.2, ease: "power3.out" }, t0);
  world.leg(tl, t0 + 0.1, { ...SHOTS.hero, x: -3.7, y: 0.0, z: 5.3, ty: 0.0 }, Math.max(0.5, l0.start - t0 - 0.1), "power2.out");
  tl.to(state, { eyes: 1, duration: 0.6, ease: "power2.out" }, l0.start - 0.5);

  // ---- "Inside:": see-through and open, a steady swing to the diagram view, open before "camera". ----
  const OPEN_AT = l0.start;
  const firstPart = word("camera");
  const openDur = Math.max(0.8, Math.min(1.4, firstPart - OPEN_AT - 0.1));
  tl.to(state, { xray: 1, duration: openDur, ease: "power2.inOut" }, OPEN_AT);
  tl.to(state, { explode: 1, duration: openDur, ease: "power2.out" }, OPEN_AT);
  world.leg(tl, OPEN_AT, { x: 5.6, y: 2.8, z: 7.4, tx: 0.2, ty: 0.75, tz: -0.1, fov: 42 }, openDur + 0.2, "power2.inOut"); // the whole exploded stack, lid and webcam included
  const CLOSE = Math.max(word("agent") + 0.6, END - 1.8);
  world.leg(tl, OPEN_AT + openDur + 0.2, { x: 5.2, y: 2.6, z: 7.0, tx: 0.2, ty: 0.75, tz: -0.1, fov: 41 }, Math.max(0.3, CLOSE - (OPEN_AT + openDur + 0.2)), "none");

  // ---- Each part lights with its label as it is named; they stay up so the diagram reads as a whole. ----
  const off = (p, dx, dy, dz) => [p[0] + dx, p[1] + dy, p[2] + dz];
  const lWeb = world.label("intro-webcam", { text: "Camera · sees", at: off(world.at("webcam", 1), -1.05, -0.02, 0.7), size: 64, scale: 0.55 });
  const lSpk = world.label("intro-speaker", { text: "Speaker · talks", at: off(world.at("speaker", 1), -0.95, 0.12, 0.75), size: 64, scale: 0.55 });
  const lJet = world.label("intro-jetson", { text: "Jetson · vision, memory, agent", at: off(world.at("jetson", 1), -1.0, 0.45, 0.55), size: 48, scale: 0.75 });
  const named = [["webcam", lWeb, word("camera")], ["speaker", lSpk, word("speaker")], ["jetson", lJet, word("Jetson")]];
  for (const [, lab] of named) tl.set(lab, { opacity: 0, lift: -0.25 }, t0);
  named.forEach(([k, lab, w], i) => {
    tl.to(hl, { [k]: 1, duration: 0.35, ease: "power2.out" }, w - 0.05);
    tl.to(lab, { opacity: 1, lift: 0, duration: 0.5, ease: "power4.out" }, w);
    if (i < 2) tl.to(hl, { [k]: 0.25, duration: 0.4, ease: "power2.inOut" }, named[i + 1][2] - 0.05);
  });
  tl.to(state, { pulse: 1, duration: 0.6, ease: "power2.out" }, word("vision"));

  // ---- Close up again and land on HOME by the end of the scene. ----
  [lWeb, lSpk, lJet].forEach((lab) => tl.to(lab, { opacity: 0, duration: 0.35, ease: "power2.in" }, CLOSE - 0.2));
  tl.to(hl, { webcam: 0, speaker: 0, jetson: 0, duration: 0.6, ease: "power2.inOut" }, CLOSE);
  tl.to(state, { pulse: 0, eyes: 0, duration: 0.8, ease: "power2.inOut" }, CLOSE);
  tl.to(state, { explode: 0, duration: Math.min(1.3, END - CLOSE - 0.2), ease: "power2.inOut" }, CLOSE);
  tl.to(state, { xray: 0, duration: Math.min(1.0, END - CLOSE - 0.3), ease: "power2.inOut" }, CLOSE + 0.2);
  world.leg(tl, CLOSE, { ...HOME }, Math.max(0.3, END - CLOSE - 0.1), "power2.inOut");
}
