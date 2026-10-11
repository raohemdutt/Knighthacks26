// Scene "intro": Atlas. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Beats (the words motivate every move; all times are read from film.json, never hard-coded):
//   0 .. first word      the slabs fall in from `drop` 1 and settle while the overlay shows the wordmark; low 3/4 hero.
//   line 0               "This is Atlas... two camera eyes": a slow push toward the eyes; the eyes warm up (`eyes`).
//   line 1               "It watches... Your video never leaves": the membrane rises while the camera orbits a quarter turn;
//                        the shell is full when "never leaves" lands (start + 0.9 * duration).
//   line 2               "Inside are four layers...": `open` to 1, layers light bottom to top on cameras / vision / memory / agent,
//                        3D labels lift with them.
//   last 3 s             layers close, membrane to 0.3, camera eases to HOME. Hand-over: HOME, membrane 0.3, everything else 0.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME } = world, t0 = scene.start, END = t0 + scene.duration;
  const L = scene.lines, l0 = L[0].start, l1 = L[1].start, l2 = L[2].start;
  const firstWord = l0; // film.yaml lead_s: the title and the box assemble before the first word
  const neverLeaves = l1 + L[1].duration * 0.9; // "Your video never leaves." closes the line
  // Word landings inside line 2 (fractions of its duration): "Cameras" / "A vision model" / "A memory" / "And an agent".
  const wordAt = [0.155, 0.34, 0.575, 0.685].map((f) => l2 + f * L[2].duration);

  // ---- The opening frame: a low three-quarter hero, the slabs still in the air. ----
  const HERO = { x: -4.6, y: 1.0, z: 7.2, tx: 0, ty: 0.55, tz: 0, fov: 34 };
  tl.set(cam, { ...HERO }, t0);
  // The slabs fall in (bottom lands first, see world.js), then a small squash settles the stack.
  const fallEnd = t0 + Math.min(2.4, firstWord * 0.8);
  tl.fromTo(state, { drop: 1 }, { drop: 0, duration: fallEnd - (t0 + 0.2), ease: "power4.out" }, t0 + 0.2);
  tl.to(state, { drop: -0.12, duration: 0.2, ease: "power2.out" }, fallEnd);
  tl.to(state, { drop: 0, duration: 0.6, ease: "power2.inOut" }, fallEnd + 0.2);
  // A gentle creep before the first word, so the hero is never a still.
  world.leg(tl, t0 + 0.2, { z: HERO.z - 0.35, y: HERO.y + 0.1 }, firstWord - 0.2, "none");

  // ---- Line 0: push toward the two camera eyes; they warm up on "two camera eyes". ----
  const EYES = { x: -1.9, y: 0.3, z: 4.9, tx: 0, ty: -0.3, tz: 0.6, fov: 32 };
  world.leg(tl, l0, EYES, L[0].duration * 0.9, "power2.inOut");
  tl.to(state, { pulse: 0.6, duration: 1.6, ease: "power2.inOut" }, l0 + 1.0);
  tl.to(state, { eyes: 1, duration: 0.9, ease: "power2.out" }, l0 + L[0].duration * 0.66);

  // ---- Line 1: pull back and orbit a quarter turn while the privacy shell rises. ----
  const R0 = 7.4, A0 = -0.62, Y0 = 2.0; // from the left-front, through the front (both eyes in view), to the right-front
  const orbitStart = l1 + 2.2;
  world.leg(tl, l1, { x: R0 * Math.sin(A0), y: Y0, z: R0 * Math.cos(A0), tx: 0, ty: 0.35, tz: 0, fov: 34 }, 2.2, "power2.inOut");
  // The orbit is a proxy angle written onto cam.x / cam.y / cam.z (an exact arc, one writer during its window).
  const orb = { a: A0, r: R0, y: Y0 };
  tl.to(orb, {
    a: A0 + Math.PI / 2, r: 7.0, y: 2.6, duration: neverLeaves - orbitStart, ease: "power1.inOut",
    onUpdate: () => { cam.x = orb.r * Math.sin(orb.a); cam.z = orb.r * Math.cos(orb.a); cam.y = orb.y; },
  }, orbitStart);
  tl.to(state, { membrane: 0.35, duration: 2.6, ease: "power2.inOut" }, l1 + 0.8);
  tl.to(state, { membrane: 1, duration: 1.8, ease: "power2.inOut" }, neverLeaves - 1.8);
  tl.to(state, { pulse: 0, duration: 1.2, ease: "power2.inOut" }, l2 - 1.4);

  // ---- Line 2: a high left three-quarter; the box opens; layers light bottom to top with their labels. ----
  const OPEN = { x: 5.4, y: 4.3, z: 6.3, tx: 0, ty: 0.45, tz: 0, fov: 36 }; // right-front-high: the cable stays behind the box
  world.leg(tl, l2 - 0.25, OPEN, 2.4, "power2.inOut");
  tl.to(state, { open: 1, duration: 1.6, ease: "power3.inOut" }, l2 + 0.55);
  // state.layer is an array (GSAP would read it as a list of targets): drive it through proxies that copy across.
  const lay = [0, 1, 2, 3].map(() => ({ v: 0 }));
  const names = ["Cameras", "Vision", "Memory", "Agent"];
  const labels = names.map((text, i) => world.label(`intro-${text.toLowerCase()}`, { text, at: [-2.15, world.layerY(i) + i * world.SLAB.open + 0.04, 1.35], scale: 0.5 }));
  names.forEach((_, i) => {
    tl.to(lay[i], { v: 1, duration: 0.5, ease: "power2.out", onUpdate: () => { state.layer[i] = lay[i].v; } }, wordAt[i]);
    tl.fromTo(labels[i], { opacity: 0, lift: -0.16 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, wordAt[i] + 0.05);
  });
  // A slow climb while the layers light, so the agent layer arrives at eye level.
  world.leg(tl, l2 + 2.3, { x: 5.0, y: 4.9, z: 5.9, ty: 0.6 }, END - 3.0 - (l2 + 2.3), "none");

  // ---- The last 3 s: close, dim, veil to 0.3, ease HOME (the hand-over). ----
  const close = END - 3.0;
  names.forEach((_, i) => {
    tl.to(labels[i], { opacity: 0, lift: 0.12, duration: 0.4, ease: "power2.in" }, close - 0.1);
    tl.to(lay[i], { v: 0, duration: 0.8, ease: "power2.inOut", onUpdate: () => { state.layer[i] = lay[i].v; } }, close + 0.1);
  });
  tl.to(state, { open: 0, duration: 1.5, ease: "power3.inOut" }, close + 0.1);
  tl.to(state, { membrane: 0.3, duration: 2.0, ease: "power2.inOut" }, close + 0.2);
  tl.to(state, { eyes: 0, duration: 1.4, ease: "power2.inOut" }, close + 0.8);
  world.leg(tl, close, { ...HOME }, 2.8, "power2.inOut");
}
