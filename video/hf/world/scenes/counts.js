// Scene "counts": Data into counts. The TIME axis again: frames stream out of the box's eyes and land on the ring
// as beads (events, episodes); then the beads grow into a bar chart (counts) and the camera goes low across the
// bars. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { state, HOME, P } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration;

  // Frames leave the box's eyes and arc onto the ring where the first bead sits (declared once).
  // From the camera eyes (front face of the bottom slab) up over the box to the first bead at the back of the ring.
  const EYES = [P.box[0], P.box[1] - 1.0, P.box[2] + 0.82];
  const frames = world.flow("counts-frames", { from: EYES, to: P.ring(-Math.PI / 2), count: 24, lift: 2.8 });

  // Shots. FOLLOW: up and to the left, so the arc from the box to the back of the ring reads in profile.
  // DOWN: lower, following the particles onto the ring. LOW: desk level, looking across the bars at the box.
  const FOLLOW = { x: -5.6, y: 5.6, z: 7.8, tx: 0.6, ty: -0.3, tz: -1.6, fov: 38 };
  const DOWN = { x: -6.6, y: 3.6, z: 6.4, tx: 0.8, ty: -0.6, tz: -1.0, fov: 40 };
  const LOW = { x: -8.6, y: 1.9, z: 5.0, tx: 1.2, ty: 0.3, tz: -0.6, fov: 40 };
  const LOW_DRIFT = { x: -7.9, y: 2.1, z: 6.2, tx: 1.1, ty: 0.3, tz: -0.5, fov: 40 };

  // ---- Line 0: "The same cameras turn a room into data": the ring rises, frames stream from the eyes. ----
  tl.to(state, { ring: 1, duration: 1.4, ease: "power2.out" }, at(0, 0.02));
  world.leg(tl, at(0, 0.02), FOLLOW, 3.2, "power2.inOut");
  tl.set(frames, { spread: 0.6, progress: 0 }, t0);
  tl.fromTo(frames, { density: 0 }, { density: 1, duration: 0.5, ease: "power2.out" }, at(0, 0.12));
  tl.fromTo(frames, { progress: 0 }, { progress: 1, duration: at(0, 0.98) - at(0, 0.12), ease: "none" }, at(0, 0.12));
  // "Frames become events. Events become episodes": the beads climb to a full day as the particles land.
  tl.to(state, { beads: 48, duration: at(0, 1.0) - at(0, 0.42), ease: "power1.inOut" }, at(0, 0.42));
  world.leg(tl, at(0, 0.02) + 3.2, DOWN, 3.2, "power2.inOut");
  tl.to(frames, { density: 0, duration: 0.5, ease: "power2.in" }, at(0, 0.98));

  // ---- Line 1: "Point it at the bins, and a campus gets a number": the beads grow into bars; camera goes low. ----
  tl.to(state, { bars: 1, duration: 1.8, ease: "power4.out" }, at(1, 0.08));
  world.leg(tl, at(1, 0.02), LOW, 3.0, "power2.inOut");
  // "sorted right, before and after a nudge. Tables share counts, never pictures": a slow drift along the bars.
  world.leg(tl, at(1, 0.02) + 3.0, LOW_DRIFT, Math.max(0.5, tEnd - 1.9 - (at(1, 0.02) + 3.0)), "none");

  // ---- Hand over: HOME, bars down, ring and beads gone, flow parked at density 0. ----
  world.leg(tl, tEnd - 1.9, { ...HOME }, 1.8, "power2.inOut");
  tl.to(state, { bars: 0, duration: 0.9, ease: "power2.inOut" }, tEnd - 1.7);
  tl.to(state, { ring: 0, beads: 0, duration: 0.9, ease: "power2.inOut" }, tEnd - 1.0);
  tl.to(frames, { progress: 0, duration: 0.2, ease: "none" }, tEnd - 0.4);
}
