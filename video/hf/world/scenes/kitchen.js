// Scene "kitchen": The kitchen. A fast travel to the kitchen set (x = KX), a planned hard cut to the wall camera's own
// view as the person throws a can into the trash, a pan to the counter Atlas keeping the count, then back home.
// Camera + world choreography on the ROOT timeline (absolute seconds). See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { cam, state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration;

  // The wall camera's view holds still (a security camera does not move); the counter device gets a slow drift.
  const C = SHOTS.counterAtlas;
  const COUNTER_DRIFT = { ...C, x: C.x + 0.5, z: C.z + 0.35, fov: 29 };

  // ---- Line 0: "Point it at the kitchen": travel to the kitchen; it rises during the travel. ----
  tl.to(state, { kitchen: 1, duration: 1.0, ease: "power2.out" }, t0);
  world.leg(tl, t0, SHOTS.kitchenWide, 1.55, "power2.inOut");
  // "The bin camera sees": a planned hard cut to the wall camera's view (announced in WORLD.md's hand-over table).
  const cut = at(0, 0.2);
  tl.set(cam, { ...SHOTS.binCam }, cut);
  // "a can go in the trash": the throw lands on "the trash".
  // He drops it in the blue bin after Atlas tells him cans go in recycling (in the hold before the count), seen from the wall camera.
  tl.fromTo(state, { canThrow: 0 }, { canThrow: 1, duration: 1.15, ease: "none", immediateRender: false }, at(0, 1) + 0.05);

  // ---- Line 1: "It keeps the count": pan to the counter Atlas; it lights up as it counts. ----
  // A second planned hard cut, out of the wall camera's view onto the counter device (a pan from the wall camera
  // swept through the person's arm and the bins), then a slow drift.
  tl.set(cam, { ...C }, L[1].start);
  world.leg(tl, L[1].start, COUNTER_DRIFT, tEnd - 1.75 - L[1].start, "none");
  tl.to(state, { binCount: 1, duration: 0.6, ease: "power2.out" }, at(1, 0.08));

  // ---- Line 2: "Only counts leave the box. Never pictures." (the overlay carries it), then home. ----
  const home = tEnd - 1.75;
  world.leg(tl, home, { ...HOME }, 1.65, "power2.inOut");
  tl.to(state, { kitchen: 0, binCount: 0, duration: 0.6, ease: "power2.inOut" }, home + 0.6);
  tl.set(state, { canThrow: 0 }, home + 1.3); // the kitchen is gone by now, so the can does not pop back into view
}
