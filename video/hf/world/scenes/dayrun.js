// Scene "dayrun": Day Run, the finale. The campus map fades in under the desk, the camera pulls up and away, the
// route draws with the words; then the thesis: pull all the way back to the hero with the box, its shell, the day
// ring and the campus all in one calm frame. Everything stays on: the "end" scene inherits it. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { state } = world, t0 = scene.start;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration;

  const HIGH = { x: -3.6, y: 15.5, z: 13.5, tx: 1.2, ty: -0.9, tz: -1.6, fov: 42 };
  const HERO = { x: 0, y: 6, z: 16, tx: 0, ty: 0, tz: 0, fov: 38 };

  // ---- Line 0: "Point it at your habits": the campus appears under the desk; the camera pulls up and away. ----
  tl.to(state, { campus: 1, duration: 2.4, ease: "power2.out" }, at(0, 0.03));
  world.leg(tl, at(0, 0.03), HIGH, 3.8, "power2.inOut");
  // "play tomorrow before you live it, on a map of your own campus": the route draws with the words.
  tl.fromTo(state, { route: 0 }, { route: 1, duration: at(0, 1.0) - at(0, 0.36), ease: "power2.inOut" }, at(0, 0.36));

  // ---- Line 1: "the same box, the same cameras, the same memory": all the way back to the hero, calm. ----
  world.leg(tl, L[1].start, HERO, 5.2, "power2.inOut");
  tl.to(state, { ring: 1, duration: 2.0, ease: "power2.out" }, L[1].start + 0.4);
  tl.to(state, { beads: 48, duration: 3.0, ease: "power2.inOut" }, L[1].start + 0.6);
  tl.to(state, { membrane: 1, duration: 2.6, ease: "power2.out" }, L[1].start + 1.2);
}
