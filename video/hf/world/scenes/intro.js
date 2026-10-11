// Scene "intro": Atlas. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { cam, state } = world, t0 = scene.start;
  world.leg(tl, t0, { x: 2.0, y: 2.0, z: 7.8, fov: 34 }, scene.duration * 0.9, "none");
}
