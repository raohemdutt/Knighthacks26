// Scene "lives": Same box, other lives. The camera rises high over the campus while three other lives pop up (the
// overlay), then the Day Run route draws itself across the map, and the camera pulls to the final wide hero shot
// with the ring, the membrane and the campus on. Everything stays on for "end".
// Camera + world choreography on the ROOT timeline (absolute seconds). See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { state } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration;

  // High over the campus, a little to the right, the whole map in frame; a slow drift while the cards pop.
  const HIGH = { x: 7.0, y: 25.0, z: 21.0, tx: 0.5, ty: -1.0, tz: -1.2, fov: 40 };
  const HIGH_DRIFT = { x: 4.5, y: 24.0, z: 22.5, tx: 0.5, ty: -1.0, tz: -1.0, fov: 40 };
  // The end scene's opening frame (WORLD.md, end row).
  const HERO = { x: 0, y: 6, z: 16, tx: 0, ty: 0, tz: 0, fov: 38 };

  // ---- Line 0: "Same box, other lives": the campus comes up under everything; the camera rises over it. ----
  tl.to(state, { campus: 1, duration: 1.4, ease: "power2.out" }, t0);
  tl.to(state, { friends: 0, friendsLink: 0, room: 0, kitchen: 0, duration: 0.8, ease: "power2.inOut" }, t0);
  world.leg(tl, t0, HIGH, 2.6, "power2.inOut");
  world.leg(tl, t0 + 2.6, HIGH_DRIFT, L[1].start - (t0 + 2.6), "none");

  // ---- Line 1: "And Day Run. Play tomorrow before you live it": the route draws across the real campus, and each
  // stop is named as the line reaches it (assets/ucf-route.json, baked from OpenStreetMap by campus_bake.py). ----
  const draw = at(1, 0.1);
  tl.to(state, { route: 1, duration: 2.0, ease: "power2.inOut" }, draw);
  world.ucfStops.forEach((s, i) => {
    const l = world.label("lives-stop-" + i, { text: s.name, at: [(s.u - 0.5) * 30, -0.2, (s.v - 0.5) * 30], size: 56, scale: 3.4 });
    tl.fromTo(l, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.45, ease: "power4.out" }, draw + 0.15 + i * 0.6);
    tl.to(l, { opacity: 0, duration: 0.4, ease: "power2.in" }, at(1, 0.6));
  });
  // "on a map of your own campus": pull to the final hero; the ring and the shell come up around the box.
  const pull = at(1, 0.6);
  world.leg(tl, pull, HERO, Math.min(2.8, tEnd - 0.1 - pull), "power2.inOut");
  tl.to(state, { ring: 1, membrane: 1, duration: 1.6, ease: "power2.out" }, pull + 0.6);
}
