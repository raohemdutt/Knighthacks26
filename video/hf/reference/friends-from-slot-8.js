// Slot 8: Friends' Atlases. Camera + world choreography on the ROOT timeline; every time is t0 + seconds. See WORLD.md.
//
// The SPACE axis toward the friends. Word timings (voice.json, slot-local):
//   line 0 at 1.5  "Point it at your friends." (1.5–2.9) "Each person's Atlas keeps their own calendar private." (2.9–6.0)
//   line 1 at 6.61 "Ask for a time that works for everyone," (6.6–9.7) "and the boxes negotiate." (9.7–11.6)
//                  "Only free or busy leaves each box." (11.6–14.3) "The event lands in everyone's plan after a tap." (14.3–17.75)
export default function choreograph(tl, world, t0) {
  const { cam, state, HOME, P } = world;
  const at = (s) => t0 + s;

  // The wide shot: up and back, the desk centred a little left so the overlay's screen card has the right third.
  // (Projected with CLOSER: box ≈ (734,551) px, friend 0 ≈ (1116,703), friend 1 ≈ (243,634), friend 2 ≈ (834,388);
  // the overlay's 2D bits are placed against those numbers, so CLOSER holds still from 9.7 s to 20 s.)
  const WIDE = { x: 6.8, y: 15.2, z: 29.5, tx: 4.5, ty: 0.2, tz: -0.9, fov: 37 };
  const CLOSER = { x: 6.5, y: 14.5, z: 28.0, tx: 4.4, ty: 0.2, tz: -0.8, fov: 37 };

  // Labels over each friend's shell (declared once).
  const labels = P.friend.map((f, i) => world.label("s8-private-" + i, { text: "Private calendar", at: [f[0], f[1] + 3.1, f[2]], scale: 1.25 }));

  // Particle flows: free/busy answers in from each friend, the agreed time back out. Declared once, pulsed many times.
  const inflow = P.friend.map((f, i) => world.flow("s8-in-" + i, { from: f, to: P.box, count: 6, size: 1.2 }));
  const outflow = P.friend.map((f, i) => world.flow("s8-out-" + i, { from: P.box, to: f, count: 6, size: 1.2 }));
  const pulse = (flows, when, stagger = 0.22, dur = 1.6) => {
    flows.forEach((f, i) => tl.fromTo(f, { progress: 0 }, { progress: 1, duration: dur, ease: "power1.inOut" }, at(when + i * stagger)));
  };
  [...inflow, ...outflow].forEach((f) => tl.set(f, { density: 1, spread: 0.35, progress: 0 }, t0));

  // ---- Title card: home, calm. ----
  tl.set(cam, { ...HOME }, t0);

  // ---- "Point it at your friends": pull up and back; the friends' boxes rise around the desk as the shot opens. ----
  world.leg(tl, at(1.5), WIDE, 3.4, "power2.inOut");
  tl.fromTo(state, { friends: 0 }, { friends: 1, duration: 2.0, ease: "power3.out" }, at(2.2));

  // ---- "Each person's Atlas keeps their own calendar private": a label settles over each shell. ----
  labels.forEach((l, i) => tl.fromTo(l, { opacity: 0, lift: -0.5 }, { opacity: 1, lift: 0, duration: 0.8, ease: "power3.out" }, at(3.3 + i * 0.25)));

  // ---- "Ask for a time that works for everyone": ease in a touch. ----
  world.leg(tl, at(6.7), CLOSER, 3.0, "power2.inOut");
  labels.forEach((l) => tl.to(l, { opacity: 0, duration: 0.6, ease: "power2.in" }, at(9.3)));

  // ---- "and the boxes negotiate": links draw shell to shell, answers stream in, the proposal streams out. ----
  tl.fromTo(state, { friendsLink: 0 }, { friendsLink: 1, duration: 1.2, ease: "power4.out" }, at(9.8));
  pulse(inflow, 10.2);
  pulse(outflow, 11.4);

  // ---- "Only free or busy leaves each box": one more quiet round trip under the overlay's tokens. ----
  pulse(inflow, 12.4, 0.18, 1.8);

  // ---- "The event lands in everyone's plan after a tap": the agreed event goes out to all three. ----
  pulse(outflow, 17.3, 0.12, 1.5);

  // ---- The hold: the recording plays in the overlay; the circle keeps talking now and then. ----
  world.leg(tl, at(20.0), { ...CLOSER, x: 5.6, y: 14.2, z: 28.6, tx: 4.2 }, 32.0, "none");
  pulse(inflow, 26.0, 0.3, 2.0); pulse(outflow, 29.0, 0.3, 2.0);
  pulse(inflow, 38.0, 0.3, 2.0); pulse(outflow, 41.0, 0.3, 2.0);
  pulse(inflow, 48.0, 0.3, 2.0);

  // ---- Home: links fade, friends sink away, flows out, everything reset by 59.0. ----
  tl.to(state, { friendsLink: 0, duration: 1.2, ease: "power2.inOut" }, at(53.6));
  world.leg(tl, at(54.0), { ...HOME }, 4.2, "power2.inOut");
  tl.to(state, { friends: 0, duration: 2.2, ease: "power2.inOut" }, at(55.4));
  [...inflow, ...outflow].forEach((f) => tl.to(f, { density: 0, duration: 0.8, ease: "power2.in" }, at(52.6)));
  [...inflow, ...outflow].forEach((f) => tl.set(f, { progress: 0 }, at(58.9)));
  labels.forEach((l) => tl.set(l, { opacity: 0, lift: 0 }, at(58.9)));
}
