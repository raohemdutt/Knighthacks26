// Scene "friends": Friends' Atlases. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// The SPACE axis toward the friends. Word offsets inside each line were measured on the narration (ffmpeg
// silencedetect on the two mp3s), line-local seconds:
//   line 0 (4.512 s): "Point it at your friends." 0.16–1.10 · "Each person's Atlas keeps their own calendar private." 1.57–4.10
//   line 1 (9.912 s): "Ask for a time that works for everyone," 0.14–2.28 · "and the boxes negotiate." 2.69–3.96
//                     "Only free or busy leaves each box." 4.53–6.51 · "The event lands" 6.74–7.44 · "after a tap." 7.90–9.49
// Hand-over in: HOME (from "counts"). Hand-over out: HOME, friends 0, friendsLink 0, every flow's density 0.
export default function choreograph(tl, world, scene) {
  const { state, HOME, P } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const MEASURED = [4.512, 9.912];
  const word = (i, s) => scene.lines[i].start + (s / MEASURED[i]) * scene.lines[i].duration;

  // The wide shot: up and back over the desk, the three shells around it, the box a touch left of centre.
  // The overlay (compositions/friends.html) places its chips, rails and calendars against the midpoint of the
  // slow push from WIDE to CLOSER: box ≈ (942,546) px, friend 0 ≈ (1418,724), friend 1 ≈ (424,656), friend 2 ≈ (992,354).
  const WIDE = { x: 1.5, y: 14.5, z: 25.0, tx: 0.3, ty: 0, tz: -0.6, fov: 38 };
  const CLOSER = { x: 1.2, y: 13.9, z: 24.0, tx: 0.3, ty: 0, tz: -0.6, fov: 38 };

  // Particle flows (declared once): free/busy answers in from each friend, the proposal and the event back out.
  const inflow = P.friend.map((f, i) => world.flow("friends-f" + i, { from: f, to: P.box, count: 6, size: 1.2 }));
  const outflow = P.friend.map((f, i) => world.flow("friends-o" + i, { from: P.box, to: f, count: 6, size: 1.2 }));
  const all = [...inflow, ...outflow];
  const pulse = (flows, when, stagger = 0.22, dur = 1.6) =>
    flows.forEach((f, i) => tl.fromTo(f, { progress: 0 }, { progress: 1, duration: dur, ease: "power1.inOut" }, when + i * stagger));
  all.forEach((f) => tl.set(f, { density: 1, spread: 0.35, progress: 0 }, t0));

  // ---- "Point it at your friends": pull up and back; the friends' boxes rise around the desk as the shot opens. ----
  world.leg(tl, t0, WIDE, 3.2, "power2.inOut");
  tl.fromTo(state, { friends: 0 }, { friends: 1, duration: 2.0, ease: "power3.out" }, t0 + 0.6);

  // ---- "Ask for a time that works for everyone": a slow push for the whole negotiation (the hold never dies). ----
  world.leg(tl, scene.lines[1].start, CLOSER, word(1, 9.0) - scene.lines[1].start, "none");

  // ---- "and the boxes negotiate": links draw shell to shell, answers stream in, the proposal streams out. ----
  tl.fromTo(state, { friendsLink: 0 }, { friendsLink: 1, duration: 1.0, ease: "power4.out" }, word(1, 2.7));
  pulse(inflow, word(1, 3.0));
  pulse(outflow, word(1, 4.1), 0.2, 1.5);

  // ---- "Only free or busy leaves each box": one quiet round of answers under the overlay's tokens. ----
  pulse(inflow, word(1, 4.9), 0.25, 1.6);

  // ---- "The event lands after a tap": the agreed event goes out to all three after the approve press. ----
  pulse(outflow, word(1, 7.8), 0.14, 1.3);

  // ---- Home: links fade, flows out, the friends sink away, everything reset before the scene ends. ----
  const homeAt = tEnd - 1.95;
  all.forEach((f) => tl.to(f, { density: 0, duration: 0.7, ease: "power2.in" }, homeAt - 0.4));
  tl.to(state, { friendsLink: 0, duration: 0.9, ease: "power2.inOut" }, homeAt - 0.2);
  world.leg(tl, homeAt, { ...HOME }, 1.85, "power2.inOut");
  tl.to(state, { friends: 0, duration: 1.4, ease: "power2.inOut" }, homeAt + 0.3);
  all.forEach((f) => tl.set(f, { progress: 0 }, tEnd - 0.05));
}
