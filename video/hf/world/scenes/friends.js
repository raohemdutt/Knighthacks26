// Scene "friends": Friends. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Word offsets inside each line (silence gaps measured on the narration mp3s, line-local seconds):
//   line 0: "Point it at your friends." 0.12–0.95 · "Each Atlas keeps its owner's calendar private." 1.33–3.36
//   line 1: "Ask for a time for everyone," 0.12–1.30 · "and the boxes talk." 1.55–2.39 · "Free." 2.85–3.13
//           "Busy then, how about three." 3.49–4.65 · "Agreed," 5.02–5.42 · "three of three in." 5.72–6.48
//           "Only free or busy leaves each box." 6.84–9.26
// Starts from whatever the previous scene left (a 3 s leg covers a trip back from the kitchen).
// Hand-over: HOME, friends 0, friendsLink 0, every flow's density 0.
// The overlay (compositions/friends.html) pins its chips, bubbles and calendars to the FRIENDS shot:
//   box (957,558) · friend 0 (1395,743) · friend 1 (482,667) · friend 2 (1005,348) px; the drift moves them < 20 px.
export default function choreograph(tl, world, scene) {
  const { state, HOME, P } = world, t0 = scene.start, END = t0 + scene.duration;
  const W = (i, s) => scene.lines[i].start + s;

  const FRIENDS = { x: 0.8, y: 9.5, z: 14.5, tx: 0, ty: -0.6, tz: -0.6, fov: 40 };
  const FRIENDS_D = { x: 0.7, y: 9.8, z: 14.8, tx: -0.05, ty: -0.65, tz: -0.65, fov: 39.6 };

  // Flows (declared once): answers in from each friend, the proposal and the event out to each.
  const inflow = P.friend.map((f, i) => world.flow("friends-in" + i, { from: f, to: P.box, count: 6, size: 1.2, lift: 1.4 }));
  const outflow = P.friend.map((f, i) => world.flow("friends-out" + i, { from: P.box, to: f, count: 6, size: 1.2, lift: 1.4 }));
  const all = [...inflow, ...outflow];
  all.forEach((f) => tl.set(f, { density: 1, spread: 0.35, progress: 0 }, t0));
  const pulse = (f, at, dur = 1.3) => tl.fromTo(f, { progress: 0 }, { progress: 1, duration: dur, ease: "power1.inOut", immediateRender: false }, at);

  // ---- line 0: "Point it at your friends": pull up; three Atlases rise around the desk, each in its own shell. ----
  world.leg(tl, t0, FRIENDS, 3.0, "power2.inOut");
  tl.to(state, { friends: 1, duration: 1.8, ease: "power4.out" }, t0 + 0.6);
  world.leg(tl, t0 + 3.0, FRIENDS_D, W(1, 9.0) - (t0 + 3.0), "none");

  // ---- line 1: "and the boxes talk": links draw, the question goes out. ----
  tl.to(state, { friendsLink: 1, duration: 1.0, ease: "power4.out" }, W(1, 1.55));
  outflow.forEach((f, i) => pulse(f, W(1, 1.7) + i * 0.12));
  // "Free." (friend 0) · "Busy then, how about three." (friend 1) · "Agreed, three of three in." (friend 2)
  pulse(inflow[0], W(1, 2.6));
  pulse(inflow[1], W(1, 3.3));
  pulse(inflow[2], W(1, 4.8));
  // The Approve tap (overlay, right after "three of three in"): the event goes out to all three.
  outflow.forEach((f, i) => pulse(f, W(1, 6.2) + i * 0.1, 1.0));
  // "Only free or busy leaves each box": one quiet round of answers under the overlay's tokens.
  inflow.forEach((f, i) => pulse(f, W(1, 6.9) + i * 0.2, 1.6));

  // ---- Home: links and flows out, the friends sink, the camera returns. ----
  const homeAt = END - 1.95;
  all.forEach((f) => tl.to(f, { density: 0, duration: 0.6, ease: "power2.in" }, homeAt - 0.3));
  tl.to(state, { friendsLink: 0, duration: 0.8, ease: "power2.inOut" }, homeAt - 0.2);
  world.leg(tl, homeAt, { ...HOME }, 1.85, "power2.inOut");
  tl.to(state, { friends: 0, duration: 1.3, ease: "power2.inOut" }, homeAt + 0.3);
  all.forEach((f) => tl.set(f, { progress: 0 }, END - 0.05));
}
