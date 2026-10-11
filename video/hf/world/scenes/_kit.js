// Shared helpers for the v4.7 scenes. Every v4.7 scene opens on a planned hard cut (WORLD.md allows announced cuts):
// `cut` resets every world number to 0 and sets the camera, so a scene never inherits state from the one before it,
// whichever order the beats are in. `at(L, i, f)` is a point inside spoken line i; `camPath` is one continuous move
// through several shots (Catmull-Rom, one ease).

const BASE = {
  explode: 0, xray: 0, spin: 0, drop: 0, eyes: 0, pulse: 0, cable: 0, open: 0,
  membrane: 0, ring: 0, beads: 0, bars: 0, lab: 0, labGlow: 0, labCurve: 0, friends: 0, friendsLink: 0, campus: 0, route: 0, blur: 0,
  room: 0, person: 0, personLook: 0, personSmile: 0, personSleepy: 0, personStrain: 0, personTalk: 0, kitchen: 0, canThrow: 0, binCount: 0,
};
const HL = { webcam: 0, jetson: 0, speaker: 0, lid: 0, front: 0 };

/** Hard cut at time t: every world number back to 0 (then `set`), the camera straight to `shot`. */
export function cut(tl, world, t, shot, set = {}) {
  tl.set(world.state, { ...BASE, ...set }, t);
  tl.set(world.state.hl, { ...HL, ...(set.hl || {}) }, t);
  tl.set(world.cam, { ...shot }, t);
  // Labels and particle flows other scenes left on (they are not on world.state) go too.
  const labels = Object.values(world.labels || {}), flows = Object.values(world.flows || {});
  if (labels.length) tl.set(labels, { opacity: 0 }, t);
  if (flows.length) tl.set(flows, { density: 0 }, t);
}

export const at = (L, i, f = 0) => { const l = L[Math.min(i, L.length - 1)]; return l.start + f * l.duration; };

const KEYS = ["x", "y", "z", "tx", "ty", "tz", "fov"];
const cr = (p0, p1, p2, p3, t) => { const t2 = t * t, t3 = t2 * t; return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3); };
export function camPath(tl, world, start, dur, keys, ease = "power2.inOut") {
  const cam = world.cam, p = { u: 0 }, n = keys.length - 1;
  tl.fromTo(p, { u: 0 }, {
    u: 1, duration: dur, ease, immediateRender: false,
    onUpdate: () => {
      const s = Math.min(Math.max(p.u, 0), 1) * n, i = Math.min(n - 1, Math.floor(s)), f = s - i;
      const k0 = keys[Math.max(0, i - 1)], k1 = keys[i], k2 = keys[i + 1], k3 = keys[Math.min(n, i + 2)];
      for (const k of KEYS) cam[k] = cr(k0[k], k1[k], k2[k], k3[k], f);
    },
  }, start);
}

export const yawTo = (a, b) => Math.atan2(b[0] - a[0], b[2] - a[2]);

/** A slow drift from `shot` toward `shot` nudged by `d` over the scene (the default move of a held shot). */
export function drift(tl, world, start, dur, shot, d = {}) {
  const to = { ...shot }; for (const k of Object.keys(d)) to[k] = shot[k] + d[k];
  world.leg(tl, start, to, Math.max(0.1, dur), "none");
}
