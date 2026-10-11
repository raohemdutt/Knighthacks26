"use client";

import { useEffect, useRef } from "react";

/**
 * The aurora behind every screen: fields of colour that drift, stretch and turn, never repeating.
 *
 * It is one small canvas (about 160 px across) stretched over the whole window by CSS, so the browser composites a
 * single tiny texture and the upscaling is what makes the fields soft. Drawing four gradients at that size costs
 * nothing, there are no blend modes or giant layers for the GPU to re-rasterise, and the frame rate is capped at
 * 30. The colours are the --aurora-* tokens, re-read when the theme flips. Hidden tabs and reduced motion stop it.
 */

const FIELDS = 4;
const FPS = 30;
const WIDTH = 160; // the canvas is this many pixels wide; its height follows the window's shape

interface Field {
  color: string;
  // the centre wanders on a sum of two slow sines per axis, so no two fields ever fall into step
  ax: number;
  bx: number;
  ay: number;
  by: number;
  fx: number;
  gx: number;
  fy: number;
  gy: number;
  spin: number;
  radius: number;
  squash: number;
}

function fields(colors: string[]): Field[] {
  const out: Field[] = [];
  for (let i = 0; i < FIELDS; i++) {
    out.push({
      color: colors[i % colors.length] as string,
      ax: 0.5 + (i % 2) * 0.3,
      bx: 0.25,
      ay: 0.45 + (i >> 1) * 0.25,
      by: 0.22,
      fx: 0.026 + i * 0.006,
      gx: 0.016 + i * 0.004,
      fy: 0.022 + i * 0.005,
      gy: 0.014 + i * 0.003,
      spin: (i % 2 ? -1 : 1) * (0.02 + i * 0.006),
      radius: 0.78 - i * 0.08,
      squash: 0.72 + i * 0.06,
    });
  }
  return out;
}

/** The same colour at a share of its alpha, for the soft middle of a field. */
function faded(color: string, share: number): string {
  const m = /rgba?\(([^)]+)\)/.exec(color);
  if (!m) return color;
  const parts = (m[1] as string).split(",").map((v) => v.trim());
  const alpha = parts.length > 3 ? Number(parts[3]) : 1;
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${(alpha * share).toFixed(3)})`;
}

function readColors(): string[] {
  const s = getComputedStyle(document.documentElement);
  return ["--aurora-a", "--aurora-b", "--aurora-c", "--aurora-b"].map((v) => s.getPropertyValue(v).trim() || "rgba(91,140,255,0.4)");
}

export function Aurora() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let list = fields(readColors());
    let light = document.documentElement.dataset.theme === "light";
    let frame = 0;
    let last = 0;
    const t0 = performance.now();

    const size = () => {
      const aspect = window.innerHeight / Math.max(1, window.innerWidth);
      canvas.width = WIDTH;
      canvas.height = Math.max(40, Math.round(WIDTH * aspect));
    };

    const paint = (now: number) => {
      const t = (now - t0) / 1000;
      const w = canvas.width;
      const h = canvas.height;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, w, h);
      // Over a light page the fields tint it like watercolour; over a dark one they add light.
      ctx.globalCompositeOperation = light ? "multiply" : "lighter";
      const big = Math.max(w, h);
      for (const f of list) {
        const x = (f.ax + f.bx * Math.sin(t * f.fx * 2 * Math.PI) * Math.cos(t * f.gx * 2 * Math.PI + 1)) * w;
        const y = (f.ay + f.by * Math.cos(t * f.fy * 2 * Math.PI) * Math.sin(t * f.gy * 2 * Math.PI + 2)) * h;
        const r = f.radius * big * (1 + 0.08 * Math.sin(t * f.gy * 4 * Math.PI));
        ctx.setTransform(1, 0, 0, 1, x, y);
        ctx.rotate(t * f.spin);
        ctx.scale(1, f.squash + 0.06 * Math.sin(t * f.fx * 3 * Math.PI));
        // a long, even fade: no edge anywhere, just colour thinning out into the page
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
        g.addColorStop(0, f.color);
        g.addColorStop(0.35, faded(f.color, 0.6));
        g.addColorStop(0.7, faded(f.color, 0.18));
        g.addColorStop(1, faded(f.color, 0));
        ctx.fillStyle = g;
        ctx.fillRect(-r, -r, r * 2, r * 2);
      }
    };

    const tick = (now: number) => {
      frame = 0;
      if (now - last >= 1000 / FPS) {
        last = now;
        paint(now);
      }
      frame = requestAnimationFrame(tick);
    };
    const start = () => {
      if (!frame && !calm && document.visibilityState === "visible") frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };

    size();
    paint(performance.now());
    start();

    const onVisibility = () => (document.visibilityState === "visible" ? start() : stop());
    const onResize = () => {
      size();
      paint(performance.now());
    };
    const watcher = new MutationObserver(() => {
      light = document.documentElement.dataset.theme === "light";
      list = fields(readColors());
      paint(performance.now());
    });
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("resize", onResize);
    return () => {
      stop();
      watcher.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("resize", onResize);
    };
  }, []);
  return <canvas ref={ref} className="aurora" aria-hidden="true" />;
}
