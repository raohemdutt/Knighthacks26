import { ThreeCanvas } from "@remotion/three";
import { useMemo } from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { Card } from "./Soft";
import { page, theme, type } from "./theme";

/**
 * The tool before the use cases: one soft box resting on the same surface as every card, two camera eyes. The box
 * opens into its layers while one line at a time names them. Depth comes the way it does on the cards: a light from
 * the top left, a soft shadow to the bottom right, no horizon. Primitives only: no model files, renders the same
 * every time.
 */

const LAYERS = ["Cameras, faces blurred", "Qwen VL understands the scene", "Memory of what it saw", "Agent that asks first"];

const LINES = [
  { from: 0, text: "This is Atlas." },
  { from: 120, text: "A private AI box that sees, remembers, talks and learns." },
  { from: 330, text: "Everything runs on the box. Your video never leaves it." },
  { from: 600, text: "Everything after this minute is something you point it at." },
];

const LAYER_H = 0.36;
const GAP = 0.04;
const LAYER = new RoundedBoxGeometry(2.4, LAYER_H, 1.6, 4, 0.08); // soft edges, like the cards' corners

/** A radial gradient, shade in the middle fading to nothing: the soft shadow under the box. */
function useSoftShadow(): THREE.Texture {
  return useMemo(() => {
    const size = 512;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(183, 192, 205, 0.95)");
    g.addColorStop(0.5, "rgba(183, 192, 205, 0.4)");
    g.addColorStop(1, "rgba(183, 192, 205, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }, []);
}

function Box({ open, spin }: { open: number; spin: number }) {
  const bottom = -1.5 * (LAYER_H + GAP + open * 0.5); // the lowest layer's centre, so the eyes stay on it
  return (
    <group position={[-1.9, 0.3, 0]} rotation={[0.38, spin, 0]}>
      {LAYERS.map((label, i) => (
        <mesh key={label} geometry={LAYER} position={[0, (i - 1.5) * (LAYER_H + GAP + open * 0.5), 0]}>
          <meshStandardMaterial color={theme.surface} roughness={1} metalness={0} />
        </mesh>
      ))}
      {[-0.45, 0.45].map((x) => (
        <mesh key={x} position={[x, bottom, 0.801]}>
          <circleGeometry args={[0.07, 48]} />
          <meshBasicMaterial color={theme.ink} />
        </mesh>
      ))}
    </group>
  );
}

function Scene({ open, spin }: { open: number; spin: number }) {
  const shadow = useSoftShadow();
  const floor = -1.35 - open * 0.75;
  return (
    <>
      <color attach="background" args={[theme.surface]} />
      {/* light from the top left, shade to the bottom right: the same rule as every card's shadows */}
      <hemisphereLight args={[theme.light, theme.shade, 2.4]} />
      <directionalLight position={[-6, 9, 7]} intensity={1.5} />
      <Box open={open} spin={spin} />
      <mesh position={[-1.4, floor + 0.001, 0.6]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[7, 7]} />
        <meshBasicMaterial map={shadow} transparent depthWrite={false} />
      </mesh>
    </>
  );
}

export function Intro3D() {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const open = spring({ frame: frame - 420, fps, config: { damping: 20 } });
  const spin = interpolate(frame, [0, 1200], [-0.5, 0.7]);
  const line = [...LINES].reverse().find((l) => frame >= l.from) ?? LINES[0];
  const lineIn = interpolate(frame - line.from, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ backgroundColor: theme.surface }}>
      <ThreeCanvas width={width} height={height} camera={{ position: [0, 1.6, 8.5], fov: 38 }}>
        <Scene open={open} spin={spin} />
      </ThreeCanvas>
      {open > 0.4 ? (
        <AbsoluteFill style={{ justifyContent: "center", alignItems: "flex-end", paddingRight: 160, gap: 28 }}>
          {[...LAYERS].reverse().map((l) => (
            <Card key={l} style={{ padding: "22px 40px", fontSize: type.body * 0.85, fontWeight: 500, opacity: Math.min(1, open) }}>
              {l}
            </Card>
          ))}
        </AbsoluteFill>
      ) : null}
      <AbsoluteFill style={{ ...page, backgroundColor: "transparent", justifyContent: "flex-end", paddingBottom: 110 }}>
        <div style={{ fontSize: type.title * 0.9, fontWeight: 600, opacity: lineIn, maxWidth: 1500 }}>{line.text}</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
}
