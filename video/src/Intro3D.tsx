import { ThreeCanvas } from "@remotion/three";
import { useMemo } from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { theme } from "./theme";

/**
 * The tool before the use cases: the Atlas box on a table, two camera eyes, light flowing in from the room and never
 * out. Then the box opens into its layers (cameras + face blur, Qwen VL, memory, agent) while the caption names them.
 * Everything is built from primitives: no model files to fetch, renders the same every time.
 */

const LAYERS = [
  { label: "Cameras + face blur", color: theme.cool },
  { label: "Qwen VL understands the scene", color: "#8f7cf2" },
  { label: "Memory of everything it saw", color: theme.warm },
  { label: "Agent that asks before it acts", color: "#5ad19a" },
];

const LINES = [
  { from: 0, text: "This is Atlas." },
  { from: 120, text: "A private AI box that sees, remembers, talks and learns." },
  { from: 330, text: "Everything runs on the box. Your video never leaves it." },
  { from: 600, text: "Everything after this minute is something you point it at." },
];

function Particles({ frame }: { frame: number }) {
  // Light from the room flowing into the box: points travel inward along fixed rays, a new wave every 90 frames.
  const rays = useMemo(
    () =>
      Array.from({ length: 160 }, (_, i) => {
        const a = (i / 160) * Math.PI * 2;
        const y = Math.sin(i * 12.9898) * 1.4;
        return new THREE.Vector3(Math.cos(a) * 6, y, Math.sin(a) * 6);
      }),
    [],
  );
  const positions = useMemo(() => new Float32Array(rays.length * 3), [rays]);
  rays.forEach((r, i) => {
    const t = ((frame + i * 7) % 90) / 90; // 0 at the edge of the room, 1 inside the box
    positions[i * 3] = r.x * (1 - t);
    positions[i * 3 + 1] = r.y * (1 - t) + 0.2;
    positions[i * 3 + 2] = r.z * (1 - t);
  });
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} count={rays.length} />
      </bufferGeometry>
      <pointsMaterial color={theme.cool} size={0.06} transparent opacity={0.8} />
    </points>
  );
}

function Box({ open, spin }: { open: number; spin: number }) {
  return (
    <group position={[-1.6, 0.35, 0]} rotation={[0.35, spin, 0]}>
      {LAYERS.map((layer, i) => (
        <mesh key={layer.label} position={[0, (i - 1.5) * (0.42 + open * 0.55), 0]}>
          <boxGeometry args={[2.4, 0.38, 1.6]} />
          <meshStandardMaterial color={open > 0.05 ? layer.color : "#2a2e36"} metalness={0.3} roughness={0.45} />
        </mesh>
      ))}
      {[-0.5, 0.5].map((x) => (
        <mesh key={x} position={[x, 0.1, 0.82 + open * 0.2]}>
          <sphereGeometry args={[0.14, 32, 32]} />
          <meshStandardMaterial color="#0b0c0e" emissive={theme.cool} emissiveIntensity={0.6} />
        </mesh>
      ))}
    </group>
  );
}

export function Intro3D() {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const open = spring({ frame: frame - 420, fps, config: { damping: 18 } });
  const spin = interpolate(frame, [0, 1200], [-0.6, 0.9]);
  const line = [...LINES].reverse().find((l) => frame >= l.from) ?? LINES[0];
  const lineIn = interpolate(frame - line.from, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <ThreeCanvas width={width} height={height} camera={{ position: [0, 1.2, 9], fov: 40 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[4, 6, 5]} intensity={1.4} />
        <pointLight position={[-4, 2, 3]} intensity={0.8} color={theme.warm} />
        <Particles frame={frame} />
        <Box open={open} spin={spin} />
        <mesh position={[0, -1.25 - open * 0.8, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[30, 30]} />
          <meshStandardMaterial color="#1a1c21" />
        </mesh>
      </ThreeCanvas>
      {open > 0.5 ? (
        <AbsoluteFill style={{ justifyContent: "center", alignItems: "flex-end", paddingRight: 140 }}>
          {[...LAYERS].reverse().map((l) => (
            <div key={l.label} style={{ color: l.color, fontSize: 40, fontWeight: 600, margin: "26px 0", opacity: open }}>
              {l.label}
            </div>
          ))}
        </AbsoluteFill>
      ) : null}
      <AbsoluteFill style={{ justifyContent: "flex-end", padding: "0 160px 110px" }}>
        <div style={{ color: theme.text, fontSize: 60, fontWeight: 600, opacity: lineIn }}>{line.text}</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
}
