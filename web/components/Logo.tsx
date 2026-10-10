import { useId } from "react";

/**
 * The Atlas mark: a camera lens ring with an "A" inside it, the dot of the lens as the A's crossbar.
 * `draw` animates the strokes in (the load splash); everywhere else it is static.
 */
export function Logo({ size = 40, draw = false, className }: { size?: number; draw?: boolean; className?: string }) {
  const gradient = useId();
  const d = draw ? "draw" : undefined;
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      role="img"
      aria-label="Atlas"
    >
      <defs>
        <linearGradient id={gradient} x1="12" y1="10" x2="88" y2="92" gradientUnits="userSpaceOnUse">
          <stop offset="0" style={{ stopColor: "var(--accent)" }} />
          <stop offset="1" style={{ stopColor: "var(--accent-text)" }} />
        </linearGradient>
      </defs>
      <circle data-logo={d && "ring"} cx="50" cy="50" r="44" pathLength="1" stroke={`url(#${gradient})`} strokeWidth="6" />
      <path data-logo={d && "a"} d="M30 73L50 27L70 73" pathLength="1" stroke={`url(#${gradient})`} strokeWidth="7" />
      <circle data-logo={d && "dot"} cx="50" cy="59" r="5.5" fill={`url(#${gradient})`} />
    </svg>
  );
}
