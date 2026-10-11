import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/**
 * Production: a static export in web/out/ that the gateway serves itself.
 * `trailingSlash` makes every route a folder with an index.html
 * (/timeline/ -> out/timeline/index.html), which is what a plain static file
 * server can resolve without rewrite rules.
 *
 * Development (`next dev`): set NEXT_PUBLIC_ATLAS_URL to a running gateway and
 * /api, /memory, /identity, /capture, /runtime and /health are proxied to it, so the browser sees one origin
 * and the gateway needs no CORS. The WebSocket connects to the gateway directly.
 */
export default function config(phase) {
  const base = {
    reactStrictMode: true,
    trailingSlash: true,
    images: { unoptimized: true },
  };
  if (phase !== PHASE_DEVELOPMENT_SERVER) return { ...base, output: "export" };

  const gateway = (process.env.NEXT_PUBLIC_ATLAS_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
  return {
    ...base,
    skipTrailingSlashRedirect: true, // keep /api/ask as /api/ask on its way to the gateway
    async rewrites() {
      return [
        { source: "/api/:path*", destination: `${gateway}/api/:path*` },
        { source: "/memory/:path*", destination: `${gateway}/memory/:path*` },
        { source: "/identity/:path*", destination: `${gateway}/identity/:path*` },
        { source: "/capture/:path*", destination: `${gateway}/capture/:path*` },
        { source: "/runtime/:path*", destination: `${gateway}/runtime/:path*` },
        { source: "/health", destination: `${gateway}/health` },
      ];
    },
  };
}
