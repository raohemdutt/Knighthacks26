import type { Metadata, Viewport } from "next";
import { Inter, Outfit } from "next/font/google";
import type { ReactNode } from "react";
import { Aurora } from "@/components/Aurora";
import { GuideLayer } from "@/components/guide/GuideLayer";
import { Shell } from "@/components/Shell";
import { Splash } from "@/components/Splash";
import { AtlasProvider } from "@/lib/atlas";
import { VoiceProvider } from "@/lib/voice";
import "./globals.css";

// Headings and the wordmark: a geometric face with some warmth. The body is Inter, tuned for screens at small sizes.
const display = Outfit({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-display", display: "swap" });
const body = Inter({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-body", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Atlas", template: "%s · Atlas" },
  description: "Ask your desk what happened, see what is due, and approve what Atlas wants to do.",
  manifest: "/manifest.webmanifest",
  applicationName: "Atlas",
  appleWebApp: { capable: true, title: "Atlas", statusBarStyle: "black-translucent" },
  icons: {
    icon: [
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/icons/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#070b14",
  colorScheme: "dark light",
};

/**
 * The theme before the first paint, so a page never flashes the other one. Night is the default; a choice made
 * with the toggle is remembered in localStorage under "atlas.theme". The same key is read by Shell.tsx.
 */
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("atlas.theme");if(t!=="light"&&t!=="dark")t="dark";document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme="dark";}})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`} data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <Aurora />
        <Splash />
        <AtlasProvider>
          <VoiceProvider>
            <Shell>{children}</Shell>
            <GuideLayer />
          </VoiceProvider>
        </AtlasProvider>
      </body>
    </html>
  );
}
