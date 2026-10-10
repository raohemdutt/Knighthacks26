import type { Metadata, Viewport } from "next";
import { Quicksand } from "next/font/google";
import type { ReactNode } from "react";
import { Shell } from "@/components/Shell";
import { Splash } from "@/components/Splash";
import { AtlasProvider } from "@/lib/atlas";
import { VoiceProvider } from "@/lib/voice";
import "./globals.css";

// Headings and the wordmark. A soft, rounded face, to match the neumorphic theme and the logo; the body stays on the system font.
const display = Quicksand({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-display", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Atlas", template: "%s · Atlas" },
  description: "Ask your desk what happened, see what is due, and approve what Atlas wants to do.",
  manifest: "/manifest.webmanifest",
  applicationName: "Atlas",
  appleWebApp: { capable: true, title: "Atlas", statusBarStyle: "default" },
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
  themeColor: "#eef2f8",
  colorScheme: "light",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={display.variable}>
      <body>
        <Splash />
        <AtlasProvider>
          <VoiceProvider>
            <Shell>{children}</Shell>
          </VoiceProvider>
        </AtlasProvider>
      </body>
    </html>
  );
}
