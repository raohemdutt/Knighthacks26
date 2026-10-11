"use client";

import type { ReactNode } from "react";
import ui from "@/components/ui.module.css";

/**
 * Every screen enters the same way: the whole page rises and fades in, and its first few blocks follow a beat
 * behind each other (ui.module.css, .page). A template remounts on every navigation, which is what restarts it.
 */
export default function Template({ children }: { children: ReactNode }) {
  return <div className={ui.page}>{children}</div>;
}
