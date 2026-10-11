"use client";

import { useEffect, useState } from "react";
import { Logo } from "./Logo";
import styles from "./splash.module.css";

const SHOW_MS = 2000; // matches the fade-out in splash.module.css

/** The logo flashes over the app on every load, then fades away and unmounts. */
export function Splash() {
  const [gone, setGone] = useState(false);
  useEffect(() => {
    const t = window.setTimeout(() => setGone(true), SHOW_MS);
    return () => window.clearTimeout(t);
  }, []);
  if (gone) return null;
  return (
    <div className={styles.splash} aria-hidden="true">
      <div className={styles.mark}>
        <Logo size={132} draw />
        <span className={styles.word}>Atlas</span>
        <span className={styles.tag}>Your desk, remembered</span>
      </div>
    </div>
  );
}
