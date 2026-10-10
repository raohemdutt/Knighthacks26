import type { ReactNode } from "react";
import type { ApiError } from "@/lib/api";
import ui from "./ui.module.css";

export function Loading({ what }: { what: string }) {
  return (
    <div className={ui.loading} role="status" aria-live="polite">
      <span className={ui.spinner} aria-hidden="true" />
      <span>Loading {what}…</span>
    </div>
  );
}

/** Nothing to show yet: say what would appear here, and when. */
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className={ui.state}>
      <p className={ui.stateTitle}>{title}</p>
      {children ? <p>{children}</p> : null}
    </div>
  );
}

export function ErrorBox({ what, error, onRetry }: { what: string; error: ApiError; onRetry?: () => void }) {
  return (
    <div className={`${ui.state} ${ui.stateError}`} role="alert">
      <p className={ui.stateTitle}>Couldn&apos;t load {what}</p>
      <p className={ui.breakWords}>
        {error.message}
        {error.status ? ` (HTTP ${error.status})` : ""}
      </p>
      {error.needsToken ? <p>This box asks for a token. Add it under Settings, top right.</p> : null}
      {onRetry ? (
        <button type="button" className={ui.btn} onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

/** A failed write (ask, approve, share): shown inline next to the control that failed. */
export function WriteError({ error }: { error: ApiError }) {
  return (
    <p className={`${ui.state} ${ui.stateError} ${ui.breakWords}`} role="alert">
      {error.needsToken
        ? "This box asks for a token before it accepts changes. Add it under Settings, top right."
        : `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`}
    </p>
  );
}
