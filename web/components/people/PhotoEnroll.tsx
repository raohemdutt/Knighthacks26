"use client";

import { useRef, useState, type ChangeEvent } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type EnrollPhotosResult } from "@/lib/api";
import p from "./people.module.css";
import { photoOf } from "./photos";

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));
const MAX = 12;

/**
 * Pick photos (or take them with the phone's camera app) and send them to the box. The box answers for each one:
 * taken, or why not. The photos are shrunk here, sent once, and kept nowhere: not on this page, not on the box.
 */
export function PhotoEnroll({ who, consent, least, onDone }: {
  who: { name?: string; member_id?: string; circle_member_id?: string };
  consent: boolean;
  least: number; // 3 for someone new, 1 when adding to someone enrolled
  onDone: (result: EnrollPhotosResult) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [names, setNames] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ done: number; of: number } | undefined>(undefined);
  const [result, setResult] = useState<EnrollPhotosResult | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const busy = progress !== undefined;

  const chosen = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).slice(0, MAX);
    e.target.value = ""; // the same photos can be picked again after a failure
    if (!files.length) return;
    setError(undefined);
    setResult(undefined);
    if (files.length < least) {
      setError(new ApiError(0, `Pick at least ${least} photos (5 to 10 is best).`));
      return;
    }
    setNames(files.map((f, i) => f.name || `Photo ${i + 1}`));
    setProgress({ done: 0, of: files.length });
    try {
      const photos: string[] = [];
      for (const file of files) {
        photos.push(await photoOf(file));
        setProgress({ done: photos.length, of: files.length });
      }
      const answer = await api.enrollPhotos(who, photos, consent);
      setResult(answer);
      if (answer.state === "done") onDone(answer);
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setProgress(undefined);
    }
  };

  return (
    <div className={p.stack}>
      <ul className={p.guide}>
        <li>5 to 10 photos, in good light, with only this person in them.</li>
        <li>Mostly straight on, plus a slight turn to each side.</li>
        <li>With and without glasses, if they wear them.</li>
      </ul>
      <input ref={input} id="face-photos" className={p.picker} type="file" accept="image/*" multiple onChange={chosen} disabled={busy || !consent} />
      <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !consent} onClick={() => input.current?.click()}>
        {busy ? "Working…" : result?.state === "failed" ? "Pick photos again" : "Choose or take photos"}
      </button>
      {progress ? (
        <div role="status" aria-live="polite">
          <p className={ui.muted}>
            {progress.done < progress.of ? `Preparing photo ${progress.done + 1} of ${progress.of}…` : "The box is looking at them…"}
          </p>
          <div className={ui.meter} role="img" aria-label={`${progress.done} of ${progress.of} prepared`}>
            <div className={`${ui.meterFill} ${ui.meterMet}`} style={{ width: `${Math.round((progress.done / progress.of) * 100)}%` }} />
          </div>
        </div>
      ) : null}
      {result ? (
        <div role="status" aria-live="polite" className={p.stack}>
          <p>
            <strong>
              {result.state === "done"
                ? `${result.accepted} of ${result.photos.length} photos taken.`
                : result.error === "already_enrolled"
                  ? "This face is already enrolled as someone else here, so nothing was saved."
                  : `Only ${result.accepted} usable ${result.accepted === 1 ? "photo" : "photos"}; ${result.needed} are needed, so nothing was saved.`}
            </strong>
          </p>
          <ul className={p.photos}>
            {result.photos.map((ph) => (
              <li key={ph.index} className={p.photo}>
                <span className={ph.accepted ? p.photoOk : p.photoNo}>{ph.accepted ? "Taken" : "Not used"}</span>
                <span className={`${ui.grow} ${ui.breakWords}`}>
                  {names[ph.index] ?? `Photo ${ph.index + 1}`}
                  {ph.message ? <span className={ui.muted}>: {ph.message}</span> : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </div>
  );
}
