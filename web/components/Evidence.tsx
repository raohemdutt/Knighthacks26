import { cameraOfFrame, type Citation } from "@/lib/api";
import { cameraName, dayTime, hostOf, safeUrl } from "@/lib/format";
import { Keyframe } from "./Keyframe";
import ui from "./ui.module.css";

/**
 * The evidence behind an answer. Every citation is shown: camera frames as
 * images, outside sources as titled links, and sources with neither (a group
 * chat message, a calendar entry) as a titled line with its time. An answer
 * with no citations says so instead of passing as fact.
 */
export function Evidence({ citations }: { citations: Citation[] }) {
  const frames = citations.filter((c) => c.frame_ref);
  const others = citations.filter((c) => !c.frame_ref);

  if (citations.length === 0) {
    return (
      <div className={ui.evidence}>
        <p className={ui.sectionTitle}>Evidence</p>
        <p className={ui.muted}>None attached. Atlas had nothing on record to back this answer.</p>
      </div>
    );
  }

  return (
    <div className={ui.evidence}>
      <p className={ui.sectionTitle}>
        Evidence · {citations.length} {citations.length === 1 ? "source" : "sources"}
      </p>
      {frames.length > 0 ? (
        <div className={ui.frames}>
          {frames.map((c, i) => {
            const camera = c.frame_ref ? cameraOfFrame(c.frame_ref) : null;
            const caption = [c.title, camera ? cameraName(camera) : null, dayTime(c.ts)].filter(Boolean).join(" · ");
            return <Keyframe key={`${c.frame_ref}-${i}`} frameRef={c.frame_ref as string} caption={caption} />;
          })}
        </div>
      ) : null}
      {others.length > 0 ? (
        <ul className={ui.sources}>
          {others.map((c, i) => {
            const url = safeUrl(c.url);
            return (
              <li key={`${c.url ?? c.title ?? c.episode_id ?? "source"}-${i}`} className={ui.source}>
                {url ? (
                  <a href={url} target="_blank" rel="noreferrer noopener">
                    {c.title || hostOf(url)}
                  </a>
                ) : (
                  <span>{c.title || c.url || (c.episode_id ? `Camera episode ${c.episode_id}` : "Source without a title")}</span>
                )}
                <span className={ui.frameCaption}>{[url ? hostOf(url) : null, dayTime(c.ts)].filter(Boolean).join(" · ")}</span>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
