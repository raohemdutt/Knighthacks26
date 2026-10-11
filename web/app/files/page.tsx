"use client";

import { useEffect, useRef, useState } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, ApiError, fileUrl, uploadFile, type InboxFile } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayTime } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** The file inbox (W1): files shared from this phone, kept encrypted on the box and searchable from Timeline and Ask. */
export default function FilesPage() {
  const files = useFetch((s) => api.files(s));
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  // Arriving from the share sheet: /files/?shared=2
  useEffect(() => {
    const n = new URLSearchParams(window.location.search).get("shared");
    if (n) setNote(`${n} shared item${n === "1" ? "" : "s"} added to the inbox.`);
  }, []);

  useAtlasEvents(["workspace.file.added"], () => files.reload());

  const add = async (list: FileList | null) => {
    if (!list || list.length === 0) return;
    setBusy(true);
    setError(null);
    setNote(null);
    let added = 0;
    let same = 0;
    try {
      for (const f of Array.from(list)) {
        const r = await uploadFile(f);
        if (r.duplicate) same += 1;
        else added += 1;
      }
      setNote(`${added} added${same ? `, ${same} already in the inbox` : ""}.`);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, e instanceof Error ? e.message : String(e)));
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
      files.reload();
    }
  };

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Files</h1>
      </div>
      <p className={ui.muted}>
        Share a file from any app (or pick one here). Atlas keeps it encrypted on the box, reads its text and makes it searchable:
        ask about it, or search for it on Timeline.
      </p>
      <div className={ui.row}>
        <input ref={input} type="file" multiple hidden onChange={(e) => add(e.target.files)} aria-label="Choose files" />
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={() => input.current?.click()} disabled={busy}>
          {busy ? "Adding..." : "Add files"}
        </button>
      </div>
      {note ? <p className={ui.small}>{note}</p> : null}
      {error ? <ErrorBox what="the file" error={error} /> : null}

      {files.error ? (
        <ErrorBox what="the inbox" error={files.error} onRetry={files.reload} />
      ) : !files.data ? (
        <Loading what="files" />
      ) : files.data.length === 0 ? (
        <Empty title="No files yet">A syllabus, a screenshot of a flyer, lecture notes: shared files show up here, newest first.</Empty>
      ) : (
        <ul className={ui.list}>
          {files.data.map((f: InboxFile) => (
            <li key={f.file_id} className={ui.card}>
              <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
                <a href={fileUrl(f.file_id)} className={ui.breakWords}>
                  <strong>{f.name}</strong>
                </a>
                <span className={ui.tag}>{size(f.size)}</span>
              </div>
              <p className={`${ui.muted} ${ui.small}`}>
                {dayTime(f.added_at)} · {f.text_chars ? `${f.text_chars.toLocaleString()} characters read` : "kept, not searchable (no text found)"}
              </p>
              {f.excerpt ? <p className={ui.small}>{f.excerpt}</p> : null}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
