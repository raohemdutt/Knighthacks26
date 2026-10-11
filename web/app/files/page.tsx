"use client";

import { useEffect, useRef, useState } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, ApiError, fileUrl, uploadFile, type InboxFile } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayTime } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import fs from "./files.module.css";

/** "pdf", "png": the extension shown on the file's badge, at most four letters. */
function extOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 && dot < name.length - 1 ? name.slice(dot + 1, dot + 5).toLowerCase() : "";
}

/** The badge's colour family, from the file's type. */
function kindOf(f: InboxFile): "doc" | "image" | "text" | "other" {
  if (f.content_type.startsWith("image/")) return "image";
  if (f.content_type === "application/pdf" || /word|document|presentation|sheet/.test(f.content_type)) return "doc";
  if (f.content_type.startsWith("text/")) return "text";
  return "other";
}

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

      <input ref={input} type="file" multiple hidden onChange={(e) => add(e.target.files)} aria-label="Choose files" />
      <button type="button" className={fs.drop} onClick={() => input.current?.click()} disabled={busy} aria-busy={busy}>
        <span className={fs.dropMark} aria-hidden="true">
          {busy ? (
            <span className={fs.spinner} />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4M7 9l5-5 5 5M4.5 15.5v3a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-3" />
            </svg>
          )}
        </span>
        <span className={fs.dropText}>
          <span className={fs.dropTitle}>{busy ? "Adding…" : "Add files"}</span>
          <span className={fs.dropSub}>Or share one from any app. Kept encrypted on the box, read, and searchable from Ask and Timeline.</span>
        </span>
      </button>
      {note ? (
        <p className={fs.note} role="status">
          {note}
        </p>
      ) : null}
      {error ? <ErrorBox what="the file" error={error} /> : null}

      {files.error ? (
        <ErrorBox what="the inbox" error={files.error} onRetry={files.reload} />
      ) : !files.data ? (
        <Loading what="files" />
      ) : files.data.length === 0 ? (
        <Empty title="No files yet">A syllabus, a screenshot of a flyer, lecture notes: shared files show up here, newest first.</Empty>
      ) : (
        <ul className={fs.list}>
          {files.data.map((f: InboxFile) => (
            <li key={f.file_id}>
              <a href={fileUrl(f.file_id)} className={fs.file}>
                <span className={`${fs.kind} ${fs[`kind_${kindOf(f)}`] ?? ""}`} aria-hidden="true">
                  {extOf(f.name) || "file"}
                </span>
                <span className={fs.body}>
                  <span className={fs.name}>{f.name}</span>
                  <span className={fs.meta}>
                    {dayTime(f.added_at)} · {size(f.size)} ·{" "}
                    {f.text_chars ? (
                      `${f.text_chars.toLocaleString()} characters read`
                    ) : (
                      <span className={fs.unread}>kept, not searchable (no text found)</span>
                    )}
                  </span>
                  {f.excerpt ? <span className={fs.excerpt}>{f.excerpt}</span> : null}
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
