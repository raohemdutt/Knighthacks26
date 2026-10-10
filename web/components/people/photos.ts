/** Pictures on their way to the box: shrunk in the browser, sent as base64, never kept by the page. */

const MAX_SIDE = 1600; // the box shrinks to this anyway; sending less keeps a phone photo from being 5 MB of upload
const QUALITY = 0.9;

function base64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** A JPEG still of whatever `source` shows now, as base64 (no "data:" prefix). Not mirrored. */
export function still(source: CanvasImageSource, width: number, height: number, maxSide = MAX_SIDE): string | null {
  if (!width || !height) return null;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  const url = canvas.toDataURL("image/jpeg", QUALITY);
  canvas.width = canvas.height = 0; // let the pixels go at once
  return url.startsWith("data:image/jpeg") ? url.slice(url.indexOf(",") + 1) : null;
}

/**
 * A chosen photo as base64 JPEG, shrunk when the browser can decode it (which also turns it upright). A file the
 * browser cannot draw is sent as it is, and the box says whether it could read it.
 */
export async function photoOf(file: File): Promise<string> {
  try {
    const bitmap = await createImageBitmap(file);
    const out = still(bitmap, bitmap.width, bitmap.height);
    bitmap.close();
    if (out) return out;
  } catch {
    /* not decodable here (HEIC on some browsers): fall through */
  }
  return base64(await file.arrayBuffer());
}
