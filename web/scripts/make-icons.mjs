// Draws the PWA icons (a lens: accent ring and dot on the app's soft light background)
// with nothing but Node's zlib, so the repo needs no image tooling.
//   node scripts/make-icons.mjs      -> public/icons/*.png
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "icons");
const BG = [0xee, 0xf2, 0xf8];
const ACCENT = [0x25, 0x63, 0xeb];

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, "ascii");
  const tail = Buffer.alloc(4);
  tail.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
  return Buffer.concat([head, data, tail]);
}

/** How much of the accent shape covers a point, 0..1. Coordinates are in units of the icon's half-width. */
function coverage(x, y, edge) {
  const d = Math.hypot(x, y);
  const band = (inner, outer) => Math.min(smooth(d - inner, edge), smooth(outer - d, edge));
  return Math.max(band(0.4, 0.52), smooth(0.2 - d, edge));
}

const smooth = (v, edge) => Math.min(1, Math.max(0, v / edge + 0.5));

function png(size) {
  const raw = Buffer.alloc((size * 3 + 1) * size);
  const edge = 2 / size; // one pixel of anti-aliasing
  for (let py = 0; py < size; py++) {
    const row = py * (size * 3 + 1);
    raw[row] = 0; // filter: none
    for (let px = 0; px < size; px++) {
      const a = coverage(((px + 0.5) / size) * 2 - 1, ((py + 0.5) / size) * 2 - 1, edge);
      for (let ch = 0; ch < 3; ch++) raw[row + 1 + px * 3 + ch] = Math.round(BG[ch] + (ACCENT[ch] - BG[ch]) * a);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr.set([8, 2, 0, 0, 0], 8); // 8 bits per channel, RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

mkdirSync(OUT, { recursive: true });
for (const [name, size] of [
  ["icon-192.png", 192],
  ["icon-512.png", 512],
  ["apple-touch-icon.png", 180],
]) {
  writeFileSync(join(OUT, name), png(size));
  console.log(`wrote public/icons/${name} (${size}x${size})`);
}
