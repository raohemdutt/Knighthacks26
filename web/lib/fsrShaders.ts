/**
 * FSR 1 (EASU + RCAS) port: GLSL ES 3.00 fragment shaders for WebGL2.
 *
 * This is a port of the algorithm AMD published as FidelityFX Super Resolution 1.0
 * (`ffx_fsr1.h`, by Timothy Lottes, Advanced Micro Devices). It is NOT AMD's file and it
 * has not been checked line by line against it: it was written from the published
 * algorithm with no copy of the header to hand. Treat it as a port, not the reference.
 *
 * Where this port differs from AMD's reference:
 *  1. Taps. The reference fetches the 12 EASU taps with four `textureGather` calls per
 *     colour channel. WebGL2 (GLSL ES 3.00) has no `textureGather`, so each of the 12 taps
 *     is one `texelFetch`. The 12-tap pattern, the four-quadrant direction and length
 *     estimate, the rotated and stretched kernel, the negative-lobe window and the final
 *     clamp to the min/max of the four nearest texels follow the reference.
 *  2. Maths. The reference uses bit-trick approximations for 1/x and 1/sqrt(x)
 *     (`APrxLoRcpF1`, `APrxLoRsqF1`, `APrxMedRcpF1`). This port uses exact division and
 *     `inversesqrt`, with small epsilons where the reference relies on the approximation
 *     never returning infinity. Results differ in the last bits.
 *  3. Precision. One `highp float` path. There is no 16-bit ("H") path and no packed
 *     variant, and the constants (`con0`..`con3`) are worked out from two uniforms in the
 *     shader instead of being set up on the CPU.
 *  4. Edges. `texelFetch` does not clamp, so tap coordinates are clamped to the image by
 *     hand. The reference leaves that to the sampler.
 *  5. RCAS. The five-tap cross, the per-channel limiter, the 0.25 - 1/16 lobe limit and the
 *     `exp2(-sharpness)` scale follow the reference, and the optional noise term
 *     (`FSR_RCAS_DENOISE`) is always on. Two divisions are guarded against zero.
 *  6. Not ported: the `FsrEasuL` / `FsrRcasL` single-channel variants, film grain
 *     (`FsrLfga`), the tone mapper (`FsrSrtm`) and temporal dither (`FsrTepd`).
 *  7. Alpha is not upscaled: the pipeline is opaque and writes alpha 1.
 *
 * Colour space: both passes work on perceptual (gamma-encoded) colour, as AMD specifies.
 * See `fsr.ts` for how the linear scene gets there.
 *
 * ---------------------------------------------------------------------------------------
 * FidelityFX Super Resolution 1.0 is
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ---------------------------------------------------------------------------------------
 */

/** One triangle that covers the screen. Every pass works from gl_FragCoord, so it needs no UVs. */
export const FULLSCREEN_VERT = /* glsl */ `
in vec3 position;
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/**
 * Linear to perceptual. The scene target is an sRGB texture, so a fetch returns linear
 * light; this writes the sRGB-encoded value into a plain RGBA8 texture, which is what
 * EASU, RCAS and the bilinear comparison all read.
 */
export const ENCODE_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uScene;
out vec4 outColor;
vec3 toSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
void main() {
  outColor = vec4(toSrgb(texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).rgb), 1.0);
}
`;

/** EASU: edge-adaptive spatial upsampling, from the low-resolution image to the output size. */
export const EASU_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uInput; // perceptual colour, at the internal resolution
uniform vec2 uInOverOut;  // internal size / output size
out vec4 outColor;

ivec2 gMax;
vec3 tap(ivec2 p) {
  return texelFetch(uInput, clamp(p, ivec2(0), gMax), 0).rgb;
}
// Twice the luma, the cheap way the reference does it.
float luma2(vec3 c) {
  return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// One of the four "+" shaped neighbourhoods around the nearest texels:
//    a
//  b c d
//    e
// It adds its gradient direction and how edge-like it is, weighted bilinearly.
void easuSet(inout vec2 dir, inout float len, float w, float lA, float lB, float lC, float lD, float lE) {
  float lenX = max(abs(lD - lC), abs(lC - lB));
  float dirX = lD - lB;
  dir.x += dirX * w;
  lenX = clamp(abs(dirX) / max(lenX, 1.0e-6), 0.0, 1.0);
  len += lenX * lenX * w;

  float lenY = max(abs(lE - lC), abs(lC - lA));
  float dirY = lE - lA;
  dir.y += dirY * w;
  lenY = clamp(abs(dirY) / max(lenY, 1.0e-6), 0.0, 1.0);
  len += lenY * lenY * w;
}

// One filter tap: rotate the offset onto the edge direction, stretch it, and weigh it with
// the reference's polynomial approximation of a windowed Lanczos 2 lobe.
void easuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len, float lob, float clp, vec3 c) {
  vec2 v = vec2(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x) * len;
  float d2 = min(dot(v, v), clp);
  float wB = (2.0 / 5.0) * d2 - 1.0;
  float wA = lob * d2 - 1.0;
  wB *= wB;
  wA *= wA;
  wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
  float w = wB * wA;
  aC += c * w;
  aW += w;
}

void main() {
  gMax = textureSize(uInput, 0) - 1;
  // Where this output pixel falls in the input image, in texels, measured from texel centres.
  vec2 pp = gl_FragCoord.xy * uInOverOut - 0.5;
  vec2 fp = floor(pp);
  pp -= fp;
  ivec2 ip = ivec2(fp);

  // The 12 taps ('f' is the texel at ip):
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  vec3 b = tap(ip + ivec2(0, -1));
  vec3 c = tap(ip + ivec2(1, -1));
  vec3 e = tap(ip + ivec2(-1, 0));
  vec3 f = tap(ip);
  vec3 g = tap(ip + ivec2(1, 0));
  vec3 h = tap(ip + ivec2(2, 0));
  vec3 i = tap(ip + ivec2(-1, 1));
  vec3 j = tap(ip + ivec2(0, 1));
  vec3 k = tap(ip + ivec2(1, 1));
  vec3 l = tap(ip + ivec2(2, 1));
  vec3 n = tap(ip + ivec2(0, 2));
  vec3 o = tap(ip + ivec2(1, 2));
  float bL = luma2(b), cL = luma2(c), eL = luma2(e), fL = luma2(f), gL = luma2(g), hL = luma2(h);
  float iL = luma2(i), jL = luma2(j), kL = luma2(k), lL = luma2(l), nL = luma2(n), oL = luma2(o);

  // Direction and edge strength, gathered around f, g, j and k.
  vec2 dir = vec2(0.0);
  float len = 0.0;
  easuSet(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
  easuSet(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
  easuSet(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);
  easuSet(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);

  // Normalise the direction; with no gradient at all, fall back to the x axis.
  float dirR = dot(dir, dir);
  bool zro = dirR < 1.0 / 32768.0;
  dirR = zro ? 1.0 : inversesqrt(dirR);
  dir.x = zro ? 1.0 : dir.x;
  dir *= dirR;
  // Edge strength from {0..2} to {0..1}, shaped with a square.
  len = len * 0.5;
  len *= len;
  // Stretch the kernel from 1 (axis-aligned edge) to sqrt(2) (diagonal edge).
  float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
  // Anisotropic kernel: along the edge it widens, across the edge it narrows to half.
  vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
  // The negative lobe gets stronger on edges; the window ends where the lobe does.
  float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
  float clp = 1.0 / lob;

  vec3 min4 = min(min(f, g), min(j, k));
  vec3 max4 = max(max(f, g), max(j, k));
  vec3 aC = vec3(0.0);
  float aW = 0.0;
  easuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b);
  easuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c);
  easuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i);
  easuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j);
  easuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f);
  easuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e);
  easuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k);
  easuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l);
  easuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h);
  easuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g);
  easuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, o);
  easuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n);
  // Normalise and de-ring: never leave the range of the four nearest texels.
  outColor = vec4(min(max4, max(min4, aC / aW)), 1.0);
}
`;

/**
 * RCAS: robust contrast-adaptive sharpening of the EASU image, written to the screen.
 *
 * This pass also draws the comparison split, which is not part of FSR: left of `uSplitX`
 * it shows a plain bilinear upscale of the same low-resolution image.
 */
export const RCAS_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uEasu;  // perceptual colour, at the output resolution
uniform sampler2D uLow;   // the low-resolution image, bilinear filtered (comparison only)
uniform vec2 uOutSize;
uniform float uCon;       // exp2(-sharpness): 1 is the sharpest
uniform float uSplitX;    // output pixels; below zero the split is off
out vec4 outColor;

const float RCAS_LIMIT = 0.25 - 1.0 / 16.0;

float luma2(vec3 c) {
  return c.b * 0.5 + (c.r * 0.5 + c.g);
}

void main() {
  if (uSplitX >= 0.0) {
    float side = gl_FragCoord.x - uSplitX;
    if (abs(side) < 1.0) {
      outColor = vec4(0.953, 0.961, 0.969, 1.0);
      return;
    }
    if (side < 0.0) {
      outColor = vec4(texture(uLow, gl_FragCoord.xy / uOutSize).rgb, 1.0);
      return;
    }
  }

  ivec2 sp = ivec2(gl_FragCoord.xy);
  ivec2 mx = textureSize(uEasu, 0) - 1;
  //    b
  //  d e f
  //    h
  vec3 b = texelFetch(uEasu, clamp(sp + ivec2(0, -1), ivec2(0), mx), 0).rgb;
  vec3 d = texelFetch(uEasu, clamp(sp + ivec2(-1, 0), ivec2(0), mx), 0).rgb;
  vec3 e = texelFetch(uEasu, sp, 0).rgb;
  vec3 f = texelFetch(uEasu, clamp(sp + ivec2(1, 0), ivec2(0), mx), 0).rgb;
  vec3 h = texelFetch(uEasu, clamp(sp + ivec2(0, 1), ivec2(0), mx), 0).rgb;

  float bL = luma2(b), dL = luma2(d), eL = luma2(e), fL = luma2(f), hL = luma2(h);
  // Noise detection: an isolated spike is sharpened only half as much.
  float nz = 0.25 * (bL + dL + fL + hL) - eL;
  float range = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);
  nz = clamp(abs(nz) / max(range, 1.0e-5), 0.0, 1.0);
  nz = -0.5 * nz + 1.0;

  // The most sharpening each channel can take without leaving {0..1}.
  vec3 mn4 = min(min(b, d), min(f, h));
  vec3 mx4 = max(max(b, d), max(f, h));
  vec3 hitMin = min(mn4, e) / max(4.0 * mx4, vec3(1.0e-5));
  vec3 hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1.0e-5));
  vec3 lobeRGB = max(-hitMin, hitMax);
  float lobe = max(-RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * uCon;
  lobe *= nz;

  vec3 pix = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
  outColor = vec4(clamp(pix, 0.0, 1.0), 1.0);
}
`;
