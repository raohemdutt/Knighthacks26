import * as THREE from "three";
import { EASU_FRAG, ENCODE_FRAG, FULLSCREEN_VERT, RCAS_FRAG } from "./fsrShaders";

/**
 * FSR 1 (EASU + RCAS) port as a three.js post-process. The shaders, and how they differ
 * from AMD's reference, are in `fsrShaders.ts`.
 *
 * A frame goes through four steps:
 *
 *   scene  ->  [1] low-resolution target, sRGB8_ALPHA8, multisampled
 *          ->  [2] encode: low-resolution RGBA8 holding perceptual (sRGB-encoded) values
 *          ->  [3] EASU:   output-resolution RGBA8, still perceptual
 *          ->  [4] RCAS:   the canvas
 *
 * Colour space. FSR 1 must be fed perceptual colour: EASU finds edges by comparing lumas
 * and RCAS limits sharpening by local contrast, and both only match what the eye sees in a
 * gamma-encoded signal. three.js does the opposite by default: into a render target it
 * writes linear light, and skips the sRGB conversion it applies when drawing to the canvas.
 * So the conversion is done here, in the open:
 *  - Target [1] is an sRGB texture (`SRGB8_ALPHA8`). three.js still shades and blends in
 *    linear light; the GPU encodes on write, so 8 bits are spent perceptually and the dark
 *    floor does not band the way linear RGBA8 would.
 *  - Sampling an sRGB texture decodes it back to linear, and WebGL2 has no way to read the
 *    stored bytes as they are. Pass [2] therefore fetches each texel (linear), applies the
 *    sRGB curve and writes a plain RGBA8 texture. It is a same-size copy at the low
 *    resolution, the cheapest pass in the chain.
 *  - EASU and RCAS read and write those perceptual values untouched, and RCAS writes them
 *    straight to the canvas. The canvas is sRGB and these materials are raw shaders, so
 *    three.js adds no second conversion.
 * Tone mapping is not applied in a render target either; this pipeline assumes
 * `NoToneMapping` (the three.js default), so values are already in 0..1.
 *
 * FSR 1 also expects anti-aliased input, which is why target [1] is multisampled.
 *
 * With the scale at 1 (FSR off) the frame still goes through target [1], at full size, and
 * pass [2] writes it to the canvas. That costs one copy, and it is there for a reason:
 * see-through surfaces blend in linear light inside target [1], but in gamma space when
 * three.js draws straight to the canvas, so the two would not be the same colour and
 * switching FSR on and off would shift every translucent surface. `setBypass(true)` draws
 * straight to the canvas instead (the benchmark uses it for an honest baseline), and that
 * is also what happens when the pipeline cannot run.
 *
 * The pipeline is opaque: give the renderer an opaque clear colour.
 */

/** Share of the display resolution rendered, per axis. 1 means FSR is off. */
export type FsrScale = 0.5 | 0.67 | 0.77 | 1;

export interface FsrOptions {
  /** Internal resolution as a share of the output, per axis. Default 0.67 (AMD's "Quality" is 1/1.5). */
  scale?: number;
  /** RCAS sharpness in stops, as AMD defines it: 0 is the sharpest, each +1 halves it. Default 0.2. */
  sharpness?: number;
  /** For testing the fallback: behave as if WebGL2 were missing, or as if a shader failed to compile. */
  simulate?: "webgl1" | "badshader" | null;
}

export interface FsrInfo {
  /** False when the pipeline cannot run here; `render` then draws the scene normally. */
  supported: boolean;
  /** Why it is not supported, in plain words. */
  reason: string | null;
  /** True when this frame goes through EASU and RCAS (supported, not bypassed, and scale below 1). */
  active: boolean;
  scale: number;
  compare: boolean;
  /** Where the comparison divides the canvas, 0..1 of its width. */
  split: number;
  /** The size the scene is rendered at, in device pixels. */
  internal: [number, number];
  /** The size of the canvas, in device pixels. */
  output: [number, number];
}

export interface FsrPipeline {
  /** Draw one frame to the canvas. */
  render(scene: THREE.Scene, camera: THREE.Camera): void;
  /** Size the canvas (CSS pixels) and every target. The device pixel ratio is capped at 2. */
  setSize(cssWidth: number, cssHeight: number, devicePixelRatio?: number): void;
  setScale(scale: number): void;
  setSharpness(stops: number): void;
  /** Draw straight to the canvas, with no targets and no passes at all. */
  setBypass(on: boolean): void;
  /** Split the canvas at `at` (0..1 of its width, default the middle): left a bilinear upscale of the low-resolution image, right FSR. */
  setCompare(on: boolean, at?: number): void;
  info(): FsrInfo;
  dispose(): void;
}

const MAX_PIXEL_RATIO = 2;
let warned = false;

function warnOnce(reason: string) {
  if (warned) return;
  warned = true;
  console.warn(`FSR 1 is off, rendering normally: ${reason}`);
}

/** Compile and link one pass with the raw context, so a failure is known before anything is drawn with it. */
function linkError(gl: WebGL2RenderingContext, name: string, fragment: string): string | null {
  const compile = (type: number, source: string): WebGLShader | string => {
    const shader = gl.createShader(type);
    if (!shader) return "could not create a shader";
    gl.shaderSource(shader, `#version 300 es\n${source}`);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
    const log = gl.getShaderInfoLog(shader) || "no log";
    gl.deleteShader(shader);
    return log;
  };
  const vs = compile(gl.VERTEX_SHADER, FULLSCREEN_VERT);
  if (typeof vs === "string") return `${name} vertex shader: ${vs}`;
  const fs = compile(gl.FRAGMENT_SHADER, fragment);
  if (typeof fs === "string") {
    gl.deleteShader(vs);
    return `${name} fragment shader: ${fs}`;
  }
  const program = gl.createProgram();
  let error: string | null = null;
  if (!program) error = `${name}: could not create a program`;
  else {
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) error = `${name} link: ${gl.getProgramInfoLog(program) || "no log"}`;
    gl.deleteProgram(program);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return error;
}

export function createFsrPipeline(renderer: THREE.WebGLRenderer, options: FsrOptions = {}): FsrPipeline {
  let scale = clampScale(options.scale ?? 0.67);
  let sharpness = Math.max(0, options.sharpness ?? 0.2);
  let compare = false;
  let split = 0.5;
  let bypass = false;
  let output: [number, number] = [1, 1];
  let internal: [number, number] = [1, 1];

  // --- Can it run here? -------------------------------------------------------------------
  let reason: string | null = null;
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  if (options.simulate === "webgl1" || !webgl2) {
    reason = "this browser gave a WebGL1 context, and the shaders need WebGL2";
  } else {
    const passes: [string, string][] = [
      ["encode", ENCODE_FRAG],
      ["EASU", options.simulate === "badshader" ? `${EASU_FRAG}\nthis is not GLSL;` : EASU_FRAG],
      ["RCAS", RCAS_FRAG],
    ];
    for (const [name, source] of passes) {
      const error = linkError(gl, name, source);
      if (error) {
        reason = `a shader did not compile (${error.trim().slice(0, 200)})`;
        break;
      }
    }
  }
  if (reason) warnOnce(reason);
  const supported = reason === null;

  // --- Targets and passes (only made when the pipeline can run) ----------------------------
  let sceneTarget: THREE.WebGLRenderTarget | null = null; // [1]
  let lowTarget: THREE.WebGLRenderTarget | null = null; // [2]
  let easuTarget: THREE.WebGLRenderTarget | null = null; // [3]
  const quadScene = new THREE.Scene();
  const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const triangle = new THREE.BufferGeometry();
  triangle.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  const pass = (fragmentShader: string, uniforms: Record<string, THREE.IUniform>) =>
    new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: FULLSCREEN_VERT, fragmentShader, uniforms, depthTest: false, depthWrite: false });
  const encode = pass(ENCODE_FRAG, { uScene: { value: null } });
  const easu = pass(EASU_FRAG, { uInput: { value: null }, uInOverOut: { value: new THREE.Vector2(1, 1) } });
  const rcas = pass(RCAS_FRAG, {
    uEasu: { value: null },
    uLow: { value: null },
    uOutSize: { value: new THREE.Vector2(1, 1) },
    uCon: { value: Math.pow(2, -sharpness) },
    uSplitX: { value: -1 },
  });
  const quad = new THREE.Mesh(triangle, encode);
  quad.frustumCulled = false;
  quadScene.add(quad);

  const piped = () => supported && !bypass;
  const active = () => piped() && scale < 1;

  function freeTargets() {
    sceneTarget?.dispose();
    lowTarget?.dispose();
    easuTarget?.dispose();
    sceneTarget = lowTarget = easuTarget = null;
  }

  function allocate() {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    output = [Math.max(1, Math.round(size.x)), Math.max(1, Math.round(size.y))];
    internal = active() ? [Math.max(1, Math.round(output[0] * scale)), Math.max(1, Math.round(output[1] * scale))] : output;
    freeTargets();
    if (!piped()) return;
    sceneTarget = new THREE.WebGLRenderTarget(internal[0], internal[1], {
      colorSpace: THREE.SRGBColorSpace, // stored as SRGB8_ALPHA8: linear in the shader, perceptual in memory
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      samples: Math.min(4, renderer.capabilities.maxSamples),
      generateMipmaps: false,
    });
    encode.uniforms.uScene!.value = sceneTarget.texture;
    if (!active()) return;
    // Bilinear here is only for the comparison half; EASU fetches whole texels.
    lowTarget = new THREE.WebGLRenderTarget(internal[0], internal[1], { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false, generateMipmaps: false });
    easuTarget = new THREE.WebGLRenderTarget(output[0], output[1], { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, generateMipmaps: false });
    easu.uniforms.uInput!.value = lowTarget.texture;
    (easu.uniforms.uInOverOut!.value as THREE.Vector2).set(internal[0] / output[0], internal[1] / output[1]);
    rcas.uniforms.uEasu!.value = easuTarget.texture;
    rcas.uniforms.uLow!.value = lowTarget.texture;
    (rcas.uniforms.uOutSize!.value as THREE.Vector2).set(output[0], output[1]);
  }

  function clampScale(value: number): number {
    return Number.isFinite(value) ? Math.min(1, Math.max(0.25, value)) : 1;
  }

  function run(material: THREE.RawShaderMaterial, target: THREE.WebGLRenderTarget | null) {
    quad.material = material;
    renderer.setRenderTarget(target);
    renderer.render(quadScene, quadCamera);
  }

  allocate();

  return {
    render(scene, camera) {
      if (!piped() || !sceneTarget) {
        renderer.setRenderTarget(null);
        renderer.render(scene, camera);
        return;
      }
      renderer.setRenderTarget(sceneTarget);
      renderer.render(scene, camera);
      if (!lowTarget || !easuTarget) {
        run(encode, null); // FSR off: the full-size picture, encoded for the canvas
        return;
      }
      run(encode, lowTarget);
      run(easu, easuTarget);
      rcas.uniforms.uSplitX!.value = compare ? Math.floor(output[0] * split) + 0.5 : -1;
      run(rcas, null);
    },
    setSize(cssWidth, cssHeight, devicePixelRatio) {
      const ratio = devicePixelRatio ?? (typeof window === "undefined" ? 1 : window.devicePixelRatio || 1);
      renderer.setPixelRatio(Math.min(Math.max(ratio, 0.5), MAX_PIXEL_RATIO));
      renderer.setSize(Math.max(1, cssWidth), Math.max(1, cssHeight), false);
      allocate();
    },
    setScale(next) {
      const value = clampScale(next);
      if (value === scale) return;
      scale = value;
      allocate();
    },
    setSharpness(stops) {
      sharpness = Math.max(0, stops);
      rcas.uniforms.uCon!.value = Math.pow(2, -sharpness);
    },
    setBypass(on) {
      if (on === bypass) return;
      bypass = on;
      allocate();
    },
    setCompare(on, at) {
      compare = on;
      if (at !== undefined && Number.isFinite(at)) split = Math.min(1, Math.max(0, at));
    },
    info() {
      return { supported, reason, active: active(), scale, compare: compare && active(), split, internal, output };
    },
    dispose() {
      freeTargets();
      triangle.dispose();
      encode.dispose();
      easu.dispose();
      rcas.dispose();
    },
  };
}
