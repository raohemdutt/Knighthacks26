import { Config } from "@remotion/cli/config";

// three.js needs a GPU-backed WebGL context in headless Chrome; ANGLE works on Windows, macOS and Linux.
Config.setChromiumOpenGlRenderer("angle");
Config.setVideoImageFormat("jpeg");
