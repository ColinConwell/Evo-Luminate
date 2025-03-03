// const { renderSDF } = require("./sdf-renderer.js");
import { renderSDF } from "./sdf-renderer.js";
// Render an SDF
renderSDF({
  sourceCode: `float scene(vec3 p) { return length(p) - 1.0; }`,
  outputPath: "sphere.png",
  width: 800,
  height: 800,
  uniforms: {
    u_diffIntensity: 0.8,
    u_specIntensity: 0.3,
  },
});
