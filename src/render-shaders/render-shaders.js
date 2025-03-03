// render-shaders.js
const fs = require("fs");
const path = require("path");
const createGl = require("gl");
const createREGL = require("regl");
const PNG = require("pngjs").PNG;

/**
 * Renders a shader spec and writes out a PNG.
 * @param {object} spec - An object with properties: fragmentShader, width, height, uniforms.
 * @param {string} outPath - Path to save the PNG.
 * @returns {Promise<void>}
 */
function renderShader(spec, outPath) {
  return new Promise((resolve, reject) => {
    const width = spec.width || 256;
    const height = spec.height || 256;

    // Create a headless WebGL context with preserveDrawingBuffer enabled.
    const gl = createGl(width, height, { preserveDrawingBuffer: true });
    if (!gl) {
      return reject(new Error("Failed to create WebGL context"));
    }

    // Initialize regl with the headless context.
    const regl = createREGL({ gl });

    // Create a draw command with a full-screen triangle.
    const draw = regl({
      frag: spec.fragmentShader,
      vert:
        spec.vertexShader ??
        `
        precision mediump float;
        attribute vec2 position;
        varying vec2 uv;
        void main() {
          // Map clip-space [-1,1] to uv [0,1]
          uv = 0.5 * (position + 1.0);
          gl_Position = vec4(position, 0, 1);
        }
      `,
      attributes: {
        // Vertices for a full-screen triangle.
        position: [
          [-1, -1],
          [3, -1],
          [-1, 3],
        ],
      },
      uniforms: spec.uniforms || {},
      count: 3,
      framebuffer: null,
    });

    // Clear the drawing buffer.
    regl.clear({ color: [0, 0, 0, 1], depth: 1 });

    // Execute the draw command.
    draw();

    // Read pixels from the framebuffer.
    const pixels = regl.read();

    // Create a PNG using pngjs. Note: flip vertically.
    const png = new PNG({ width, height });
    for (let y = 0; y < height; y++) {
      const srcY = height - 1 - y; // flip y coordinate
      for (let x = 0; x < width; x++) {
        const srcIdx = (srcY * width + x) * 4;
        const dstIdx = (y * width + x) * 4;
        png.data[dstIdx] = pixels[srcIdx]; // red
        png.data[dstIdx + 1] = pixels[srcIdx + 1]; // green
        png.data[dstIdx + 2] = pixels[srcIdx + 2]; // blue
        png.data[dstIdx + 3] = pixels[srcIdx + 3]; // alpha
      }
    }

    // Encode the PNG synchronously.
    const buffer = PNG.sync.write(png);

    fs.writeFile(outPath, buffer, (err) => {
      // Clean up the resources.
      regl.destroy();
      gl.destroy();
      if (err) {
        return reject(err);
      }
      console.log(`Saved ${outPath}`);
      resolve();
    });
  });
}

/**
 * Main entry point.
 *
 * Usage:
 *   node render-shaders.js [outputDir]
 *
 * If no standard input is provided, a test shader is rendered as "test.png".
 * Otherwise, expects JSON input (a shader spec or an array of specs).
 */
function main() {
  const outPath = process.argv[2] || ".";

  // If there's no piped input, render the default test shader.
  if (process.stdin.isTTY) {
    const testShader = {
      fragmentShader: `
        precision mediump float;
        varying vec2 uv;
        void main(){
          gl_FragColor = vec4(uv, 0.5, 1.0);
        }
      `,
      width: 256,
      height: 256,
      uniforms: {},
    };
    renderShader(testShader, outPath).catch((err) => {
      console.error("Error rendering test shader:", err.message);
    });
  } else {
    // Otherwise, read JSON input from stdin.
    let inputData = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      inputData += chunk;
    });
    process.stdin.on("end", () => {
      try {
        const parsed = JSON.parse(inputData);
        const specs = Array.isArray(parsed) ? parsed : [parsed];
        (async function () {
          for (const spec of specs) {
            await renderShader(spec, outPath);
          }
        })();
      } catch (err) {
        console.error("Error parsing input:", err.message);
      }
    });
  }
}

main();
