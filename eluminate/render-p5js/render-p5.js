// p5-sketch-renderer.js
const fs = require("fs");
const path = require("path");
const p5 = require("node-p5");

/**
 * Renders a p5.js sketch and writes out image files.
 * @param {object} spec - An object with properties defining the sketch.
 * @param {string} outPath - Path to save the output images.
 * @returns {Promise<void>}
 */
function renderP5Sketch(spec, outPath) {
  console.log("Rendering sketch", outPath);
  return new Promise((resolve, reject) => {
    const width = spec.width || 800;
    const height = spec.height || 600;
    const frameRate = spec.frameRate || 60;
    const maxFrames = spec.frames || 1; // Number of frames to render
    const outputFormat = spec.format || "png";

    if (!spec.sketchCode) {
      return reject(
        new Error("Sketch specification must include sketchCode string")
      );
    }

    // Ensure output directory exists
    const dirPath = path.dirname(outPath);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }

    const sketch = (p) => {
      // This will hold the return value from createCanvas()
      let myCanvas;
      let currentFrame = 0;
      let finished = false;

      // Create a function that will be called for each frame save
      const saveNextFrame = () => {
        if (currentFrame >= maxFrames || finished) {
          if (!finished) {
            console.log("All frames rendered");
            finished = true;
            setTimeout(() => {
              p.remove();
              resolve();
            }, 100);
          }
          return;
        }

        console.log(`Saving frame ${currentFrame + 1}/${maxFrames}`);

        // Calculate output filename
        let frameName;
        if (maxFrames > 1) {
          const frameNumberStr = currentFrame.toString().padStart(5, "0");
          frameName =
            path.basename(outPath, path.extname(outPath)) +
            "_" +
            frameNumberStr;
        } else {
          frameName = path.basename(outPath, path.extname(outPath));
        }

        // Follow exactly the pattern in the documentation
        p.saveCanvas(myCanvas, frameName, outputFormat)
          .then((filename) => {
            console.log(`Saved the canvas as ${filename}`);

            // Move the file if necessary to the target location
            const savedPath = path.resolve(filename);
            const targetDir = path.dirname(outPath);
            const targetPath = path.join(targetDir, path.basename(filename));

            if (savedPath !== targetPath) {
              console.log(`Moving from ${savedPath} to ${targetPath}`);
              fs.renameSync(savedPath, targetPath);
            }

            // Increment frame counter
            currentFrame++;

            // If there are more frames, continue
            if (currentFrame < maxFrames) {
              setTimeout(saveNextFrame, 100);
            } else {
              finished = true;
              setTimeout(() => {
                p.remove();
                resolve();
              }, 100);
            }
          })
          .catch((err) => {
            console.error("Error saving canvas:", err);
            currentFrame++;
            if (currentFrame < maxFrames) {
              setTimeout(saveNextFrame, 100);
            } else {
              finished = true;
              setTimeout(() => {
                p.remove();
                resolve();
              }, 100);
            }
          });
      };

      // Define default setup and draw
      p.setup = function () {
        myCanvas = p.createCanvas(width, height);
        p.frameRate(frameRate);
      };

      p.draw = function () {
        p.background(255);
      };

      // Evaluate user code, which might override setup and draw
      try {
        // Inject user code with access to width and height
        const userCode = `
          // Make width and height available
          const width = ${width};
          const height = ${height};
          
          ${spec.sketchCode}
        `;

        eval(userCode);

        // Keep references to user functions
        const userSetup = p.setup;
        const userDraw = p.draw;

        // Replace setup to save canvas reference
        p.setup = function () {
          // First call user setup which might create a canvas
          userSetup.call(p);

          // Ensure we have a canvas reference from createCanvas
          if (!myCanvas || !myCanvas.elt) {
            // If user didn't create a canvas or we lost reference
            myCanvas = p.createCanvas(width, height);
            console.log("Created fallback canvas");
          }

          console.log("Canvas captured:", !!myCanvas, "Type:", typeof myCanvas);

          // Start saving frames after a short delay
          setTimeout(saveNextFrame, 500);
        };

        // Replace draw to just call user draw
        p.draw = function () {
          userDraw.call(p);
        };
      } catch (error) {
        console.error("Error evaluating sketch code:", error);
        reject(error);
      }
    };

    // Start the sketch
    try {
      p5.createSketch(sketch);
    } catch (err) {
      reject(new Error(`Failed to create p5.js sketch: ${err.message}`));
    }
  });
}

/**
 * Main entry point.
 *
 * Usage:
 *   node p5-sketch-renderer.js [outputPath]
 *
 * If no standard input is provided, a test sketch is rendered as "output.png".
 * Otherwise, expects JSON input (a sketch spec or an array of specs).
 */
async function main() {
  const outPath = process.argv[2] || "output.png";

  // If there's no piped input, render the default test sketch
  if (process.stdin.isTTY) {
    const testSketch = {
      width: 800,
      height: 600,
      frames: 1,
      sketchCode: `
        // Setup function runs once at the beginning
        p.setup = function() {
          myCanvas = p.createCanvas(width, height);
          p.background(240);
        };
        
        // Draw function handles rendering
        p.draw = function() {
          // Draw a gradient background
          for (let y = 0; y < height; y++) {
            const c = p.lerpColor(p.color(63, 191, 191), p.color(63, 63, 191), y / height);
            p.stroke(c);
            p.line(0, y, width, y);
          }
          
          // Draw some circles
          p.noStroke();
          for (let i = 0; i < 10; i++) {
            let x = width * (0.1 + 0.8 * (i / 9));
            let size = width * 0.1 * p.sin(p.PI * i / 9);
            p.fill(255, 220, 120, 200);
            p.circle(x, height * 0.5, size * 2);
            p.fill(255, 120, 180, 200);
            p.circle(x, height * 0.5, size);
          }
          
          // Draw text
          p.fill(255);
          p.textSize(32);
          p.textAlign(p.CENTER, p.CENTER);
          p.text("P5.js Renderer Test", width/2, height/2);
        };
      `,
    };

    renderP5Sketch(testSketch, outPath)
      .catch((err) => {
        console.error("Error rendering test sketch:", err.message);
      })
      .then(() => {
        console.log("Test sketch rendered successfully");
      });
  } else {
    // Otherwise, read JSON input from stdin
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
          for (let i = 0; i < specs.length; i++) {
            const spec = specs[i];
            const specOutPath =
              specs.length > 1
                ? path.join(
                    path.dirname(outPath),
                    `${path.basename(
                      outPath,
                      path.extname(outPath)
                    )}_${i}${path.extname(outPath)}`
                  )
                : outPath;

            await renderP5Sketch(spec, specOutPath);
          }
        })().catch((err) => {
          console.error("Error rendering sketches:", err.message);
        });
      } catch (err) {
        console.error("Error parsing input:", err.message);
      }
    });
  }
}

main()
  .catch((err) => {
    console.error("Error rendering sketches:", err.message);
  })
  .then(() => {
    console.log("Rendering sketches completed");
  });
