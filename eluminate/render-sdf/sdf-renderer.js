// sdf-renderer.js
// Use import syntax for ES modules
import fs from "fs";
import path from "path";
import { createCanvas } from "canvas";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Default SDF shader code
// You can replace this with your own default SDF or load from file
const defaultSDF = `
// SDF for a simple sphere
float scene(vec3 p) {
    return length(p) - 1.0;
}
`;

/**
 * Renders an SDF to an image file
 * @param {Object} options - Rendering options
 * @param {string} options.sourceCode - GLSL SDF code to render
 * @param {string} options.outputPath - Path to save the rendered image
 * @param {number} options.width - Width of the output image
 * @param {number} options.height - Height of the output image
 * @param {Object} options.uniforms - Additional uniforms to pass to the shader
 * @param {boolean} options.animate - Whether to animate the scene (requires multiple frames)
 * @param {number} options.frames - Number of frames to render if animating
 * @param {number} options.cornerRadius - Radius of rounded corners
 */
async function renderSDF({
  sourceCode = defaultSDF,
  outputPath = "sdf-output.png",
  width = 800,
  height = 800,
  uniforms = {},
  animate = false,
  frames = 1,
  cornerRadius = 8,
}) {
  // Create a canvas to render to
  const canvas = createCanvas(width, height);
  const context = canvas.getContext("2d");

  // Set up renderer to use the canvas
  // Note: NodeCanvas patch is not needed as we're directly creating the renderer

  // Initialize the renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setSize(width, height);
  renderer.setClearColor(0xffffff, 1);

  // Create the scene
  const scene = new THREE.Scene();

  // Set up the camera
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.z = 4;
  camera.position.y = 1;

  // Set up OrbitControls for the offline renderer
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.maxDistance = 10;
  controls.minDistance = 2;
  controls.enableDamping = true;

  // Add a directional light
  const light = new THREE.DirectionalLight(0xffffff, 1);
  scene.add(light);

  // Set up the raymarching plane with ShaderMaterial
  const geometry = new THREE.PlaneGeometry();
  const material = new THREE.ShaderMaterial({ transparent: true, opacity: 1 });
  const rayMarchPlane = new THREE.Mesh(geometry, material);

  // Compute the size of the near plane
  const nearPlaneWidth =
    camera.near *
    Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) *
    camera.aspect *
    2;
  const nearPlaneHeight = nearPlaneWidth / camera.aspect;
  rayMarchPlane.scale.set(nearPlaneWidth, nearPlaneHeight, 1);

  // Set up uniforms
  const backgroundColor = new THREE.Color(0xffffff);
  const all_uniforms = {
    u_eps: { value: 0.001 },
    u_maxDis: { value: 1000 },
    u_maxSteps: { value: 100 },

    u_clearColor: { value: backgroundColor },

    u_camPos: { value: camera.position.clone() },
    u_camToWorldMat: { value: camera.matrixWorld.clone() },
    u_camInvProjMat: { value: camera.projectionMatrixInverse.clone() },

    u_lightDir: { value: camera.position.clone() },
    u_lightColor: { value: light.color.clone() },

    u_diffIntensity: { value: 0.6 },
    u_specIntensity: { value: 0.2 },
    u_ambientIntensity: { value: 0.3 },
    u_shininess: { value: 5.0 },

    u_time: { value: 0 },

    u_canvasSize: { value: { x: width, y: height } },
    u_cornerRadius: { value: cornerRadius },
  };

  // Add custom uniforms
  for (const [key, value] of Object.entries(uniforms)) {
    all_uniforms[key] = { value: value };
  }

  material.uniforms = all_uniforms;

  // Define the vertex shader
  const vertCode = `
    // Pass UV coordinates to the fragment shader.
    out vec2 vUv;
    
    void main() {
        // Compute view-space position.
        vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPos;
        vUv = uv;
    }
  `;

  // Define the fragment shader
  const fragCode = `
    precision mediump float;

    // From vertex shader.
    in vec2 vUv;

    // Uniforms.
    uniform vec3 u_clearColor;
    uniform vec2 u_canvasSize;
    uniform float u_cornerRadius;
    uniform float u_eps;
    uniform float u_maxDis;
    uniform int u_maxSteps;

    uniform vec3 u_camPos;
    uniform mat4 u_camToWorldMat;
    uniform mat4 u_camInvProjMat;

    uniform vec3 u_lightDir;
    uniform vec3 u_lightColor;

    uniform float u_diffIntensity;
    uniform float u_specIntensity;
    uniform float u_ambientIntensity;
    uniform float u_shininess;

    uniform float u_time;

    ${sourceCode}

    float rayMarch(vec3 ro, vec3 rd) {
        float d = 0.0;
        float cd;
        vec3 p;
        for (int i = 0; i < u_maxSteps; ++i) {
            p = ro + d * rd;
            cd = scene(p);
            if (cd < u_eps || d >= u_maxDis) break;
            d += cd;
        }
        return d;
    }

    vec3 sceneCol(vec3 p) {
        // Changed to a lighter, more visible color
        return vec3(1.0, 0.65, 0.6);
    }

    vec3 normal(vec3 p) {
        vec3 n = vec3(0.0);
        vec3 e;
        for (int i = 0; i < 4; i++) {
            e = 0.5773 * (2.0 * vec3(float((i+3) >> 1 & 1),
                                       float((i >> 1) & 1),
                                       float(i & 1)) - 1.0);
            n += e * scene(p + e * u_eps);
        }
        return normalize(n);
    }

    // Phong lighting calculation
    vec3 phongLighting(vec3 p, vec3 n, vec3 v) {
        vec3 matColor = sceneCol(p);
        vec3 ambient = matColor * u_ambientIntensity;
        
        // Diffuse
        vec3 l = normalize(u_lightDir);
        float diff = max(dot(n, l), 0.0);
        vec3 diffuse = matColor * u_lightColor * diff * u_diffIntensity;
        
        // Specular
        vec3 r = reflect(-l, n);
        float spec = pow(max(dot(r, v), 0.0), u_shininess);
        vec3 specular = u_lightColor * spec * u_specIntensity;
        
        // Rim lighting
        float rim = 1.0 - max(dot(n, v), 0.0);
        rim = pow(rim, 3.0) * 0.5;
        vec3 rimLight = rim * u_lightColor;
        
        return ambient + diffuse + specular + rimLight;
    }
    
    float roundedBoxSDF(vec2 CenterPosition, vec2 Size, float Radius) {
        return length(max(abs(CenterPosition)-Size+Radius,0.0))-Radius;
    }

    void main() {
        // Get UV from vertex shader
        vec2 uv = vUv.xy;

        // Get ray origin and direction from camera uniforms
        vec3 ro = u_camPos;
        vec3 rd = (u_camInvProjMat * vec4(uv*2.-1., 0, 1)).xyz;
        rd = (u_camToWorldMat * vec4(rd, 0)).xyz;
        rd = normalize(rd);
        
        // Ray marching and find total distance traveled
        float disTraveled = rayMarch(ro, rd);

        // Find the hit position
        vec3 hp = ro + disTraveled * rd;
        
        // Calculate the final color as before
        vec4 finalColor;
        if (disTraveled >= u_maxDis) {
            finalColor = vec4(u_clearColor, 1);
        } else {
            vec3 n = normal(hp);
            vec3 v = normalize(ro - hp);
            vec3 color = phongLighting(hp, n, v);
            finalColor = vec4(color, 1);
        }

        // Apply rounded corners using the existing roundedBoxSDF
        vec2 pos = vUv * u_canvasSize;
        float distance = roundedBoxSDF(
            pos - (u_canvasSize * 0.5), // Center position
            u_canvasSize * 0.5,         // Size
            u_cornerRadius              // Radius
        );
        //Use the SDF to create a smooth alpha transition
        float alpha = 1.0 - smoothstep(-1.0, 0.0, distance);
        gl_FragColor = vec4(finalColor.rgb, finalColor.a * alpha);
    }
  `;

  material.vertexShader = vertCode;
  material.fragmentShader = fragCode;

  scene.add(rayMarchPlane);

  // Utility variables for animation
  const cameraForwardPos = new THREE.Vector3(0, 0, -1);
  const VECTOR3ZERO = new THREE.Vector3(0, 0, 0);
  const startTime = Date.now();

  // Render function
  function renderFrame(frameNum) {
    // Update controls
    controls.update();

    // Add automatic rotation
    const rotationSpeed = 0.01;
    camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), rotationSpeed);

    // Update camera matrices for ray marching
    camera.updateMatrixWorld();
    all_uniforms.u_camToWorldMat.value.copy(camera.matrixWorld);
    all_uniforms.u_camInvProjMat.value.copy(camera.projectionMatrixInverse);
    all_uniforms.u_camPos.value.copy(camera.position);

    // Update light position to match camera
    light.position.copy(camera.position);
    all_uniforms.u_lightDir.value.copy(camera.position);

    // Update screen plane position and rotation
    const forward = camera.position
      .clone()
      .add(camera.getWorldDirection(VECTOR3ZERO).multiplyScalar(camera.near));
    rayMarchPlane.position.copy(forward);
    rayMarchPlane.rotation.copy(camera.rotation);

    // Update time uniform
    all_uniforms.u_time.value = (Date.now() - startTime) / 1000;

    // Render the scene
    renderer.render(scene, camera);

    return {
      buffer: canvas.toBuffer("image/png"),
      filename: animate
        ? `${path.basename(
            outputPath,
            path.extname(outputPath)
          )}_${frameNum}${path.extname(outputPath)}`
        : outputPath,
    };
  }

  // Render frames
  const renderedFrames = [];
  for (let i = 0; i < (animate ? frames : 1); i++) {
    const result = renderFrame(i);
    renderedFrames.push(result);
    fs.writeFileSync(result.filename, result.buffer);
    console.log(
      `Frame ${i + 1}/${animate ? frames : 1} saved to ${result.filename}`
    );
  }

  // Clean up resources
  renderer.dispose();
  material.dispose();
  geometry.dispose();

  return renderedFrames;
}

// Example usage
async function main() {
  const args = process.argv.slice(2);
  let sdfCode;

  // Check if input file was provided
  if (args.length > 0) {
    try {
      sdfCode = fs.readFileSync(args[0], "utf8");
    } catch (error) {
      console.error(`Error reading SDF file: ${error.message}`);
      process.exit(1);
    }
  } else {
    // Use default SDF if no file provided
    sdfCode = defaultSDF;
  }

  // Output path from arguments or default
  const outputPath = args[1] || "sdf-output.png";

  try {
    await renderSDF({
      sourceCode: sdfCode,
      outputPath,
      width: 800,
      height: 800,
      animate: false,
    });
    console.log(`SDF successfully rendered to ${outputPath}`);
  } catch (error) {
    console.error(`Error rendering SDF: ${error.message}`);
    process.exit(1);
  }
}

// Since this is now an ESM module, use a different approach to detect if run directly
// Check if this module is being run directly
import { fileURLToPath } from "url";
const currentFilePath = fileURLToPath(import.meta.url);
const isRunningDirectly = process.argv[1] === currentFilePath;

if (isRunningDirectly) {
  main();
}

export { renderSDF };
