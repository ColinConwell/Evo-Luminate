vertexShader = """
attribute vec2 position;
varying vec2 vUv;

void main() {
    // Map clip-space [-1,1] position to UV coordinates [0,1]
    vUv = 0.5 * (position + 1.0);
    
    // Simply output the position for a full-screen triangle
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

shaderTemplate = """
precision mediump float;
varying vec2 vUv;

// Basic uniforms needed for ray marching
uniform vec3 u_camPos;
uniform float u_eps;
uniform float u_maxDis;
uniform int u_maxSteps;

{{CONSTANTS}}

{{LIBRARY_METHODS}}

{{SOURCE_CODE}}

// Simple ray marching function
float rayMarch(vec3 ro, vec3 rd) {
    float d = 0.0;
    
    for (int i = 0; i < 100; i++) {
        if (i >= u_maxSteps) break;
        
        vec3 p = ro + d * rd;
        float cd = scene(p);
        
        if (cd < u_eps || d >= u_maxDis) break;
        d += cd;
    }
    
    return d;
}

// Simple normal calculation
vec3 normal(vec3 p) {
    float d = scene(p);
    vec2 e = vec2(u_eps, 0.0);
    
    vec3 n = d - vec3(
        scene(p - e.xyy),
        scene(p - e.yxy),
        scene(p - e.yyx)
    );
    
    return normalize(n);
}

void main2() {
    // Create a simple ray from camera position through the screen
    vec3 ro = u_camPos;
    vec3 rd = normalize(vec3(vUv * 2.0 - 1.0, -1.0)); // Simple perspective projection
    
    // Ray march to find distance
    float d = rayMarch(ro, rd);
    
    if (d < u_maxDis) {
        // We hit something - basic Lambertian shading
        vec3 p = ro + d * rd;
        vec3 n = normal(p);
        vec3 light = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(n, light), 0.0);
        
        vec3 color = vec3(1.0, 0.5, 0.2) * (diff * 0.8 + 0.2);
        gl_FragColor = vec4(color, 1.0);
    } else {
        // We hit nothing - background color
        gl_FragColor = vec4(0.1, 0.1, 0.1, 1.0);
    }
}
void main() {
    // Create a look-at camera system
    vec3 ro = u_camPos;                     // Camera position from uniform
    vec3 target = vec3(0.0, 0.0, 0.0);      // Always look at origin
    vec3 forward = normalize(target - ro);  // Forward vector (z-axis)
    
    // Create an arbitrary up vector (typically world up)
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    
    // Construct camera basis
    vec3 right = normalize(cross(forward, worldUp)); // Right vector (x-axis)
    vec3 up = normalize(cross(right, forward));      // Up vector (y-axis)
    
    // Use the basis to transform the ray direction
    vec2 uv = vUv * 2.0 - 1.0;  // Map UV from [0,1] to [-1,1]
    float aspect = 1.0;         // Assuming square viewport
    
    // Create ray direction using camera basis and field of view
    float fov = 1.0;
    vec3 rd = normalize(right * uv.x * aspect * fov + up * uv.y * fov + forward);
    
    // Ray march to find distance
    float d = rayMarch(ro, rd);
    
    if (d < u_maxDis) {
        // We hit something - enhanced shading
        vec3 p = ro + d * rd;
        vec3 n = normal(p);
        vec3 mat = vec3(1.0, 0.5, 0.2);
        
        // Light direction - adjust to always come from camera side
        vec3 light1 = normalize(u_camPos);
        vec3 light2 = normalize(vec3(-1.0, 0.5, -0.5));
        
        // Diffuse lighting from two light sources
        float diff1 = max(dot(n, light1), 0.0);
        float diff2 = max(dot(n, light2), 0.0) * 0.5; // Second light is dimmer
        
        // Ambient occlusion approximation
        float ao = 0.5 + 0.5 * n.y;
        
        // Combine lighting components
        vec3 color = mat * (diff1 + diff2 + 0.2) * ao;
        
        // Add specular highlight
        vec3 ref = reflect(rd, n);
        float spec = pow(max(dot(ref, light1), 0.0), 32.0);
        color += vec3(spec) * 0.5;
        
        gl_FragColor = vec4(color, 1.0);
    } else {
        // We hit nothing - gradient background
        vec3 bg = mix(vec3(0.5, 0.7, 1.0), vec3(0.1, 0.2, 0.3), vUv.y);
        gl_FragColor = vec4(bg, 1.0);
    }
}
"""

sdfLibraryHeaders = """
// Utility Functions
float dot2(in vec2 v);
float dot2(in vec3 v);
float ndot(in vec2 a, in vec2 b);

// Primitives
float sdSphere(vec3 p, float s);
float sdBox(vec3 p, vec3 b);
float sdRoundBox(vec3 p, vec3 b, float r);
float sdBoxFrame(vec3 p, vec3 b, float e);
float sdTorus(vec3 p, vec2 t);
float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb);
float sdLink(vec3 p, float le, float r1, float r2);
float sdCylinder(vec3 p, vec3 c);
float sdCone(vec3 p, vec2 c, float h);
float sdPlane(vec3 p, vec3 n, float h);
float sdHexPrism(vec3 p, vec2 h);
float sdTriPrism(vec3 p, vec2 h);
float sdCapsule(vec3 p, vec3 a, vec3 b, float r);
float sdVerticalCapsule(vec3 p, float h, float r);
float sdCappedCylinder(vec3 p, float h, float r);
float sdRoundedCylinder(vec3 p, float ra, float rb, float h);
float sdCappedCone(vec3 p, float h, float r1, float r2);
float sdSolidAngle(vec3 p, vec2 c, float ra);
float sdCutSphere(vec3 p, float r, float h);
float sdEllipsoid(vec3 p, vec3 r);
float sdOctahedron(vec3 p, float s);
float sdOctahedronBound(vec3 p, float s);

// Boolean Operations
float opUnion(float d1, float d2);
float opSubtraction(float d1, float d2);
float opIntersection(float d1, float d2);
float opSmoothUnion(float d1, float d2, float k);
float opSmoothSubtraction(float d1, float d2, float k);
float opSmoothIntersection(float d1, float d2, float k);

// Transformations
float opScale(vec3 p, float s, float d);
float opOnion(float sdf, float thickness);
"""

# // Domain Operations
# vec3 opRepetition(vec3 p, vec3 s);
# vec3 opLimitedRepetition(vec3 p, float s, vec3 limit);
# vec3 opSymX(vec3 p);
# vec3 opSymXZ(vec3 p);

# // Deformations
# vec3 opTwistY(vec3 p, float k);
# vec3 opCheapBendX(vec3 p, float k);

sdfLibrary = """
// SDF Library - Based on Inigo Quilez's distance functions
// GLSL ES 1.0 compatible

// --------
// Utilities
// --------

// Squared length of vector
float dot2(in vec2 v) { return dot(v, v); }
float dot2(in vec3 v) { return dot(v, v); }

// Special dot product for certain primitives
float ndot(in vec2 a, in vec2 b) { return a.x*b.x - a.y*b.y; }

// --------
// Primitives
// --------

// Sphere - exact
// p: evaluation point, s: radius
float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

// Box - exact
// p: evaluation point, b: box dimensions (half-lengths)
float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Round Box - exact
// p: evaluation point, b: box dimensions (half-lengths), r: corner radius
float sdRoundBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Box Frame - exact
// p: evaluation point, b: box dimensions (half-lengths), e: frame thickness
float sdBoxFrame(vec3 p, vec3 b, float e) {
  p = abs(p) - b;
  vec3 q = abs(p + e) - e;
  return min(min(
      length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
      length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
      length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

// Torus - exact
// p: evaluation point, t: vec2(major radius, minor radius)
float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

// Capped Torus - exact
// p: evaluation point, sc: vec2(sin, cos) of angle, ra: major radius, rb: minor radius
float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb) {
  p.x = abs(p.x);
  float k = (sc.y*p.x > sc.x*p.y) ? dot(p.xy, sc) : length(p.xy);
  return sqrt(dot(p, p) + ra*ra - 2.0*ra*k) - rb;
}

// Link - exact
// p: evaluation point, le: half-length, r1: major radius, r2: minor radius
float sdLink(vec3 p, float le, float r1, float r2) {
  vec3 q = vec3(p.x, max(abs(p.y) - le, 0.0), p.z);
  return length(vec2(length(q.xy) - r1, q.z)) - r2;
}

// Infinite Cylinder - exact
// p: evaluation point, c: vec3(center.x, center.y, radius)
float sdCylinder(vec3 p, vec3 c) {
  return length(p.xz - c.xy) - c.z;
}

// Cone - exact
// p: evaluation point, c: vec2(sin, cos) of angle, h: height
float sdCone(vec3 p, vec2 c, float h) {
  // c is the sin/cos of the angle, h is height
  vec2 q = h*vec2(c.x/c.y, -1.0);
    
  vec2 w = vec2(length(p.xz), p.y);
  vec2 a = w - q*clamp(dot(w, q)/dot(q, q), 0.0, 1.0);
  vec2 b = w - q*vec2(clamp(w.x/q.x, 0.0, 1.0), 1.0);
  float k = sign(q.y);
  float d = min(dot(a, a), dot(b, b));
  float s = max(k*(w.x*q.y - w.y*q.x), k*(w.y - q.y));
  return sqrt(d)*sign(s);
}

// Plane - exact
// p: evaluation point, n: normal (must be normalized), h: offset
float sdPlane(vec3 p, vec3 n, float h) {
  // n must be normalized
  return dot(p, n) + h;
}

// Hexagonal Prism - exact
// p: evaluation point, h: vec2(horizontal half-width, vertical half-height)
float sdHexPrism(vec3 p, vec2 h) {
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  vec2 d = vec2(
       length(p.xy - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x))*sign(p.y - h.x),
       p.z - h.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Triangular Prism - bound
// p: evaluation point, h: vec2(horizontal half-width, vertical half-height)
float sdTriPrism(vec3 p, vec2 h) {
  vec3 q = abs(p);
  return max(q.z - h.y, max(q.x*0.866025 + p.y*0.5, -p.y) - h.x*0.5);
}

// Capsule / Line - exact
// p: evaluation point, a: line start, b: line end, r: radius
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  return length(pa - ba*h) - r;
}

// Vertical Capsule - exact
// p: evaluation point, h: half-height, r: radius
float sdVerticalCapsule(vec3 p, float h, float r) {
  p.y -= clamp(p.y, 0.0, h);
  return length(p) - r;
}

// Vertical Capped Cylinder - exact
// p: evaluation point, h: half-height, r: radius
float sdCappedCylinder(vec3 p, float h, float r) {
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Rounded Cylinder - exact
// p: evaluation point, ra: outer radius, rb: corner radius, h: half-height
float sdRoundedCylinder(vec3 p, float ra, float rb, float h) {
  vec2 d = vec2(length(p.xz) - 2.0*ra + rb, abs(p.y) - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - rb;
}

// Capped Cone - exact
// p: evaluation point, h: height, r1: base radius, r2: top radius
float sdCappedCone(vec3 p, float h, float r1, float r2) {
  vec2 q = vec2(length(p.xz), p.y);
  vec2 k1 = vec2(r2, h);
  vec2 k2 = vec2(r2 - r1, 2.0*h);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
  vec2 cb = q - k1 + k2*clamp(dot(k1 - q, k2)/dot2(k2), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s*sqrt(min(dot2(ca), dot2(cb)));
}

// Solid Angle - exact
// p: evaluation point, c: vec2(sin, cos) of angle, ra: radius
float sdSolidAngle(vec3 p, vec2 c, float ra) {
  vec2 q = vec2(length(p.xz), p.y);
  float l = length(q) - ra;
  float m = length(q - c*clamp(dot(q, c), 0.0, ra));
  return max(l, m*sign(c.y*q.x - c.x*q.y));
}

// Cut Sphere - exact
// p: evaluation point, r: radius, h: cut height
float sdCutSphere(vec3 p, float r, float h) {
  // Sampling independent computations
  float w = sqrt(r*r - h*h);

  // Sampling dependent computations
  vec2 q = vec2(length(p.xz), p.y);
  float s = max((h-r)*q.x*q.x + w*w*(h+r-2.0*q.y), h*q.x - w*q.y);
  return (s < 0.0) ? length(q) - r :
         (q.x < w) ? h - q.y :
                    length(q - vec2(w, h));
}

// Ellipsoid - bound (not exact!)
// p: evaluation point, r: radii
float sdEllipsoid(vec3 p, vec3 r) {
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0 - 1.0)/k1;
}

// Octahedron - exact
// p: evaluation point, s: size
float sdOctahedron(vec3 p, float s) {
  p = abs(p);
  float m = p.x + p.y + p.z - s;
  vec3 q;
       if(3.0*p.x < m) q = p.xyz;
  else if(3.0*p.y < m) q = p.yzx;
  else if(3.0*p.z < m) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z - q.y + s), 0.0, s); 
  return length(vec3(q.x, q.y - s + k, q.z - k)); 
}

// Octahedron - bound (not exact)
// p: evaluation point, s: size
float sdOctahedronBound(vec3 p, float s) {
  p = abs(p);
  return (p.x + p.y + p.z - s)*0.57735027;
}

// --------
// Operations
// --------

// Union - exact
// Combines two shapes by taking the closest point
float opUnion(float d1, float d2) {
  return min(d1, d2);
}

// Subtraction - bound
// Subtracts d1 from d2
float opSubtraction(float d1, float d2) {
  return max(-d1, d2);
}

// Intersection - bound
// Returns the intersection of two shapes
float opIntersection(float d1, float d2) {
  return max(d1, d2);
}

// Smooth Union - bound
// Smoothly blends between two shapes
float opSmoothUnion(float d1, float d2, float k) {
  float h = clamp(0.5 + 0.5*(d2 - d1)/k, 0.0, 1.0);
  return mix(d2, d1, h) - k*h*(1.0 - h);
}

// Smooth Subtraction - bound
// Smoothly subtracts d1 from d2
float opSmoothSubtraction(float d1, float d2, float k) {
  float h = clamp(0.5 - 0.5*(d2 + d1)/k, 0.0, 1.0);
  return mix(d2, -d1, h) + k*h*(1.0 - h);
}

// Smooth Intersection - bound
// Smoothly intersects two shapes
float opSmoothIntersection(float d1, float d2, float k) {
  float h = clamp(0.5 - 0.5*(d2 - d1)/k, 0.0, 1.0);
  return mix(d2, d1, h) + k*h*(1.0 - h);
}

// Scale - exact
// Uniformly scales a shape
float opScale(vec3 p, float s, float d) {
  return d/s;
}

// Onion - exact
// Creates shell/hollow version of a shape with given thickness
float opOnion(float sdf, float thickness) {
  return abs(sdf) - thickness;
}
"""

# // --------
# // Domain Operations
# // --------

# // Repeat - exact for symmetric primitives
# // Creates infinite repetitions of a shape
# vec3 opRepetition(vec3 p, vec3 s) {
#   return p - s*round(p/s);
# }

# // Limited Repetition - exact for symmetric primitives
# // Creates limited repetitions of a shape
# vec3 opLimitedRepetition(vec3 p, float s, vec3 limit) {
#   return p - s*clamp(round(p/s), -limit, limit);
# }

# // Symmetry X - exact if object doesn't cross symmetry plane
# // Creates a mirrored copy along X axis
# vec3 opSymX(vec3 p) {
#   p.x = abs(p.x);
#   return p;
# }

# // Symmetry XZ - exact if object doesn't cross symmetry plane
# // Creates a mirrored copy along X and Z axes
# vec3 opSymXZ(vec3 p) {
#   p.xz = abs(p.xz);
#   return p;
# }

# // --------
# // Deformations
# // --------

# // Twist Y - bound
# // Twists the space around the Y axis
# vec3 opTwistY(vec3 p, float k) {
#   float c = cos(k*p.y);
#   float s = sin(k*p.y);
#   mat2 m = mat2(c, -s, s, c);
#   return vec3(m * p.xz, p.y);
# }

# // Bend X - bound
# // Bends the space around the X axis
# vec3 opCheapBendX(vec3 p, float k) {
#   float c = cos(k*p.x);
#   float s = sin(k*p.x);
#   mat2 m = mat2(c, -s, s, c);
#   return vec3(p.x, m * p.yz);
# }
