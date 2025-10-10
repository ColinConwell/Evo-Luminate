precision mediump float;

// This shader variation generates a visually stimulating cosmic panorama 
// with additional effects: dynamic twist and rainbow color effect.

// Original helper functions from the previous shader variation.

float hash(float n) { 
    return fract(sin(n) * 43758.5453123); 
}

vec2 rotate2D(vec2 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

vec2 randomPoint(float seed) {
    return vec2(hash(seed), hash(seed * 2.0));
}

float gentleGlow(float distance, float intensity, float time) {
    return 0.5 + 0.5 * sin(9.0 * distance - time * 5.0 + intensity);
}

vec2 gravitationalLens(vec2 coord, vec2 lensCenter, float lensStrength) {
    vec2 offset = coord - lensCenter;
    float dist = length(offset);
    return coord + (offset / (dist + 0.2)) * lensStrength / (dist * dist + 0.01);
}

vec3 blackHole(vec2 coord, vec2 blackHoleCenter, float eventHorizonRadius) {
    float dist = distance(coord, blackHoleCenter);
    float accretionGlow = smoothstep(eventHorizonRadius * 1.2, eventHorizonRadius, dist);
    vec3 accretionColor = vec3(0.8, 0.5, 0.2) * accretionGlow * accretionGlow;
    float distortion = 1.0 - smoothstep(eventHorizonRadius, eventHorizonRadius * 1.2, dist);
    coord = mix(coord, blackHoleCenter, distortion * distortion);
    return accretionColor;
}

vec3 temporalBlur(vec3 currentColor, vec3 previousColor, float blurAmount) {
    return mix(previousColor, currentColor, blurAmount);
}

vec3 colorWarp(vec3 color, vec2 coord, float time) {
    float distortion = sin(coord.x * 15.0 + time * 2.0) * cos(coord.y * 15.0 - time * 2.0) * 0.1;
    return color + vec3(distortion, -distortion, distortion * 0.5);
}

vec3 generateNebulaColor(vec2 coord, float time) {
    float n = sin(coord.x * 10.0 + time) * cos(coord.y * 10.0 - time);
    vec3 baseColor = vec3(0.5, 0.3, 0.7);
    vec3 secondaryColor = vec3(0.2, 0.6, 0.9);
    return mix(baseColor, secondaryColor, n * 0.5 + 0.5);
}

vec3 nebulaEffect(vec2 coord, float time) {
    vec3 nebulaColor = generateNebulaColor(coord, time);
    float intensity = smoothstep(0.2, 0.8, sin(coord.y * 5.0 + time * 2.0) * 0.5 + 0.5);
    
    // Add fBm-based cloud-like wisps to the nebula
    float fbmValue = fbm(coord * 3.0, time);
    float wisp = smoothstep(0.3, 0.7, fbmValue);
    
    // Create color variation based on fBm
    vec3 wispColor = mix(
        vec3(0.1, 0.4, 0.8), // Cool blue
        vec3(0.8, 0.2, 0.7), // Warm purple
        fbm(coord * 2.0 + vec2(time * 0.2), time * 0.5)
    );
    
    // Blend the original nebula with the fBm wisps
    return mix(nebulaColor * intensity, wispColor, wisp * 0.6);
}

vec3 vignette(vec3 color, vec2 coord) {
    float d = distance(coord, vec2(0.5));
    float vig = smoothstep(0.8, 0.4, d);
    return color * vig;
}

vec3 starBurstEffect(vec2 coord, float time) {
    float pulse = abs(sin(time * 3.0 + dot(coord, coord) * 10.0));
    return vec3(pulse, 1.0 - pulse, 0.5 + 0.5 * sin(time + coord.x * 5.0));
}

vec3 enhanceCosmicRays(vec3 color, vec2 coord, float intensity) {
    return color + vec3(intensity * sin(coord.x * 10.0),
                          intensity * cos(coord.y * 10.0),
                          intensity * sin(coord.x * 10.0 + coord.y * 10.0));
}

float galacticNoise(vec2 coord, float time) {
    return fract(sin(dot(coord, vec2(12.9898,78.233))) * 43758.5453 + time);
}

vec3 improveSupernova(vec3 color, float time) {
    float factor = abs(sin(time * 2.0));
    return mix(color, vec3(1.0, 0.8, 0.5), factor);
}

// NEW FUNCTIONS ADDED BEFORE USAGE

vec3 dynamicTwist(vec3 color, vec2 coord, float time) {
    float twist = sin(coord.y * 10.0 + time) * 0.1;
    return color + vec3(twist, twist * 0.5, -twist);
}

// Noise function for fBm
float noise2D(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    
    // Smoothstep for better interpolation
    fp = fp * fp * (3.0 - 2.0 * fp);
    
    float n00 = hash(dot(ip, vec2(1.0, 113.0)));
    float n01 = hash(dot(ip + vec2(0.0, 1.0), vec2(1.0, 113.0)));
    float n10 = hash(dot(ip + vec2(1.0, 0.0), vec2(1.0, 113.0)));
    float n11 = hash(dot(ip + vec2(1.0, 1.0), vec2(1.0, 113.0)));
    
    float nx0 = mix(n00, n10, fp.x);
    float nx1 = mix(n01, n11, fp.x);
    
    return mix(nx0, nx1, fp.y);
}

// Fractal Brownian Motion
float fbm(vec2 p, float time) {
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    // Time-based evolution of the pattern
    p += time * 0.1;
    
    // Sum multiple octaves of noise
    for(int i = 0; i < 6; i++) {
        sum += amp * noise2D(p * freq);
        freq *= 2.0;
        amp *= 0.5;
        // Rotate coordinates slightly for each octave to add variation
        p = rotate2D(p, 0.1);
    }
    
    return sum;
}

vec3 rainbowEffect(vec2 coord, float time) {
    return vec3(
        sin(time + coord.x * 10.0),
        sin(time + coord.y * 10.0 + 2.0),
        sin(time + coord.x * 10.0 + coord.y * 10.0 + 4.0)
    );
}

void main() {
    vec2 fragCoord = vec2((uv.x - 0.5) * max(width / height, 1.0) + 0.5, 
                          (uv.y - 0.5) * max(height / width, 1.0) + 0.5);
    vec3 baseColor = mix(vec3(0.01, 0.0, 0.02), vec3(0.0, 0.005, 0.03), fragCoord.y);
    vec3 color = baseColor;

    color = colorWarp(color, fragCoord, time);

    vec2 rotatedCoord = rotate2D(fragCoord, time * 0.1);
    vec3 nebula = nebulaEffect(rotatedCoord, time * 2.0);
    
    // Use fBm to create a more dynamic nebula blend mask
    float nebulaBlendMask = 0.3 + 0.2 * fbm(fragCoord * 4.0 - time * 0.05, time * 0.3);
    color = mix(color, nebula, nebulaBlendMask);

    vec3 blackHoleColor = blackHole(fragCoord, vec2(0.5, 0.5), 0.1);
    color = mix(color, blackHoleColor, 0.5);

    vec2 lensCenter = vec2(0.5 + 0.1 * sin(time), 0.5 + 0.1 * cos(time));
    vec2 lensCoord = gravitationalLens(fragCoord, lensCenter, 0.05);

    for (int i = 0; i < 200; ++i) {
        vec2 point = lensCenter + randomPoint(float(i) * 100.0) * 0.4;
        float dist = distance(lensCoord, point);
        float glowIntensity = gentleGlow(dist, hash(float(i)), time) * (0.5 + 0.5 * sin(time * 2.0 + float(i)));
        float star = smoothstep(0.01, 0.02, sin(dist * 20.0 - time * 5.0 + hash(float(i)) * 5.0) * 0.5 + 0.5) * glowIntensity;
        vec3 starColor = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), hash(float(i) * 3.0));
        starColor *= glowIntensity * glowIntensity;
        vec3 dynamicColor = mix(starColor, vec3(0.8, 0.2, 0.9), (sin(time * 0.5 + float(i) * 2.0) + 1.0) / 2.0);
        color = mix(color, dynamicColor, star);
    }

    float angle = atan(lensCoord.y - lensCenter.y, lensCoord.x - lensCenter.x);
    float radius = length(lensCoord - lensCenter);
    float radialSymmetry = sin(7.0 * angle + radius * 7.0 + time * 1.2) * 0.5 + 0.5;
    color = mix(color, vec3(0.3, 0.6, 0.9), radialSymmetry * 0.4);

    float radialGlow = 1.0 - smoothstep(0.1, 0.5, radius);
    vec3 glowColor = mix(vec3(0.9, 0.3, 0.7), vec3(0.1, 0.7, 0.9), sin(time + radius) * 0.5 + 0.5);
    color += glowColor * 0.2;

    float noise = galacticNoise(fragCoord, time);
    vec3 cosmicRayEnhanced = enhanceCosmicRays(color, fragCoord, noise);
    color = mix(color, cosmicRayEnhanced, 0.15);

    vec3 supernovaColor = improveSupernova(vec3(1.0, 0.5, 0.2), time);
    color = mix(color, supernovaColor, 0.05);
    
    // NEW EFFECTS: apply dynamic twist and rainbow effects before final blurring
    vec3 twistedColor = dynamicTwist(color, fragCoord, time);
    vec3 rainbow = rainbowEffect(fragCoord, time);
    color = mix(twistedColor, rainbow, 0.3);

    vec3 blurredColor = temporalBlur(color, vec3(0.4, 0.6, 0.8), 0.5);
    blurredColor = vignette(blurredColor, fragCoord);
    vec3 starBurst = starBurstEffect(fragCoord, time);
    blurredColor = mix(blurredColor, starBurst, 0.2);
    
    gl_FragColor = vec4(blurredColor, 1.0);
}
