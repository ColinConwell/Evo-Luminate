"""Render the shader to an image"""

import os, math

from src.shaderToImage import shader_to_image
from src.artifacts.sdf_code import (
    vertexShader,
    shaderTemplate,
    sdfLibrary,
)

output_dir = "test_sdf_render"
os.makedirs(output_dir, exist_ok=True)

source_code = """
float scene(vec3 p) {
    float sphere = sdSphere(p - vec3(0.0, 0.0, 0.0), 1.0);
    float box = sdBox(p - vec3(2.0, 0.0, 0.0), vec3(0.5));
    float torus = sdTorus(p - vec3(-2.0, 0.0, 0.0), vec2(0.7, 0.3));
    
    return opUnion(opUnion(sphere, box), torus);
    
}
"""


def make_fragment_shader(source_code: str) -> str:
    return (
        shaderTemplate.replace("{{SOURCE_CODE}}", source_code)
        .replace("{{LIBRARY_METHODS}}", sdfLibrary)
        .replace("{{CONSTANTS}}", "")
    )


fragmentShader = make_fragment_shader(source_code)


def render_from_angles(output_dir, angles):
    """Render the scene from multiple angles around the origin"""
    os.makedirs(output_dir, exist_ok=True)

    for i, angle in enumerate(angles):
        frame_path = f"{output_dir}/frame_{i:03d}.png"

        # Calculate camera position based on angle
        # Orbit camera at fixed distance from origin
        distance = 5.0
        camera_x = distance * math.sin(angle)
        camera_z = distance * math.cos(angle)
        camera_y = 1.0  # Slightly above the object

        # Camera position
        camera_position = [camera_x, camera_y, camera_z]

        shader_to_image(
            fragmentShader,
            vertexShader,
            frame_path,
            512,
            512,
            uniforms={
                "u_camPos": camera_position,
                "u_eps": 0.001,  # Small epsilon for surface detection
                "u_maxDis": 20.0,  # Maximum ray distance
                "u_maxSteps": 100,
            },
        )


# Example usage
angles = [
    0,
    math.pi / 4,
    math.pi / 2,
    3 * math.pi / 4,
    math.pi,
    5 * math.pi / 4,
    3 * math.pi / 2,
    7 * math.pi / 4,
]
render_from_angles("test_sdf_render", angles)
