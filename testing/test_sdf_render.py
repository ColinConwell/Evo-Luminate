"""Slow render test (optional). Run with: pytest -m slow"""

import os, math, pytest

from eluminate.shaderToImage import shader_to_image
from eluminate.artifacts.sdf_code import (
    vertexShader,
    shaderTemplate,
    sdfLibrary,
)


def make_fragment_shader(source_code: str) -> str:
    return (
        shaderTemplate.replace("{{SOURCE_CODE}}", source_code)
        .replace("{{LIBRARY_METHODS}}", sdfLibrary)
        .replace("{{CONSTANTS}}", "")
    )


@pytest.mark.slow
def test_render_small_single_frame(tmp_path):
    output_dir = tmp_path
    source_code = """
    float scene(vec3 p) {
        return sdSphere(p, 1.0);
    }
    """
    fragmentShader = make_fragment_shader(source_code)
    frame_path = os.path.join(output_dir, "frame.png")

    # Render a tiny image quickly
    shader_to_image(
        fragmentShader,
        vertexShader,
        frame_path,
        128,
        128,
        uniforms={"u_camPos": [0.0, 1.0, 4.0], "u_eps": 0.002, "u_maxDis": 10.0, "u_maxSteps": 64},
    )
    assert os.path.exists(frame_path)
