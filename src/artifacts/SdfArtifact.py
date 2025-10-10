import os
import math
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Optional, Union

from src.utils import extractBlocks, saveCodeBlocks, loadCodeBlocks

from src.artifacts.Artifact import Artifact
from src.artifacts.ShaderArtifact import ShaderArtifact
from ..shaderToImage import shader_to_image
from .sdf_code import (
    shaderTemplate,
    vertexShader,
    shaderTemplate,
    sdfLibrary,
    sdfLibraryHeaders,
)
from src.models import get_llm_client, defaultModel, get_image_embedder


# - Use combinations of primitive SDFs (sphere, box, torus, etc.)
# - Apply operations like union, subtraction, intersection, and smooth blending
# - Consider symmetry, repetition, and domain manipulation for complex effects
# - Add surface details using noise or mathematical patterns
# - Consider performance - avoid excessive operations or recursion


class SDFArtifact(Artifact):
    name = "sdf"
    systemPrompt = """
    You are an expert in creating 3D objects using WebGL 1.0 and signed distance functions (SDFs).
    
    Technical requirements:
    - Create a function called scene(vec3 p) that returns a float representing the SDF
    - Use only GLSL 1.0 compatible syntax
    - Do not create any uniforms - all design variables should be constants
    - Consider every part of the object and its requirements.
    - Don't include your thinking in the comments.
    - Only return the code. No chatting. 
    
    Creative guidelines:
    - Utilize the full capabilities of glsl and 3d signed distance fields.
    - Create a visually interesting 3D object with multiple components
    
    Return your code in the following format:
    <CONSTANTS>
    // Design parameters as constants
    // Example: const float RADIUS = 1.0;
    </CONSTANTS>
    <SOURCE>
    // Helper functions
    
    // Main scene function
    float scene(vec3 p) {
        // Your SDF code here
    }
    </SOURCE>
    
    You will have access to this library of SDF functions:
    {{HEADERS}}
    """.strip().replace(
        "{{HEADERS}}", ""
    )
    # .replace(
    #     "{{HEADERS}}", sdfLibraryHeaders
    # )

    def _make_fragment_shader(self, genome: str) -> str:
        blocks = extractBlocks(genome)
        assert "SOURCE" in blocks, "SOURCE block is missing"
        assert "CONSTANTS" in blocks, "CONSTANTS block is missing"
        return (
            shaderTemplate.replace("{{SOURCE_CODE}}", blocks["SOURCE"])
            .replace("{{CONSTANTS}}", blocks["CONSTANTS"])
            .replace("{{LIBRARY_METHODS}}", "")
            # .replace("{{LIBRARY_METHODS}}", sdfLibrary)
        )

    def render_phenotype(self, output_dir: str, **kwargs) -> Optional[str]:
        """Render the shader to an image"""
        os.makedirs(output_dir, exist_ok=True)

        # Define 3 different viewing angles (in radians)
        angles = [0, 2.0944, 4.18879]  # 0°, 120°, 240° around the object
        frame_paths = []
        fragmentShader = self._make_fragment_shader(self.genome)
        os.makedirs(output_dir, exist_ok=True)
        for i, angle in enumerate(angles):
            frame_path = f"{output_dir}/{self.id}_frame_{i:03d}.png"

            # Calculate camera position based on angle
            # Orbit camera at fixed distance from origin
            distance = 4.0
            camera_x = distance * math.sin(angle)
            camera_z = distance * math.cos(angle)
            camera_y = 1.5  # Slightly above the object

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
            frame_paths.append(frame_path)

        self.phenome = frame_paths
        return self.phenome

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, image_url: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt
        userContent = [
            {"type": "text", "text": f"User prompt: {prompt}"},
        ]
        if image_url:
            userContent.append({"type": "image_url", "image_url": {"url": image_url}})

        response = get_llm_client().chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": cls.systemPrompt},
                {"role": "user", "content": userContent},
            ],
        )
        artifact.genome = response.choices[0].message.content.strip()
        # print(response.choices[0].message.content.strip())

        os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
        genome_path = os.path.join(output_dir, f"source/{artifact.id}.txt")
        with open(genome_path, "w") as f:
            f.write(artifact.genome)

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        artifact.render_phenotype(os.path.join(output_dir, "images"), **kwargs)
        artifact.compute_embedding()

        # if self.embedding is not None:
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
        embedding_path = os.path.join(output_dir, f"embeddings/{artifact.id}.npy")
        np.save(embedding_path, artifact.embedding.cpu().numpy())

        # Save prompt to output dir
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
        prompt_path = os.path.join(output_dir, f"prompts/{artifact.id}.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        return artifact

    def compute_embedding(self) -> torch.Tensor:
        """Compute embedding for this shader artifact"""
        if self.embedding is not None:
            return self.embedding

        frame_embeddings = []
        for frame_path in self.phenome:
            if os.path.exists(frame_path):
                frame_emb = get_image_embedder().embedImage(frame_path)
                frame_embeddings.append(frame_emb)
            else:
                logging.warning(f"Frame path not found: {frame_path}")

        # Concatenate
        concat_embedding = torch.cat(frame_embeddings, dim=0)
        # Normalize
        normalized = torch.nn.functional.normalize(concat_embedding, dim=0)
        self.embedding = normalized
        return self.embedding
