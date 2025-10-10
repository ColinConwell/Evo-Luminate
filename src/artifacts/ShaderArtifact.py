import os
import logging
import numpy as np
from enum import Enum
import torch
from typing import List, Dict, Any, Optional, Union

from src.shaderToImage import shader_to_image
from src.models import get_llm_client, get_image_embedder
from src.utils import extractCode
from src.artifacts.Artifact import Artifact

defaultModel = "openai:o3-mini"

vertex_code = """
precision mediump float;
attribute vec2 position;
varying vec2 uv;
void main() {
    // Map clip-space [-1,1] to uv [0,1]
    uv = 0.5 * (position + 1.0);
    gl_Position = vec4(position, 0, 1);
}
"""


class ShaderArtifact(Artifact):
    name = "shader"
    systemPrompt = """You are an expert in creating WebGL 1.0 fragment shaders.
    Return valid webgl fragment shader.
    Provide the full fragment shader code without explanation.
    You can only use these uniforms:
	varying vec2 uv;
	uniform float time;
    Start the file with  "precision mediump float;"
    """

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt

        response = get_llm_client().chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": cls.systemPrompt},
                {"role": "user", "content": f"User prompt: {prompt}"},
            ],
        )

        artifact.genome = extractCode(response.choices[0].message.content.strip())
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        artifact.render_phenotype(os.path.join(output_dir, "images"), **kwargs)
        artifact.compute_embedding()

        os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
        genome_path = os.path.join(output_dir, f"source/{artifact.id}.glsl")
        with open(genome_path, "w") as f:
            f.write(artifact.genome)

        # if self.embedding is not None:
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
        embedding_path = os.path.join(output_dir, f"embeddings/{artifact.id}.npy")
        np.save(embedding_path, artifact.embedding.cpu().numpy())

        # Save prompt to output dir
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
        prompt_path = os.path.join(output_dir, f"prompts/{artifact.id}.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        # artifacts_dir = os.path.join(output_dir, "artifacts")
        # os.makedirs(artifacts_dir, exist_ok=True)

        # for artifact in self.artifacts:
        #     artifact.save(artifacts_dir)
        return artifact

    def render_phenotype(self, output_dir: str, **kwargs) -> Optional[str]:
        """Render the shader to an image"""
        os.makedirs(output_dir, exist_ok=True)

        time_points = [0, 3]
        frame_paths = []

        for i, t in enumerate(time_points):
            frame_path = f"{output_dir}/{self.id}_t{i}.png"
            shader_to_image(
                self.genome, vertex_code, frame_path, 768, 768, uniforms={"time": t}
            )
            frame_paths.append(frame_path)

        self.phenome = frame_paths

        return self.phenome

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
