import os
import json
import uuid
import time
import logging
import numpy as np
from enum import Enum
import torch
from typing import List, Dict, Any, Optional, Union

from .shaderToImage import shader_to_image
from .models import llm_client, text_embedder, image_embedder
from .utils import extractCode

defaultModel = "openai:o3-mini"
# defaultModel = "openai:gpt-4o-mini"


class Artifact:
    name = "Artifact"

    def __init__(self, id: str = None):
        self.id = id or str(uuid.uuid4())
        self.genome = None  # Code or prompt
        self.phenome = None  # Path to the rendered phenotype
        self.prompt = None  # Original generation prompt
        self.embedding = None
        self.creation_time = time.time()
        self.metadata = {}

    @classmethod
    def create_random(cls, prompt: str, output_dir: str, **kwargs):
        """Generate a random artifact directly (no explicit idea) and render it"""
        raise NotImplementedError("Subclasses must implement this")

    @classmethod
    def from_genome(cls, genome: str, output_dir: str, prompt: str = None, **kwargs):
        """Create an artifact from an existing genome and render it"""
        raise NotImplementedError("Subclasses must implement this")

    def render_phenotype(self, output_dir: str, **kwargs) -> Optional[str]:
        """Render the phenotype from the genome"""
        raise NotImplementedError("Subclasses must implement this")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from a dictionary"""
        raise NotImplementedError("Subclasses must implement this")

    def save(self, output_dir: str):
        raise NotImplementedError("Subclasses must implement this")
        # """Save to disk"""
        # os.makedirs(output_dir, exist_ok=True)

        # artifact_path = os.path.join(output_dir, f"{self.id}.json")
        # with open(artifact_path, "w") as f:
        #     json.dump(self.to_dict(), f, indent=2)

        # return artifact_path

    def crossover_with(
        self,
        other_artifacts: List["Artifact"],
        crossover_idea: str,
        output_dir: str,
        **kwargs,
    ):
        """Create a new artifact by crossing over this artifact with others"""
        raise NotImplementedError("Subclasses must implement this")


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

        response = llm_client.chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": ShaderArtifact.systemPrompt},
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
            shader_to_image(self.genome, frame_path, 768, 768, uniforms={"time": t})
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
                frame_emb = image_embedder.embedImage(frame_path)
                frame_embeddings.append(frame_emb)
            else:
                logging.warning(f"Frame path not found: {frame_path}")

        # Concatenate
        concat_embedding = torch.cat(frame_embeddings, dim=0)
        # Normalize
        normalized = torch.nn.functional.normalize(concat_embedding, dim=0)
        self.embedding = normalized
        return self.embedding

    # def save(self, output_dir: str):
    #     """Save to disk"""
    #     os.makedirs(output_dir, exist_ok=True)
    #     genome_path = os.path.join(output_dir, f"{self.id}.glsl")
    #     with open(genome_path, "w") as f:
    #         f.write(self.genome)

    #     if self.embedding is not None:
    #         embedding_path = os.path.join(output_dir, f"{self.id}_embedding.npy")
    #         np.save(embedding_path, self.embedding.cpu().numpy())


class GameIdeaArtifact(Artifact):
    name = "game idea"
    systemPrompt = """You are an expert in designing p5.js games
    Return a description of a game that can be implemented in p5.js
    Include mechanics, art style, rules, win conditions, and any other relevant details
    """

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt

        response = llm_client.chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": GameIdeaArtifact.systemPrompt},
                {"role": "user", "content": f"User prompt: {prompt}"},
            ],
        )

        artifact.genome = response.choices[0].message.content.strip()

        os.makedirs(os.path.join(output_dir, "ideas"), exist_ok=True)
        idea_path = os.path.join(output_dir, f"ideas/{artifact.id}.txt")
        with open(idea_path, "w") as f:
            f.write(artifact.genome)

        artifact.compute_embedding()
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
        embedding_path = os.path.join(output_dir, f"embeddings/{artifact.id}.npy")
        np.save(embedding_path, artifact.embedding.cpu().numpy())

        return artifact

    def compute_embedding(self) -> torch.Tensor:
        """Compute embedding for this game idea artifact"""
        if self.embedding is not None:
            return self.embedding

        self.embedding = text_embedder.embedText(self.genome)[0]
        return self.embedding
