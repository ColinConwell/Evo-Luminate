import os
import numpy as np
import torch

from src.models import get_llm_client, defaultModel, get_text_embedder
from src.artifacts.Artifact import Artifact


class GaArtifact(Artifact):
    name = "image generating genetic algorithms"
    systemPrompt = """You are an expert in creating genetic representations for images. You may use any python libraries you want.
    Only return the python class.
    Return a python file with a class that has the following methods:
    @classmethod
    def create_random(cls) -> ImageGenome:
    def mutate(self) -> ImageGenome:
    def crossover(self, other) -> ImageGenome:
    def render(self): # returns a PIL Image or numpy array
    """

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt
        userContent = [
            {"type": "text", "text": f"User prompt: {prompt}"},
        ]

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

        os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
        genome_path = os.path.join(output_dir, f"source/{artifact.id}.txt")
        with open(genome_path, "w") as f:
            f.write(artifact.genome)

        return artifact

    def compute_embedding(self) -> torch.Tensor:
        """Compute embedding for this shader artifact"""
        if self.embedding is not None:
            return self.embedding

        emb = get_text_embedder().embedText(self.genome)[0]

        self.embedding = emb
        return self.embedding
