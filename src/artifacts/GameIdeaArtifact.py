import os
import numpy as np
import torch


from src.models import text_embedder, llm_client, defaultModel
from src.artifacts.Artifact import Artifact


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
