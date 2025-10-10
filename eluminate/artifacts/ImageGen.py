import os
import logging
import numpy as np
from enum import Enum
import torch
from typing import Optional

from eluminate.models import llm_client, image_embedder, make_image
from eluminate.artifacts.Artifact import Artifact

defaultModel = "openai:o3-mini"


class ImageGenArtifact(Artifact):
    name = "AI image prompt"
    systemPrompt = """You create image prompts for AI image generation.
    Prompt Structure Guidelines:
    Core Components (Required)
    Subject: Main focus/central element (what/who is being depicted)
    Style: Artistic approach or visual aesthetic
    Composition: Arrangement of elements within frame
    Lighting: Type, direction, and quality of light
    Color Palette: Dominant colors or color scheme
    Mood/Atmosphere: Emotional tone or ambiance
    Technical Details: Camera settings, perspective, techniques
    Additional Elements: Supporting details or background context

    Best Practices:
    Be specific and descriptive rather than vague
    Maintain concise wording (max 1024 characters)
    Structure information in clear, labeled sections
    Focus on visual characteristics rather than backstory
    Prioritize concrete details over abstract concepts
    Use terminology relevant to visual arts and photography
    
    Return only the image prompt. No other text.

    """
    # Format Example
    # Subject: [specific description]
    # Style: [artistic approach]
    # Composition: [arrangement]
    # Lighting: [light qualities]
    # Color Palette: [key colors]
    # Mood: [emotional tone]
    # Technical Details: [camera/rendering specifics]
    # Additional Elements: [supporting details]

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        artifact = cls()
        artifact.prompt = prompt

        response = llm_client.chat.completions.create(
            model=defaultModel,
            max_completion_tokens=20000,
            reasoning_effort=kwargs.get("reasoning_effort", "low"),
            messages=[
                {"role": "system", "content": cls.systemPrompt},
                {"role": "user", "content": f"User prompt: {prompt}"},
            ],
        )
        imagePrompt = response.choices[0].message.content.strip()
        artifact.genome = imagePrompt

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        artifact.render_phenotype(os.path.join(output_dir, "images"), **kwargs)
        artifact.compute_embedding()

        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
        prompt_path = os.path.join(output_dir, f"prompts/{artifact.id}.txt")
        with open(prompt_path, "w") as f:
            f.write(imagePrompt)

        # if self.embedding is not None:
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
        embedding_path = os.path.join(output_dir, f"embeddings/{artifact.id}.npy")
        np.save(embedding_path, artifact.embedding.cpu().numpy())

        return artifact

    def render_phenotype(self, output_dir: str, **kwargs) -> Optional[str]:
        """Render the shader to an image"""
        os.makedirs(output_dir, exist_ok=True)

        image = make_image(self.genome)

        save_path = os.path.join(output_dir, f"{self.id}.jpg")
        image.save(save_path)

        self.phenome = save_path

        return save_path

    def compute_embedding(self) -> torch.Tensor:
        """Compute embedding for this shader artifact"""
        if self.embedding is not None:
            return self.embedding

        image_emb = image_embedder.embedImage(self.phenome)
        self.embedding = image_emb
        return self.embedding
