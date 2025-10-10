import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any


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
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        """Generate a random artifact directly (no explicit idea) and render it"""
        raise NotImplementedError("Subclasses must implement this")

    def render_phenotype(self, output_dir: str, **kwargs) -> Optional[str]:
        """Render the phenotype from the genome"""
        raise NotImplementedError("Subclasses must implement this")
