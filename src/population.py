import os
import json
import uuid
import time
import subprocess
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union

# from Genomes import Genome, ShaderGenome, IdeaGenome, PromptGenome
from artifacts import Artifact, ShaderArtifact


class Population:
    """Manages a population of genomes"""

    def __init__(self):
        self.artifacts = []
        self.id_to_artifact = {}

    def add(self, artifact: Artifact):
        """Add a genome to the population"""
        self.artifacts.append(artifact)
        self.id_to_artifact[artifact.id] = artifact

    def add_all(self, artifacts: List[Artifact]):
        """Add multiple genomes to the population"""
        for artifact in artifacts:
            self.add(artifact)

    def remove(self, artifact: Artifact):
        """Remove a genome from the population"""
        if artifact.id in self.id_to_artifact:
            self.artifacts.remove(artifact)
            del self.id_to_artifact[artifact.id]

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Get a genome by ID"""
        return self.id_to_artifact.get(artifact_id)

    def get_all(self) -> List[Artifact]:
        """Get all genomes"""
        return self.artifacts.copy()

    def get_random(self, count: int = 1) -> List[Artifact]:
        """Get random genomes"""
        if count >= len(self.artifacts):
            return self.get_all()

        indices = np.random.choice(len(self.artifacts), size=count, replace=False)
        return [self.artifacts[i] for i in indices]

    def get_best(self, count: int = 1) -> List[Artifact]:
        """Get genomes with highest fitness"""
        sorted_artifacts = sorted(
            self.artifacts,
            key=lambda a: a.fitness if a.fitness is not None else float("-inf"),
            reverse=True,
        )
        return sorted_artifacts[:count]

    def select_by_novelty(
        self,
        embeddings: torch.Tensor,
        k_neighbors: int = 3,
        return_distances: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Select genomes by novelty score (distance to k nearest neighbors)
        Returns indices of genomes sorted by novelty (highest first)

        If return_distances is True, also returns the average distance to k nearest neighbors
        """
        if len(self.artifacts) <= k_neighbors:
            indices = list(range(len(self.artifacts)))
            if return_distances:
                # If not enough artifacts for meaningful distances, return zeros
                return indices, torch.zeros(len(self.artifacts))
            return indices

        norm_emb = torch.nn.functional.normalize(embeddings, dim=1)
        similarity = torch.mm(norm_emb, norm_emb.t())
        distances = 1 - similarity  # cosine distance

        # Set self-distance to a high value to exclude from nearest neighbor calculation
        for i in range(distances.shape[0]):
            distances[i, i] = 1e6

        # Get k nearest neighbors for each genome
        sorted_dist, _ = torch.sort(distances, dim=1)
        k_nearest = sorted_dist[:, :k_neighbors]

        # Compute novelty as average distance to k nearest neighbors
        novelty_scores = k_nearest.mean(dim=1)

        # Get indices sorted by novelty (highest first)
        sorted_indices = torch.argsort(novelty_scores, descending=True).tolist()

        if return_distances:
            return sorted_indices, novelty_scores

        return sorted_indices

    def save(self, output_dir: str):
        """Save all genomes to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save list of all genome IDs
        genome_ids = [g.id for g in self.artifacts]
        meta_path = os.path.join(output_dir, "population.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "genome_ids": genome_ids,
                    "count": len(self.artifacts),
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )

        # Save each genome
        artifacts_dir = os.path.join(output_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        for artifact in self.artifacts:
            artifact.save(artifacts_dir)

        return meta_path

    @classmethod
    def load(cls, input_dir: str) -> "Population":
        """Load population from disk"""
        population = cls()

        # Load list of genome IDs
        meta_path = os.path.join(input_dir, "population.json")
        if not os.path.exists(meta_path):
            logging.error("Population metadata file not found: %s", meta_path)
            return population

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load each genome
        artifacts_dir = os.path.join(input_dir, "artifacts")
        for artifact_id in meta.get("artifact_ids", []):
            artifact_path = os.path.join(artifacts_dir, f"{artifact_id}.json")

            if not os.path.exists(artifact_path):
                logging.warning("Artifact file not found: %s", artifact_path)
                continue

            with open(artifact_path, "r") as f:
                artifact_data = json.load(f)

            # Create appropriate artifact type based on the data
            if artifact_data.get("type") == "ShaderArtifact":
                artifact = ShaderArtifact.from_dict(artifact_data)
            # elif artifact_data.get("type") == "IdeaGenome":
            #     artifact = IdeaGenome.from_dict(artifact_data)
            # elif artifact_data.get("type") == "PromptGenome":
            #     artifact = PromptGenome.from_dict(artifact_data)
            else:
                raise ValueError(f"Unknown artifact type: {artifact_data.get('type')}")

            population.add(artifact)

        return population
