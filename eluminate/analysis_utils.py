"""Shared utilities for analyzing and visualizing evolution experiments."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_population_data(results_dir: str) -> List[List[str]]:
    """
    Load population data from a results directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of lists, where each inner list contains genome IDs for a generation
    """
    results_path = Path(results_dir)
    pop_data_path = results_path / "population_data.jsonl"

    if not pop_data_path.exists():
        print(f"Error: population_data.jsonl not found at {pop_data_path}")
        return []

    generations = []
    with open(pop_data_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                if "genome_ids" in data:
                    generations.append(data["genome_ids"])

    print(f"Loaded {len(generations)} generations from {pop_data_path}")
    return generations


def load_latents(
    results_dir: str,
    generations: Optional[List[List[str]]] = None,
    skip_first_generation: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load latent vectors for all genomes.

    Args:
        results_dir: Path to the results directory
        generations: List of lists of genome IDs (loaded if None)
        skip_first_generation: If True, skip loading the first generation

    Returns:
        Dictionary of {genome_id: latent_vector}
        Dictionary of {genome_id: generation_number}
    """
    results_path = Path(results_dir)
    embeddings_dir = results_path / "artifacts" / "embeddings"

    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found at {embeddings_dir}")
        return {}, {}

    # Load generations if not provided
    if generations is None:
        generations = load_population_data(results_dir)

    # Create a mapping from genome ID to generation number
    genome_to_gen = {}
    start_gen = 1 if skip_first_generation else 0
    for gen_idx, genome_ids in enumerate(generations[start_gen:], start=start_gen):
        for genome_id in genome_ids:
            genome_to_gen[genome_id] = gen_idx

    # Load latent vectors
    latents = {}
    for genome_id in genome_to_gen:
        file_path = embeddings_dir / f"{genome_id}.npy"
        if file_path.exists():
            latent = np.load(file_path)
            # Convert multi-dimensional latents to 1D vectors if needed
            if len(latent.shape) > 1:
                latent = latent.flatten()
            latents[genome_id] = latent

    print(f"Loaded {len(latents)} latent vectors from {embeddings_dir}")
    return latents, genome_to_gen


def load_novelty_metrics(results_dir: str) -> List[Dict]:
    """
    Load novelty metrics from a single JSONL file.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of dictionaries containing metrics for each generation
    """
    results_path = Path(results_dir)
    metrics_list = []

    # Look for the novelty_metrics.jsonl file
    metrics_file = results_path / "novelty_metrics.jsonl"

    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    metrics = json.loads(line)
                    metrics_list.append(metrics)
    else:
        print(f"No novelty_metrics.jsonl file found in {results_dir}")

    # Sort by generation number
    metrics_list.sort(key=lambda x: x.get("generation", 0))
    return metrics_list


def reduce_dimensionality(
    latents: Dict[str, np.ndarray],
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Reduce dimensionality of latent vectors using UMAP or PCA.

    Args:
        latents: Dictionary of {genome_id: latent_vector}
        method: 'umap' or 'pca' for dimensionality reduction method
        n_neighbors: UMAP parameter for local neighborhood size
        min_dist: UMAP parameter for minimum distance between points
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of {genome_id: [x, y]} coordinates
    """
    # Stack latents and prepare for dimensionality reduction
    genome_ids = list(latents.keys())
    latent_stack = np.stack([latents[genome_id] for genome_id in genome_ids])

    # Standardize the latent vectors
    scaler = StandardScaler()
    latent_stack_scaled = scaler.fit_transform(latent_stack)

    # Apply dimensionality reduction to 2D
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(latent_stack_scaled)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:  # default to UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(latent_stack_scaled)

    # Create mapping from genome ID to 2D coordinates
    coordinates = {}
    for i, genome_id in enumerate(genome_ids):
        coordinates[genome_id] = embedding[i]

    return coordinates

