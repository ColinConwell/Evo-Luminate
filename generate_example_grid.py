#!/usr/bin/env python3
"""
Representative Grid-based UMAP for Latent Vectors

This script:
1. Selects N most representative latent vectors using K-means clustering
2. Performs a grid-based UMAP projection on these representatives
3. Outputs a JSON file with grid positions

Usage:
    python representative_grid_umap.py --input-dir /path/to/latents --output grid_positions.json --num-representatives 50
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math
from scipy.optimize import linear_sum_assignment
import shutil


def load_latents(results_dir):
    directory = os.path.join(results_dir, "artifacts", "embeddings")
    print(f"Loading latents from {directory}")
    """Load latent vectors for all genomes except the first generation"""

    # Get the parent directory (results_dir)
    results_dir = Path(directory).parent.parent

    # Find population data file
    pop_data_path = results_dir / "population_data.jsonl"

    if not pop_data_path.exists():
        print(f"Error: population_data.jsonl not found at {pop_data_path}")
        return {}

    # Extract genome IDs, skipping the first generation
    all_genome_ids = []

    # Read the jsonl file line by line
    with open(pop_data_path, "r") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if "genome_ids" in item:
                    if i > 0:  # Skip the first generation (i == 0)
                        all_genome_ids.extend(item["genome_ids"])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    all_genome_ids = list(set(all_genome_ids))
    print(f"Found {len(all_genome_ids)} genome IDs from generations after the first")

    # Load latent vectors for the selected genome IDs
    latents = {}
    for genome_id in all_genome_ids:
        file_path = Path(directory) / f"{genome_id}.npy"
        if file_path.exists():
            latent = np.load(file_path)
            # Convert multi-dimensional latents to 1D vectors if needed
            if len(latent.shape) > 1:
                latent = latent.flatten()
            latents[genome_id] = latent

    print(f"Loaded {len(latents)} latent vectors from {directory}")
    return latents


def find_representative_latents(latents, n_representatives):
    """
    Find the N most representative latent vectors using K-means clustering

    Args:
        latents: Dictionary of {key: latent_vector}
        n_representatives: Number of representatives to select

    Returns:
        Dictionary of {key: latent_vector} for the selected representatives
    """
    if n_representatives >= len(latents):
        print(
            f"Requested {n_representatives} representatives but only {len(latents)} latents available."
        )
        return latents

    # Stack latents and prepare for clustering
    keys = list(latents.keys())
    latent_stack = np.stack([latents[key] for key in keys])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_representatives, random_state=42)
    kmeans.fit(latent_stack)

    # Find closest embedding to each cluster center
    centers = kmeans.cluster_centers_
    representative_indices = []

    for i in range(n_representatives):
        # Calculate distance from center to all points
        distances = np.linalg.norm(latent_stack - centers[i], axis=1)
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        representative_indices.append(closest_idx)

    # Create dictionary of representative latents
    representative_latents = {}
    for idx in representative_indices:
        key = keys[idx]
        representative_latents[key] = latents[key]

    print(f"Selected {len(representative_latents)} representative latent vectors")
    return representative_latents


def create_grid_umap(latents, n_neighbors=15, min_dist=0.1):
    """
    Project latents to 2D using UMAP and assign grid positions

    Args:
        latents: Dictionary of {key: latent_vector}
        n_neighbors: UMAP parameter for local neighborhood size
        min_dist: UMAP parameter for minimum distance between points

    Returns:
        Dictionary with grid dimensions and positions
    """
    # Stack latents and normalize
    keys = list(latents.keys())
    latent_stack = np.stack([latents[key] for key in keys])

    # Standardize the latent vectors
    scaler = StandardScaler()
    latent_stack_scaled = scaler.fit_transform(latent_stack)

    # Apply UMAP for dimensionality reduction to 2D
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
    )

    embedding = reducer.fit_transform(latent_stack_scaled)

    # Scale embedding to [0, 1] range
    min_vals = embedding.min(axis=0)
    max_vals = embedding.max(axis=0)
    embedding_scaled = (embedding - min_vals) / (max_vals - min_vals)

    # Determine grid size (square grid)
    n_items = len(keys)
    grid_size = math.ceil(math.sqrt(n_items))

    # Create cost matrix for grid assignment
    cost_matrix = np.zeros((n_items, grid_size * grid_size))

    # Generate all grid positions
    grid_positions_array = np.array(
        [(i, j) for i in range(grid_size) for j in range(grid_size)]
    )

    # Normalize grid positions to [0,1] range (same as embedding)
    if grid_size > 1:  # Avoid division by zero
        grid_positions_normalized = grid_positions_array / (grid_size - 1)
    else:
        grid_positions_normalized = grid_positions_array

    # Calculate cost as Euclidean distance between UMAP embeddings and grid positions
    for i in range(n_items):
        for j in range(len(grid_positions_normalized)):
            cost_matrix[i, j] = np.sqrt(
                (embedding_scaled[i, 0] - grid_positions_normalized[j, 0]) ** 2
                + (embedding_scaled[i, 1] - grid_positions_normalized[j, 1]) ** 2
            )

    # Use the Hungarian algorithm to find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Assign grid positions based on optimal assignment
    grid_positions = {}
    for idx, key in enumerate(keys):
        grid_idx = col_indices[idx]
        i, j = grid_positions_array[grid_idx]
        grid_positions[key] = {"i": int(i), "j": int(j)}

    # Create final result
    result = {
        "rows": grid_size,
        "cols": grid_size,
        "grid_positions": grid_positions,
        "representative_keys": keys,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Select representative latent vectors and create grid-based UMAP visualization"
    )
    parser.add_argument(
        "--results-dir",
        "-i",
        required=True,
        help="Directory containing output directory",
    )
    # parser.add_argument(
    #     "--output", "-o", default="grid_positions.json", help="Output JSON file path"
    # )
    parser.add_argument(
        "--num-representatives",
        "-n",
        type=int,
        default=64,
        help="Number of representative latent vectors to select",
    )
    parser.add_argument(
        "--neighbors", type=int, default=15, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist parameter"
    )
    args = parser.parse_args()

    # Load all latents
    latents = load_latents(args.results_dir)

    if not latents:
        print(f"No .npy files found")
        return

    # Find representative latents
    representative_latents = find_representative_latents(
        latents, args.num_representatives
    )

    # Create grid UMAP for the representative latents
    result = create_grid_umap(representative_latents, args.neighbors, args.min_dist)

    output_path = os.path.join(args.results_dir, "grid_positions.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Grid positions saved to {output_path}")
    print(f"Grid dimensions: {result['rows']}x{result['cols']}")
    print(f"Total positions: {len(result['grid_positions'])}")


if __name__ == "__main__":
    main()
