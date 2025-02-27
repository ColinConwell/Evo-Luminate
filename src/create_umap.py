#!/usr/bin/env python3
"""
Grid-based UMAP for Latent Vectors

This script processes a directory of .npy files containing latent vectors,
performs a grid-based UMAP projection, and outputs a JSON file with grid positions.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import umap

from sklearn.preprocessing import StandardScaler
import math
from scipy.optimize import linear_sum_assignment


def load_latents(directory):
    """Load all .npy files from directory and return dict of {key: latent_vector}"""
    latents = {}
    for file_path in Path(directory).glob("*.npy"):
        key = file_path.stem  # Get filename without extension
        latent = np.load(file_path)
        latents[key] = latent

    print(f"Loaded {len(latents)} latent vectors from {directory}")
    return latents


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

    # Create a grid assignment problem
    # We'll use the Hungarian algorithm (via scipy) to find optimal assignments
    # from scipy.optimize import linear_sum_assignment

    # Create cost matrix for grid assignment
    # For each latent vector and each grid position, calculate distance
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
    result = {"rows": grid_size, "cols": grid_size, "grid_positions": grid_positions}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create grid-based UMAP visualization of latent vectors"
    )
    parser.add_argument(
        "latent_dir", help="Directory containing .npy latent vector files"
    )
    parser.add_argument(
        "--output", "-o", default="grid_positions.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--neighbors", "-n", type=int, default=15, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist", "-d", type=float, default=0.1, help="UMAP min_dist parameter"
    )
    args = parser.parse_args()

    print(f"Loading latents from {args.latent_dir}")
    # Load latents
    latents = load_latents(args.latent_dir)

    if not latents:
        print(f"No .npy files found in {args.latent_dir}")
        return

    # Create grid UMAP
    result = create_grid_umap(latents, args.neighbors, args.min_dist)

    # Save results
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Grid positions saved to {args.output}")
    print(f"Grid dimensions: {result['rows']}x{result['cols']}")
    print(f"Total positions: {len(result['grid_positions'])}")


if __name__ == "__main__":
    main()
