#!/usr/bin/env python3
"""
Plot Latent Vectors by Generation

This script:
1. Loads all latent vectors from embeddings directory
2. Performs dimensionality reduction using UMAP
3. Plots the vectors on a 2D grid, colored by generation

Usage:
    python plot_latents_by_generation.py /path/to/results_dir
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.colors as mcolors


def load_population_data(results_dir):
    """
    Load population data from a results directory

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


def load_latents(results_dir, generations):
    """
    Load latent vectors for all genomes

    Args:
        results_dir: Path to the results directory
        generations: List of lists of genome IDs

    Returns:
        Dictionary of {genome_id: latent_vector}
        Dictionary of {genome_id: generation_number}
    """
    embeddings_dir = Path(results_dir) / "artifacts" / "embeddings"

    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found at {embeddings_dir}")
        return {}, {}

    # Create a mapping from genome ID to generation number
    genome_to_gen = {}
    for gen_idx, genome_ids in enumerate(generations):
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


def reduce_dimensionality(latents, method="umap", n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality of latent vectors using UMAP or PCA

    Args:
        latents: Dictionary of {genome_id: latent_vector}
        method: 'umap' or 'pca' for dimensionality reduction method
        n_neighbors: UMAP parameter for local neighborhood size
        min_dist: UMAP parameter for minimum distance between points

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
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_stack_scaled)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:  # default to UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
        )
        embedding = reducer.fit_transform(latent_stack_scaled)

    # Create mapping from genome ID to 2D coordinates
    coordinates = {}
    for i, genome_id in enumerate(genome_ids):
        coordinates[genome_id] = embedding[i]

    return coordinates


def plot_latents_by_generation(
    coordinates, genome_to_gen, output_path=None, label_interval=5
):
    """
    Plot latent vectors on a 2D grid, colored by generation

    Args:
        coordinates: Dictionary of {genome_id: [x, y]} coordinates
        genome_to_gen: Dictionary of {genome_id: generation_number}
        output_path: Path to save the plot
        label_interval: Show every Nth label on the colorbar
    """
    # Group coordinates by generation
    gen_to_coordinates = defaultdict(list)
    for genome_id, coord in coordinates.items():
        if genome_id in genome_to_gen:
            gen_idx = genome_to_gen[genome_id]
            gen_to_coordinates[gen_idx].append(coord)

    # Convert lists to numpy arrays for plotting
    for gen_idx in gen_to_coordinates:
        gen_to_coordinates[gen_idx] = np.array(gen_to_coordinates[gen_idx])

    # Create a colormap for generations
    num_generations = max(gen_to_coordinates.keys()) + 1
    cmap = plt.get_cmap("viridis", num_generations)
    norm = mcolors.Normalize(vmin=0, vmax=num_generations - 1)

    # Create the plot with explicit figure and axes objects
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each generation with a different color
    for gen_idx, coords in sorted(gen_to_coordinates.items()):
        color = cmap(norm(gen_idx))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=[color],
            # label=f"Generation {gen_idx}",
            alpha=0.7,
            s=50,
        )

    ax.set_title("Latent Space Visualization by Generation", fontsize=16)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    # ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Add colorbar with explicit axes reference and show only every Nth label
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create tick positions at every label_interval
    tick_positions = np.arange(0, num_generations, label_interval)
    if num_generations - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_generations - 1)

    cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions)
    cbar.set_label("Generation", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot latent vectors by generation")
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the plot (default: display plot)",
    )
    parser.add_argument(
        "--neighbors", type=int, default=80, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.6, help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "pca"],
        help="Dimensionality reduction method (umap or pca)",
    )
    parser.add_argument(
        "--label-interval",
        type=int,
        default=5,
        help="Show every Nth label on the colorbar",
    )
    args = parser.parse_args()

    # Load the generations, an array of arrays where each string is a genome id
    generations = load_population_data(args.results_dir)

    if not generations:
        print(f"No population data found in {args.results_dir}")
        return

    # Load latent vectors and generation mapping
    latents, genome_to_gen = load_latents(args.results_dir, generations)

    if not latents:
        print(f"No latent vectors found in {args.results_dir}")
        return

    # Reduce dimensionality using selected method
    coordinates = reduce_dimensionality(
        latents, method=args.method, n_neighbors=args.neighbors, min_dist=args.min_dist
    )

    # If no output path specified, create one in the results directory
    output_path = args.output
    if not output_path:
        method_name = args.method.lower()
        output_path = os.path.join(
            args.results_dir, f"latents_by_generation_{method_name}.png"
        )

    # Plot latents by generation
    plot_latents_by_generation(
        coordinates, genome_to_gen, output_path, args.label_interval
    )


if __name__ == "__main__":
    main()
