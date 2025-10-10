#!/usr/bin/env python3
"""
Consolidated plotting and visualization script for evolution experiments.

This script provides multiple visualization modes:
- novelty: Plot novelty metrics across generations
- umap-generations: Plot latent space colored by generation
- umap-grid: Create a grid-based UMAP visualization
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eluminate.analysis_utils import (
    load_population_data,
    load_latents,
    load_novelty_metrics,
    reduce_dimensionality,
)


# ============================================================================
# Novelty Plotting
# ============================================================================


def plot_novelty_metrics(
    metrics_list: list, output_path: str = None, show: bool = False
):
    """Plot novelty metrics across generations."""
    if not metrics_list:
        print("No metrics data found")
        return

    generations = [m.get("generation", i + 1) for i, m in enumerate(metrics_list)]
    mean_novelty = [m.get("mean_novelty", 0) for m in metrics_list]
    mean_genome_length = [m.get("mean_genome_length", 0) for m in metrics_list]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot mean novelty on primary y-axis
    color = "blue"
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Novelty Score (Cosine Distance)", fontsize=12, color=color)
    ax1.plot(generations, mean_novelty, color=color, linewidth=2, label="Mean Novelty")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create secondary y-axis and plot mean genome length
    ax2 = ax1.twinx()
    color = "red"
    ax2.set_ylabel("Mean Genome Length", fontsize=12, color=color)
    ax2.plot(
        generations,
        mean_genome_length,
        color=color,
        linestyle="-",
        linewidth=2,
        label="Mean Genome Length",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Population Novelty and Genome Length Across Generations", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_strategy_comparison(
    metrics_list: list, output_path: str = None, show: bool = False
):
    """Create a bar chart comparing average novelty scores for different strategies."""
    if not metrics_list:
        print("No metrics data found")
        return

    # Collect strategy data across all generations
    strategy_data = {}

    for metrics in metrics_list:
        if "strategy_metrics" in metrics:
            for strategy, stats in metrics["strategy_metrics"].items():
                if strategy == "None":
                    continue
                if strategy not in strategy_data:
                    strategy_data[strategy] = {"novelty_scores": [], "counts": []}

                strategy_data[strategy]["novelty_scores"].append(stats["avg_novelty"])
                strategy_data[strategy]["counts"].append(stats["count"])

    if not strategy_data:
        print("No strategy metrics found in the data")
        return

    # Calculate average novelty and standard deviation for each strategy
    strategies = []
    avg_novelties = []
    std_novelties = []
    avg_counts = []

    for strategy, data in strategy_data.items():
        if data["novelty_scores"]:
            strategies.append(strategy)
            avg_novelties.append(np.mean(data["novelty_scores"]))
            std_novelties.append(np.std(data["novelty_scores"]))
            avg_counts.append(np.mean(data["counts"]))

    # Sort strategies by average novelty (descending)
    sorted_indices = np.argsort(avg_novelties)[::-1]
    strategies = [strategies[i] for i in sorted_indices]
    avg_novelties = [avg_novelties[i] for i in sorted_indices]
    std_novelties = [std_novelties[i] for i in sorted_indices]
    avg_counts = [avg_counts[i] for i in sorted_indices]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.7
    positions = np.arange(len(strategies))

    bars = ax.bar(
        positions,
        avg_novelties,
        bar_width,
        yerr=std_novelties,
        capsize=5,
        color="skyblue",
        edgecolor="navy",
        alpha=0.8,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, avg_counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std_novelties[i] + 0.005,
            f"n≈{count:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Average Novelty Score", fontsize=12)
    ax.set_title("Comparison of Novelty Scores by Strategy", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        strategy_output = output_path.replace(".png", "_strategy_comparison.png")
        plt.savefig(strategy_output, dpi=300, bbox_inches="tight")
        print(f"Strategy comparison plot saved to {strategy_output}")

    if show:
        plt.show()
    plt.close()


# ============================================================================
# UMAP by Generation Plotting
# ============================================================================


def plot_latents_by_generation(
    coordinates: dict,
    genome_to_gen: dict,
    output_path: str = None,
    label_interval: int = 5,
    show: bool = False,
):
    """Plot latent vectors on a 2D grid, colored by generation."""
    # Group coordinates by generation
    gen_to_coordinates = defaultdict(list)
    for genome_id, coord in coordinates.items():
        if genome_id in genome_to_gen:
            gen_idx = genome_to_gen[genome_id]
            gen_to_coordinates[gen_idx].append(coord)

    # Convert lists to numpy arrays
    for gen_idx in gen_to_coordinates:
        gen_to_coordinates[gen_idx] = np.array(gen_to_coordinates[gen_idx])

    # Create colormap
    num_generations = max(gen_to_coordinates.keys()) + 1
    cmap = plt.get_cmap("viridis", num_generations)
    norm = mcolors.Normalize(vmin=0, vmax=num_generations - 1)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each generation
    for gen_idx, coords in sorted(gen_to_coordinates.items()):
        color = cmap(norm(gen_idx))
        ax.scatter(coords[:, 0], coords[:, 1], c=[color], alpha=0.7, s=50)

    ax.set_title("Latent Space Visualization by Generation", fontsize=16)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.grid(alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    tick_positions = np.arange(0, num_generations, label_interval)
    if num_generations - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_generations - 1)

    cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions)
    cbar.set_label("Generation", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================================
# UMAP Grid Visualization
# ============================================================================


def find_representative_latents(latents: dict, n_representatives: int) -> dict:
    """Find the N most representative latent vectors using K-means clustering."""
    if n_representatives >= len(latents):
        print(
            f"Requested {n_representatives} representatives but only {len(latents)} latents available."
        )
        return latents

    keys = list(latents.keys())
    latent_stack = np.stack([latents[key] for key in keys])

    kmeans = KMeans(n_clusters=n_representatives, random_state=42)
    kmeans.fit(latent_stack)

    centers = kmeans.cluster_centers_
    representative_indices = []

    for i in range(n_representatives):
        distances = np.linalg.norm(latent_stack - centers[i], axis=1)
        closest_idx = np.argmin(distances)
        representative_indices.append(closest_idx)

    representative_latents = {}
    for idx in representative_indices:
        key = keys[idx]
        representative_latents[key] = latents[key]

    print(f"Selected {len(representative_latents)} representative latent vectors")
    return representative_latents


def create_grid_umap(
    latents: dict, n_neighbors: int = 15, min_dist: float = 0.1, aspect_ratio: float = 1.0
) -> dict:
    """Project latents to 2D using UMAP and assign grid positions."""
    from sklearn.preprocessing import StandardScaler
    import umap

    keys = list(latents.keys())
    latent_stack = np.stack([latents[key] for key in keys])

    scaler = StandardScaler()
    latent_stack_scaled = scaler.fit_transform(latent_stack)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
    )
    embedding = reducer.fit_transform(latent_stack_scaled)

    # Scale embedding to [0, 1]
    min_vals = embedding.min(axis=0)
    max_vals = embedding.max(axis=0)
    embedding_scaled = (embedding - min_vals) / (max_vals - min_vals)

    # Determine grid size
    n_items = len(keys)
    height = math.sqrt(n_items / aspect_ratio)
    width = height * aspect_ratio

    cols = math.ceil(width)
    rows = math.ceil(height)

    while rows * cols < n_items:
        if cols / rows < aspect_ratio:
            cols += 1
        else:
            rows += 1

    grid_size_rows = rows + 1
    grid_size_cols = cols + 1

    # Create cost matrix
    cost_matrix = np.zeros((n_items, grid_size_rows * grid_size_cols))
    grid_positions_array = np.array(
        [(i, j) for i in range(grid_size_rows) for j in range(grid_size_cols)]
    )

    # Apply non-linear scaling for organic look
    center_i = (grid_size_rows - 1) / 2
    center_j = (grid_size_cols - 1) / 2

    grid_positions_scaled = np.zeros_like(grid_positions_array, dtype=float)
    for idx, (i, j) in enumerate(grid_positions_array):
        di = (i - center_i) / max(center_i, 1)
        dj = (j - center_j) / max(center_j, 1)

        scale_factor = 0.3

        di_scaled = di * (abs(di) ** scale_factor) / abs(di) if di != 0 else 0
        dj_scaled = dj * (abs(dj) ** scale_factor) / abs(dj) if dj != 0 else 0

        grid_positions_scaled[idx, 0] = center_i + di_scaled * center_i
        grid_positions_scaled[idx, 1] = center_j + dj_scaled * center_j

    # Normalize grid positions
    grid_positions_normalized = np.zeros_like(grid_positions_scaled, dtype=float)
    if grid_size_rows > 1:
        grid_positions_normalized[:, 0] = grid_positions_scaled[:, 0] / (
            grid_size_rows - 1
        )
    if grid_size_cols > 1:
        grid_positions_normalized[:, 1] = grid_positions_scaled[:, 1] / (
            grid_size_cols - 1
        )

    # Calculate cost
    for i in range(n_items):
        for j in range(len(grid_positions_normalized)):
            cost_matrix[i, j] = np.sqrt(
                (embedding_scaled[i, 0] - grid_positions_normalized[j, 0]) ** 2
                + (embedding_scaled[i, 1] - grid_positions_normalized[j, 1]) ** 2
            )

    # Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Assign grid positions
    grid_positions = {}
    for idx, key in enumerate(keys):
        grid_idx = col_indices[idx]
        i, j = grid_positions_array[grid_idx]
        grid_positions[key] = {"i": int(i), "j": int(j)}

    return {
        "rows": grid_size_rows,
        "cols": grid_size_cols,
        "grid_positions": grid_positions,
        "representative_keys": keys,
    }


def create_grid_image(results_dir: str, grid_positions: dict, rows: int, cols: int) -> str:
    """Create a composite image with all representative images laid out on the grid."""
    images_dir = os.path.join(results_dir, "artifacts", "images")

    if not os.path.exists(images_dir):
        print(f"Images directory not found at {images_dir}")
        return None

    # Check for sample image
    sample_id = next(iter(grid_positions.keys()))
    sample_path = os.path.join(images_dir, f"{sample_id}.jpg")

    if not os.path.exists(sample_path):
        sample_path = os.path.join(images_dir, f"{sample_id}.png")
        if not os.path.exists(sample_path):
            print(f"No images found for representative genomes in {images_dir}")
            return None

    # Get image dimensions
    with Image.open(sample_path) as img:
        img_width, img_height = img.size

    # Create grid
    grid_img = Image.new("RGB", (cols * img_width, rows * img_height), color="white")

    missing_images = 0
    for genome_id, position in grid_positions.items():
        i, j = position["i"], position["j"]

        img_path = os.path.join(images_dir, f"{genome_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{genome_id}.png")

        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height))

                    x = j * img_width
                    y = i * img_height
                    grid_img.paste(img, (x, y))
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                missing_images += 1
        else:
            missing_images += 1

    if missing_images > 0:
        print(f"Warning: {missing_images} images were missing or could not be processed")

    output_path = os.path.join(results_dir, "grid_visualization.jpg")
    grid_img.save(output_path, quality=95)
    print(f"Grid visualization saved to {output_path}")

    return output_path


# ============================================================================
# Main CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evolution experiment results"
    )
    parser.add_argument(
        "mode",
        choices=["novelty", "umap-generations", "umap-grid"],
        help="Visualization mode",
    )
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in results_dir)",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    # UMAP parameters
    parser.add_argument(
        "--neighbors", type=int, default=80, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "pca"],
        help="Dimensionality reduction method",
    )

    # UMAP grid parameters
    parser.add_argument(
        "--num-representatives",
        "-n",
        type=int,
        default=64,
        help="Number of representative samples for grid",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=1.0,
        help="Grid aspect ratio (width/height)",
    )

    # Display parameters
    parser.add_argument(
        "--label-interval",
        type=int,
        default=5,
        help="Show every Nth label on colorbar",
    )

    args = parser.parse_args()

    if args.mode == "novelty":
        metrics_list = load_novelty_metrics(args.results_dir)
        if not metrics_list:
            print(f"No novelty metrics found in {args.results_dir}")
            return

        output_path = args.output or os.path.join(args.results_dir, "novelty_plot.png")
        plot_novelty_metrics(metrics_list, output_path, show=args.show)
        plot_strategy_comparison(metrics_list, output_path, show=args.show)

    elif args.mode == "umap-generations":
        generations = load_population_data(args.results_dir)
        if not generations:
            print(f"No population data found in {args.results_dir}")
            return

        latents, genome_to_gen = load_latents(args.results_dir, generations)
        if not latents:
            print(f"No latent vectors found in {args.results_dir}")
            return

        coordinates = reduce_dimensionality(
            latents,
            method=args.method,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
        )

        output_path = args.output or os.path.join(
            args.results_dir, f"latents_by_generation_{args.method}.png"
        )
        plot_latents_by_generation(
            coordinates, genome_to_gen, output_path, args.label_interval, show=args.show
        )

    elif args.mode == "umap-grid":
        latents, _ = load_latents(args.results_dir, skip_first_generation=True)
        if not latents:
            print(f"No latent vectors found in {args.results_dir}")
            return

        representative_latents = find_representative_latents(
            latents, args.num_representatives
        )
        result = create_grid_umap(
            representative_latents, args.neighbors, args.min_dist, args.aspect_ratio
        )

        output_path = args.output or os.path.join(
            args.results_dir, "grid_positions.json"
        )
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Grid positions saved to {output_path}")
        print(f"Grid dimensions: {result['rows']}x{result['cols']}")
        print(f"Total positions: {len(result['grid_positions'])}")

        create_grid_image(
            args.results_dir, result["grid_positions"], result["rows"], result["cols"]
        )


if __name__ == "__main__":
    main()

