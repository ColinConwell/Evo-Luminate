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
from PIL import Image


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


def create_grid_umap(latents, n_neighbors=15, min_dist=0.1, aspect_ratio=1.0):
    """
    Project latents to 2D using UMAP and assign grid positions

    Args:
        latents: Dictionary of {key: latent_vector}
        n_neighbors: UMAP parameter for local neighborhood size
        min_dist: UMAP parameter for minimum distance between points
        aspect_ratio: Desired width/height ratio of the grid (default: 1.0 for square)

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

    # Determine grid size based on aspect ratio
    n_items = len(keys)

    # Calculate grid dimensions that maintain the aspect ratio
    # width/height = aspect_ratio, and width * height >= n_items
    height = math.sqrt(n_items / aspect_ratio)
    width = height * aspect_ratio

    # Round up to ensure we have enough cells
    cols = math.ceil(width)
    rows = math.ceil(height)

    # Ensure we have enough cells
    while rows * cols < n_items:
        # Increase the smaller dimension to maintain aspect ratio as closely as possible
        if cols / rows < aspect_ratio:
            cols += 1
        else:
            rows += 1

    # Add extra space for organic look
    grid_size_rows = rows + 1
    grid_size_cols = cols + 1

    # Create cost matrix for grid assignment
    cost_matrix = np.zeros((n_items, grid_size_rows * grid_size_cols))

    # Generate all grid positions
    grid_positions_array = np.array(
        [(i, j) for i in range(grid_size_rows) for j in range(grid_size_cols)]
    )

    # Calculate the center of the grid
    center_i = (grid_size_rows - 1) / 2
    center_j = (grid_size_cols - 1) / 2

    # Apply a non-linear scaling to grid positions to make them denser toward the center
    # This creates a more organic, circular layout with fewer holes in the middle
    grid_positions_scaled = np.zeros_like(grid_positions_array, dtype=float)
    for idx, (i, j) in enumerate(grid_positions_array):
        # Calculate distance from center (0 to 1 scale)
        di = (i - center_i) / max(center_i, 1)
        dj = (j - center_j) / max(center_j, 1)

        # Apply non-linear scaling - points further from center get compressed inward
        # This makes the center denser by compressing the outer regions
        scale_factor = (
            0.3  # Controls how much scaling to apply (0.0 = no scaling, 1.0 = maximum)
        )

        # Apply a power function that compresses the outer regions
        # For values < 1, power > 1 compresses toward zero
        di_scaled = di * (abs(di) ** scale_factor) / abs(di) if di != 0 else 0
        dj_scaled = dj * (abs(dj) ** scale_factor) / abs(dj) if dj != 0 else 0

        # Convert back to grid coordinates
        grid_positions_scaled[idx, 0] = center_i + di_scaled * center_i
        grid_positions_scaled[idx, 1] = center_j + dj_scaled * center_j

    # Normalize grid positions to [0,1] range (same as embedding)
    grid_positions_normalized = np.zeros_like(grid_positions_scaled, dtype=float)
    if grid_size_rows > 1:  # Avoid division by zero
        grid_positions_normalized[:, 0] = grid_positions_scaled[:, 0] / (
            grid_size_rows - 1
        )
    if grid_size_cols > 1:  # Avoid division by zero
        grid_positions_normalized[:, 1] = grid_positions_scaled[:, 1] / (
            grid_size_cols - 1
        )

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
        "rows": grid_size_rows,
        "cols": grid_size_cols,
        "grid_positions": grid_positions,
        "representative_keys": keys,
    }

    return result


def create_grid_image(results_dir, grid_positions, rows, cols):
    """
    Create a composite image with all representative images laid out on the grid

    Args:
        results_dir: Directory containing the results
        grid_positions: Dictionary mapping genome IDs to grid positions
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Path to the saved composite image
    """
    images_dir = os.path.join(results_dir, "artifacts", "images")

    if not os.path.exists(images_dir):
        print(f"Images directory not found at {images_dir}")
        return None

    # Check if at least one image exists
    sample_id = next(iter(grid_positions.keys()))
    sample_path = os.path.join(images_dir, f"{sample_id}.jpg")

    if not os.path.exists(sample_path):
        # Try PNG if JPG doesn't exist
        sample_path = os.path.join(images_dir, f"{sample_id}.png")
        if not os.path.exists(sample_path):
            print(f"No images found for representative genomes in {images_dir}")
            return None

    # Get image dimensions from the first image
    with Image.open(sample_path) as img:
        img_width, img_height = img.size

    # Create a blank canvas for the grid
    grid_img = Image.new("RGB", (cols * img_width, rows * img_height), color="white")

    # Place each image in its grid position
    missing_images = 0
    for genome_id, position in grid_positions.items():
        i, j = position["i"], position["j"]

        # Try JPG first, then PNG
        img_path = os.path.join(images_dir, f"{genome_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{genome_id}.png")

        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    # Resize if necessary to ensure consistent grid
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height))

                    # Calculate position in the grid
                    x = j * img_width
                    y = i * img_height

                    # Paste the image onto the grid
                    grid_img.paste(img, (x, y))
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                missing_images += 1
        else:
            missing_images += 1

    if missing_images > 0:
        print(
            f"Warning: {missing_images} images were missing or could not be processed"
        )

    # Save the composite grid image
    output_path = os.path.join(results_dir, "grid_visualization.jpg")
    grid_img.save(output_path, quality=95)
    print(f"Grid visualization saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Select representative latent vectors and create grid-based UMAP visualization"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing output directory",
    )
    parser.add_argument(
        "--num-representatives",
        "-n",
        type=int,
        default=64,
        help="Number of representative latent vectors to select",
    )
    parser.add_argument(
        "--neighbors", type=int, default=80, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=1.0,
        help="Desired width/height ratio of the grid (default: 1.0 for square)",
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
    result = create_grid_umap(
        representative_latents, args.neighbors, args.min_dist, args.aspect_ratio
    )

    output_path = os.path.join(args.results_dir, "grid_positions.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Grid positions saved to {output_path}")
    print(f"Grid dimensions: {result['rows']}x{result['cols']}")
    print(f"Total positions: {len(result['grid_positions'])}")

    # Create and save the grid visualization image
    create_grid_image(
        args.results_dir, result["grid_positions"], result["rows"], result["cols"]
    )


if __name__ == "__main__":
    main()
