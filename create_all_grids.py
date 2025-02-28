#!/usr/bin/env python3
"""
Batch Processing Script for Project Embeddings

This script:
1. Finds all project folders in the results directory
2. For each project, reads population_data.jsonl
3. Splits data into "initial" (first line) and "optimized" (remaining lines)
4. Runs representative_grid_umap.py for each group
5. Saves output JSON files in the project root

Usage:
    python batch_process_projects.py --results-dir /path/to/results --num-representatives 50
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
import tempfile
import shutil


def find_project_folders(results_dir):
    """Find all project folders that contain population_data.jsonl and config.json"""
    project_folders = []

    for path in Path(results_dir).iterdir():
        if path.is_dir():
            pop_data_path = path / "population_data.jsonl"
            config_path = path / "config.json"

            if pop_data_path.exists() and config_path.exists():
                project_folders.append(path)

    print(f"Found {len(project_folders)} project folders in {results_dir}")
    return project_folders


def process_population_data(pop_data_path):
    """
    Read population_data.jsonl and split into initial and optimized genome_ids

    Returns:
        Tuple of (initial_genome_ids, optimized_genome_ids)
    """
    initial_genome_ids = []
    optimized_genome_ids = []

    # Read the jsonl file line by line
    with open(pop_data_path, "r") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if "genome_ids" in item:
                    if i == 0:
                        # First line is "initial"
                        initial_genome_ids.extend(item["genome_ids"])
                    else:
                        # All other lines are "optimized"
                        optimized_genome_ids.extend(item["genome_ids"])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue

    print(f"Extracted {len(initial_genome_ids)} initial genome IDs")
    print(f"Extracted {len(optimized_genome_ids)} optimized genome IDs")

    return initial_genome_ids, optimized_genome_ids


def create_temp_embedding_dir(project_folder, genome_ids):
    """
    Create a temporary directory with symbolic links to the requested embeddings

    Args:
        project_folder: Path to the project folder
        genome_ids: List of genome IDs to include

    Returns:
        Path to the temporary directory
    """
    # Path to the embeddings folder
    embeddings_dir = project_folder / "artifacts" / "embeddings"

    if not embeddings_dir.exists():
        print(f"Warning: Embeddings directory not found at {embeddings_dir}")
        return None

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create symbolic links for each requested genome ID
    links_created = 0
    for genome_id in genome_ids:
        embedding_file = embeddings_dir / f"{genome_id}.npy"
        if embedding_file.exists():
            # Create a symbolic link in the temp directory
            os.symlink(str(embedding_file), str(Path(temp_dir) / f"{genome_id}.npy"))
            links_created += 1

    print(f"Created temporary directory with {links_created} embedding links")
    return temp_dir


def run_representative_grid_umap(input_dir, output_json, num_representatives):
    """
    Run the representative_grid_umap.py script

    Args:
        input_dir: Directory containing embedding .npy files
        output_json: Path to save the output JSON
        num_representatives: Number of representatives to select

    Returns:
        True if successful, False otherwise
    """
    # Construct the command - convert all paths to strings
    cmd = [
        "python",
        "representative_grid_umap.py",
        "--input-dir",
        str(input_dir),
        "--output",
        str(output_json),
        "--num-representatives",
        str(num_representatives),
    ]

    # Run the command
    try:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running representative_grid_umap.py: {e}")
        return False


def process_project(project_folder, num_representatives):
    """
    Process a single project folder

    Args:
        project_folder: Path to the project folder
        num_representatives: Number of representatives to select
    """
    print(f"\n--- Processing project: {project_folder.name} ---")

    # Path to population data
    pop_data_path = project_folder / "population_data.jsonl"

    # Get initial and optimized genome IDs
    initial_ids, optimized_ids = process_population_data(pop_data_path)

    # Define output JSON paths
    initial_json = project_folder / "grid_initial.json"
    optimized_json = project_folder / "grid_optimized.json"

    # Process initial embeddings
    if initial_ids:
        print("\nProcessing initial embeddings...")
        temp_dir = create_temp_embedding_dir(project_folder, initial_ids)
        if temp_dir:
            success = run_representative_grid_umap(
                temp_dir, initial_json, min(num_representatives, len(initial_ids))
            )
            shutil.rmtree(temp_dir)  # Clean up temp directory
            if success:
                print(f"Initial grid saved to {initial_json}")

    # Process optimized embeddings
    if optimized_ids:
        print("\nProcessing optimized embeddings...")
        temp_dir = create_temp_embedding_dir(project_folder, optimized_ids)
        if temp_dir:
            success = run_representative_grid_umap(
                temp_dir, optimized_json, min(num_representatives, len(optimized_ids))
            )
            shutil.rmtree(temp_dir)  # Clean up temp directory
            if success:
                print(f"Optimized grid saved to {optimized_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process project embeddings into representative grid UMAPs"
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        required=True,
        help="Directory containing project folders",
    )
    parser.add_argument(
        "--num-representatives",
        "-n",
        type=int,
        default=50,
        help="Number of representative latent vectors to select",
    )
    args = parser.parse_args()

    # Find all project folders
    project_folders = find_project_folders(args.results_dir)

    # Process each project
    for project_folder in project_folders:
        print(f"Processing project: {project_folder.name}")
        # process_project(project_folder, args.num_representatives)

    print("\nAll projects processed successfully!")


if __name__ == "__main__":
    main()
