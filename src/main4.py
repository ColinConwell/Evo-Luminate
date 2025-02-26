import os
import json
import uuid
import time
import logging
import numpy as np
from enum import Enum
import torch
from typing import List, Dict, Any, Optional, Union

from population import Population
from evolution4 import generate_evolve_ideas
from artifacts import Artifact, ShaderArtifact

# Global LLM client
from models import llm_client, text_embedder, image_embedder


def compute_embeddings(artifacts: List[Artifact]) -> torch.Tensor:
    """Compute embeddings for a list of artifacts"""
    if not artifacts:
        return torch.tensor([])

    embeddings = []

    for artifact in artifacts:
        if artifact.embedding is not None:
            embeddings.append(artifact.embedding)
            continue
        # try:
        if artifact.phenome:
            embedding = image_embedder.embedImage(artifact.phenome)
        else:
            embedding = text_embedder.encodeText(artifact.genome)
        artifact.embedding = embedding
        embeddings.append(embedding)
    # Stack embeddings into a single tensor
    return torch.stack(embeddings)


def run_evolution_experiment(
    initial_prompt: str,
    output_dir: str,
    config: Dict[str, Any] = None,
) -> Population:
    """
    Run a complete evolution experiment

    Parameters:
    - initial_prompt: Prompt to generate initial population
    - output_dir: Directory to save results
    - config: Configuration parameters

    Returns:
    - Final population
    """
    # Default configuration
    default_config = {
        "initial_population_size": 10,
        "population_size": 20,
        "num_generations": 10,
        "children_per_generation": 10,
        "mutation_probability": 0.7,
        "crossover_probability": 0.3,
        "k_neighbors": 3,
        "random_seed": 42,
    }

    # Update with user config
    if config:
        default_config.update(config)
    config = default_config

    # Set random seed
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    # log_path = os.path.join(output_dir, "experiment.log")
    # logging.basicConfig(
    #     filename=log_path,
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     filemode="w",
    # )

    # Log experiment start
    logging.info("Starting evolution experiment")
    logging.info("Initial prompt: %s", initial_prompt)

    # Create initial population directory
    initial_dir = os.path.join(output_dir, "initial")
    os.makedirs(initial_dir, exist_ok=True)

    # Create initial population
    logging.info("Generating initial population...")
    initial_artifacts = []
    for i in range(config["initial_population_size"]):
        try:
            # Create and render in one step
            artifact = ShaderArtifact.create_random(
                prompt=initial_prompt, output_dir=initial_dir
            )
            initial_artifacts.append(artifact)
            logging.info(
                f"Created initial artifact {i+1}/{config['initial_population_size']}"
            )
        except Exception as e:
            logging.error(f"Failed to create artifact {i}: {e}")

    # Initialize population
    population = Population()
    population.add_all(initial_artifacts)

    # Compute embeddings
    embeddings = compute_embeddings(population.get_all())

    # Run evolution
    for generation in range(config["num_generations"]):
        logging.info("Generation %d of %d", generation + 1, config["num_generations"])

        # Create generation directory
        gen_dir = os.path.join(output_dir, f"generation_{generation+1:03d}")
        os.makedirs(gen_dir, exist_ok=True)

        # Generate new artifacts
        new_artifacts = []

        while len(new_artifacts) < config["children_per_generation"]:
            strategy = ""
            # Decide if this is variation (1 parent) or combination (2 parents)
            if np.random.random() < config["mutation_probability"]:
                # Variation (mutation-like): select one parent
                parent = population.get_random(1)[0]
                other_artifacts = []  # Empty - just evolve the parent itself

            else:
                # Combination (crossover-like): select multiple parents
                parents = population.get_random(2)
                parent = parents[0]
                other_artifacts = [parents[1]]

            # Generate evolution idea
            evolution_ideas = generate_evolve_ideas(
                [parent] + other_artifacts, strategy=strategy, count=2
            )
            print("evolution_ideas", evolution_ideas)
            for idea in evolution_ideas:
                # Apply the evolution idea
                child = parent.crossover_with(
                    other_artifacts=other_artifacts,
                    crossover_idea=idea,
                    output_dir=gen_dir,
                )
                new_artifacts.append(child)

        # Add new artifacts to population
        population.add_all(new_artifacts)

        # Compute embeddings for all artifacts
        all_artifacts = population.get_all()
        embeddings = compute_embeddings(all_artifacts)

        print("embeddings", embeddings.shape)

        # Select diverse subset based on novelty
        novelty_indices, avg_distances = population.select_by_novelty(
            embeddings, k_neighbors=config["k_neighbors"], return_distances=True
        )
        keep_indices = novelty_indices[: config["population_size"]]

        # Save novelty metrics for this generation
        novelty_metrics = {
            "generation": generation + 1,
            "avg_distance_to_neighbors": avg_distances.tolist(),
            "mean_novelty": float(avg_distances.mean().item()),
            "max_novelty": float(avg_distances.max().item()),
            "min_novelty": float(avg_distances.min().item()),
        }
        metrics_path = os.path.join(gen_dir, "novelty_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(novelty_metrics, f, indent=2)

        logging.info(
            f"Generation {generation+1} mean novelty: {novelty_metrics['mean_novelty']:.4f}"
        )

        # Create new population with selected artifacts
        new_population = Population()
        for idx in keep_indices:
            new_population.add(all_artifacts[idx])

        # Update population
        population = new_population

        # Save generation results
        population.save(gen_dir)

        logging.info(
            "Generation %d complete. Population size: %d",
            generation + 1,
            len(population.get_all()),
        )

    # Save final population
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    population.save(final_dir)

    logging.info("Experiment complete. Results saved to %s", output_dir)

    return population


if __name__ == "__main__":
    run_evolution_experiment(
        "Create a colorful abstract shader with flowing patterns",
        "results",
        config={
            "initial_population_size": 10,
            "population_size": 10,
            "num_generations": 16,
            "children_per_generation": 10,
        },
    )
