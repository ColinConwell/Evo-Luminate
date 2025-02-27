import os
import json
import random
import uuid
import time
import logging
import numpy as np
from enum import Enum
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from population import Population
from artifacts import Artifact, ShaderArtifact

from models import llm_client


def artifacts_to_string(artifacts):
    s = ""
    for i, artifact in enumerate(artifacts):
        s += f"Example {i+1}:\n"
        s += f"{artifact.genome}\n\n"
    return s


def construct_evolution_prompt(artifacts, user_prompt, summary, creative_strategy=None):
    prompt = f"I'm exploring diverse possibilities for {user_prompt}s.\n\n"
    prompt += f"Summary of the current population: {summary}\n\n"
    prompt += "Make this shader significantly more interesting and less like what is done before:\n\n"
    prompt += artifacts_to_string(artifacts)

    # if creative_strategy:
    #     prompt += f"{creative_strategy}\n\n"
    # else:
    #     prompt += "create a new implementation that is significantly different from these examples.\n\n"

    # prompt += f"Generate a complete, novel {domain_prompt.split()[0]} that explores an area not represented in the examples above."

    return prompt


def get_embeddings(artifacts: List[Artifact]) -> torch.Tensor:
    """Compute embeddings for a list of artifacts"""
    embeddings = []
    for artifact in artifacts:
        embedding = artifact.compute_embedding()
        embeddings.append(embedding)
    return torch.stack(embeddings)


def complete_prompt(prompt: str, model: str = "openai:gpt-4o-mini") -> str:
    return (
        llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        .choices[0]
        .message.content.strip()
    )


def save_novelty_metrics(population: Population, output_dir: str, k_neighbors: int = 3):
    _, avg_distances = population.select_by_novelty(
        get_embeddings(population.get_all()),
        k_neighbors=k_neighbors,
        return_distances=True,
    )
    novelty_metrics = {
        "generation": 0,
        "avg_distance_to_neighbors": avg_distances.tolist(),
        "mean_novelty": float(avg_distances.mean().item()),
        "mean_genome_length": np.mean(
            [len(artifact.genome) for artifact in population.get_all()]
        ),
    }

    metrics_path = os.path.join(output_dir, "novelty_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(novelty_metrics, f, indent=2)


def run_evolution_experiment(
    output_dir: str, config: Dict[str, Any] = None
) -> Population:
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

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
    logging.info("Output directory: %s", output_dir)

    # Create initial population directory
    initial_dir = os.path.join(output_dir, "initial")
    os.makedirs(initial_dir, exist_ok=True)

    # Create initial population
    logging.info("Generating initial population...")
    initial_artifacts = []
    while len(initial_artifacts) < config["initial_population_size"]:
        try:
            artifact = ShaderArtifact.create_from_prompt(
                prompt=config["prompt"], output_dir=initial_dir
            )
            initial_artifacts.append(artifact)
        except Exception as e:
            logging.error(f"Failed to create artifact: {e}")

    population = Population()
    population.add_all(initial_artifacts)

    save_novelty_metrics(population, initial_dir, k_neighbors=config["k_neighbors"])

    # Run evolution
    for generation in range(config["num_generations"]):
        logging.info("Generation %d of %d", generation + 1, config["num_generations"])

        # Create generation directory
        gen_dir = os.path.join(output_dir, f"generation_{generation+1:03d}")
        os.makedirs(gen_dir, exist_ok=True)

        # Generate new artifacts
        new_artifacts = []

        summary = complete_prompt(
            f"Concisely summarize the population of artifacts: {artifacts_to_string(population.get_all())}",
            model="openai:gpt-4o-mini",
        )
        print(summary)

        to_evolve = random.sample(
            population.get_all(), config["children_per_generation"]
        )
        for artifact in to_evolve:
            try:
                evolution_prompt = construct_evolution_prompt(
                    artifacts=[artifact],
                    user_prompt=config["prompt"],
                    summary=summary,
                    creative_strategy=None,
                )
                new_artifact = ShaderArtifact.create_from_prompt(
                    prompt=evolution_prompt, output_dir=gen_dir
                )
                new_artifacts.append(new_artifact)
            except Exception as e:
                logging.error(f"Failed to create artifact: {e}")
                continue

        # Add new artifacts to population
        population.add_all(new_artifacts)

        # Compute embeddings for all artifacts
        all_artifacts = population.get_all()
        embeddings = get_embeddings(all_artifacts)

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
            "mean_genome_length": np.mean(
                [len(artifact.genome) for artifact in all_artifacts]
            ),
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
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    run_evolution_experiment(
        output_dir=output_dir,
        config={
            "prompt": "Create an interesting shader",
            "initial_population_size": 6,
            "population_size": 12,
            "children_per_generation": 6,
            "num_generations": 16,
            "k_neighbors": 3,
        },
    )
