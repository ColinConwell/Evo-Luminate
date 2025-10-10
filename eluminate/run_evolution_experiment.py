import os
import json
import random
import logging
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .population import Population
from .artifacts.load_artifacts import get_artifact_class
from .creative_strategies_manager import CreativityStrategyManager
from .utils import load_image_path_base64

manager = CreativityStrategyManager("eluminate/creativity_strategies.json")


def artifacts_to_string(artifacts):
    s = ""
    for i, artifact in enumerate(artifacts):
        s += f"Example {i+1}:\n"
        s += f"{artifact.genome}\n\n"
    return s


def construct_evolution_prompt(
    artifacts, user_prompt, summary, evolution_mode="variation", creative_strategy=None
):
    if summary is None and evolution_mode == "creation":
        raise ValueError("Summary is required for evolution_mode=creation")

    prompt = ""
    if len(user_prompt):
        prompt = f"I'm exploring diverse possibilities for {user_prompt}\n\n"

    if summary:
        prompt += f"Summary of the current population:\n{summary}\n"

    if evolution_mode == "variation":
        prompt += f"Make this {artifacts[0].name} significantly different from what is done before:\n\n"
    elif evolution_mode == "creation":
        prompt += f"Create a new {artifacts[0].name} that is significantly different from what is done before:\n\n"

    if creative_strategy:
        prompt += f"\n{creative_strategy}\n"

    if evolution_mode == "variation":
        prompt += "Current " + (artifacts[0].name) + ":\n\n"
        prompt += artifacts[0].genome + "\n\n"

    return prompt


def construct_crossover_prompt(artifacts, user_prompt, summary, creative_strategy=None):
    prompt = ""
    if len(user_prompt):
        prompt = f"I'm exploring diverse possibilities for {user_prompt}s.\n\n"
    prompt += f"Combine these {artifacts[0].name}s to create a new {artifacts[0].name}"
    if summary:
        prompt += f"Summary of the current population: {summary}\n\n"
    for i, artifact in enumerate(artifacts):
        prompt += f"Example {i+1}:\n"
        prompt += f"{artifact.genome}\n\n"
    if creative_strategy:
        prompt += f"\n{creative_strategy}\n"
    return prompt


def construct_repair_prompt(
    artifact, user_prompt, summary, evolution_mode, creative_strategy
):
    prompt = f'Improve this {artifact.name} to be more like "{user_prompt}"\n'

    prompt += (
        f"once it satisfies the goals make it novel from what has been done before:\n"
    )
    if summary:
        prompt += f"Summary of the current population: {summary}\n\n"

    prompt += f"Current {artifact.name}:\n\n"
    prompt += f"{artifact.genome}\n\n"
    prompt += f"A render of the current SDF is attached. Examine it for any issues and fix them"

    # if creative_strategy:
    #     prompt += f"\n{creative_strategy}\n"

    return prompt


def get_embeddings(artifacts: List) -> torch.Tensor:
    """Compute embeddings for a list of artifacts and return device-resident float32 tensor"""
    from .utils import get_device

    device = get_device()
    embeddings = []
    for artifact in artifacts:
        embedding = artifact.compute_embedding()
        # Ensure tensor and move to device as float32
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        embeddings.append(embedding.to(device=device, dtype=torch.float32))
    return torch.stack(embeddings)


def complete_prompt(prompt: str, model: str = "openai:gpt-4o-mini") -> str:
    # Lazy import to avoid heavy dependencies during fast test imports
    from .models import llm_client
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


def generate_population_summary(
    artifacts, artifact_class_name, model="openai:gpt-4o-mini"
):
    """Generate a summary of the current population."""
    summary_prompt = f"Create a concise overview of the collection of {artifact_class_name}s including specific goals and methods: {artifacts_to_string(artifacts)}"
    return complete_prompt(summary_prompt, model=model)


def save_novelty_metrics(
    population: Population, output_dir: str, generation: int = 0, k_neighbors: int = 3
):
    """Calculate and save novelty metrics for the current population."""
    _, avg_distances = population.select_by_novelty(
        get_embeddings(population.get_all()),
        k_neighbors=k_neighbors,
        return_distances=True,
    )

    # Group artifacts by creative strategy
    strategy_to_distances = {}

    all_artifacts = population.get_all()
    for i, artifact in enumerate(all_artifacts):
        strategy_name = artifact.metadata.get("creative_strategy_name", "None")
        if strategy_name not in strategy_to_distances:
            strategy_to_distances[strategy_name] = []
        strategy_to_distances[strategy_name].append(avg_distances[i].item())

    # Calculate average novelty per strategy
    strategy_metrics = {}
    for strategy, distances in strategy_to_distances.items():
        if distances:
            strategy_metrics[strategy] = {
                "count": len(distances),
                "avg_novelty": np.mean(distances),
                "std_novelty": np.std(distances),
            }

    novelty_metrics = {
        "generation": generation,
        "timestamp": datetime.now().isoformat(),
        "avg_distance_to_neighbors": avg_distances.tolist(),
        "mean_novelty": float(avg_distances.mean().item()),
        "mean_genome_length": np.mean(
            [len(artifact.genome) for artifact in population.get_all()]
        ),
        "strategy_metrics": strategy_metrics,
    }

    metrics_path = os.path.join(output_dir, "novelty_metrics.jsonl")
    with open(metrics_path, "a") as f:
        f.write(json.dumps(novelty_metrics) + "\n")

    return float(avg_distances.mean().item())


def create_initial_population(config, artifacts_dir, ArtifactClass):
    """Create the initial population of artifacts."""
    logging.info("Generating initial population...")
    initial_artifacts = []

    def create_artifact():
        try:
            return ArtifactClass.create_from_prompt(
                prompt=config["prompt"],
                output_dir=artifacts_dir,
                reasoning_effort=config["reasoning_effort"],
                image_url=None,
            )
        except Exception as e:
            logging.error(f"Failed to create artifact: {e}\n{traceback.format_exc()}")
            return None

    with ThreadPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
        future_to_artifact = {
            executor.submit(create_artifact): i
            for i in range(config["initial_population_size"])
        }

        for future in as_completed(future_to_artifact):
            artifact = future.result()
            if artifact:
                initial_artifacts.append(artifact)

    # If there are errors, fill in the rest with random parents
    while len(initial_artifacts) < config["initial_population_size"]:
        if not initial_artifacts:  # Handle case where no artifacts were created
            logging.error("Failed to create any initial artifacts")
            raise RuntimeError("Could not create initial population")

        child = create_artifact()
        if child:
            initial_artifacts.append(child)

    return initial_artifacts


def load_artifact_image(artifact) -> Optional[str]:
    image_url = (
        random.choice(artifact.phenome)
        if isinstance(artifact.phenome, list)
        else artifact.phenome
    )
    return load_image_path_base64(image_url)


def evolve_population(population, config, artifacts_dir, ArtifactClass, summary=None):
    """Generate a new set of evolved artifacts from the current population."""
    new_artifacts = []
    to_evolve = random.sample(population.get_all(), config["children_per_generation"])

    def evolve_artifact(artifact):
        try:
            do_crossover = random.random() < config.get("crossover_rate", 0.0)
            creative_strategy = (
                manager.get_random_strategy()
                if config["use_creative_strategies"]
                else None
            )
            creative_strategy_prompt = (
                manager.to_prompt(creative_strategy) if creative_strategy else None
            )
            if do_crossover:
                mate = random.choice(population.get_all())
                evolution_prompt = construct_crossover_prompt(
                    artifacts=[artifact, mate],
                    user_prompt=config["prompt"],
                    summary=summary,
                    creative_strategy=creative_strategy_prompt,
                )
            else:
                evolution_prompt = construct_evolution_prompt(
                    artifacts=[artifact],
                    user_prompt=config["prompt"],
                    summary=summary,
                    evolution_mode=config["evolution_mode"],
                    creative_strategy=creative_strategy_prompt,
                )
                # evolution_prompt = construct_repair_prompt(
                #     artifact=artifact,
                #     user_prompt=config["prompt"],
                #     summary=summary,
                #     evolution_mode=config["evolution_mode"],
                #     creative_strategy=creative_strategy_prompt,
                # )
            new_artifact = ArtifactClass.create_from_prompt(
                prompt=evolution_prompt,
                output_dir=artifacts_dir,
                reasoning_effort=config["reasoning_effort"],
                image_url=None,
                # image_url=(load_artifact_image(artifact)),
                # if config["use_images"] else None
            )
            new_artifact.metadata["creative_strategy_name"] = (
                creative_strategy["name"] if creative_strategy else None
            )
            return new_artifact
        except Exception as e:
            logging.error(f"Failed to create artifact: {e}\n{traceback.format_exc()}")
            return None

    with ThreadPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
        future_to_artifact = {
            executor.submit(evolve_artifact, artifact): artifact
            for artifact in to_evolve
        }

        for future in as_completed(future_to_artifact):
            new_artifact = future.result()
            if new_artifact:
                new_artifacts.append(new_artifact)

    # If there are errors, fill in the rest
    while len(new_artifacts) < config["children_per_generation"]:
        parent = random.choice(population.get_all())
        child = evolve_artifact(parent)
        if child:
            new_artifacts.append(child)

    return new_artifacts


def select_next_generation(population, config):
    """Select the next generation based on novelty."""
    all_artifacts = population.get_all()
    embeddings = get_embeddings(all_artifacts)

    # Select diverse subset based on novelty
    novelty_indices, _ = population.select_by_novelty(
        embeddings, k_neighbors=config["k_neighbors"], return_distances=True
    )
    keep_indices = novelty_indices[: config["population_size"]]

    # Create new population with selected artifacts
    new_population = Population()
    for idx in keep_indices:
        new_population.add(all_artifacts[idx])

    return new_population


def run_evolution_experiment(
    output_dir: str, config: Dict[str, Any] = None
) -> Population:
    """Run an evolutionary experiment to generate diverse artifacts."""
    # Setup experiment
    os.makedirs(output_dir, exist_ok=True)
    import random as _py_random
    _py_random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    # Get device information for logging
    from .utils import get_device
    device = get_device()
    logging.info(f"Using device: {device}")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Setup logging
    logging.info("Starting evolution experiment")
    logging.info("Output directory: %s", output_dir)

    # Create artifacts directory
    artifacts_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Get the artifact class based on config
    ArtifactClass = get_artifact_class(config)

    # Create initial population
    initial_artifacts = create_initial_population(config, artifacts_dir, ArtifactClass)
    population = Population()
    population.add_all(initial_artifacts)

    # Save initial population data and metrics
    population.save(output_dir, generation=0)
    mean_novelty = save_novelty_metrics(
        population, output_dir, generation=0, k_neighbors=config["k_neighbors"]
    )
    logging.info(f"Initial population mean novelty: {mean_novelty:.4f}")

    # Run evolution
    for generation in range(config["num_generations"]):
        logging.info("Generation %d of %d", generation + 1, config["num_generations"])

        # Generate population summary if enabled
        summary = None
        if config["use_summary"]:
            summary = generate_population_summary(
                population.get_all(), ArtifactClass.name
            )
            print("-" * 80)
            print(summary)
            print("-" * 80)

            summary_path = os.path.join(output_dir, "summaries.jsonl")
            with open(summary_path, "a") as f:
                f.write(
                    json.dumps({"summary": summary, "generation": generation}) + "\n"
                )

        # Evolve population
        new_artifacts = evolve_population(
            population, config, artifacts_dir, ArtifactClass, summary
        )
        population.add_all(new_artifacts)

        # Select next generation
        population = select_next_generation(population, config)

        # Save novelty metrics for this generation
        current_gen = generation + 1
        mean_novelty = save_novelty_metrics(
            population,
            output_dir,
            generation=current_gen,
            k_neighbors=config["k_neighbors"],
        )

        logging.info(f"Generation {current_gen} mean novelty: {mean_novelty:.4f}")

        # Save generation results
        population.save(output_dir, generation=current_gen)
        logging.info(
            "Generation %d complete. Population size: %d",
            current_gen,
            len(population.get_all()),
        )

    logging.info("Experiment complete. Results saved to %s", output_dir)
    return population
