import os
import json
import uuid
import time
import logging
import numpy as np
from enum import Enum
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from artifacts import Artifact, ShaderArtifact

from IlluminationArchive import IlluminationArchive
from models import llm_client, text_embedder, image_embedder

defaultModel = "gpt-4o-mini"


def construct_evolution_prompt(artifacts, user_prompt, creative_strategy=None):
    prompt = f"I'm exploring diverse possibilities for {user_prompt}s.\n\n"
    prompt += "Create something significantly different from my current examples:\n\n"

    for i, artifact in enumerate(artifacts):
        prompt += f"Example {i+1}:\n"
        prompt += f"{artifact.genome}\n\n"

    # if creative_strategy:
    #     prompt += f"{creative_strategy}\n\n"
    # else:
    #     prompt += "create a new implementation that is significantly different from these examples.\n\n"

    # prompt += f"Generate a complete, novel {domain_prompt.split()[0]} that explores an area not represented in the examples above."

    return prompt


def run_llm_illumination(
    user_prompt,
    generations=50,
    samples_per_gen=10,
    initial_population_size=20,
    creative_strategies=None,
):
    # Initialize archive
    archive = IlluminationArchive(distance_threshold=0.3)

    # Create initial population
    print(f"Generating initial population of {initial_population_size} artifacts...")
    initial_artifacts = []
    for i in range(initial_population_size):
        artifact = ShaderArtifact.create_from_prompt(user_prompt, output_dir="")
        artifact.compute_embedding()
        initial_artifacts.append(artifact)

    archive.add_generation(initial_artifacts)

    # Track metrics across generations
    metrics = {
        "avg_novelty": [],
        "max_novelty": [],
        "cluster_count": [],
        "unique_artifacts": [],
    }

    for gen in range(1, generations + 1):
        print(f"\nGeneration {gen}/{generations}")
        sampling_n = 4
        sampling_strategy = "random"
        examples = archive.get_samples(n=sampling_n, strategy=sampling_strategy)

        # Select creative strategy if provided
        creative_strategy = None
        # if creative_strategies:
        #     creative_strategy = creative_strategies[gen % len(creative_strategies)]
        #     print(f"Using creative strategy: {creative_strategy[:50]}...")

        # Generate new artifacts
        offspring = []
        for i in range(samples_per_gen):
            print(f"  Generating artifact {i+1}/{samples_per_gen}...")

            # Construct evolution prompt
            prompt = construct_evolution_prompt(
                examples, user_prompt, creative_strategy
            )

            artifact = ShaderArtifact.create_from_prompt(prompt, output_dir="")
            artifact.compute_embedding()

            offspring.append(artifact)

        # Add to archive
        archive.add_generation(offspring)

        # Log statistics
        novelty_scores = [a.novelty_score for a in offspring]
        metrics["avg_novelty"].append(sum(novelty_scores) / len(novelty_scores))
        metrics["max_novelty"].append(max(novelty_scores))
        metrics["cluster_count"].append(len(archive.cluster_members))
        metrics["unique_artifacts"].append(len(archive.artifacts))

        print(
            f"  Stats: Avg Novelty={metrics['avg_novelty'][-1]:.4f}, "
            f"Max Novelty={metrics['max_novelty'][-1]:.4f}, "
            f"Clusters={metrics['cluster_count'][-1]}, "
            f"Total Artifacts={metrics['unique_artifacts'][-1]}"
        )

        # Periodic detailed analysis
        if gen % 10 == 0 or gen == generations:
            analyze_population(archive, llm)

    return archive, metrics
