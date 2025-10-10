import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

from eluminate.run_evolution_experiment import run_evolution_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_experiment_name(config: Dict[str, Any]) -> str:
    """Create a descriptive name for the experiment based on its configuration."""
    domain = config[
        "artifact_class"
    ]  # "shader" if config["artifact_class"] == "ShaderArtifact" else "website"
    strat = "strat" if config["use_creative_strategies"] else "no-strat"
    mode = config["evolution_mode"]
    reasoning = config["reasoning_effort"]
    summary = "summ" if config.get("use_summary", True) else "no-summ"
    crossover = "_crossover" if config.get("crossover_rate", 0.0) > 0.0 else ""
    return f"{domain}_{strat}_{mode}_{reasoning}_{summary}{crossover}"


def run_from_config(study_dir: str, config: Dict[str, Any]):
    exp_name = create_experiment_name(config)
    seed_suffix = f"_seed{config['random_seed']}"
    exp_dir = os.path.join(study_dir, exp_name + seed_suffix)
    if not os.path.exists(exp_dir):
        logging.info(
            f"Starting experiment: {exp_name} with seed {config['random_seed']}"
        )

        # Run the experiment
        run_evolution_experiment(output_dir=exp_dir, config=config)

        logging.info(
            f"Completed experiment: {exp_name} with seed {config['random_seed']}"
        )


def run_ablation_study(base_output_dir: str, random_seeds: list = [42, 43, 44]):
    """Run the full ablation study with multiple seeds for statistical significance."""

    # Create timestamp for this batch of experiments
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = os.path.join(base_output_dir, f"ablation_study")
    os.makedirs(study_dir, exist_ok=True)

    # Save study metadata
    study_metadata = {
        # "timestamp": timestamp,
        "random_seeds": random_seeds,
        "description": "Ablation study testing creative strategies, reasoning effort, summaries, and evolution mode",
    }
    with open(os.path.join(study_dir, "study_metadata.json"), "w") as f:
        json.dump(study_metadata, f, indent=2)

    # Log the start of the ablation study
    logging.info(f"Starting ablation study in {study_dir}")
    logging.info(
        f"Using low reasoning effort for additional experiments to reduce costs"
    )

    # Define the domains to test
    domains = [
        {"artifact_class": "shader", "prompt": "Create an interesting shader"},
        # {
        #     "artifact_class": "GameIdeaArtifact",
        #     "prompt": "a creative variation of the game snake",
        # },
        {
            "artifact_class": "website",
            "prompt": "a creative time display. Only show a 512px canvas and any instructions should appear on hover.",
        },
    ]

    # Common configuration parameters
    common_config = {
        "initial_population_size": 20,
        "population_size": 20,
        "children_per_generation": 10,
        "num_generations": 30,
        "k_neighbors": 3,
        "max_workers": 10,
        "use_summary": True,
        "evolution_mode": "variation",
        "use_creative_strategies": True,
        "reasoning_effort": "low",
        "crossover_rate": 0.0,
    }

    # Run the core 2x2 experiments
    for seed in random_seeds:
        for domain_config in domains:
            for reasoning_effort in ["low", "medium"]:
                for use_creative_strategies in [True, False]:
                    # Create full configuration
                    full_config = {
                        **common_config,
                        **domain_config,
                        "use_creative_strategies": use_creative_strategies,
                        "reasoning_effort": reasoning_effort,
                        "random_seed": seed,
                    }

                    run_from_config(study_dir, full_config)

    for seed in random_seeds:
        for domain_config in domains:
            for use_summary in [False]:
                full_config = {
                    **common_config,
                    **domain_config,
                    "use_summary": use_summary,
                    "random_seed": seed,
                }
                run_from_config(study_dir, full_config)

    for seed in random_seeds:
        for domain_config in domains:
            for evolution_mode in ["creation"]:
                full_config = {
                    **common_config,
                    **domain_config,
                    "evolution_mode": evolution_mode,
                    "random_seed": seed,
                }
                run_from_config(study_dir, full_config)

    for seed in random_seeds:
        for domain_config in domains:
            full_config = {
                **common_config,
                **domain_config,
                "random_seed": seed,
                "crossover_rate": 0.3,
            }
            run_from_config(study_dir, full_config)

    # Log completion of core experiments
    logging.info("Core 2x2 experiments complete.")

    return study_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Base output directory for all experiments",
    )
    parser.add_argument(
        "--seeds", default="42,43,44", help="Comma-separated list of random seeds"
    )

    args = parser.parse_args()

    # Parse random seeds
    random_seeds = [int(seed) for seed in args.seeds.split(",")]

    # Run the ablation study
    study_dir = run_ablation_study(args.output_dir, random_seeds)
