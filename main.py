import os
from datetime import datetime

from src.run_evolution_experiment import run_evolution_experiment

if __name__ == "__main__":
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"run_{timestamp}")

    # run_evolution_experiment(
    #     output_dir=output_dir,
    #     config={
    #         "random_seed": 42,
    #         "prompt": "Create an interesting shader",
    #         "initial_population_size": 12,
    #         "population_size": 12,
    #         "children_per_generation": 6,
    #         "num_generations": 20,
    #         "k_neighbors": 3,
    #         "max_workers": 4,  # Control parallelism level
    #         "artifact_class": "ShaderArtifact",  # Default to ShaderArtifact
    #         "reasoning_effort": "low",
    #         "use_creative_strategies": True,
    #     },
    # )
    run_evolution_experiment(
        output_dir=output_dir,
        config={
            "random_seed": 42,
            "prompt": "a creative variation of the game snake",
            "initial_population_size": 4,
            "population_size": 4,
            "children_per_generation": 2,
            "num_generations": 4,
            "k_neighbors": 3,
            "max_workers": 4,  # Control parallelism level
            "artifact_class": "GameIdeaArtifact",  # Default to ShaderArtifact
            "evolution_mode": "creation",
            "reasoning_effort": "low",
            "use_creative_strategies": True,
            "use_summary": True,
        },
    )
