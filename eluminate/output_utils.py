import json
import os
from pathlib import Path


def load_population_data(results_dir):
    results_dir = Path(results_dir)

    # Find population data file
    pop_data_path = results_dir / "population_data.jsonl"

    if not pop_data_path.exists():
        raise FileNotFoundError(
            f"Error: population_data.jsonl not found at {pop_data_path}"
        )

    generations = []

    # Read the jsonl file line by line
    with open(pop_data_path, "r") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if "genome_ids" in item:
                    generations.append(item["genome_ids"])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue

    print(f"Found {len(generations)} generations")
    return generations
