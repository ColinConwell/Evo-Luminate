import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def gather_genome_ids(results_dir: str) -> Dict[str, Any]:
    """
    Gather genome IDs from all generation directories including initial
    """
    results_path = Path(results_dir)
    all_genomes = {"generations": {}, "all_genome_ids": [], "total_count": 0}

    # Find all generation directories and the initial directory
    gen_dirs = sorted(
        [
            d
            for d in results_path.iterdir()
            if d.is_dir() and (d.name.startswith("generation_") or d.name == "initial")
        ]
    )

    for gen_dir in gen_dirs:
        population_file = gen_dir / "population.json"
        if population_file.exists():
            with open(population_file, "r") as f:
                population_data = json.load(f)

                # Determine generation number
                if gen_dir.name == "initial":
                    gen_num = 0
                else:
                    gen_num = int(gen_dir.name.split("_")[1])

                # Store genome IDs for this generation
                all_genomes["generations"][gen_num] = {
                    "genome_ids": population_data["genome_ids"],
                    "count": population_data["count"],
                    "timestamp": population_data.get("timestamp"),
                }

                # Add to the complete list of genome IDs
                all_genomes["all_genome_ids"].extend(population_data["genome_ids"])
                all_genomes["total_count"] += population_data["count"]

    # Remove any duplicate genome IDs (in case some genomes survived multiple generations)
    all_genomes["all_genome_ids"] = list(set(all_genomes["all_genome_ids"]))
    all_genomes["total_count"] = len(all_genomes["all_genome_ids"])

    return all_genomes


def main():
    parser = argparse.ArgumentParser(
        description="Gather genome IDs from all generation directories"
    )
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the gathered genome IDs (default: results_dir/all_genomes.json)",
    )
    args = parser.parse_args()

    all_genomes = gather_genome_ids(args.results_dir)

    if not all_genomes["generations"]:
        print(f"No population data found in {args.results_dir}")
        return

    print(f"Found population data for {len(all_genomes['generations'])} generations")
    print(f"Total unique genomes: {all_genomes['total_count']}")

    # If no output path specified, create one in the results directory
    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.results_dir, "all_genomes.json")

    with open(output_path, "w") as f:
        json.dump(all_genomes, f, indent=2)

    print(f"Genome IDs saved to {output_path}")


if __name__ == "__main__":
    main()
