import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


def load_novelty_metrics(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load novelty metrics from all generation directories
    """
    results_path = Path(results_dir)
    metrics_list = []

    # Find all generation directories and the initial directory
    gen_dirs = sorted(
        [
            d
            for d in results_path.iterdir()
            if d.is_dir() and (d.name.startswith("generation_") or d.name == "initial")
        ]
    )

    for gen_dir in gen_dirs:
        metrics_file = gen_dir / "novelty_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                # If this is the "initial" directory, set generation to 0
                if gen_dir.name == "initial":
                    metrics["generation"] = 0
                metrics_list.append(metrics)

    # Sort by generation number
    metrics_list.sort(key=lambda x: x.get("generation", 0))
    return metrics_list


def plot_novelty_metrics(metrics_list: List[Dict[str, Any]], output_path: str = None):
    """
    Plot novelty metrics across generations
    """
    if not metrics_list:
        print("No metrics data found")
        return

    generations = [m.get("generation", i + 1) for i, m in enumerate(metrics_list)]
    mean_novelty = [m.get("mean_novelty", 0) for m in metrics_list]
    # Extract mean genome length data
    mean_genome_length = [m.get("mean_genome_length", 0) for m in metrics_list]
    # max_novelty = [m.get("max_novelty", 0) for m in metrics_list]
    # min_novelty = [m.get("min_novelty", 0) for m in metrics_list]

    # Create figure with primary y-axis
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot mean novelty on primary y-axis
    color = "blue"
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Novelty Score (Cosine Distance)", fontsize=12, color=color)
    ax1.plot(generations, mean_novelty, color=color, linewidth=2, label="Mean Novelty")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create secondary y-axis and plot mean genome length
    ax2 = ax1.twinx()
    color = "red"
    ax2.set_ylabel("Mean Genome Length", fontsize=12, color=color)
    ax2.plot(
        generations,
        mean_genome_length,
        color=color,
        linestyle="-",
        linewidth=2,
        label="Mean Genome Length",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Add title and grid
    plt.title("Population Novelty and Genome Length Across Generations", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Add annotations for highest and lowest points (commented out for now)
    # max_gen_idx = np.argmax(mean_novelty)
    # min_gen_idx = np.argmin(mean_novelty)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    # else:
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot novelty metrics across generations"
    )
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the plot (default: display plot)",
    )
    args = parser.parse_args()

    metrics_list = load_novelty_metrics(args.results_dir)

    if not metrics_list:
        print(f"No novelty metrics found in {args.results_dir}")
        return

    print(f"Found novelty metrics for {len(metrics_list)} generations")

    # If no output path specified, create one in the results directory
    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.results_dir, "novelty_plot.png")

    plot_novelty_metrics(metrics_list, output_path)


if __name__ == "__main__":
    main()
