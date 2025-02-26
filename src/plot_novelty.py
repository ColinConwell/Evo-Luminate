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

    # Find all generation directories
    gen_dirs = sorted(
        [
            d
            for d in results_path.iterdir()
            if d.is_dir() and d.name.startswith("generation_")
        ]
    )

    for gen_dir in gen_dirs:
        metrics_file = gen_dir / "novelty_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
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
    max_novelty = [m.get("max_novelty", 0) for m in metrics_list]
    min_novelty = [m.get("min_novelty", 0) for m in metrics_list]

    plt.figure(figsize=(12, 7))

    # Plot mean novelty with confidence interval
    plt.plot(generations, mean_novelty, "b-", linewidth=2, label="Mean Novelty")

    # # Plot min and max as a shaded area
    # plt.fill_between(
    #     generations,
    #     min_novelty,
    #     max_novelty,
    #     color="blue",
    #     alpha=0.2,
    #     label="Min-Max Range",
    # )

    # # Add individual lines for min and max
    # plt.plot(generations, max_novelty, "g--", linewidth=1, label="Max Novelty")
    # plt.plot(generations, min_novelty, "r--", linewidth=1, label="Min Novelty")

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Novelty Score (Cosine Distance)", fontsize=12)
    plt.title("Population Novelty Across Generations", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Add annotations for highest and lowest points
    max_gen_idx = np.argmax(mean_novelty)
    min_gen_idx = np.argmin(mean_novelty)

    # plt.annotate(
    #     f"Max: {mean_novelty[max_gen_idx]:.4f}",
    #     xy=(generations[max_gen_idx], mean_novelty[max_gen_idx]),
    #     xytext=(10, 10),
    #     textcoords="offset points",
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    # )

    # plt.annotate(
    #     f"Min: {mean_novelty[min_gen_idx]:.4f}",
    #     xy=(generations[min_gen_idx], mean_novelty[min_gen_idx]),
    #     xytext=(10, -20),
    #     textcoords="offset points",
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    # )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
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
