import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


def load_novelty_metrics(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load novelty metrics from a single JSONL file
    """
    results_path = Path(results_dir)
    metrics_list = []

    # Look for the novelty_metrics.jsonl file
    metrics_file = results_path / "novelty_metrics.jsonl"

    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    metrics = json.loads(line)
                    metrics_list.append(metrics)
    else:
        print(f"No novelty_metrics.jsonl file found in {results_dir}")

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


def plot_strategy_comparison(
    metrics_list: List[Dict[str, Any]], output_path: str = None
):
    """
    Create a bar chart comparing the average novelty scores for different strategies
    with standard deviation error bars.
    """
    if not metrics_list:
        print("No metrics data found")
        return

    # Collect strategy data across all generations
    strategy_data = {}

    for metrics in metrics_list:
        if "strategy_metrics" in metrics:
            for strategy, stats in metrics["strategy_metrics"].items():
                if strategy == "None":
                    continue
                if strategy not in strategy_data:
                    strategy_data[strategy] = {"novelty_scores": [], "counts": []}

                strategy_data[strategy]["novelty_scores"].append(stats["avg_novelty"])
                strategy_data[strategy]["counts"].append(stats["count"])

    if not strategy_data:
        print("No strategy metrics found in the data")
        return

    # Calculate average novelty and standard deviation for each strategy
    strategies = []
    avg_novelties = []
    std_novelties = []
    avg_counts = []

    for strategy, data in strategy_data.items():
        if data["novelty_scores"]:  # Only include strategies with data
            strategies.append(strategy)
            avg_novelties.append(np.mean(data["novelty_scores"]))
            std_novelties.append(np.std(data["novelty_scores"]))
            avg_counts.append(np.mean(data["counts"]))

    # Sort strategies by average novelty (descending)
    sorted_indices = np.argsort(avg_novelties)[::-1]
    strategies = [strategies[i] for i in sorted_indices]
    avg_novelties = [avg_novelties[i] for i in sorted_indices]
    std_novelties = [std_novelties[i] for i in sorted_indices]
    avg_counts = [avg_counts[i] for i in sorted_indices]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set bar width and positions
    bar_width = 0.7
    positions = np.arange(len(strategies))

    # Create bars with error bars
    bars = ax.bar(
        positions,
        avg_novelties,
        bar_width,
        yerr=std_novelties,
        capsize=5,
        color="skyblue",
        edgecolor="navy",
        alpha=0.8,
    )

    # Add strategy labels
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Add count annotations on top of each bar
    for i, (bar, count) in enumerate(zip(bars, avg_counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std_novelties[i] + 0.005,
            f"n≈{count:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add labels and title
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Average Novelty Score", fontsize=12)
    ax.set_title("Comparison of Novelty Scores by Strategy", fontsize=14)

    # Add grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        # Create a modified output path for the strategy comparison
        strategy_output = output_path.replace(".png", "_strategy_comparison.png")
        plt.savefig(strategy_output, dpi=300, bbox_inches="tight")
        print(f"Strategy comparison plot saved to {strategy_output}")

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

    # Plot both the original novelty metrics and the strategy comparison
    plot_novelty_metrics(metrics_list, output_path)
    plot_strategy_comparison(metrics_list, output_path)


if __name__ == "__main__":
    main()
