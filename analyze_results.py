import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import scipy.stats
import csv


# Removed parse_experiment_name function as we now read directly from config.json


def load_study_metrics(study_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load metrics from all experiment directories in the study,
    grouping by configuration (ignoring seed differences)
    """
    study_path = Path(study_dir)
    configs = defaultdict(list)

    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(study_path) if os.path.isdir(study_path / d)]
    print("num_dirs ", len(exp_dirs))
    for exp_dir in exp_dirs:
        full_exp_dir = study_path / exp_dir

        # Load config.json if it exists
        config_file = full_exp_dir / "config.json"
        if not config_file.exists():
            print(f"No config.json found in {exp_dir}, skipping")
            continue

        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            # defaultCrossover = 0.3 if config.get("artifact_class") == "website" else 0.0
            defaultCrossover = 0.0
            # Extract the information we need
            config_info = {
                "domain": config.get("artifact_class"),
                "creative_strategies": config.get("use_creative_strategies", False),
                "evolution_mode": config.get("evolution_mode", "creation"),
                "reasoning_effort": config.get("reasoning_effort", "low"),
                "use_summary": config.get("use_summary", True),
                "seed": config.get("random_seed", None),
                "crossover": config.get("crossover_rate", defaultCrossover) > 0.0,
            }

            # Create a config key for grouping similar configurations (ignoring seed)
            config_key = "__".join(
                [f"{k}:{v}" for k, v in config_info.items() if k != "seed"]
            )
        except Exception as e:
            print(f"Error reading config in {exp_dir}: {e}")
            continue

        # Load novelty metrics for this experiment
        metrics_file = full_exp_dir / "novelty_metrics.jsonl"
        if not metrics_file.exists():
            print(f"No metrics file found in {exp_dir}")
            continue

        metrics_list = []
        with open(metrics_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    metrics = json.loads(line)
                    metrics_list.append(metrics)

        # Sort by generation
        metrics_list.sort(key=lambda x: x.get("generation", 0))

        if config_info["domain"] == "GameIdeaArtifact":
            continue
        # Add to configs using config_key to group by configuration
        configs[config_key].append(
            {
                "config": config_info,
                "metrics": metrics_list,
                "full_config": config,  # Store the full configuration for reference
                "exp_dir": exp_dir,
            }
        )

    for key in configs.keys():
        if len(configs[key]) > 1:
            print(len(configs[key]), key)
    print(
        f"Found {len(configs)} unique configurations across {len(exp_dirs)} experiment directories"
    )
    return configs


def aggregate_metrics_by_generation(
    experiments: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate metrics across different seeds for the same configuration,
    organized by generation.
    """
    agg_metrics = defaultdict(lambda: defaultdict(list))

    for exp in experiments:
        for metrics in exp["metrics"]:
            gen = metrics.get("generation", 0)

            # Collect all values for this generation
            agg_metrics[gen]["mean_novelty"].append(metrics.get("mean_novelty", 0))
            agg_metrics[gen]["mean_genome_length"].append(
                metrics.get("mean_genome_length", 0)
            )

            # Also collect strategy metrics if available
            if "strategy_metrics" in metrics:
                if "strategy_metrics" not in agg_metrics[gen]:
                    agg_metrics[gen]["strategy_metrics"] = defaultdict(list)

                for strategy, stats in metrics["strategy_metrics"].items():
                    if "avg_novelty" in stats:
                        agg_metrics[gen]["strategy_metrics"][strategy].append(
                            {
                                "avg_novelty": stats["avg_novelty"],
                                "count": stats.get("count", 0),
                            }
                        )

    # Calculate averages and standard deviations
    result = {}
    for gen, values in agg_metrics.items():
        result[gen] = {
            "generation": gen,
            "mean_novelty": np.mean(values["mean_novelty"]),
            "std_novelty": np.std(values["mean_novelty"]),
            "mean_genome_length": np.mean(values["mean_genome_length"]),
            "std_genome_length": np.std(values["mean_genome_length"]),
            "sample_count": len(values["mean_novelty"]),
        }

        # Aggregate strategy metrics if available
        if "strategy_metrics" in values:
            result[gen]["strategy_metrics"] = {}
            for strategy, stats_list in values["strategy_metrics"].items():
                novelty_values = [stats["avg_novelty"] for stats in stats_list]
                count_values = [stats["count"] for stats in stats_list]

                result[gen]["strategy_metrics"][strategy] = {
                    "avg_novelty": np.mean(novelty_values),
                    "std_novelty": np.std(novelty_values),
                    "count": np.mean(count_values),
                    "sample_count": len(novelty_values),
                }

    return result


def plot_normalized_comparison(
    configs: Dict[str, Dict[str, Any]],
    output_dir: str,
    plot_genome_length: bool = True,
):
    """
    Create a bar chart showing percentage increase in novelty from first to last generation.
    This shows how much each configuration improved over the course of evolution.

    Parameters:
        configs: Dictionary of configuration data
        output_dir: Directory to save plots
        plot_genome_length: If True, also plot genome length alongside novelty
    """
    # Extract metrics for each configuration
    domain_metrics = defaultdict(list)

    # Track crossover settings by domain
    domain_crossover_settings = defaultdict(set)

    # First pass: collect all crossover settings for each domain
    for config_key, experiments in configs.items():
        config = experiments[0]["config"]
        domain = config["domain"]
        domain_crossover_settings[domain].add(config["crossover"])

    average_start_novelty = defaultdict(list)
    for config_key, experiments in configs.items():
        # Get sample configuration
        config = experiments[0]["config"]
        domain = config["domain"]

        # Check if there are multiple different crossover settings across this domain
        multiple_crossovers = len(domain_crossover_settings[domain]) > 1

        # Calculate novelty increase and genome length for each experiment
        novelty_increases = []
        genome_lengths = []

        for exp in experiments:
            metrics = exp["metrics"]
            if not metrics or len(metrics) < 2:
                continue

            # Sort metrics by generation
            sorted_metrics = sorted(metrics, key=lambda m: m.get("generation", 0))

            # Get first and last generation novelty
            first_gen_novelty = sorted_metrics[0].get("mean_novelty", 0)
            last_gen_novelty = sorted_metrics[-1].get("mean_novelty", 0)

            # Get last generation genome length
            last_gen_genome_length = sorted_metrics[-1].get("mean_genome_length", 0)

            # Calculate percentage increase
            novelty_increases.append(last_gen_novelty)
            genome_lengths.append(last_gen_genome_length)

            average_start_novelty[domain].append(first_gen_novelty)

        # Create a readable configuration label
        config_label = (
            f"{'Strategies On' if config['creative_strategies'] else 'Strategies Off'}\n"
            f"Reasoning: {config['reasoning_effort'].capitalize()}\n"
            f"Mode: {config['evolution_mode'].capitalize()}\n"
            f"{'Summary On' if config['use_summary'] else 'Summary Off'}\n"
        )

        if multiple_crossovers:
            config_label += f"\nCross: {config['crossover']}"

        domain_metrics[domain].append(
            {
                "config": config,
                "config_label": config_label,
                "mean_novelty_increase": np.mean(novelty_increases)
                / np.mean(average_start_novelty[domain]),
                "std_novelty_increase": np.std(novelty_increases),
                "mean_genome_length": np.mean(genome_lengths),
                "std_genome_length": np.std(genome_lengths),
                "sample_count": len(novelty_increases),
            }
        )

    # Create a plot for each domain
    for domain, metrics in domain_metrics.items():
        if not metrics:
            continue

        # Sort by mean novelty increase (highest first)
        metrics.sort(key=lambda x: x["mean_novelty_increase"], reverse=True)

        # Set up the plot with a clean style
        plt.style.use("seaborn-v0_8-whitegrid")

        # Create a single figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract data for plotting
        config_labels = [m["config_label"] for m in metrics]
        novelty_means = [m["mean_novelty_increase"] for m in metrics]
        novelty_stds = [m["std_novelty_increase"] for m in metrics]
        sample_counts = [m["sample_count"] for m in metrics]

        if plot_genome_length:
            genome_means = [m["mean_genome_length"] for m in metrics]
            genome_stds = [m["std_genome_length"] for m in metrics]

        # Set width and positions
        bar_width = 0.8

        # Calculate positions for bars
        if plot_genome_length:
            novelty_positions = np.arange(len(config_labels)) * 2  # Even positions
            genome_positions = novelty_positions + bar_width  # Odd positions
        else:
            novelty_positions = np.arange(len(config_labels))

        # Colors for bars
        novelty_color = "#3498db"  # Blue
        genome_color = "#e74c3c"  # Red

        # Create novelty bars
        novelty_bars = ax.bar(
            novelty_positions,
            novelty_means,
            bar_width,
            yerr=novelty_stds,
            capsize=5,
            color=novelty_color,
            edgecolor="none",
            alpha=0.85,
            label="Novelty",
        )

        # Create genome length bars if requested
        if plot_genome_length:
            # Create a second y-axis for genome length
            ax2 = ax.twinx()

            genome_bars = ax2.bar(
                genome_positions,
                genome_means,
                bar_width,
                yerr=genome_stds,
                capsize=5,
                color=genome_color,
                edgecolor="none",
                alpha=0.85,
                label="Genome Length",
            )

            # Set y-axis label for genome length
            ax2.set_ylabel(
                "Genome Length", fontsize=12, fontweight="bold", color=genome_color
            )

            # Format the second y-axis
            ax2.tick_params(axis="y", colors=genome_color)
            ax2.spines["right"].set_color(genome_color)
            ax2.spines["right"].set_alpha(0.3)

            # Add annotations for genome lengths
            for i, (pos, value, std) in enumerate(
                zip(genome_positions, genome_means, genome_stds)
            ):
                # Add genome length value
                ax2.annotate(
                    f"{value:.0f}",
                    xy=(pos, value + std + 5),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=genome_color,
                )

        # Add annotations for novelty values and sample counts
        for i, (pos, value, std, count) in enumerate(
            zip(novelty_positions, novelty_means, novelty_stds, sample_counts)
        ):
            # Add novelty value
            ax.annotate(
                f"{value:.3f}",
                xy=(pos, value + std + 0.005),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=novelty_color,
            )

            # Add sample count
            ax.annotate(
                f"n={count}",
                xy=(pos, value / 2),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.1 else "#333333",
            )

        # Set x-tick positions and labels
        if plot_genome_length:
            # Place ticks between the pairs of bars
            ax.set_xticks(novelty_positions + bar_width / 2)
        else:
            ax.set_xticks(novelty_positions)

        ax.set_xticklabels(config_labels, fontsize=9)

        # Set labels and title
        # ax.set_xlabel("Configuration", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Final Novelty / Start Novelty",
            fontsize=12,
            fontweight="bold",
            color=novelty_color,
        )

        # Format the first y-axis
        ax.tick_params(axis="y", colors=novelty_color)
        ax.spines["left"].set_color(novelty_color)
        ax.spines["left"].set_alpha(0.3)

        # Set title based on what we're plotting
        if plot_genome_length:
            title = f"{domain.upper()} Domain - Novelty and Genome Length Comparison"
        else:
            title = f"{domain.upper()} Domain - Novelty Comparison"

        # ax.set_title(title, fontsize=14, fontweight="bold")

        # Remove grid lines for a cleaner look
        ax.grid(False)
        if plot_genome_length:
            ax2.grid(False)

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        if not plot_genome_length:
            ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_alpha(0.3)

        # Add legend if plotting both metrics
        if plot_genome_length:
            # Combine legends from both axes
            handles = [novelty_bars[0], genome_bars[0]]
            labels = ["Novelty", "Genome Length"]
            ax.legend(handles, labels, loc="upper right", frameon=False)

        # Adjust layout and save
        plt.tight_layout()

        # Determine filename based on whether we're plotting genome length
        if plot_genome_length:
            output_path = os.path.join(
                output_dir, f"{domain}_novelty_and_length_comparison.png"
            )
        else:
            output_path = os.path.join(
                output_dir, f"{domain}_normalized_comparison.png"
            )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot for {domain} saved to {output_path}")
        plt.close()


def plot_strategy_comparison(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create enhanced visualizations comparing strategy performance across domain pairs.
    Only considers experiments matching the specified configuration.

    Parameters:
        configs: Dictionary of configuration data
        output_dir: Directory to save plots
    """
    to_evaluate = {
        "strategies": True,
        "reasoning": "low",
        "mode": "variation",
        "summary": True,
    }

    # Filter experiments that match our criteria
    matching_experiments = []
    for config_key, experiments in configs.items():
        config = experiments[0]["config"]

        # Check if this configuration matches our criteria
        if (
            config["creative_strategies"] == to_evaluate["strategies"]
            and config["reasoning_effort"] == to_evaluate["reasoning"]
            and config["evolution_mode"] == to_evaluate["mode"]
            and config["use_summary"] == to_evaluate["summary"]
        ):

            matching_experiments.extend(experiments)

    # Group experiments by domain
    domain_experiments = defaultdict(list)
    for exp in matching_experiments:
        domain = exp["config"]["domain"]
        domain_experiments[domain].append(exp)

    # Get all domains
    domains = list(domain_experiments.keys())

    if len(domains) < 2:
        print("Not enough domains found for comparison")
        return

    # Collect strategy metrics across all generations for each domain
    domain_strategy_data = {}
    domain_baseline_novelty = {}
    domain_strategy_usage = {}

    for domain, exps in domain_experiments.items():
        # Track metrics by experiment and generation
        exp_gen_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        baseline_values = []

        for exp_idx, exp in enumerate(exps):
            # Get the metrics sorted by generation
            if not exp["metrics"]:
                continue

            sorted_metrics = sorted(
                exp["metrics"], key=lambda m: m.get("generation", 0)
            )

            # Get baseline novelty from first generation
            if sorted_metrics and "mean_novelty" in sorted_metrics[0]:
                baseline_values.append(sorted_metrics[0]["mean_novelty"])

            # Process all generations
            for gen_metrics in sorted_metrics:
                generation = gen_metrics.get("generation", 0)

                if "strategy_metrics" not in gen_metrics:
                    continue

                # Calculate total count of artifacts in this generation
                total_count = sum(
                    stats.get("count", 0)
                    for strategy, stats in gen_metrics["strategy_metrics"].items()
                    if strategy != "None" and strategy is not None
                )

                if total_count == 0:
                    continue

                # Collect metrics and usage percentage for each strategy
                for strategy, stats in gen_metrics["strategy_metrics"].items():
                    # Skip None strategy
                    if strategy == "None" or strategy is None:
                        continue

                    if "avg_novelty" in stats:
                        # Calculate percentage of generation using this strategy
                        count = stats.get("count", 0)
                        usage_percent = (
                            (count / total_count) * 100 if total_count > 0 else 0
                        )

                        # Store by experiment and generation
                        exp_gen_metrics[exp_idx][generation][strategy] = {
                            "novelty": stats["avg_novelty"],
                            "usage": usage_percent,
                            "count": count,
                        }

        # Calculate average baseline novelty for this domain
        domain_baseline_novelty[domain] = (
            np.mean(baseline_values) if baseline_values else 0
        )

        # First, average across generations for each experiment
        exp_avg_metrics = defaultdict(lambda: defaultdict(list))
        for exp_idx in exp_gen_metrics:
            # Get all generations for this experiment
            generations = exp_gen_metrics[exp_idx].keys()
            if not generations:
                continue

            # Get all strategies used in this experiment
            all_exp_strategies = set()
            for gen_data in exp_gen_metrics[exp_idx].values():
                all_exp_strategies.update(gen_data.keys())

            # Average each strategy across all generations
            for strategy in all_exp_strategies:
                strategy_novelty = []
                strategy_usage = []

                for gen in generations:
                    if strategy in exp_gen_metrics[exp_idx][gen]:
                        metrics = exp_gen_metrics[exp_idx][gen][strategy]
                        strategy_novelty.append(metrics["novelty"])
                        strategy_usage.append(metrics["usage"])

                if strategy_novelty and strategy_usage:
                    exp_avg_metrics[exp_idx][strategy] = {
                        "novelty": np.mean(strategy_novelty),
                        "usage": np.mean(strategy_usage),
                    }

        # Then, average across experiments
        strategy_data = defaultdict(list)
        strategy_usage = defaultdict(list)

        for exp_idx in exp_avg_metrics:
            for strategy, metrics in exp_avg_metrics[exp_idx].items():
                strategy_data[strategy].append(metrics["novelty"])
                strategy_usage[strategy].append(metrics["usage"])

        # Store final averaged data
        domain_strategy_data[domain] = {
            strategy: values for strategy, values in strategy_data.items() if values
        }

        domain_strategy_usage[domain] = {
            strategy: values for strategy, values in strategy_usage.items() if values
        }

    # Get all unique strategies across all domains
    all_strategies = set()
    for domain_metrics in domain_strategy_data.values():
        all_strategies.update(domain_metrics.keys())

    # Remove None strategy if it somehow got included
    if "None" in all_strategies:
        all_strategies.remove("None")
    if None in all_strategies:
        all_strategies.remove(None)

    # Create a summary file for correlation analysis across all domains
    correlation_summary = []

    # Create plots for each pair of domains
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            domain_a = domains[i]
            domain_b = domains[j]

            # Skip if either domain doesn't have strategy metrics
            if not domain_strategy_data[domain_a] or not domain_strategy_data[domain_b]:
                continue

            # Prepare data for correlation analysis - using both novelty and usage
            strategies = []
            x_means_novelty = []
            y_means_novelty = []
            x_stds_novelty = []
            y_stds_novelty = []

            x_means_usage = []
            y_means_usage = []
            x_stds_usage = []
            y_stds_usage = []

            for strategy in all_strategies:
                # Skip if strategy doesn't exist in both domains
                if (
                    strategy not in domain_strategy_data[domain_a]
                    or strategy not in domain_strategy_data[domain_b]
                    or strategy not in domain_strategy_usage[domain_a]
                    or strategy not in domain_strategy_usage[domain_b]
                ):
                    continue

                # Calculate percentage improvement over baseline for novelty
                x_values_novelty = [
                    (v / domain_baseline_novelty[domain_a] - 1) * 100
                    for v in domain_strategy_data[domain_a][strategy]
                ]
                y_values_novelty = [
                    (v / domain_baseline_novelty[domain_b] - 1) * 100
                    for v in domain_strategy_data[domain_b][strategy]
                ]

                # Get usage percentages
                x_values_usage = domain_strategy_usage[domain_a][strategy]
                y_values_usage = domain_strategy_usage[domain_b][strategy]

                if (
                    len(x_values_novelty) > 0
                    and len(y_values_novelty) > 0
                    and len(x_values_usage) > 0
                    and len(y_values_usage) > 0
                ):
                    strategies.append(strategy)

                    # Novelty metrics
                    x_means_novelty.append(np.mean(x_values_novelty))
                    y_means_novelty.append(np.mean(y_values_novelty))
                    x_stds_novelty.append(
                        np.std(x_values_novelty) if len(x_values_novelty) > 1 else 0.5
                    )
                    y_stds_novelty.append(
                        np.std(y_values_novelty) if len(y_values_novelty) > 1 else 0.5
                    )

                    # Usage metrics
                    x_means_usage.append(np.mean(x_values_usage))
                    y_means_usage.append(np.mean(y_values_usage))
                    x_stds_usage.append(
                        np.std(x_values_usage) if len(x_values_usage) > 1 else 0.5
                    )
                    y_stds_usage.append(
                        np.std(y_values_usage) if len(y_values_usage) > 1 else 0.5
                    )

            # Calculate correlations if we have enough data points
            if (
                len(strategies) >= 3
            ):  # Need at least 3 points for meaningful correlation
                # Novelty correlation
                novelty_corr, novelty_p = scipy.stats.pearsonr(
                    x_means_novelty, y_means_novelty
                )

                # Usage correlation
                usage_corr, usage_p = scipy.stats.pearsonr(x_means_usage, y_means_usage)

                # Add to summary
                correlation_summary.append(
                    {
                        "domain_pair": f"{domain_a} vs {domain_b}",
                        "novelty_pearson_r": novelty_corr,
                        "novelty_pearson_p": novelty_p,
                        "usage_pearson_r": usage_corr,
                        "usage_pearson_p": usage_p,
                        "strategy_count": len(strategies),
                    }
                )

                # Correlation text for plots
                novelty_corr_text = (
                    f"Novelty Correlation: r={novelty_corr:.2f} (p={novelty_p:.3f})"
                )
                usage_corr_text = (
                    f"Usage Correlation: r={usage_corr:.2f} (p={usage_p:.3f})"
                )
            else:
                novelty_corr_text = "Insufficient data for correlation"
                usage_corr_text = "Insufficient data for correlation"

            # 1. Create normalized scatter plot showing percentage improvement in novelty
            plt.figure(figsize=(10, 10))

            # Create scatter plot with error bars
            plt.errorbar(
                x_means_novelty,
                y_means_novelty,
                xerr=x_stds_novelty,
                yerr=y_stds_novelty,
                fmt="o",
                ecolor="lightgray",
                elinewidth=1,
                capsize=3,
                markersize=8,
                alpha=0.7,
            )

            # Add strategy labels
            for strategy, x, y in zip(strategies, x_means_novelty, y_means_novelty):
                plt.annotate(
                    strategy,
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8,
                )

            # Calculate focused axis limits with padding
            if x_means_novelty and y_means_novelty:
                # Calculate padding as a percentage of the data range
                x_padding = (
                    (max(x_means_novelty) - min(x_means_novelty)) * 0.2
                    if len(x_means_novelty) > 1
                    else 1.0
                )
                y_padding = (
                    (max(y_means_novelty) - min(y_means_novelty)) * 0.2
                    if len(y_means_novelty) > 1
                    else 1.0
                )

                # Set focused limits with padding
                x_min = min(x_means_novelty) - x_padding - max(x_stds_novelty)
                x_max = max(x_means_novelty) + x_padding + max(x_stds_novelty)
                y_min = min(y_means_novelty) - y_padding - max(y_stds_novelty)
                y_max = max(y_means_novelty) + y_padding + max(y_stds_novelty)

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

            # Add reference lines at x=0 and y=0
            plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            plt.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

            # Add diagonal line (y=x) if it passes through the visible area
            if x_means_novelty and y_means_novelty:
                min_val = min(x_min, y_min)
                max_val = max(x_max, y_max)
                plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

            # Set labels and title
            plt.xlabel(f"{domain_a} % Improvement", fontsize=12, fontweight="bold")
            plt.ylabel(f"{domain_b} % Improvement", fontsize=12, fontweight="bold")
            plt.title(
                f"Strategy % Improvement: {domain_a} vs {domain_b}\n{novelty_corr_text}",
                fontsize=14,
                fontweight="bold",
            )

            # Add grid
            plt.grid(True, alpha=0.3)

            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(
                output_dir, f"strategy_improvement_{domain_a}_vs_{domain_b}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(
                f"Strategy improvement plot for {domain_a} vs {domain_b} saved to {output_path}"
            )
            plt.close()

            # 2. Create scatter plot showing usage percentages
            plt.figure(figsize=(10, 10))

            # Create scatter plot with error bars
            plt.errorbar(
                x_means_usage,
                y_means_usage,
                xerr=x_stds_usage,
                yerr=y_stds_usage,
                fmt="o",
                ecolor="lightgray",
                elinewidth=1,
                capsize=3,
                markersize=8,
                alpha=0.7,
            )

            # Add strategy labels
            for strategy, x, y in zip(strategies, x_means_usage, y_means_usage):
                plt.annotate(
                    strategy,
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8,
                )

            # Calculate focused axis limits with padding
            if x_means_usage and y_means_usage:
                # Calculate padding as a percentage of the data range
                x_padding = (
                    (max(x_means_usage) - min(x_means_usage)) * 0.2
                    if len(x_means_usage) > 1
                    else 1.0
                )
                y_padding = (
                    (max(y_means_usage) - min(y_means_usage)) * 0.2
                    if len(y_means_usage) > 1
                    else 1.0
                )

                # Set focused limits with padding
                x_min = max(0, min(x_means_usage) - x_padding - max(x_stds_usage))
                x_max = min(100, max(x_means_usage) + x_padding + max(x_stds_usage))
                y_min = max(0, min(y_means_usage) - y_padding - max(y_stds_usage))
                y_max = min(100, max(y_means_usage) + y_padding + max(y_stds_usage))

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

            # Add diagonal line (y=x) if it passes through the visible area
            if x_means_usage and y_means_usage:
                min_val = min(x_min, y_min)
                max_val = max(x_max, y_max)
                plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

            # Set labels and title
            plt.xlabel(f"{domain_a} Strategy Usage %", fontsize=12, fontweight="bold")
            plt.ylabel(f"{domain_b} Strategy Usage %", fontsize=12, fontweight="bold")
            plt.title(
                f"Strategy Usage Percentage: {domain_a} vs {domain_b}\n{usage_corr_text}",
                fontsize=14,
                fontweight="bold",
            )

            # Add grid
            plt.grid(True, alpha=0.3)

            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(
                output_dir, f"strategy_usage_{domain_a}_vs_{domain_b}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(
                f"Strategy usage plot for {domain_a} vs {domain_b} saved to {output_path}"
            )
            plt.close()

    # Create a summary table of correlations and strategy performance
    if correlation_summary:
        plt.figure(figsize=(12, len(correlation_summary) * 0.5 + 2))
        plt.axis("off")

        # Create table data
        table_data = [
            ["Domain Pair", "Novelty r", "p-value", "Usage r", "p-value", "Strategies"]
        ]
        for entry in correlation_summary:
            table_data.append(
                [
                    entry["domain_pair"],
                    f"{entry['novelty_pearson_r']:.3f}",
                    f"{entry['novelty_pearson_p']:.3f}",
                    f"{entry['usage_pearson_r']:.3f}",
                    f"{entry['usage_pearson_p']:.3f}",
                    str(entry["strategy_count"]),
                ]
            )

        # Create table
        table = plt.table(
            cellText=table_data,
            colLabels=None,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Add title
        plt.title(
            "Strategy Performance Correlation Across Domains",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Save the table
        plt.tight_layout()
        output_path = os.path.join(output_dir, "strategy_correlation_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Correlation summary saved to {output_path}")
        plt.close()

        # Also save as CSV for further analysis
        csv_path = os.path.join(output_dir, "strategy_correlation_summary.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table_data)
        print(f"Correlation data saved to {csv_path}")

        # Create a comprehensive strategy performance summary across all domains
        strategy_summary = []

        # Collect data for all strategies across all domains
        for strategy in all_strategies:
            strategy_data = {"strategy": strategy, "domains": {}}

            for domain in domains:
                if (
                    strategy in domain_strategy_data[domain]
                    and strategy in domain_strategy_usage[domain]
                ):

                    # Calculate percentage improvement
                    novelty_values = domain_strategy_data[domain][strategy]
                    pct_improvement = [
                        (v / domain_baseline_novelty[domain] - 1) * 100
                        for v in novelty_values
                    ]

                    # Get usage percentages
                    usage_values = domain_strategy_usage[domain][strategy]

                    strategy_data["domains"][domain] = {
                        "avg_pct_improvement": np.mean(pct_improvement),
                        "std_pct_improvement": (
                            np.std(pct_improvement) if len(pct_improvement) > 1 else 0
                        ),
                        "avg_usage": np.mean(usage_values),
                        "std_usage": (
                            np.std(usage_values) if len(usage_values) > 1 else 0
                        ),
                        "sample_count": len(novelty_values),
                    }

            # Only include strategies that appear in at least two domains
            if len(strategy_data["domains"]) >= 2:
                strategy_summary.append(strategy_data)

        # Create a CSV with the comprehensive strategy summary
        if strategy_summary:
            csv_path = os.path.join(output_dir, "strategy_performance_summary.csv")
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ["Strategy"]
                for domain in domains:
                    header.extend(
                        [
                            f"{domain} Avg % Improvement",
                            f"{domain} Std % Improvement",
                            f"{domain} Avg Usage %",
                            f"{domain} Std Usage %",
                            f"{domain} Samples",
                        ]
                    )
                writer.writerow(header)

                # Write data for each strategy
                for strategy_data in strategy_summary:
                    row = [strategy_data["strategy"]]

                    for domain in domains:
                        if domain in strategy_data["domains"]:
                            domain_data = strategy_data["domains"][domain]
                            row.extend(
                                [
                                    f"{domain_data['avg_pct_improvement']:.2f}",
                                    f"{domain_data['std_pct_improvement']:.2f}",
                                    f"{domain_data['avg_usage']:.2f}",
                                    f"{domain_data['std_usage']:.2f}",
                                    str(domain_data["sample_count"]),
                                ]
                            )
                        else:
                            row.extend(["N/A", "N/A", "N/A", "N/A", "0"])

                    writer.writerow(row)

            print(f"Comprehensive strategy performance summary saved to {csv_path}")


# Add this to the main function
def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument(
        "study_dir", type=str, help="Path to the ablation study directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: study_dir/analysis)",
    )
    args = parser.parse_args()

    # Set up output directory
    output_dir = (
        args.output if args.output else os.path.join(args.study_dir, "analysis")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics from all experiments
    configs = load_study_metrics(args.study_dir)

    if not configs:
        print(f"No valid configurations found in {args.study_dir}")
        return

    # Create plots
    plot_normalized_comparison(configs, output_dir)
    # plot_strategy_comparison(configs, output_dir)

    # print(f"Analysis complete. Results saved to {output_dir}")
    # print(configs.keys())


if __name__ == "__main__":
    main()
