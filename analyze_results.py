import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple


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

            # Extract the information we need
            config_info = {
                "domain": (
                    "shader"
                    if config.get("artifact_class") == "ShaderArtifact"
                    else "game"
                ),
                "creative_strategies": config.get("use_creative_strategies", False),
                "evolution_mode": config.get("evolution_mode", "creation"),
                "reasoning_effort": config.get("reasoning_effort", "low"),
                "use_summary": config.get("use_summary", True),
                "seed": config.get("random_seed", None),
            }

            # Create a config key for grouping similar configurations (ignoring seed)
            config_key = (
                f"{config_info['domain']}_"
                f"{'strat' if config_info['creative_strategies'] else 'nostrat'}_"
                f"{config_info['evolution_mode']}_"
                f"{config_info['reasoning_effort']}_"
                f"{'sum' if config_info['use_summary'] else 'nosum'}"
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

        # Add to configs using config_key to group by configuration
        configs[config_key].append(
            {
                "config": config_info,
                "metrics": metrics_list,
                "full_config": config,  # Store the full configuration for reference
                "exp_dir": exp_dir,
            }
        )

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


def plot_configuration_comparison(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create a bar chart comparing the final novelty scores for different configurations.
    """
    # Extract final generation metrics for each configuration
    final_metrics = []

    for config_key, experiments in configs.items():
        # Get sample configuration (they're all the same except for seed)
        config = experiments[0]["config"]

        # Aggregate metrics
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Get the final generation metrics
        final_gen = max(agg_metrics.keys())
        metrics = agg_metrics[final_gen]

        # Create a more readable configuration label
        config_label = (
            f"{config['domain']}\n"
            f"{'Strat' if config['creative_strategies'] else 'No-Strat'}, "
            f"{config['evolution_mode']}, "
            f"{config['reasoning_effort']}, "
            f"{'Sum' if config['use_summary'] else 'No-Sum'}"
        )

        final_metrics.append(
            {
                "config": config,
                "config_label": config_label,
                "mean_novelty": metrics["mean_novelty"],
                "std_novelty": metrics["std_novelty"],
                "sample_count": metrics["sample_count"],
            }
        )

    # Group by domain
    domains = set(m["config"]["domain"] for m in final_metrics)
    domain_metrics = {domain: [] for domain in domains}

    for metrics in final_metrics:
        domain_metrics[metrics["config"]["domain"]].append(metrics)

    # Create a plot for each domain
    for domain, metrics in domain_metrics.items():
        # Sort by mean novelty (descending)
        metrics.sort(key=lambda x: x["mean_novelty"], reverse=True)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Extract data for plotting
        config_labels = [m["config_label"] for m in metrics]
        novelty_means = [m["mean_novelty"] for m in metrics]
        novelty_stds = [m["std_novelty"] for m in metrics]

        # Create bars
        bar_positions = np.arange(len(config_labels))
        bars = ax.bar(
            bar_positions,
            novelty_means,
            yerr=novelty_stds,
            capsize=5,
            color="skyblue",
            edgecolor="navy",
            alpha=0.8,
        )

        # Add annotations for sample counts
        for i, metrics in enumerate(metrics):
            ax.annotate(
                f"n={metrics['sample_count']}",
                xy=(i, novelty_means[i] + novelty_stds[i] + 0.005),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Label the axes
        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("Mean Novelty", fontsize=12)
        ax.set_title(
            f"Comparison of Configurations - {domain.upper()} Domain", fontsize=14
        )

        # Set x-tick positions and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(config_labels, fontsize=10)

        # Add grid for readability
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout and save
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{domain}_config_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Configuration comparison plot for {domain} saved to {output_path}")
        plt.close()


def plot_consolidated_novelty_and_length(
    configs: Dict[str, Dict[str, Any]], output_dir: str
):
    """
    Create consolidated plots for each domain with all configurations on the same chart.
    This makes it easier to compare different configurations directly.
    """
    # Group configurations by domain
    domain_configs = defaultdict(list)

    for config_key, experiments in configs.items():
        config = experiments[0]["config"]
        domain = config["domain"]

        # Aggregate metrics
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Only include if we have data
        if agg_metrics:
            domain_configs[domain].append(
                {"config": config, "metrics": agg_metrics, "config_key": config_key}
            )

    # For each domain, create consolidated plots
    for domain, configs_list in domain_configs.items():
        # 1. Consolidated Novelty Plot
        plt.figure(figsize=(14, 10))

        # Define a color cycle for different configs
        # More distinct colors for better separation
        colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # Track min/max y values for consistent axes
        min_novelty, max_novelty = float("inf"), float("-inf")

        # Create a legend mapping
        legend_handles = []
        legend_labels = []

        # Plot each configuration
        for i, config_data in enumerate(configs_list):
            config = config_data["config"]
            metrics = config_data["metrics"]

            # Sort generations
            generations = sorted(metrics.keys())
            if not generations:
                continue

            # Extract novelty data
            novelty_values = [metrics[g]["mean_novelty"] for g in generations]
            novelty_stds = [metrics[g]["std_novelty"] for g in generations]

            # Update min/max
            min_novelty = min(
                min_novelty, min([v - s for v, s in zip(novelty_values, novelty_stds)])
            )
            max_novelty = max(
                max_novelty, max([v + s for v, s in zip(novelty_values, novelty_stds)])
            )

            # Create a readable label
            label = (
                f"{'Strat' if config['creative_strategies'] else 'No-Strat'} | "
                f"{config['evolution_mode']} | "
                f"{config['reasoning_effort']} | "
                f"{'Sum' if config['use_summary'] else 'No-Sum'}"
            )

            # Choose color and line style
            color = colors[i % len(colors)]
            linestyle = "-" if config["creative_strategies"] else "--"

            # Plot the line
            (line,) = plt.plot(
                generations,
                novelty_values,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )

            # Add error band
            plt.fill_between(
                generations,
                [v - s for v, s in zip(novelty_values, novelty_stds)],
                [v + s for v, s in zip(novelty_values, novelty_stds)],
                color=color,
                alpha=0.1,
            )

            # Add to legend mapping
            legend_handles.append(line)
            legend_labels.append(label)

        # Add labels and title
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Novelty Score", fontsize=12)
        plt.title(
            f"{domain.upper()} Domain - Novelty Across All Configurations", fontsize=14
        )

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add legend
        plt.legend(
            handles=legend_handles, labels=legend_labels, loc="best", fontsize=10
        )

        # Adjust y-axis for better comparison
        plt.ylim([max(0, min_novelty - 0.02), max_novelty + 0.02])

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir, f"{domain}_consolidated_novelty.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Consolidated novelty plot for {domain} saved to {output_path}")
        plt.close()

        # 2. Consolidated Genome Length Plot
        plt.figure(figsize=(14, 10))

        # Track min/max y values
        min_length, max_length = float("inf"), float("-inf")

        # Clear legend mapping
        legend_handles = []
        legend_labels = []

        # Plot each configuration
        for i, config_data in enumerate(configs_list):
            config = config_data["config"]
            metrics = config_data["metrics"]

            # Sort generations
            generations = sorted(metrics.keys())
            if not generations:
                continue

            # Extract genome length data
            length_values = [metrics[g]["mean_genome_length"] for g in generations]
            length_stds = [metrics[g]["std_genome_length"] for g in generations]

            # Update min/max
            min_length = min(
                min_length, min([v - s for v, s in zip(length_values, length_stds)])
            )
            max_length = max(
                max_length, max([v + s for v, s in zip(length_values, length_stds)])
            )

            # Create a readable label
            label = (
                f"{'Strat' if config['creative_strategies'] else 'No-Strat'} | "
                f"{config['evolution_mode']} | "
                f"{config['reasoning_effort']} | "
                f"{'Sum' if config['use_summary'] else 'No-Sum'}"
            )

            # Choose color and line style
            color = colors[i % len(colors)]
            linestyle = "-" if config["creative_strategies"] else "--"

            # Plot the line
            (line,) = plt.plot(
                generations,
                length_values,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )

            # Add error band
            plt.fill_between(
                generations,
                [v - s for v, s in zip(length_values, length_stds)],
                [v + s for v, s in zip(length_values, length_stds)],
                color=color,
                alpha=0.1,
            )

            # Add to legend mapping
            legend_handles.append(line)
            legend_labels.append(label)

        # Add labels and title
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Genome Length", fontsize=12)
        plt.title(
            f"{domain.upper()} Domain - Genome Length Across All Configurations",
            fontsize=14,
        )

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add legend
        plt.legend(
            handles=legend_handles, labels=legend_labels, loc="best", fontsize=10
        )

        # Adjust y-axis for better comparison
        plt.ylim([max(0, min_length - 50), max_length + 50])

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(
            output_dir, f"{domain}_consolidated_genome_length.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Consolidated genome length plot for {domain} saved to {output_path}")
        plt.close()


def plot_novelty_over_generations(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Plot novelty over generations for each configuration.
    Group similar configurations together for comparison.
    """
    # Group by domain
    domain_configs = defaultdict(list)

    for config_key, experiments in configs.items():
        # Get sample configuration
        config = experiments[0]["config"]
        domain = config["domain"]

        # Aggregate metrics by generation
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Sort by generation
        generations = sorted(agg_metrics.keys())
        gen_values = [agg_metrics[g] for g in generations]

        domain_configs[domain].append(
            {"config": config, "generations": generations, "metrics": gen_values}
        )

    # Create plots for each domain
    for domain, configs_list in domain_configs.items():
        # 1. First plot: Core 2x2 experiments (creative strategies × reasoning effort)
        core_configs = [
            c
            for c in configs_list
            if c["config"]["evolution_mode"] == "creation"
            and c["config"]["use_summary"]
        ]

        if core_configs:
            fig, ax = plt.subplots(figsize=(12, 7))

            for config_data in core_configs:
                config = config_data["config"]
                generations = config_data["generations"]
                metrics = config_data["metrics"]

                # Create label for the line
                label = (
                    f"{'Strat' if config['creative_strategies'] else 'No-Strat'}, "
                    f"{config['reasoning_effort']}"
                )

                # Set line style based on configuration
                linestyle = "-" if config["creative_strategies"] else "--"
                color = "blue" if config["reasoning_effort"] == "high" else "green"

                # Extract values to plot
                mean_values = [m["mean_novelty"] for m in metrics]
                std_values = [m["std_novelty"] for m in metrics]

                # Plot mean with error band
                ax.plot(
                    generations,
                    mean_values,
                    label=label,
                    linestyle=linestyle,
                    color=color,
                    linewidth=2,
                )
                ax.fill_between(
                    generations,
                    [m - s for m, s in zip(mean_values, std_values)],
                    [m + s for m, s in zip(mean_values, std_values)],
                    color=color,
                    alpha=0.2,
                )

            # Add labels and title
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Mean Novelty", fontsize=12)
            ax.set_title(
                f"{domain.upper()} Domain - Core Experiments (Strategies × Reasoning)",
                fontsize=14,
            )

            # Add grid and legend
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="best")

            # Save the plot
            output_path = os.path.join(output_dir, f"{domain}_core_experiments.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Core experiments plot for {domain} saved to {output_path}")
            plt.close()

        # 2. Second plot: Summary and Evolution Mode effects (for best configuration)
        # For simplicity, we'll use all variations here
        variations = [
            c
            for c in configs_list
            if c["config"]["evolution_mode"] == "variation"
            or not c["config"]["use_summary"]
        ]

        # Get the best core configuration
        if core_configs and variations:
            # Find best core config based on final generation novelty
            best_core = max(
                core_configs,
                key=lambda c: c["metrics"][-1]["mean_novelty"] if c["metrics"] else 0,
            )

            best_config = best_core["config"]

            # Plot best core config and its variations
            fig, ax = plt.subplots(figsize=(12, 7))

            # First plot the best core configuration
            generations = best_core["generations"]
            metrics = best_core["metrics"]
            mean_values = [m["mean_novelty"] for m in metrics]
            std_values = [m["std_novelty"] for m in metrics]

            label = f"Best Core: Strat={best_config['creative_strategies']}, Reasoning={best_config['reasoning_effort']}"
            ax.plot(
                generations,
                mean_values,
                label=label,
                linestyle="-",
                color="blue",
                linewidth=2,
            )
            ax.fill_between(
                generations,
                [m - s for m, s in zip(mean_values, std_values)],
                [m + s for m, s in zip(mean_values, std_values)],
                color="blue",
                alpha=0.2,
            )

            # Now plot the variations
            for config_data in variations:
                config = config_data["config"]

                # Create a descriptive label
                if not config["use_summary"]:
                    label = "No Summary"
                    color = "red"
                else:
                    label = "Variation Mode"
                    color = "green"

                generations = config_data["generations"]
                metrics = config_data["metrics"]
                mean_values = [m["mean_novelty"] for m in metrics]
                std_values = [m["std_novelty"] for m in metrics]

                ax.plot(
                    generations,
                    mean_values,
                    label=label,
                    linestyle="--",
                    color=color,
                    linewidth=2,
                )
                ax.fill_between(
                    generations,
                    [m - s for m, s in zip(mean_values, std_values)],
                    [m + s for m, s in zip(mean_values, std_values)],
                    color=color,
                    alpha=0.2,
                )

            # Add labels and title
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Mean Novelty", fontsize=12)
            ax.set_title(
                f"{domain.upper()} Domain - Effect of Summary & Evolution Mode",
                fontsize=14,
            )

            # Add grid and legend
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="best")

            # Save the plot
            output_path = os.path.join(output_dir, f"{domain}_variations.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Variations plot for {domain} saved to {output_path}")
            plt.close()


def plot_strategy_metrics(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Plot the effectiveness of different creative strategies.
    """
    # Collect strategy metrics from all configurations that use creative strategies
    strategy_metrics = defaultdict(list)

    for config_key, experiments in configs.items():
        config = experiments[0]["config"]

        # Skip if not using creative strategies
        if not config["creative_strategies"]:
            continue

        # Get aggregated metrics
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Get final generation metrics
        final_gen = max(agg_metrics.keys())
        metrics = agg_metrics[final_gen]

        # Collect strategy metrics if available
        if "strategy_metrics" in metrics:
            for strategy, stats in metrics["strategy_metrics"].items():
                if strategy == "None":
                    continue
                strategy_metrics[strategy].append(
                    {
                        "domain": config["domain"],
                        "avg_novelty": stats["avg_novelty"],
                        "std_novelty": stats.get("std_novelty", 0),
                        "count": stats["count"],
                    }
                )

    # Group by domain
    domains = set(
        item["domain"] for items in strategy_metrics.values() for item in items
    )

    for domain in domains:
        # Aggregate strategy metrics for this domain
        domain_strategy_metrics = {}

        for strategy, metrics_list in strategy_metrics.items():
            # Filter to this domain
            domain_metrics = [m for m in metrics_list if m["domain"] == domain]

            if domain_metrics:
                novelty_values = [m["avg_novelty"] for m in domain_metrics]
                count_values = [m["count"] for m in domain_metrics]

                domain_strategy_metrics[strategy] = {
                    "avg_novelty": np.mean(novelty_values),
                    "std_novelty": np.std(novelty_values),
                    "count": np.mean(count_values),
                    "sample_count": len(domain_metrics),
                }

        # Plot if we have metrics
        if domain_strategy_metrics:
            # Sort strategies by average novelty
            sorted_strategies = sorted(
                domain_strategy_metrics.items(),
                key=lambda x: x[1]["avg_novelty"],
                reverse=True,
            )

            # Extract data for plotting
            strategy_names = [s for s, _ in sorted_strategies]
            novelty_means = [m["avg_novelty"] for _, m in sorted_strategies]
            novelty_stds = [m["std_novelty"] for _, m in sorted_strategies]
            counts = [m["count"] for _, m in sorted_strategies]

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Set bar width and positions
            bar_width = 0.7
            positions = np.arange(len(strategy_names))

            # Create bars with error bars
            bars = ax.bar(
                positions,
                novelty_means,
                bar_width,
                yerr=novelty_stds,
                capsize=5,
                color="skyblue",
                edgecolor="navy",
                alpha=0.8,
            )

            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + novelty_stds[i] + 0.005,
                    f"n≈{count:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Add labels and title
            ax.set_xticks(positions)
            ax.set_xticklabels(strategy_names, rotation=45, ha="right")
            ax.set_xlabel("Strategy", fontsize=12)
            ax.set_ylabel("Average Novelty", fontsize=12)
            ax.set_title(
                f"{domain.upper()} Domain - Effectiveness of Creative Strategies",
                fontsize=14,
            )

            # Add grid for readability
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(output_dir, f"{domain}_strategy_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Strategy comparison plot for {domain} saved to {output_path}")
            plt.close()


def plot_normalized_comparison(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create a bar chart showing percentage increase over baseline configuration.
    This makes relative differences more apparent.
    """
    # Extract final generation metrics for each configuration
    domain_metrics = defaultdict(list)

    for config_key, experiments in configs.items():
        # Get sample configuration
        config = experiments[0]["config"]
        domain = config["domain"]

        # Aggregate metrics
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Get the final generation metrics
        if agg_metrics:
            final_gen = max(agg_metrics.keys())
            metrics = agg_metrics[final_gen]

            # Create a more readable configuration label
            config_label = (
                f"{'Strat' if config['creative_strategies'] else 'No-Strat'}, "
                f"{config['evolution_mode']}, "
                f"{config['reasoning_effort']}, "
                f"{'Sum' if config['use_summary'] else 'No-Sum'}"
            )

            domain_metrics[domain].append(
                {
                    "config": config,
                    "config_label": config_label,
                    "mean_novelty": metrics["mean_novelty"],
                    "std_novelty": metrics.get("std_novelty", 0),
                    "sample_count": metrics.get("sample_count", 1),
                }
            )

    # Create a plot for each domain
    for domain, metrics in domain_metrics.items():
        if not metrics:
            continue

        # Sort by mean novelty (ascending to find baseline)
        metrics.sort(key=lambda x: x["mean_novelty"])

        # Set the baseline as the worst-performing configuration
        baseline = metrics[0]["mean_novelty"]
        baseline_label = metrics[0]["config_label"]

        # Calculate percentage increases
        for m in metrics:
            m["percent_increase"] = ((m["mean_novelty"] - baseline) / baseline) * 100

        # Sort by percentage increase (highest first for visualization)
        metrics.sort(key=lambda x: x["percent_increase"], reverse=True)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Extract data for plotting
        config_labels = [m["config_label"] for m in metrics]
        percent_increases = [m["percent_increase"] for m in metrics]

        # Add absolute values to the config labels
        for i, m in enumerate(metrics):
            config_labels[i] = f"{config_labels[i]}\n(novelty: {m['mean_novelty']:.4f})"

        # Create bars
        bar_positions = np.arange(len(config_labels))
        bars = ax.bar(
            bar_positions,
            percent_increases,
            color=["green" if p > 0 else "red" for p in percent_increases],
            edgecolor="navy",
            alpha=0.8,
        )

        # Add annotations for percentage increases
        for i, value in enumerate(percent_increases):
            ax.annotate(
                f"{value:.1f}%",
                xy=(i, value + 0.5),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            # Add sample count
            ax.annotate(
                f"n={metrics[i]['sample_count']}",
                xy=(i, value / 2),  # Place in middle of bar
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 10 else "black",  # Ensure readability
            )

        # Add baseline indicator
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.annotate(
            f"Baseline: {baseline_label} (novelty: {baseline:.4f})",
            xy=(0, -5),
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

        # Label the axes
        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("% Increase in Novelty over Baseline", fontsize=12)
        ax.set_title(
            f"Relative Improvement in Novelty - {domain.upper()} Domain", fontsize=14
        )

        # Set x-tick positions and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(config_labels, fontsize=9)

        # Add grid for readability
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Set y-axis to start slightly below 0 to show the baseline annotation
        ax.set_ylim(bottom=-7)

        # Adjust layout and save
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{domain}_normalized_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Normalized comparison plot for {domain} saved to {output_path}")
        plt.close()


def plot_strategy_artifact_counts(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create a bar chart showing the number of artifacts produced by each creative strategy.
    Only consider experiments that used creative strategies.
    """
    # Collect strategy metrics from experiments that used creative strategies
    domain_strategy_counts = defaultdict(lambda: defaultdict(int))
    domain_experiment_counts = defaultdict(int)

    for config_key, experiments in configs.items():
        config = experiments[0]["config"]
        domain = config["domain"]

        # Skip if not using creative strategies
        if not config.get("creative_strategies", False):
            continue

        domain_experiment_counts[domain] += 1

        # Process each experiment
        for experiment in experiments:
            # Aggregate metrics by generation to get the most recent data
            agg_metrics = aggregate_metrics_by_generation([experiment])

            if not agg_metrics:
                continue

            # Get the final generation metrics
            final_gen = max(agg_metrics.keys())
            metrics = agg_metrics[final_gen]

            # Extract strategy counts
            if "strategy_metrics" in metrics:
                for strategy, stats in metrics["strategy_metrics"].items():
                    if strategy == "None":
                        continue
                    count = stats.get("count", 0)
                    domain_strategy_counts[domain][strategy] += count

    # Create a plot for each domain
    for domain, strategy_counts in domain_strategy_counts.items():
        if not strategy_counts:
            print(f"No strategy count data found for {domain} domain")
            continue

        # Sort strategies by count (descending)
        sorted_strategies = sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Extract data for plotting
        strategy_names = [s for s, _ in sorted_strategies]
        counts = [c for _, c in sorted_strategies]

        # Use a colorful palette for better distinction
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(strategy_names)))

        # Set up the plot
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create bars
        bar_positions = np.arange(len(strategy_names))
        bars = ax.bar(
            bar_positions, counts, color=colors, width=0.7, edgecolor="none", alpha=0.85
        )

        # Add count annotations on top of bars
        for i, count in enumerate(counts):
            ax.annotate(
                f"{count}",
                xy=(i, count + 0.5),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

            # Add percentage of total
            total_count = sum(counts)
            percentage = (count / total_count) * 100

            ax.annotate(
                f"({percentage:.1f}%)",
                xy=(i, count / 2),  # Place in middle of bar
                ha="center",
                va="center",
                fontsize=9,
                color="white" if count > 5 else "#333333",  # Ensure readability
            )

        # Add labels and title
        ax.set_xlabel("Creative Strategy", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel(
            "Number of Artifacts", fontsize=12, fontweight="bold", labelpad=10
        )

        experiment_text = f"From {domain_experiment_counts[domain]} experiments"
        ax.set_title(
            f"Artifacts Produced by Each Creative Strategy - {domain.upper()} Domain\n{experiment_text}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Set x-tick positions and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(strategy_names, fontsize=9, rotation=45, ha="right")

        # Cleaner grid
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)

        # Add total count as text
        fig.text(
            0.5,
            0.01,
            f"Total artifacts: {total_count}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
            style="italic",
        )

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])

        output_path = os.path.join(output_dir, f"{domain}_strategy_artifact_counts.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Strategy artifact count plot for {domain} saved to {output_path}")
        plt.close()

        # Also create a pie chart alternative
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=None,
            autopct="",
            colors=colors,
            wedgeprops={"width": 0.6, "edgecolor": "w", "linewidth": 1},
            startangle=90,
        )

        # Add percentage and count labels inside wedges
        for i, autotext in enumerate(autotexts):
            count = counts[i]
            percentage = (count / total_count) * 100
            autotext.set_text(f"{percentage:.1f}%\n({count})")

        # Add legend outside the pie
        ax.legend(
            wedges,
            strategy_names,
            title="Creative Strategies",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        ax.set_title(
            f"Artifacts by Creative Strategy - {domain.upper()} Domain\n{experiment_text}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()

        pie_output_path = os.path.join(
            output_dir, f"{domain}_strategy_artifact_pie.png"
        )
        plt.savefig(pie_output_path, dpi=300, bbox_inches="tight")
        print(f"Strategy artifact pie chart for {domain} saved to {pie_output_path}")
        plt.close()


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
    plot_configuration_comparison(configs, output_dir)
    plot_normalized_comparison(configs, output_dir)  # Add this line
    plot_consolidated_novelty_and_length(configs, output_dir)
    plot_novelty_over_generations(configs, output_dir)
    plot_strategy_metrics(configs, output_dir)

    print(f"Analysis complete. Results saved to {output_dir}")


def plot_normalized_comparison(configs: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Create a bar chart showing percentage increase over baseline configuration.
    Simplified version with consistent colors and improved spacing.
    """
    # Extract final generation metrics for each configuration
    domain_metrics = defaultdict(list)

    for config_key, experiments in configs.items():
        # Get sample configuration
        config = experiments[0]["config"]
        domain = config["domain"]

        # Aggregate metrics
        agg_metrics = aggregate_metrics_by_generation(experiments)

        # Get the final generation metrics
        if agg_metrics:
            final_gen = max(agg_metrics.keys())
            metrics = agg_metrics[final_gen]

            # Create a more readable configuration label
            config_label = (
                f"{'Strategies ON' if config['creative_strategies'] else 'No Strategies'}\n"
                f"{config['evolution_mode'].capitalize()}, {config['reasoning_effort'].capitalize()}\n"
                f"{'Summary ON' if config['use_summary'] else 'No Summary'}"
            )

            domain_metrics[domain].append(
                {
                    "config": config,
                    "config_label": config_label,
                    "mean_novelty": metrics["mean_novelty"],
                    "std_novelty": metrics.get("std_novelty", 0),
                    "sample_count": metrics.get("sample_count", 1),
                }
            )

    # Create a plot for each domain
    for domain, metrics in domain_metrics.items():
        if not metrics:
            continue

        # Sort by mean novelty (ascending to find baseline)
        metrics.sort(key=lambda x: x["mean_novelty"])

        # Set the baseline as the worst-performing configuration
        baseline = metrics[0]["mean_novelty"]
        baseline_label = metrics[0]["config_label"]
        baseline_novelty = metrics[0]["mean_novelty"]

        # Calculate percentage increases
        for m in metrics:
            m["percent_increase"] = ((m["mean_novelty"] - baseline) / baseline) * 100

        # Sort by percentage increase (highest first for visualization)
        metrics.sort(key=lambda x: x["percent_increase"], reverse=True)

        # Set up the plot with a clean style
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        # Extract data for plotting
        config_labels = [m["config_label"] for m in metrics]
        percent_increases = [m["percent_increase"] for m in metrics]
        novelty_values = [m["mean_novelty"] for m in metrics]

        # Use a single consistent color for all bars
        bar_color = "#3498db"  # A nice blue color

        # Create bars with improved aesthetics
        bar_positions = np.arange(len(config_labels))
        bars = ax.bar(
            bar_positions,
            percent_increases,
            color=bar_color,
            width=0.7,
            edgecolor="none",
            alpha=0.85,
        )

        # Add annotations for percentage increases
        for i, value in enumerate(percent_increases):
            ax.annotate(
                f"{value:.1f}%",
                xy=(i, value + 0.1),  # Reduced spacing
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

            # Add novelty value with minimal spacing
            ax.annotate(
                f"novelty: {novelty_values[i]:.4f}",
                xy=(i, value / 3),  # Place in lower part of bar
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 5 else "#333333",  # Ensure readability
            )

        # Add baseline indicator
        ax.axhline(y=0, color="#555555", linestyle="-", linewidth=1, alpha=0.5)

        # Add baseline text at bottom
        baseline_text = f"Baseline: {baseline_label.replace('\n', ' ')} (novelty: {baseline_novelty:.4f})"
        fig.text(
            0.5,
            0.01,
            baseline_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#e74c3c",
            style="italic",
        )

        # Labels and title
        ax.set_xlabel("Configuration", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel(
            "% Improvement over Baseline", fontsize=12, fontweight="bold", labelpad=10
        )
        ax.set_title(
            f"Novelty Improvement - {domain.upper()} Domain",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Set x-tick positions and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(config_labels, fontsize=10)

        # Cleaner grid
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)

        # Set y-axis to start at 0
        max_value = max(percent_increases) if percent_increases else 0
        y_margin = max_value * 0.1  # 10% margin
        ax.set_ylim(bottom=-1, top=max_value + y_margin)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Make room for baseline annotation

        output_path = os.path.join(output_dir, f"{domain}_normalized_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Normalized comparison plot for {domain} saved to {output_path}")
        plt.close()


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
    # plot_configuration_comparison(configs, output_dir)
    # plot_normalized_comparison(configs, output_dir)
    # plot_consolidated_novelty_and_length(configs, output_dir)
    # plot_novelty_over_generations(configs, output_dir)
    # plot_strategy_metrics(configs, output_dir)
    plot_strategy_artifact_counts(configs, output_dir)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
