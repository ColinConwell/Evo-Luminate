import os
import argparse
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run an evolutionary experiment")
    parser.add_argument(
        "--output_name",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Name of output directory (default: timestamp)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt for the experiment",
    )
    parser.add_argument(
        "--initial_population_size",
        type=int,
        default=20,
        help="Size of the initial population",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=20,
        help="Size of the population in each generation",
    )
    parser.add_argument(
        "--children_per_generation",
        type=int,
        default=10,
        help="Number of children to create per generation",
    )
    parser.add_argument(
        "--num_generations", type=int, default=20, help="Number of generations to run"
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=3, help="Number of neighbors for selection"
    )
    parser.add_argument(
        "--max_workers", type=int, default=5, help="Maximum number of parallel workers"
    )

    parser.add_argument(
        "--artifact_class",
        type=str,
        default="ShaderArtifact",
        help="Class of artifact to evolve",
    )
    parser.add_argument(
        "--evolution_mode", type=str, default="variation", help="Mode of evolution"
    )
    parser.add_argument(
        "--reasoning_effort", type=str, default="low", help="Level of reasoning effort"
    )
    parser.add_argument(
        "--no_strategies",
        action="store_true",
        default=False,
        help="Whether to use creative strategies",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        default=False,
        help="Disable summary usage",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.3,
        help="Probability of crossover during reproduction",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=False,
        help="Force CPU device (workaround for platform-specific issues)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create a timestamped directory for this run
    output_dir = os.path.join("results", f"{args.artifact_class}_{args.output_name}")
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists")
        exit(1)

    # Set conservative thread env to avoid native mutex issues on some macOS setups
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Allow CPU fallback for unsupported MPS ops to avoid native mutex issues
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Workaround for some macOS + Objective-C frameworks during fork
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    # Ensure safer multiprocessing behavior
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Log PyTorch device (CUDA, MPS, or CPU)
    from eluminate.utils import get_device

    device = get_device(cpu_only=args.force_cpu)
    print(f"Using PyTorch device: {device}")

    # Lazily import the heavy experiment after environment setup
    from eluminate.run_evolution_experiment import run_evolution_experiment

    # Run the experiment with the parsed arguments
    run_evolution_experiment(
        output_dir=output_dir,
        config={
            "random_seed": args.random_seed,
            "prompt": args.prompt,
            "initial_population_size": args.initial_population_size,
            "population_size": args.population_size,
            "children_per_generation": args.children_per_generation,
            "num_generations": args.num_generations,
            "k_neighbors": args.k_neighbors,
            "max_workers": args.max_workers,
            "artifact_class": args.artifact_class,
            "evolution_mode": args.evolution_mode,
            "reasoning_effort": args.reasoning_effort,
            "use_creative_strategies": not args.no_strategies,
            "use_summary": not args.no_summary,
            "crossover_rate": args.crossover_rate,
        },
    )
