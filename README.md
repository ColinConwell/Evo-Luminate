# Evo-Luminate

Evo-Luminate is an evolutionary computation framework that evolves artifacts using generative AI. It enables running experiments with various parameters to guide the evolutionary process.

## Installation

```bash
# Clone the repository
git clone https://github.com/joel-simon/lluminate.git
cd lluminate

# Install package with dependencies
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,analysis]"
```

### macOS Apple Silicon Support

This project supports Apple Silicon (M1/M2/M3) Macs using Metal Performance Shaders (MPS). The code will automatically detect and use MPS acceleration when running on Apple Silicon hardware. No additional configuration is needed.

When you run the application, you should see:

```bash
Using PyTorch device: mps
```

Note: On Apple Silicon, Metal Performance Shaders provide significant speedups for tensor operations compared to CPU execution.

### Renderers Installation

Install all renderers using the setup script:

```bash
bash scripts/setup/install-renderers.sh
```

Or with just:

```bash
just setup
```

#### System Requirements for Renderers

The renderers require some system dependencies.
On macOS, install them using Homebrew:

```bash
brew install pkg-config cairo pango libpng jpeg giflib librsvg pixman
```

Note: Node.js 18.x is required for the renderers to work properly.

## Usage

Run an evolutionary experiment using the main script:

```bash
python scripts/main_experiment.py --prompt "Your creative prompt here" --artifact_class shader --num_generations 20
```

Or with just:

```bash
just experiment --prompt "Your creative prompt here" --artifact_class shader --num_generations 20
```

### Command Line Arguments

| Argument                    | Description                                  | Default           |
| --------------------------- | -------------------------------------------- | ----------------- |
| `--output_name`             | Name of output directory                     | Current timestamp |
| `--random_seed`             | Random seed for reproducibility              | 42                |
| `--prompt`                  | Prompt for the experiment                    | ""                |
| `--initial_population_size` | Size of the initial population               | 20                |
| `--population_size`         | Size of the population in each generation    | 20                |
| `--children_per_generation` | Number of children to create per generation  | 10                |
| `--num_generations`         | Number of generations to run                 | 20                |
| `--k_neighbors`             | Number of neighbors for selection            | 3                 |
| `--max_workers`             | Maximum number of parallel workers           | 5                 |
| `--artifact_class`          | Class of artifact to evolve                  | "shader"          |
| `--evolution_mode`          | Mode of evolution                            | "variation"       |
| `--reasoning_effort`        | Level of reasoning effort                    | "low"             |
| `--no_strategies`           | Disable use of creative strategies           | False             |
| `--no_summary`              | Disable summary usage                        | False             |
| `--crossover_rate`          | Probability of crossover during reproduction | 0.3               |

## Examples

```bash
# Run an evolution experiment with a custom prompt and 30 generations
just experiment --prompt "Create a sunset shader with mountain silhouettes" --num_generations 30 --reasoning_effort high

# Run an ablation study
just ablation --output_dir results --seeds 42,43,44

# Plot novelty metrics for a results directory
just plot-novelty results/ShaderArtifact_20251010_014010

# Analyze ablation study results
just analyze results/ablation_study
```

## Project Structure

```
evo-luminate/
├── eluminate/              # Core library
│   ├── artifacts/          # Artifact implementations
│   ├── analysis_utils.py   # Analysis utilities
│   ├── run_evolution_experiment.py
│   └── ...
├── scripts/                # Executable scripts
│   ├── main_experiment.py  # Main entry point
│   ├── run_experiments.py  # Ablation studies
│   ├── plot_results.py     # Visualization tools
│   ├── analyze_results.py  # Analysis tools
│   └── setup/              # Setup scripts
├── testing/                # Test suite
├── results/                # Generated artifacts and experiment results
├── justfile                # Task runner commands
└── pyproject.toml          # Package configuration
```

## Output

Results are saved in the `results/` directory, organized by artifact class and timestamp (or custom name if provided). Each experiment creates its own directory with:

- Generated artifacts for each generation
- Logs of the evolutionary process
- Summary of experiment parameters and results

## License

Apache 2.0
