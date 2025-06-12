# LLuminate

LLuminate is an evolutionary computation framework that evolves artifacts using generative AI. It enables running experiments with various parameters to guide the evolutionary process.

## Installation

```bash
# Clone the repository
git clone https://github.com/joel-simon/lluminate.git
cd lluminate

# Install dependencies
pip install -r requirements.txt
```

### macOS Apple Silicon Support

This project supports Apple Silicon (M1/M2/M3) Macs using Metal Performance Shaders (MPS). The code will automatically detect and use MPS acceleration when running on Apple Silicon hardware. No additional configuration is needed.

When you run the application, you should see:

```bash
Using PyTorch device: mps
```

Note: On Apple Silicon, Metal Performance Shaders provide significant speedups for tensor operations compared to CPU execution.

### Renderers Installation

To install certain renderers, cd to one of the subdirectories and run npm install:

```bash
cd src/render-shaders && npm install
cd src/render-sdf && npm install
cd src/render-p5js && npm install
```

#### System Requirements for Renderers

The renderers require some system dependencies.
On macOS, install them using Homebrew:

```bash
brew install pkg-config cairo pango libpng jpeg giflib librsvg pixman
```

Note: Node.js 18.x is required for the renderers to work properly.

## Usage

Run an evolutionary experiment using the `main.py` script:

```bash
python main.py --prompt "Your creative prompt here" --artifact_class shader --num_generations 20
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

## Example

```bash
# Run an evolution experiment with a custom prompt and 30 generations
python main.py --prompt "Create a sunset shader with mountain silhouettes" --num_generations 30 --reasoning_effort high
```

## Project Structure

```
lluminate/
├── main.py                 # Main entry point for running experiments
├── src/
│   ├── run_evolution_experiment.py  # Implementation of the evolution experiment
│   └── ...                 # Other source files
├── results/                # Generated artifacts and experiment results
└── ...
```

## Output

Results are saved in the `results/` directory, organized by artifact class and timestamp (or custom name if provided). Each experiment creates its own directory with:

- Generated artifacts for each generation
- Logs of the evolutionary process
- Summary of experiment parameters and results

## License

Apache 2.0
