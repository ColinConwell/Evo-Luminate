# Evo-Luminate Task Runner
# Install `just` from: https://github.com/casey/just

# Show available commands
default:
    @just --list

# Setup: Install renderer dependencies
setup:
    bash scripts/setup/install-renderers.sh

# Run a single evolution experiment
experiment *ARGS:
    python scripts/main_experiment.py {{ARGS}}

# Run ablation study with multiple experiments
ablation *ARGS:
    python scripts/run_experiments.py {{ARGS}}

# Evolve clock artifacts (demo)
clocks *ARGS:
    python scripts/evolve_clocks.py {{ARGS}}

# Plot novelty metrics for a results directory
plot-novelty RESULTS_DIR *ARGS:
    python scripts/plot_results.py novelty {{RESULTS_DIR}} {{ARGS}}

# Plot UMAP colored by generation
plot-umap RESULTS_DIR *ARGS:
    python scripts/plot_results.py umap-generations {{RESULTS_DIR}} {{ARGS}}

# Create UMAP grid visualization
plot-grid RESULTS_DIR *ARGS:
    python scripts/plot_results.py umap-grid {{RESULTS_DIR}} {{ARGS}}

# Analyze ablation study results
analyze STUDY_DIR *ARGS:
    python scripts/analyze_results.py {{STUDY_DIR}} {{ARGS}}

# Run tests
test:
    pytest testing/

# Run specific test file
test-file FILE:
    pytest testing/{{FILE}}

# Clean up results and cached files
clean:
    rm -rf results/__pycache__
    rm -rf testing/__pycache__
    rm -rf eluminate/__pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Format code with black (if installed)
fmt:
    @command -v black >/dev/null 2>&1 && black eluminate/ scripts/ testing/ || echo "black not installed, skipping"

# Lint code with ruff (if installed)
lint:
    @command -v ruff >/dev/null 2>&1 && ruff check eluminate/ scripts/ testing/ || echo "ruff not installed, skipping"

