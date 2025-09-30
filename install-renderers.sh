#!/bin/bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "$0")" && pwd)

echo "Installing renderers (using Node 18 for each install)..."

have_cmd() { command -v "$1" >/dev/null 2>&1; }

run_with_node18() {
  # Run a single command using Node 18 without changing your global Node
  if have_cmd volta; then
    # Volta can run a one-off command with a specific Node
    volta run --node 18 "$@"
  elif have_cmd nvm; then
    # nvm exec uses a specific Node version for a single command
    # shellcheck disable=SC1090
    [ -n "${NVM_DIR:-}" ] && . "$NVM_DIR/nvm.sh" || true
    nvm install 18 >/dev/null
    nvm exec 18 "$@"
  else
    echo "Error: Neither Volta nor nvm found. Please install one of them to use Node 18." >&2
    echo "- Volta: https://volta.sh" >&2
    echo "- nvm: https://github.com/nvm-sh/nvm" >&2
    exit 1
  fi
}

install_one() {
  local dir="$1"
  local name="$2"
  local dir_path="$THIS_DIR/$dir"

  if [ ! -d "$dir_path" ]; then
    echo "Skipping $name (directory not found: $dir_path)"
    return 0
  fi

  echo "==> Installing $name in $dir_path with Node 18"
  pushd "$dir_path" >/dev/null
  run_with_node18 npm install
  popd >/dev/null
}

install_one "src/render-shaders" "render-shaders"
install_one "src/render-sdf" "render-sdf"
install_one "src/render-p5js" "render-p5js"

echo "All renderer installs complete."