import argparse
import os
import json

from src.output_utils import load_population_data
from src.artifacts.load_artifacts import get_artifact_class


def main():
    parser = argparse.ArgumentParser(
        description="Post-process artifacts in the results directory"
    )
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    args = parser.parse_args()

    generations = load_population_data(args.results_dir)

    # Load config.json to get the artifact_class
    config_path = os.path.join(args.results_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(config["artifact_class"])
    ArtifactClass = get_artifact_class(config)

    output_dir = os.path.join(args.results_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for id in generations[-1]:
        artifact = ArtifactClass.load(id, os.path.join(args.results_dir, "artifacts"))
        artifact.post_process(output_dir)
        print("Post-processed artifact", id)


if __name__ == "__main__":
    main()
