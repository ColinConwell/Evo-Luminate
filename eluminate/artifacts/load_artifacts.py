from typing import Dict, Any
import importlib


def get_artifact_class(config: Dict[str, Any]):
    """Return the appropriate Artifact class based on configuration (lazy import)."""
    artifact_class_name = config.get("artifact_class", "ShaderArtifact")
    name_to_module = {
        "GeneticArtifact": ("eluminate.artifacts.GeneticArtifact", "GeneticArtifact"),
        "ShaderArtifact": ("eluminate.artifacts.ShaderArtifact", "ShaderArtifact"),
        "GameIdeaArtifact": ("eluminate.artifacts.GameIdeaArtifact", "GameIdeaArtifact"),
        "SDFArtifact": ("eluminate.artifacts.SDFArtifact", "SDFArtifact"),
        "ImageGenArtifact": ("eluminate.artifacts.ImageGen", "ImageGenArtifact"),
    }
    if artifact_class_name not in name_to_module:
        raise ValueError(f"Unknown artifact class: {artifact_class_name}")
    module_name, class_name = name_to_module[artifact_class_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
