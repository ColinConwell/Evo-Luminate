from typing import Dict, Any
import importlib


def get_artifact_class(config: Dict[str, Any]):
    """Return the appropriate Artifact class based on configuration (lazy import)."""
    artifact_class_name = config.get("artifact_class", "ShaderArtifact")
    name_to_module = {
        "GeneticArtifact": ("src.artifacts.GeneticArtifact", "GeneticArtifact"),
        "ShaderArtifact": ("src.artifacts.ShaderArtifact", "ShaderArtifact"),
        "GameIdeaArtifact": ("src.artifacts.GameIdeaArtifact", "GameIdeaArtifact"),
        "SDFArtifact": ("src.artifacts.SDFArtifact", "SDFArtifact"),
        "ImageGenArtifact": ("src.artifacts.ImageGen", "ImageGenArtifact"),
    }
    if artifact_class_name not in name_to_module:
        raise ValueError(f"Unknown artifact class: {artifact_class_name}")
    module_name, class_name = name_to_module[artifact_class_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
