from typing import Dict, Any
from .Artifact import Artifact
from .ShaderArtifact import ShaderArtifact
from .GameIdeaArtifact import GameIdeaArtifact
from .SdfArtifact import SdfArtifact
from .Ga import GaArtifact


def get_artifact_class(config: Dict[str, Any]) -> Artifact:
    """Return the appropriate Artifact class based on configuration."""
    artifact_class_name = config.get("artifact_class", "ShaderArtifact")
    if artifact_class_name == "ShaderArtifact":
        return ShaderArtifact
    elif artifact_class_name == "GameIdeaArtifact":
        return GameIdeaArtifact
    elif artifact_class_name == "SdfArtifact":
        return SdfArtifact
    elif artifact_class_name == "GaArtifact":
        return GaArtifact
    else:
        raise ValueError(f"Unknown artifact class: {artifact_class_name}")
