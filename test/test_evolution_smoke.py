import torch


class DummyArtifact:
    name = "dummy"

    def __init__(self, vec):
        self.id = f"d{hash(tuple(vec)) % 100000}"
        self.genome = "GENOME"
        self.phenome = None
        self.prompt = None
        self.embedding = torch.tensor(vec, dtype=torch.float32)
        self.metadata = {}

    @classmethod
    def create_from_prompt(cls, prompt: str, output_dir: str, **kwargs):
        # Deterministic small vectors in 3D
        import random

        random.seed(0)
        base = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return cls(base)

    def compute_embedding(self) -> torch.Tensor:
        return self.embedding


def test_run_evolution_small(tmp_path, monkeypatch):
    # Monkeypatch artifact class to dummy to avoid LLM and rendering
    from src import run_evolution_experiment as ree

    def fake_get_artifact_class(config):
        return DummyArtifact

    monkeypatch.setattr("src.run_evolution_experiment.get_artifact_class", fake_get_artifact_class)

    config = {
        "random_seed": 42,
        "prompt": "",
        "initial_population_size": 5,
        "population_size": 5,
        "children_per_generation": 3,
        "num_generations": 1,
        "k_neighbors": 2,
        "max_workers": 2,
        "artifact_class": "DummyArtifact",
        "evolution_mode": "variation",
        "reasoning_effort": "low",
        "use_creative_strategies": False,
        "use_summary": False,
        "crossover_rate": 0.0,
    }

    pop = ree.run_evolution_experiment(output_dir=str(tmp_path), config=config)
    assert len(pop.get_all()) == config["population_size"]

