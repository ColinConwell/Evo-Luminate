import torch


def test_get_embeddings_device_float32():
    # Define a dummy Artifact with deterministic tensor embedding
    class DummyArtifact:
        def __init__(self, vec):
            self.vec = torch.tensor(vec, dtype=torch.float64)  # wrong dtype intentionally

        def compute_embedding(self):
            return self.vec

    from src.run_evolution_experiment import get_embeddings
    from src.utils import get_device

    artifacts = [DummyArtifact([1, 0, 0]), DummyArtifact([0, 1, 0])]

    emb = get_embeddings(artifacts)
    assert isinstance(emb, torch.Tensor)
    assert emb.dtype == torch.float32
    # Some backends may return mps:0 vs mps; compare type only
    assert emb.device.type == get_device().type
    assert emb.shape == (2, 3)


