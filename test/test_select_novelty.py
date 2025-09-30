import torch


def test_select_by_novelty_basic():
    # Create synthetic embeddings for 5 items in 3D
    # Two clusters far apart to test novelty ordering
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    from src.population import Population

    pop = Population()
    # Simulate artifacts list length for the selection logic edge cases
    class Dummy:
        def __init__(self, i):
            self.id = f"a{i}"

    for i in range(emb.shape[0]):
        pop.add(Dummy(i))

    # Ask for k=2 neighbors
    order, distances = pop.select_by_novelty(emb, k_neighbors=2, return_distances=True)

    assert isinstance(order, list)
    assert len(order) == emb.shape[0]
    assert distances.shape[0] == emb.shape[0]

    # The singleton-ish point near [0,0,1] should be among the most novel
    assert order[0] in {4, 2, 0}


