# Perform hierarchical clustering
import numpy as np
import random
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cosine


def cosine_distance(a, b):
    """Calculate the cosine distance between two vectors."""
    return cosine(a, b)


class IlluminationArchive:
    def __init__(self, distance_threshold=0.3):
        self.artifacts = []  # All discovered artifacts
        self.distance_threshold = distance_threshold  # For hierarchical clustering
        self.clusters = None  # Cluster assignments
        self.cluster_members = None  # Mapping of cluster IDs to artifacts

    def add_generation(self, new_artifacts):
        """Add new artifacts and calculate their novelty."""
        # Calculate novelty scores for new artifacts
        self.calculate_novelty(new_artifacts)

        # Add to archive
        self.artifacts.extend(new_artifacts)

        # Update clusters
        self.update_clusters()

    # def calculate_novelty(self, candidates, k=5):
    #     """Calculate novelty based on distance to k-nearest neighbors in archive."""
    #     for candidate in candidates:
    #         if not self.artifacts:  # First generation
    #             candidate.novelty_score = 1.0
    #             continue

    #         neighbors = self.find_k_nearest_neighbors(candidate, k)
    #         candidate.novelty_score = sum(dist for _, dist in neighbors) / len(
    #             neighbors
    #         )

    def find_k_nearest_neighbors(self, artifact, k=5):
        """Find k nearest neighbors for an artifact."""
        distances = []
        for other in self.artifacts:
            if other is not artifact:  # Don't compare with self
                dist = cosine_distance(artifact.embedding, other.embedding)
                distances.append((other, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        # Return k nearest
        return distances[: min(k, len(distances))]

    def update_clusters(self):
        """Update cluster assignments using hierarchical clustering."""
        if len(self.artifacts) <= 1:
            # Not enough artifacts to cluster
            self.clusters = [0] * len(self.artifacts)
            self.cluster_members = {0: self.artifacts.copy()} if self.artifacts else {}
            return

        # Compute distance matrix
        embeddings = np.array([a.embedding for a in self.artifacts])

        distances = pdist(embeddings, metric="cosine")
        Z = linkage(distances, method="average")

        # Cut dendrogram at distance threshold
        self.clusters = fcluster(Z, t=self.distance_threshold, criterion="distance")

        # Map artifacts to clusters
        self.cluster_members = {}
        for i, c in enumerate(self.clusters):
            if c not in self.cluster_members:
                self.cluster_members[c] = []
            self.cluster_members[c].append(self.artifacts[i])

    def get_samples(self, n=5, strategy="random"):
        """Get samples using the specified strategy.

        Args:
            n: Number of samples to return
            strategy: One of "random", "random_cluster", or "most_novel"

        Returns:
            List of sampled artifacts
        """
        if len(self.artifacts) <= n:
            return self.artifacts.copy()

        if strategy == "random":
            # Simple random sampling
            return random.sample(self.artifacts, n)

        elif strategy == "random_cluster":
            # Make sure we have clusters
            if not self.cluster_members or len(self.cluster_members) < 2:
                return random.sample(self.artifacts, n)

            # Sample from cluster representatives
            samples = []
            # Get clusters sorted by size (largest first)
            clusters = sorted(
                self.cluster_members.keys(),
                key=lambda c: len(self.cluster_members[c]),
                reverse=True,
            )

            # Take samples from different clusters
            for c in clusters[: min(n, len(clusters))]:
                members = self.cluster_members[c]
                if members:
                    # Get a representative member
                    center = np.mean([m.embedding for m in members], axis=0)
                    representative = min(
                        members, key=lambda x: cosine_distance(x.embedding, center)
                    )
                    samples.append(representative)

            # Fill remaining slots if needed
            if len(samples) < n:
                remaining = [a for a in self.artifacts if a not in samples]
                samples.extend(
                    random.sample(remaining, min(n - len(samples), len(remaining)))
                )

            return samples

        elif strategy == "most_novel":
            # Return the most novel artifacts
            sorted_artifacts = sorted(
                self.artifacts, key=lambda x: x.novelty_score, reverse=True
            )
            return sorted_artifacts[:n]

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
