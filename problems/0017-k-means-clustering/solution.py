import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    centroids = np.array([list(row) for row in initial_centroids])
    points = np.array([list(row) for row in points])
    i = 0
    while i < max_iterations:
        clusters = {i: [] for i in range(k)}
        for point in points:
            cluster_idx = np.argmin([
                np.linalg.norm(point - centroid) for centroid in centroids
            ])
            clusters[cluster_idx].append(point)

        centroids = [
            np.mean(cluster, axis=0) for cluster in clusters.values()
        ]

        i += 1

    return [tuple(elem) for elem in np.round(centroids, 4).tolist()]