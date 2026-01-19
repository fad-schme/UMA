"""
Episode clusterer for consolidation.

This module groups episodes by embedding similarity so they can be
summarized together during consolidation.

Coding agent instructions:
--------------------------
- The current implementation is a production-ready, incremental
  centroid-based clustering using cosine similarity.
- It is designed for consolidation batch sizes (O(N^2) is acceptable
  for a few hundred episodes).
- Episodes without embeddings are always placed in singleton clusters.
- If you need more scalable clustering for very large histories,
  you may replace the core logic with:
    - FAISS-based k-means
    - Agglomerative clustering
    - DBSCAN/HDBSCAN
  while preserving the public API:
    cluster(episodes: List[Episode]) -> List[List[Episode]]
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ...types_episode import Episode

logger = logging.getLogger(__name__)


class EpisodeClusterer:
    """
    Clusters episodes based on cosine similarity between embeddings.

    Strategy
    --------
    - Maintain clusters with a running centroid (in normalized embedding space).
    - For each episode with a valid embedding:
        - Compute cosine similarity to each cluster centroid.
        - Assign to the first cluster whose similarity >= threshold,
          preferring the highest similarity.
        - If none meet the threshold, create a new cluster.
    - Episodes without an embedding are always placed into singleton clusters.

    Notes
    -----
    - This is O(N^2) in the number of episodes, but consolidation is
      typically run on a bounded window (max_episodes_per_cycle), so
      this is acceptable in practice.
    - All embeddings are normalized before similarity computation.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        """
        Parameters
        ----------
        similarity_threshold:
            Minimum cosine similarity required to join an existing cluster.
        """
        self.threshold = float(similarity_threshold)
        logger.info(
            "EpisodeClusterer initialized with similarity_threshold=%.2f",
            self.threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cluster(self, episodes: List[Episode]) -> List[List[Episode]]:
        """
        Cluster episodes into groups based on embedding cosine similarity.

        Parameters
        ----------
        episodes:
            List of Episode objects. Each may or may not have an embedding.

        Returns
        -------
        List[List[Episode]]
            A list of clusters, each cluster being a list of Episode
            instances. Episodes without embeddings are placed in their
            own singleton clusters.
        """
        if not episodes:
            return []

        clusters: List[List[Episode]] = []
        cluster_centroids: List[Optional[np.ndarray]] = []
        cluster_sizes: List[int] = []

        embedded_count = 0
        no_embedding_count = 0

        for ep in episodes:
            emb = self._extract_embedding(ep)
            if emb is None:
                # No usable embedding: singleton cluster
                clusters.append([ep])
                cluster_centroids.append(None)
                cluster_sizes.append(1)
                no_embedding_count += 1
                continue

            embedded_count += 1
            emb_norm = self._normalize(emb)
            if emb_norm is None:
                # Degenerate embedding (zero norm, etc.)
                clusters.append([ep])
                cluster_centroids.append(None)
                cluster_sizes.append(1)
                no_embedding_count += 1
                continue

            # Find the best matching cluster
            best_idx = -1
            best_sim = -1.0

            for idx, centroid in enumerate(cluster_centroids):
                if centroid is None:
                    # This cluster has no centroid (singleton non-embedded, etc.)
                    continue

                sim = self._cosine_similarity(emb_norm, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            # Assign to cluster or create new
            if best_idx != -1 and best_sim >= self.threshold:
                # Merge into existing cluster
                clusters[best_idx].append(ep)
                size = cluster_sizes[best_idx]
                # Incremental centroid update in normalized space
                new_centroid = (centroid * size + emb_norm) / float(size + 1)
                new_centroid = self._normalize(new_centroid)
                cluster_centroids[best_idx] = new_centroid
                cluster_sizes[best_idx] = size + 1
            else:
                # Start a new cluster for this embedding
                clusters.append([ep])
                cluster_centroids.append(emb_norm)
                cluster_sizes.append(1)

        logger.info(
            "EpisodeClusterer: clustered %d episodes into %d clusters "
            "(embedded=%d, without_embedding=%d)",
            len(episodes),
            len(clusters),
            embedded_count,
            no_embedding_count,
        )

        return clusters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_embedding(self, episode: Episode) -> Optional[np.ndarray]:
        """
        Safely extract an embedding vector from an episode.

        Returns
        -------
        np.ndarray or None
            Returns None if the episode has no embedding or an invalid one.
        """
        emb = getattr(episode, "embedding", None)
        if emb is None:
            return None

        try:
            arr = np.asarray(emb, dtype=float)
        except Exception:
            logger.warning(
                "EpisodeClusterer: failed to convert embedding to array for episode id=%s",
                getattr(episode, "id", "<unknown>"),
            )
            return None

        if arr.ndim != 1 or arr.size == 0:
            logger.warning(
                "EpisodeClusterer: invalid embedding shape=%s for episode id=%s",
                arr.shape,
                getattr(episode, "id", "<unknown>"),
            )
            return None

        return arr

    def _normalize(self, vec: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize a vector to unit length.

        Returns
        -------
        np.ndarray or None
            Normalized vector, or None if the norm is too close to zero.
        """
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return None
        return vec / norm

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.

        Both vectors are assumed to be normalized. If not, the similarity
        will still be in [-1, 1] but may be less interpretable.
        """
        # With normalized vectors, this is just the dot product
        return float(np.dot(v1, v2))