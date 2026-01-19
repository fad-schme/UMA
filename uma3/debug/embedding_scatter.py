"""
Embedding Scatter Plot Utility (Optional, offline debugging)

Generates scatter plots of embeddings using:
- PCA
- UMAP (if installed)
- TSNE (if installed)

Coding agent instructions:
--------------------------
- Use this offline to inspect memory structure visually.
- Avoid heavy dependencies in production code path.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class EmbeddingScatter:
    """Embedding scatter plot helper.

    Notes
    -----
    - Intended for offline debugging and local inspection only.
    - Avoid importing heavy plotting libs at module import time in production.
    """

    @staticmethod
    def plot_2d(
        vectors: List[List[float]],
        labels: List[str],
        title: str = "Embeddings",
        save_to: Optional[str] = None,
        show: bool = True,
        random_state: Optional[int] = None,
        annotate: bool = True,
    ) -> Tuple[object, List[Tuple[float, float]]]:
        """Plot embeddings in 2D using PCA.

        Parameters
        ----------
        vectors:
            list of embedding vectors
        labels:
            list of labels (same length as vectors)
        save_to:
            if provided, save the figure to this path instead of or in addition to showing
        show:
            if True, call `plt.show()` (interactive). Set False for headless runs.
        random_state:
            optional random_state passed to PCA for reproducibility
        annotate:
            whether to draw text labels for each point

        Returns
        -------
        (fig, coords): Matplotlib figure (or None if plotting libs missing) and list of (x,y) coords
        """
        if not vectors:
            logger.warning("No vectors provided to plot.")
            return None, []

        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except Exception as exc:  # pragma: no cover - optional deps
            logger.error("EmbeddingScatter requires matplotlib and sklearn: %s", exc)
            raise

        pca = PCA(n_components=2, random_state=random_state) if random_state is not None else PCA(n_components=2)
        xy = pca.fit_transform(vectors)

        xs, ys = xy[:, 0], xy[:, 1]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(xs, ys, alpha=0.7)

        if annotate:
            for i, label in enumerate(labels):
                ax.annotate(str(label), (xs[i], ys[i]))

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

        coords = list(zip(xs.tolist(), ys.tolist()))

        if save_to:
            try:
                fig.savefig(save_to, bbox_inches="tight")
                logger.info("Saved embedding scatter to %s", save_to)
            except Exception:
                logger.exception("Failed to save figure to %s", save_to)

        if show and save_to is None:
            plt.show()
        elif show and save_to is not None:
            # allow both saving and showing
            plt.show()

        return fig, coords