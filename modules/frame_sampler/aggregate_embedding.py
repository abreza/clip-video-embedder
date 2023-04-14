import numpy as np


def aggregate_embeddings(embeddings, strategy="mean"):
    if strategy == "mean":
        return np.mean(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
