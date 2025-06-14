from sklearn.utils import resample
import numpy as np
import pandas as pd

class EmbeddingOversampler:
    def __init__(self, label_col='label', target_label='ASD'):
        self.label_col = label_col
        self.target_label = target_label

    def oversample(self, X_embed: np.ndarray, X_ling: np.ndarray, y: np.ndarray):
        idx_minority = np.where(y == 1)[0]
        idx_majority = np.where(y == 0)[0]

        n_majority = len(idx_majority)
        n_minority = len(idx_minority)

        X_embed_min = X_embed[idx_minority]
        X_ling_min = X_ling[idx_minority]
        y_min = y[idx_minority]

        X_embed_os, X_ling_os, y_os = resample(
            X_embed_min, X_ling_min, y_min,
            replace=True,
            n_samples=n_majority,
            random_state=42
        )

        X_embed_final = np.vstack([X_embed[idx_majority], X_embed_os])
        X_ling_final = np.vstack([X_ling[idx_majority], X_ling_os])
        y_final = np.concatenate([y[idx_majority], y_os])

        return X_embed_final, X_ling_final, y_final
