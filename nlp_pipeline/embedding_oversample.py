from sklearn.utils import resample
import numpy as np
from collections import Counter
import pandas as pd

class EmbeddingOversampler:
    def oversample(self, X_embed, X_ling, y):
        assert len(X_embed) == len(X_ling) == len(y), "Mismatch input shape!"

        idx_minority = np.where(y == 1)[0]  
        idx_majority = np.where(y == 0)[0]  

        print("Before Oversampling:", Counter(y))

        X_embed_min = X_embed[idx_minority]
        X_ling_min = X_ling[idx_minority]
        y_min = y[idx_minority]

        X_embed_os, X_ling_os, y_os = resample(
            X_embed_min, X_ling_min, y_min,
            replace=True,
            n_samples=len(idx_majority),
            random_state=42
        )

        X_embed_final = np.vstack([X_embed[idx_majority], X_embed_os])
        X_ling_final = np.vstack([X_ling[idx_majority], X_ling_os])
        y_final = np.concatenate([y[idx_majority], y_os])

        print("After Oversampling :", Counter(y_final))
        return X_embed_final, X_ling_final, y_final