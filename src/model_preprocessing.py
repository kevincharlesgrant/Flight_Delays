from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class ExplicitSparseNumeric(BaseEstimator, TransformerMixin):
    """Keep numeric zeros explicit so XGBoost does not read them as sparse missing values."""

    def fit(self, x, y=None):
        self.feature_names_in_ = np.asarray(getattr(x, "columns", np.arange(x.shape[1])), dtype=object)
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        rows, columns = values.shape
        indices = np.tile(np.arange(columns, dtype=np.int32), rows)
        indptr = np.arange(0, (rows + 1) * columns, columns, dtype=np.int64)
        return sparse.csr_matrix((values.ravel(), indices, indptr), shape=values.shape)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)
