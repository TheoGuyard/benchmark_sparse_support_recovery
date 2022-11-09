import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_features, n_nnz": [
            (500, 1000, 10),
            (1000, 500, 10),
        ],
        "M": [1., 2., 3.],
    }

    def __init__(self, n_samples=10, n_features=50, n_nnz=2, M=1., random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_nnz = n_nnz
        self.M = M
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        w = np.zeros(self.n_features)
        s = (
            np.linspace(0, self.n_features - 1, num=self.n_nnz)
            .round()
            .astype(int)
        )
        w[s] = np.sign(rng.randn(self.n_nnz))
        X = rng.randn(self.n_samples, self.n_features)
        X /= np.linalg.norm(X, 2, axis=0)
        y = X @ w
        e = rng.randn(self.n_samples)
        e *= np.sqrt(w.T @ w) / np.sqrt(10.0 * (e.T @ e))
        y += e
        M = self.M
        return dict(X=X, y=y, M=M)
