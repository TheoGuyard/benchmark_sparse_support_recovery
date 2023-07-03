import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {"n_samples, n_features, n_nnz": [(10, 30, 2)]}

    def __init__(self, n_samples=10, n_features=50, n_nnz=2, random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_nnz = n_nnz
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        w = np.zeros(self.n_features)
        s = np.linspace(0, self.n_features-1, num=self.n_nnz).round().astype(int)
        w[s] = np.random.randn(self.n_nnz)
        w[s] += np.sign(w[s])
        X = rng.randn(self.n_samples, self.n_features)
        X /= np.linalg.norm(X, 2, axis=0)
        y = X @ w
        e = rng.randn(self.n_samples)
        e *= np.sqrt(w.T @ w) / np.sqrt(10. * (e.T @ e))
        y += e
        M = np.linalg.norm(w, np.inf)
        return dict(X=X, y=y, M=M)
