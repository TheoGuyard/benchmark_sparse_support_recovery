import numpy as np

from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        "n_samples, n_features, n_nnz, rho": [
            (50, 100, 5, 0.1),
            (50, 100, 5, 0.9),
            (50, 100, 5, 0.99),
        ],
    }

    def __init__(
        self,
        n_samples=10,
        n_features=50,
        n_nnz=2,
        rho=0.9,
        snr=10,
        random_state=27,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_nnz = n_nnz
        self.rho = rho
        self.snr = snr
        self.random_state = random_state

    def get_data(self):
        X, y, w_true = make_correlated_data(
            n_samples=self.n_samples,
            n_features=self.n_features,
            rho=self.rho,
            snr=self.snr,
            density=self.n_nnz / self.n_features,
            random_state=self.random_state,
        )

        M = np.linalg.norm(w_true, np.inf)
        return dict(X=X, y=y, w_true=w_true, M=M)
