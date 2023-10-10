from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):
    name = "simulated"
    parameters = {
        "n_samples, n_features, density, rho, snr, random_state": [
            (20, 50, 0.1, 0.9, 10.0, None),
        ],
    }

    def __init__(self, **params):
        self.params = params

    def get_data(self):
        X, y, w_true = make_correlated_data(**self.params)
        return dict(X=X, y=y, w_true=w_true)
