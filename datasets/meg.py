from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml
    import numpy as np
    from benchmark_utils.datasets import generate_sources


class Dataset(BaseDataset):
    name = "meg"
    parameters = {
        "k, snr, semi_simulated, random_state": [
            (None, None, False, None),
            (0.001, 100.0, True, None),
        ],
    }

    def __init__(self, k, snr, semi_simulated, random_state=None):
        self.k = k
        self.snr = snr
        self.semi_simulated = semi_simulated
        self.random_state = random_state

    @staticmethod
    def _load_meg_data(self):
        dataset = fetch_openml(data_id=43884)
        all_data = dataset.data.to_numpy()
        X = all_data[:, :7498]
        n = X.shape[1]
        if self.semi_simulated:
            w_true = generate_sources(n, int(self.k * n), self.random_state)
            y = X @ w_true
            e = np.random.randn(X.shape[0])
            e *= np.sqrt((y @ y) / (self.snr * (e @ e)))
            y += e
        else:
            y = all_data[:, 7498]
            w_true = None

        return X, y, w_true

    def get_data(self):
        X, y, w_true = self._load_meg_data(self)
        self.X, self.y, self.w_true = X, y, w_true

        return dict(X=X, y=y, w_true=w_true)
