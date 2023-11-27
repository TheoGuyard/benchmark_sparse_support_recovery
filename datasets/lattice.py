from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pathlib


class Dataset(BaseDataset):
    name = "lattice"
    parameters = {
        "semi_simulated, k, snr, random_state": [
            (False, None, None, None),
            (True, 10, 10.0, None),
        ],
    }

    def get_data(self):
        if self.random_state:
            np.random.seed(self.random_state)
        f = np.load(pathlib.Path(__file__).parent.joinpath("lattice.npz"))
        X = f['X']
        if self.semi_simulated:
            w_true = np.zeros(X.shape[1])
            s_true = np.random.choice(X.shape[1], self.k, replace=False)
            w_true[s_true] = np.random.randn(self.k)
            y = X @ w_true
            e = np.random.randn(X.shape[0])
            e *= np.sqrt((y @ y) / (self.snr * (e @ e)))
            y += e
        else:
            w_true = None
            y = f['y']
        return dict(X=X, y=y, w_true=w_true, w_l0pb=None)
