from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pathlib
    from scipy.io import loadmat
    from benchmark_utils.datasets import compute_w_l0pb


class Dataset(BaseDataset):
    name = "deconvolution"
    parameters = {
        "k, snr, random_state": [
            (5, 10.0, None),
        ],
    }

    def __init__(self, k, snr, random_state=None):
        self.k = k
        self.snr = snr
        self.random_state = random_state

    def get_data(self):
        if self.random_state:
            np.random.seed(self.random_state)
        M = loadmat(
            pathlib.Path(__file__).parent.joinpath("deconvolution.mat")
        )
        X = np.array(M["H"])
        w_true = np.zeros(X.shape[1])
        s_true = np.random.choice(X.shape[1], self.k, replace=False)
        w_true[s_true] = np.random.randn(self.k)
        w_true[s_true] += np.sign(w_true[s_true])
        y = X @ w_true
        e = np.random.randn(X.shape[0])
        e *= np.sqrt((y @ y) / (self.snr * (e @ e)))
        y += e
        w_l0pb = compute_w_l0pb(y, X, w_true)
        return dict(X=X, y=y, w_true=w_true, w_l0pb=w_l0pb)
