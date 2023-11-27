from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pathlib


class Dataset(BaseDataset):
    name = "portfolio"
    parameters = {
        "instance": ["1", "2", "3", "4", "5"],
        "ratio": [0.25, 0.5, 0.75],
    }

    def get_data(self):
        f = np.load(pathlib.Path(__file__).parent.joinpath("portfolio.npz"))
        S = f["S" + self.instance]
        p = f["p" + self.instance]
        X = np.linalg.cholesky(self.ratio * S).T
        y = np.linalg.lstsq(X.T, (1.0 - self.ratio) * p, rcond=None)[0]
        return dict(X=X, y=y, w_true=None, w_l0pb=None)
