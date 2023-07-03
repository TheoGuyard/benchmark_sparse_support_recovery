from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "L0-penalized Least-Squares"

    parameters = {
        "lmbd_ratio": [0.1, 0.05],
    }

    def __init__(self, lmbd_ratio):
        self.lmbd_ratio = lmbd_ratio

    def _get_lmbd_max(self):
        return 1.5 * np.linalg.norm(self.X.T.dot(self.y), np.inf)

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y, M, w_true):
        self.X = X
        self.y = y
        self.M = M
        self.w_true = w_true
        self.L = np.linalg.norm(self.X, ord=2) ** 2
        self.lmbd = self.lmbd_ratio * self._get_lmbd_max()
        
    def compute(self, w):
        r = self.y - self.X.dot(w)
        S = w != 0.0
        Strue = self.w_true != 0.0
        acc = np.sum(S * Strue) / np.sum(S) if np.any(S) else 1.0
        rec = np.sum(S * Strue) / np.sum(Strue) if np.any(S) else 1.0
        fscore = 0.0 if acc + rec == 0.0 else (2 * acc * rec) / (acc + rec)

        return dict(
            value=0.5 * r.dot(r) + self.lmbd * np.count_nonzero(w),
            datafit_loss=0.5 * r.dot(r) ** 2,
            n_nnz=np.linalg.norm(w, ord=0),
            fscore=fscore,
        )

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            M=self.M,
            lmbd=self.lmbd,
            L=self.L,
        )
