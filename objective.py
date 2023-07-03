from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "L0-penalized Least-Squares"

    parameters = {
        "fit_intercept": [False],
        "lmbd_ratio": [0.1, 0.05, 0.01],
    }

    def __init__(self, fit_intercept, lmbd_ratio):
        self.fit_intercept = fit_intercept
        self.lmbd_ratio = lmbd_ratio

    def _get_lmbd_max(self):
        return 1.5 * np.linalg.norm(self.X.T.dot(self.y), np.inf)

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y, M):
        self.X, self.y = X, y
        self.M = M
        self.lmbd = self.lmbd_ratio * self._get_lmbd_max()

    def compute(self, w):
        r = self.y - self.X.dot(w)
        return dict(
            value = 0.5 * r.dot(r) + self.lmbd * np.count_nonzero(w),
            saturation = np.linalg.norm(w, np.inf) > self.M,
            n_nnz = np.linalg.norm(w, ord=0)
        )

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            M=self.M,
            lmbd=self.lmbd,
            fit_intercept=self.fit_intercept,
        )
