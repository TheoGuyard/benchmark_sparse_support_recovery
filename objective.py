from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "L0-penalized Least-squares problem with Big-M constraint"

    parameters = {
        "fit_intercept": [False],
        "reg": [0.5, 0.1, 0.05],
    }

    def __init__(self, reg=0.1, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def _get_lambda_max(self):
        return self.M * np.linalg.norm(self.X.T.dot(self.y), np.inf)

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y, M):
        self.X, self.y = X, y
        self.M = M
        self.lmbd = self.reg * self._get_lambda_max()

    def compute(self, beta):
        if np.linalg.norm(beta, np.inf) > self.M:
            return np.inf
        r = self.y - self.X.dot(beta)
        return r.dot(r) / (2.0 * r.shape[0]) + self.lmbd * np.count_nonzero(
            beta
        )

    def to_dict(self):
        return dict(
            X=self.X,
            y=self.y,
            M=self.M,
            lmbd=self.lmbd,
            fit_intercept=self.fit_intercept,
        )
