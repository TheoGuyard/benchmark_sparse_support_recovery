from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    name = "iht"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {}

    def set_objective(self, X, y, M, lmbd):
        self.X, self.y, self.M, self.lmbd = X, y, M, lmbd

    def run(self, tolerance):
        w = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2) ** 2
        old_obj = np.inf

        while True:
            r = self.y - self.X @ w
            w = w + (self.X.T @ r) / L
            w = w * (abs(w) > np.sqrt(2 * self.lmbd / L))
            w = np.clip(w, -self.M, self.M)
            obj = 0.5 * r.dot(r) + self.lmbd * np.linalg.norm(w, 0)
            if np.abs(old_obj - obj) < tolerance:
                break
            old_obj = obj

        self.w = w

    def get_result(self):
        return self.w
