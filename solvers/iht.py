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

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L

    def run(self, tolerance):
        w = np.zeros(self.X.shape[1])
        old_obj = np.inf

        while True:
            r = self.y - self.X @ w
            w = w + (self.X.T @ r) / self.L
            w = w * (abs(w) > np.sqrt(2 * self.lmbd / self.L))
            w = np.clip(w, -self.M, self.M)
            obj = 0.5 * r.dot(r) + self.lmbd * np.linalg.norm(w, 0)
            if np.abs(old_obj - obj) < tolerance:
                break
            old_obj = obj

        self.w = w

    def get_result(self):
        return self.w
