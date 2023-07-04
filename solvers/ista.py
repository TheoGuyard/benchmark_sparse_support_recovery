from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize as sci_opt


class Solver(BaseSolver):
    name = "ista"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {
        "adapt_lmbd": [True, False],
        "acceleration": [True, False],
        "debiasing": [True, False],
    }

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L

    def run(self, tolerance):
        w = np.zeros(self.X.shape[1])
        z = np.copy(w)
        old_obj = np.inf

        it = 0
        while True:
            it += 1
            w_old = np.copy(w)
            r = self.y - self.X @ z
            w = z + (self.X.T @ r) / self.L
            w = w * np.maximum(1.0 - self.lmbd / self.L / (abs(w) + 1e-10), 0.0)
            w = np.clip(w, -self.M, self.M)

            if self.acceleration:
                t = (it - 1) / (it + 5)
                z = w + t * (w - w_old)
            else:
                z = w

            if self.adapt_lmbd:
                obj = 0.5 * r.dot(r) + np.sqrt(
                    2.0 * self.lmbd
                ) * np.linalg.norm(w, 1)
            else:
                obj = 0.5 * r.dot(r) + self.lmbd * np.linalg.norm(w, 1)

            if np.abs(old_obj - obj) < tolerance:
                break
            old_obj = obj

        if self.debiasing:
            S = w != 0.0
            if np.any(S):
                res = sci_opt.lsq_linear(
                    self.X[:, S], self.y, (-self.M, self.M)
                )
                w[S] = res.x

        self.w = w

    def get_result(self):
        return self.w
