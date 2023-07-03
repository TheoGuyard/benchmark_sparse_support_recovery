from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize as sci_opt


class Solver(BaseSolver):
    name = "omp"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {}

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L

    def run(self, tolerance):
        w = np.zeros(self.X.shape[1])
        r = self.y - self.X @ w
        S = np.zeros(w.shape, dtype=bool)
        old_obj = np.inf

        for it in range(self.X.shape[1]):
            new_i = np.argmax(self.X.T @ r)
            S[new_i] = True
            res = sci_opt.lsq_linear(self.X[:, S], self.y, (-self.M, self.M))
            w[S] = res.x
            r = self.y - self.X @ w
            obj = 0.5 * r.dot(r) + self.lmbd * np.sum(S)
            if np.abs(old_obj - obj) < tolerance:
                break
            old_obj = obj

        self.w = w

    def get_result(self):
        return self.w
