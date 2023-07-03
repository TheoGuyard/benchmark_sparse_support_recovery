from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from l0bnb import BNBTree


class Solver(BaseSolver):
    name = "l0bnb"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {"ws_iter": [0, 5, 10]}

    def set_objective(self, X, y, M, lmbd):
        self.X, self.y, self.M, self.lmbd = X, y, M, lmbd

    def run(self, tolerance):
        w_ws = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2) ** 2
        for it in range(self.ws_iter):
            r = self.y - self.X @ w_ws
            w_ws = w_ws + (self.X.T @ r) / L
            w_ws = w_ws * (abs(w_ws) > np.sqrt(2 * self.lmbd / L))
            w_ws = np.clip(w_ws, -self.M, self.M)

        solver = BNBTree(self.X, self.y, int_tol=1e-6)
        result = solver.solve(
            self.lmbd, 0.0, self.M, gap_tol=tolerance, warm_start=w_ws
        )
        self.w = result.beta * (np.abs(result.beta) > 1e-5)

    def get_result(self):
        return self.w
