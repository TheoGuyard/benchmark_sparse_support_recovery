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
    parameters = {}

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L

    def run(self, tolerance):
        solver = BNBTree(self.X, self.y, int_tol=1e-6)
        result = solver.solve(self.lmbd, 0.0, self.M, gap_tol=tolerance)
        self.w = result.beta * (np.abs(result.beta) > 1e-5)

    def get_result(self):
        return self.w
