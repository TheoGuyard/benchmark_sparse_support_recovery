from benchopt import BaseSolver, safe_import_context
from l0bnb import BNBTree

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    name = 'l0bnb'
    stopping_strategy = "tolerance"
    parameters = {"ws_iter": [0, 5, 10]}

    def set_objective(self, X, y, M, lmbd, fit_intercept):
        self.X, self.y, self.M, self.lmbd, self.fit_intercept = X, y, M, lmbd, fit_intercept

    def run(self, tolerance):

        beta_ws = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2)**2
        for it in range(self.ws_iter):
            r = self.y - self.X @ beta_ws
            beta_ws = beta_ws + (self.X.T @ r)/L
            beta_ws = beta_ws*(abs(beta_ws)>np.sqrt(2*self.lmbd/L))
            beta_ws = np.clip(beta_ws,-self.M,self.M)    

        solver = BNBTree(self.X, self.y, int_tol=1e-5, rel_tol=1e-4)
        result = solver.solve(self.lmbd, 0.0, self.M, gap_tol=tolerance, warm_start=beta_ws)
        self.w = result.beta * (np.abs(result.beta) > 1e-5)

    def get_result(self):
        return self.w
