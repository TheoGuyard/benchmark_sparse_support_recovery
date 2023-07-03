from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    name = 'iht'
    stopping_strategy = "callback"
    parameters = {}

    def set_objective(self, X, y, M, lmbd, fit_intercept):
        self.X, self.y, self.M, self.lmbd, self.fit_intercept = X, y, M, lmbd, fit_intercept

    def run(self, cb):

        beta = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2)**2

        while cb(beta):
            r = self.y - self.X @ beta
            beta = beta + (self.X.T @ r)/L
            beta = beta*(abs(beta)>np.sqrt(2*self.lmbd/L))
            beta = np.clip(beta,-self.M,self.M)

        self.w=beta                  

    def get_result(self):
        return self.w