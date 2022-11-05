import numpy as np


from benchopt import BaseSolver


class Solver(BaseSolver):
    """Test solver."""
    name = 'testsolver'

    parameters = {}

    def set_objective(self, X, y, M, lmbd, fit_intercept=False):
        self.X, self.y = X, y
        self.M = M
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

    def run(self):
        self.w = np.zeros(self.X.shape[1])

    def get_result(self):
        return self.w
