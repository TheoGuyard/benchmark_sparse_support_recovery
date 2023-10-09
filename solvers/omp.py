from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from sklearn.linear_model import OrthogonalMatchingPursuit


class Solver(BaseSolver):

    name = "omp"
    parameters = {}
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="iteration"
    )

    def set_objective(self, X, y, fit_intercept):
        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept

    def get_next(self, iteration):
        return min(iteration + 1, self.X.shape[1])

    def run(self, iteration):

        if iteration == 0:
            start_time = time.time()
            self.w = np.zeros(self.X.shape[1])
            self.time = time.time() - start_time
        else:
            solver = OrthogonalMatchingPursuit(
                n_nonzero_coefs = iteration,
                fit_intercept = self.fit_intercept
            )
            start_time = time.time()
            solver.fit(self.X, self.y)
            self.w = solver.coef_.flatten()
            self.time = time.time() - start_time

    def get_result(self):
        return [self.time, self.w]
