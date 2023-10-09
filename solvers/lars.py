from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from sklearn.linear_model import Lars
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "lars"
    stopping_criterion = RunOnGridCriterion()

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.stopping_criterion.reset_grid(list(range(self.X.shape[0] + 1)))

    def run(self, k):
        if k == 0:
            start_time = time.time()
            self.w = np.zeros(self.X.shape[1])
            self.solve_time = time.time() - start_time
        else:
            solver = Lars(n_nonzero_coefs=k, fit_intercept=False)
            start_time = time.time()
            solver.fit(self.X, self.y)
            self.w = solver.coef_.flatten()
            self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(w=self.w, solve_time=self.solve_time)
