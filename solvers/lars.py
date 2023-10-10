from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    import warnings
    from sklearn.linear_model import Lars
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "lars"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 1, 10))

    def set_objective(self, X, y):
        self.X = X
        self.y = y

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

        start_time = time.time()
        if k == 0:
            w = np.zeros(self.X.shape[1])
        else:
            solver = Lars(n_nonzero_coefs=k, fit_intercept=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.fit(self.X, self.y)
            w = solver.coef_.flatten()
        self.k = k
        self.w = w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(k=self.k, w=self.w, solve_time=self.solve_time)
