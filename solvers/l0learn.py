from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import l0learn
    import numpy as np
    import warnings
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "l0learn"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))

    def set_objective(self, X, y):
        self.X = X
        self.y = y

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_result = l0learn.cvfit(
                self.X,
                self.y,
                loss="SquaredError",
                penalty="L0",
                intercept=False,
                max_support_size=k + 1,
            )
        best_w = None
        best_cv = np.inf
        for i, gamma in enumerate(fit_result.gamma):
            for j, lmbda in enumerate(fit_result.lambda_0[i]):
                w = fit_result.coeff(lmbda, gamma)
                w = np.array(w.todense()).reshape(w.shape[0])[1:]
                if np.linalg.norm(w, 0) <= k:
                    if fit_result.cv_means[i][j] < best_cv:
                        best_cv = fit_result.cv_means[i][j]
                        best_w = np.copy(w)
        self.k = k
        self.w = best_w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(k=self.k, w=self.w, solve_time=self.solve_time)
