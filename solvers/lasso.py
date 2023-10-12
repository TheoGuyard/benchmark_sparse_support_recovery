from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    import warnings
    from sklearn.linear_model import Lasso
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "lasso"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.3, 10))
    parameters = {"maxiter": [10_000]}
    requirements = ["scikit-learn"]

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.alphaMax = (
            np.linalg.norm(self.X.T @ self.y, np.inf) / self.y.shape[0]
        )
        self.alphaMin = self.alphaMax * 1e-15
        self.alphaNum = 1_000

    def run(self, iteration):
        start_time = time.time()
        k = int(np.floor(iteration * self.X.shape[0]))
        w = np.zeros(self.X.shape[1])
        for lamb in np.logspace(
            np.log10(self.alphaMax),
            np.log10(self.alphaMin),
            self.alphaNum
        ):
            w_old = w
            solver = Lasso(
                alpha=lamb,
                warm_start=True,
                fit_intercept=False,
                max_iter=self.maxiter,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.fit(self.X, self.y)
            w = solver.coef_.flatten()
            if np.sum(w != 0) > k:
                w = w_old
                break
        self.k = k
        self.w = w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(k=self.k, w=self.w, solve_time=self.solve_time)
