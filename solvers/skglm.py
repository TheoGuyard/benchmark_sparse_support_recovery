from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from skglm import Lasso, ElasticNet, MCPRegression
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "skglm"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))
    parameters = {
        "estimator": ["lasso", "enet", "mcp"],
        "max_iter": [1_000],
        "alphaNum": [1_000],
        "alphaRatio": [1e-10],
    }

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.alphaMax = np.linalg.norm(self.X.T @ self.y, np.inf) / y.size
        self.alphaMin = self.alphaRatio * self.alphaMax
        self.alphaGrid = np.logspace(
            np.log10(self.alphaMax),
            np.log10(self.alphaMin),
            self.alphaNum,
        )

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

        if self.estimator == "lasso":
            solver_class = Lasso
        elif self.estimator == "enet":
            solver_class = ElasticNet
            self.alphaGrid *= 2.0
        elif self.estimator == "mcp":
            solver_class = MCPRegression
        else:
            raise ValueError(f"Unknown estimator {self.estimator}")

        best_w = np.zeros(self.X.shape[1])
        best_v = np.linalg.norm(self.y, 2) ** 2 / (2.0 * self.y.size)
        start_time = time.time()
        for alpha in self.alphaGrid:
            solver = solver_class(
                alpha=alpha, max_iter=self.max_iter, fit_intercept=False
            )
            solver.fit(self.X, self.y)
            w = solver.coef_.flatten()
            v = np.linalg.norm(self.y - self.X @ w, 2) ** 2 / (
                2.0 * self.y.size
            )
            if np.sum(w != 0) <= k and v < best_v:
                best_w = w
                best_v = v
            if np.sum(w != 0) > k:
                break
        self.k = k
        self.w = best_w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(k=self.k, w=self.w, solve_time=self.solve_time)
