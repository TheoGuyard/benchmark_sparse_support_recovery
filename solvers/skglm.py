from benchopt import BaseSolver, safe_import_context
from benchmark_utils.stopping_criterion import RunOnGridCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import lstsq
    from skglm import Lasso, ElasticNet, MCPRegression


class Solver(BaseSolver):
    name = "skglm"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.3, 10))
    parameters = {
        "estimator": ["lasso", "enet", "mcp"],
        "max_iter": [1_000],
        "alphaNum": [1_000],
        "alphaRatio": [1e-10],
        "debiasing_step": [False, True],
    }
    install_cmd = "conda"
    requirements = ["pip:skglm", "scipy"]

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

        w = np.zeros(self.X.shape[1])
        for alpha in self.alphaGrid:
            w_old = w
            solver = solver_class(
                alpha=alpha, max_iter=self.max_iter, fit_intercept=False
            )
            solver.fit(self.X, self.y)
            w = solver.coef_.flatten()
            if np.sum(w != 0) > k:
                w = w_old
                break

        if self.debiasing_step:
            if sum(w != 0) > 0:
                XX = self.X[:, w != 0]
                ww = lstsq(XX, self.y)
                ww = ww[0]
                w[w != 0] = ww

        self.w = w

    def get_result(self):
        return dict(w=self.w)
