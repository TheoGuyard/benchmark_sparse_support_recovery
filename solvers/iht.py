from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import lstsq
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "iht"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))
    parameters = {
        "maxit": [1_000],
        "rel_tol": [1e-8],
        "debiasing_step": [False, True],
    }

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.L = np.linalg.norm(self.X, ord=2) ** 2

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

        w = np.zeros(self.X.shape[1])
        old_obj = np.inf
        for k_ws in range(k + 1):
            for _ in range(self.maxit):
                r = self.y - self.X @ w
                z = w + (self.X.T @ r) / self.L
                s = np.argsort(np.abs(z))[::-1][:k_ws]
                w = np.zeros(self.X.shape[1])
                w[s] = z[s]
                obj = 0.5 * (r @ r)
                if (np.abs(old_obj - obj) / obj) < self.rel_tol:
                    break
                old_obj = obj

        if self.debiasing_step:
            if sum(w != 0) > 0:
                XX = self.X[:, w != 0]
                ww = lstsq(XX, self.y)
                ww = ww[0]
                w[w != 0] = ww

        self.w = w

    def get_result(self):
        return dict(w=self.w)
