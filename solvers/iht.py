from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "iht"
    parameters = {"maxit": [1_000], "rel_tol": [1e-8]}

    def set_objective(self, X, y, grid):
        self.X = X
        self.y = y
        self.L = np.linalg.norm(self.X, ord=2) ** 2
        self.stopping_criterion = RunOnGridCriterion(grid=grid)

    def run(self, iteration):
        start_time = time.time()
        k = int(np.floor(iteration * self.X.shape[1]))
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
        self.w = w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(w=self.w, solve_time=self.solve_time)
