from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "iht"
    stopping_criterion = RunOnGridCriterion()
    parameters = {"maxit": [1_000], "rel_tol": [1e-4]}

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.L = np.linalg.norm(self.X, ord=2) ** 2
        self.stopping_criterion.reset_grid(list(range(self.X.shape[0] + 1)))

    def run(self, k):
        start_time = time.time()
        w = np.zeros(self.X.shape[1])
        old_obj = np.inf
        for _ in range(self.maxit):
            r = self.y - self.X @ w
            z = w + (self.X.T @ r) / self.L
            s = np.argsort(np.abs(w))[::-1][:k]
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
