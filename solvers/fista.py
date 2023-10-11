from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "fista"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))
    parameters = {"maxit": [500]}

    def set_objective(self, X, y):
        self.X = X
        self.y = y
        self.L = np.linalg.norm(self.X, 2) ** 2
        self.lambdaMax = np.linalg.norm(self.X.T @ self.y, np.inf)
        self.lambdaMin = self.lambdaMax * 1e-20
        self.lambdaNum = 1_000

    def run(self, iteration):
        start_time = time.time()
        k = int(np.floor(iteration * self.X.shape[0]))
        w = np.zeros(self.X.shape[1])
        for lamb in np.logspace(
            self.lambdaMax, self.lambdaMin, self.lambdaNum
        ):
            wold = w
            z = w
            for it in range(0, self.maxit):
                wprev = w
                z = z + self.X.T @ (self.y - self.X @ z) / self.L
                w = z * np.maximum(0, 1.0 - lamb / self.L / abs(z))
                z = w + it / (it + 5) * (w - wprev)
                if np.linalg.norm(w - wprev, 2) < 1e-4 * np.linalg.norm(w, 2):
                    break
                if lamb == self.lambdaMax:
                    break
            if np.sum(w != 0) > k:
                w = wold
                break
        self.k = k
        self.w = w
        self.solve_time = time.time() - start_time

    def get_result(self):
        return dict(k=self.k, w=self.w, solve_time=self.solve_time)
