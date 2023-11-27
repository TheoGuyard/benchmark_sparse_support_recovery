from benchopt import BaseSolver, safe_import_context
from benchmark_utils.stopping_criterion import RunOnGridCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import warnings
    from scipy.linalg import lstsq
    import l0learn


class Solver(BaseSolver):
    name = "l0learn"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))
    parameters = {
        "debiasing_step": [False, True],
    }
    install_cmd = "conda"
    requirements = ["pip:l0learn"]

    def set_objective(self, X, y):
        self.X = X
        self.y = y

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

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

        # L0learn fits a regularization path. We return the best k-sparse
        # solution among the path with respect to the cross-validation error
        # computed on the least-squares term.
        best_w = np.zeros(self.X.shape[1])
        best_cv = np.inf
        for i, gamma in enumerate(fit_result.gamma):
            for j, lmbda in enumerate(fit_result.lambda_0[i]):
                if fit_result.cv_means[i][j]:
                    w = fit_result.coeff(lmbda, gamma)
                    w = np.array(w.todense()).reshape(w.shape[0])[1:]
                    if np.linalg.norm(w, 0) <= k:
                        if fit_result.cv_means[i][j] < best_cv:
                            best_cv = fit_result.cv_means[i][j]
                            best_w = np.copy(w)

        if self.debiasing_step:
            if sum(w != 0) > 0:
                XX = self.X[:, w != 0]
                ww = lstsq(XX, self.y)
                ww = ww[0]
                w[w != 0] = ww

        self.w = best_w

    def get_result(self):
        return dict(w=self.w)
