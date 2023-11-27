from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import warnings
    from sklearn.linear_model import Lars
    from scipy.linalg import lstsq
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "lars"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.3, 10))
    parameters = {
        "debiasing_step": [False, True]
    }
    install_cmd = "conda"
    requirements = ["scikit-learn", "scipy"]

    def set_objective(self, X, y):
        self.X = X
        self.y = y

    def run(self, grid_value):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        k = int(np.floor(grid_value * self.X.shape[1]))

        if k == 0:
            w = np.zeros(self.X.shape[1])
        else:
            solver = Lars(n_nonzero_coefs=k, fit_intercept=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.fit(self.X, self.y)
            w = solver.coef_.flatten()

        if self.debiasing_step:
            if sum(w != 0) > 0:
                XX = self.X[:, w != 0]
                ww = lstsq(XX, self.y)
                ww = ww[0]
                w[w != 0] = ww

        self.w = w

    def get_result(self):
        return dict(w=self.w)
