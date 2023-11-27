from benchopt import BaseSolver, safe_import_context
from benchmark_utils.stopping_criterion import RunOnGridCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import warnings
    from sklearn.linear_model import OrthogonalMatchingPursuit


class Solver(BaseSolver):
    name = "omp"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))
    install_cmd = "conda"
    requirements = ["scikit-learn"]

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
            solver = OrthogonalMatchingPursuit(
                n_nonzero_coefs=k, fit_intercept=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.fit(self.X, self.y)
            w = solver.coef_.flatten()

        self.w = w

    def get_result(self):
        return dict(w=self.w)
