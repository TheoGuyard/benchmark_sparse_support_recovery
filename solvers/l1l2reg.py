from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import time, sys
    import numpy as np
    from sklearn.linear_model import ElasticNet


class Solver(BaseSolver):
    """L1L2-regression

    This solver returns the solution of the problem

            min_w (1/2) ||y-Xw||_2^2 + lmbd * (
    (P)         l1_ratio * ||w||_1   + (1 - l1_ratio)  * ||w||_2^2
            )

    where `0 < l1_ratio <= 1`. The value `rho` represents the proportion 
    `lmbd / lmbd_max` where `lmbd_max` is the value of `lmbd` above which the 
    all-zero vector is always the solution of (P). If `debiasing=True`, the 
    result returned is the solution of the Least-squares problem restricted 
    to the support of the solution of problem (P).
    """

    name = "l1l2reg"
    parameters = {
        "debiasing": [True, False],
        "l1_ratio": [0.2, 0.5, 0.8],
    }
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="iteration"
    )

    def set_objective(self, X, y, rho_grid, fit_intercept):
        self.X = X
        self.y = y
        self.rho_grid = rho_grid
        self.fit_intercept = fit_intercept

        # Current value of rho evaluated at an iteration. It is returned with
        # the result to compute the different objectives.
        self.rho_curr = rho_grid[0]

        # Solution of the solver corresponding to the current value of rho. It
        # is stored to be used as warm-start for the next-value of rho.
        self.w_curr = np.zeros(self.X.shape[1])

        # Maximum value of `lmbd`
        self.lmbd_max = np.linalg.norm(self.X.T @ self.y, np.inf)

    def get_next(self, iteration):
        return min(iteration + 1, self.rho_grid.size - 1)

    def run(self, iteration):
        self.rho_curr = self.rho_grid[iteration]
        lmbd = self.rho_curr * self.lmbd_max
        
        solver = ElasticNet(
            alpha = lmbd / self.X.shape[0],
            l1_ratio = self.l1_ratio,
            tol = 1e-12,
            warm_start = True,
            fit_intercept = self.fit_intercept,
        )

        start_time = time.time()
        solver.coef_ = self.w_curr  # set warm-start
        solver.fit(self.X, self.y)
        w = solver.coef_.flatten()
        if self.debiasing and np.any(w != 0.):
            s = w != 0.
            w[s] = np.linalg.lstsq(self.X[:, s], self.y, rcond=None)[0]
        self.w_curr = w
        self.time_curr = time.time() - start_time

    def get_result(self):
        return [self.rho_curr, self.time_curr, self.w_curr]
