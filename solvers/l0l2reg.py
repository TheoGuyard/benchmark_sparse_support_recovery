from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    import gurobipy as gp


class Solver(BaseSolver):
    name = "l0l2reg"
    parameters = {
        "l0_ratio": [0.2, 0.5, 0.8],
    }
    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy="iteration"
    )

    def skip(self, X, y, rho_grid, fit_intercept):
        if fit_intercept:
            return True, "l0norm does not support fit_intercept=True"
        return False, None

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

        # For the l0-regularized least-squares, rho=lmbd/lmbd_max, where 
        # lmbd_max is the maximum regularization weight above which the 
        # all-zero vector is always the problem solution.
        sol_lstsq = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        Xty_infnorm = np.linalg.norm(self.X.T @ self.y, np.inf)
        self.lmbd_max = np.linalg.norm(sol_lstsq, np.inf) * Xty_infnorm

        self.build_model()

    def build_model(self):
        model = gp.Model()
        w_var = model.addMVar(shape=self.X.shape[1], vtype="C")
        z_var = model.addMVar(shape=self.X.shape[1], vtype="B")
        t_var = model.addVar(vtype="C")
        u_var = model.addVar(vtype="C")
        s_var = model.addMVar(shape=self.X.shape[1], vtype="C")
        r_var = self.y - self.X @ w_var
        model.setObjective(
            0.5 * gp.quicksum(ri * ri for ri in r_var) + t_var + u_var,
            gp.GRB.MINIMIZE
        )
        model.addConstr(s_var >= 0.)
        model.addConstr(t_var >= sum(z_var))
        model.addConstr(u_var >= sum(s_var))
        model.addConstr(w_var * w_var <= s_var * z_var)
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("IntFeasTol", 1e-6)
        self.model = model
        self.t_var = t_var
        self.w_var = w_var
        self.u_var = u_var

    def get_next(self, iteration):
        return min(iteration + 1, self.rho_grid.size - 1)

    def run(self, iteration):
        self.rho_curr = self.rho_grid[iteration]
        lmbd_curr = self.rho_curr * self.lmbd_max 
        start_time = time.time()
        self.t_var.Obj = self.l0_ratio * lmbd_curr
        self.u_var.Obj = 0.5 * (1. - self.l0_ratio) * lmbd_curr
        self.model.optimize()
        self.w_curr = np.array(self.w_var.X)
        self.time_curr = time.time() - start_time

    def get_result(self):
        return [self.rho_curr, self.time_curr, self.w_curr]
