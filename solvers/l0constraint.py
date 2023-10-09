from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import time
    import numpy as np
    import gurobipy as gp


class Solver(BaseSolver):
    name = "l0constraint"
    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy="iteration"
    )

    def skip(self, X, y, fit_intercept):
        if fit_intercept:
            return True, "l0constraint does not support fit_intercept=True"
        return False, None

    def set_objective(self, X, y, fit_intercept):
        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept

    def build_model(self, iteration):
        model = gp.Model()
        w_var = model.addMVar(shape=self.X.shape[1], vtype="C")
        z_var = model.addMVar(shape=self.X.shape[1], vtype="B")
        r_var = self.y - self.X @ w_var
        model.setObjective(
            0.5 * gp.quicksum(ri * ri for ri in r_var), gp.GRB.MINIMIZE
        )
        model.addConstr(self.X.shape[1] - iteration <= sum(z_var))
        for i in range(self.X.shape[1]):
            model.addSOS(gp.GRB.SOS_TYPE1, [w_var[i], z_var[i]])
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 1e-16)
        model.setParam("IntFeasTol", 1e-9)
        return model, w_var

    def get_next(self, iteration):
        return min(iteration + 1, self.X.shape[1])

    def run(self, iteration):
        model, w_var = self.build_model(iteration)
        start_time = time.time()
        model.optimize()
        s = np.array(w_var.X) != 0.
        w = np.zeros(self.X.shape[1])
        if np.any(s):
            w[s] = np.linalg.lstsq(self.X[:, s], self.y, rcond=None)[0]
        self.w = w
        self.time = time.time() - start_time

        print()
        for i in range(self.X.shape[1]):
            print(self.w[i])

    def get_result(self):
        return [self.time, self.w]
