from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from gurobipy import Model, GRB
    from benchmark_utils.stopping_criterion import RunOnGridCriterion


class Solver(BaseSolver):
    name = "l0constraint"
    stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.1, 10))

    def set_objective(self, X, y):
        self.X = X
        self.y = y

    def run(self, iteration):
        m, n = self.X.shape
        k = int(np.floor(iteration * m))
        M = 10.0 * np.max(
            np.abs(np.linalg.lstsq(self.X, self.y, rcond=None)[0])
        )

        model = Model()
        w_var = model.addMVar(n, name="w", vtype="C", lb=-np.inf, ub=np.inf)
        z_var = model.addMVar(n, name="z", vtype="B")
        r_var = self.y - self.X @ w_var
        model.setObjective(0.5 * (r_var @ r_var), GRB.MINIMIZE)
        model.addConstr(w_var <= M * z_var)
        model.addConstr(w_var >= -M * z_var)
        model.addConstr(sum(z_var) <= k)
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 1e-8)
        model.setParam("IntFeasTol", 1e-8)
        model.optimize()

        self.w = w_var.X * (z_var.X > 0.5)
        self.solve_time = model.Runtime

    def get_result(self):
        return dict(w=self.w, solve_time=self.solve_time)
