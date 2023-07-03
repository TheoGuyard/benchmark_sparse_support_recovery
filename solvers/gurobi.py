from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import gurobipy as gp


class Solver(BaseSolver):
    name = "gurobi"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {}

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L

    def run(self, tolerance):
        m = gp.Model()
        w = m.addMVar(shape=self.X.shape[1], vtype="C")
        z = m.addMVar(shape=self.X.shape[1], vtype="B")
        r = self.y - self.X @ w
        m.setObjective(
            0.5 * gp.quicksum(ri * ri for ri in r) + self.lmbd * sum(z),
            gp.GRB.MINIMIZE,
        )
        m.addConstr(w <= z * self.M)
        m.addConstr(-w <= z * self.M)
        m.setParam("OutputFlag", 0)
        m.setParam("MIPGap", tolerance)
        m.setParam("IntFeasTol", 1e-6)
        m.optimize()
        self.w = np.array(w.X)

    def get_result(self):
        return self.w
