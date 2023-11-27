import numpy as np
from gurobipy import Model, GRB


def compute_w_l0pb(y, X, w_true):
    k = np.sum(w_true != 0.0)
    n = X.shape[1]
    M = 1.5 * np.max(np.abs(w_true))
    model = Model()
    w_var = model.addMVar(n, name="w", vtype="C", lb=-np.inf, ub=np.inf)
    z_var = model.addMVar(n, name="z", vtype="B")
    r_var = y - X @ w_var
    model.setObjective(0.5 * (r_var @ r_var), GRB.MINIMIZE)
    model.addConstr(w_var <= M * z_var)
    model.addConstr(w_var >= -M * z_var)
    model.addConstr(sum(z_var) <= k)
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-8)
    model.setParam("IntFeasTol", 1e-8)
    model.optimize()
    return w_var.X * (z_var.X > 0.5)


def generate_sources(n, k, random_state=None):
    rng = np.random.RandomState(random_state)
    w_true = rng.randn(n)
    w_true[rng.choice(n, n - k, replace=False)] = 0.0

    return w_true
