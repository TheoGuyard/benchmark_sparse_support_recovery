from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metrics import snr, fpr, fnr


class Objective(BaseObjective):
    name = "Sparse support recovery"
    parameters = {}

    def __init__(self, grid=np.linspace(0, 0.1, 10)):
        self.grid = grid

    def set_data(self, instance_uuid, X, y, w_true=None):
        self.instance_uuid = instance_uuid
        self.X = X
        self.y = y
        self.w_true = w_true

    def get_one_result(self):
        return np.zeros(self.X.shape[1])

    def evaluate_result(self, w, solve_time):
        metrics = {}

        # Solver metrics
        metrics["value"] = 0.5 * np.linalg.norm(self.y - self.X @ w, 2) ** 2
        metrics["n_nnz"] = np.sum(w != 0)
        metrics["snr_y"] = snr(self.y, self.X @ w)
        metrics["solve_time"] = solve_time

        # Metrics with respect to the L0-problem solution
        w_l0pb = None
        if w_l0pb is not None:
            metrics["snr_y_l0pb"] = snr(self.y, self.X @ w_l0pb)
            metrics["snr_w_l0pb"] = snr(w_l0pb, w)
            metrics["fpr_w_l0pb"] = fpr(w_l0pb, w)
            metrics["fnr_w_l0pb"] = fnr(w_l0pb, w)

        # Metrics with respect to the ground truth solution (if available)
        if self.w_true is not None:
            metrics["snr_y_true"] = snr(self.y, self.X @ self.w_true)
            metrics["snr_w_true"] = snr(self.w_true, w)
            metrics["fpr_w_true"] = fpr(self.w_true, w)
            metrics["fnr_w_true"] = fnr(self.w_true, w)

        return metrics

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            grid=self.grid,
        )
