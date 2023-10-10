from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metrics import snr, fpr, fnr


class Objective(BaseObjective):
    name = "Sparse support recovery"
    parameters = {}

    def __init__(self):
        pass

    def set_data(self, X, y, w_true=None):
        self.X = X
        self.y = y
        self.w_true = w_true

    def get_one_result(self):
        return np.zeros(self.X.shape[1])

    def evaluate_result(self, w, solve_time):
        metrics = {}
        metrics["solve_time"] = solve_time
        metrics["value"] = 0.5 * np.linalg.norm(self.y - self.X @ w, 2) ** 2
        metrics["n_nnz"] = np.linalg.norm(w, 0)
        metrics["snr_y"] = snr(self.y, self.X @ w)
        if self.w_true is not None:
            metrics["snr_w_true"] = snr(self.w_true, w)
            metrics["fpr_true"] = fpr(self.w_true, w)
            metrics["fnr_true"] = fnr(self.w_true, w)
        return metrics

    def get_objective(self):
        return dict(X=self.X, y=self.y)
