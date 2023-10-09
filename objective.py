from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Sparse support recovery"
    parameters = {}

    def __init__(self):
        pass

    def set_data(self, X, y, w_true):
        self.X = X
        self.y = y
        self.w_true = w_true

    def get_one_result(self):
        return np.zeros(self.X.shape[1])

    def evaluate_result(self, w, solve_time):
        value = 0.5 * np.linalg.norm(self.y - self.X @ w, 2) ** 2
        n_nnz = np.linalg.norm(w, 0)
        snr_w = np.linalg.norm(self.w_true, 2) / np.linalg.norm(
            self.w_true - w, 2
        )
        snr_y = np.linalg.norm(self.y, 2) / np.linalg.norm(
            self.y - self.X @ w, 2
        )
        fpr = np.sum((self.w_true == 0.0) * (w != 0.0)) / np.sum(
            self.w_true != 0.0
        )
        fnr = np.sum((self.w_true != 0.0) * (w == 0.0)) / np.sum(
            self.w_true == 0.0
        )
        return dict(
            value=value,
            n_nnz=n_nnz,
            solve_time=solve_time,
            snr_w=snr_w,
            snr_y=snr_y,
            fpr=fpr,
            fnr=fnr,
        )

    def get_objective(self):
        return dict(X=self.X, y=self.y)
