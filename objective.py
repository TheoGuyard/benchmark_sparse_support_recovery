from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):

    name = "Sparse linear methods"
    parameters = {
        "fit_intercept": [False, True],
    }

    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y, w_true):
        self.X = X
        self.y = y
        self.w_true = w_true
        
    def get_w_true_statistics(self, w):
        """Returns reconstruction statistics of `w` with respect to `w_true`.
        If `w_true` is not available, no statistics are returned."""
        if self.w_true is None:
            statistics = dict()
        else:
            w_snr = 10. * np.log10(
                np.linalg.norm(self.w_true, 2)**2 / 
                np.linalg.norm(w - self.w_true, 2)**2
            )  # snr in w regarding w_true
            Xw_snr = 10. * np.log10(
                np.linalg.norm(self.X @ self.w_true, 2)**2 / 
                np.linalg.norm(self.X @ (w - self.w_true), 2)**2
            )  # snr in Xw regarding Xw_true
            r_zer = self.w_true != 0.0  # real non-zeros
            r_nnz = self.w_true == 0.0  # real zeros
            d_zer = w != 0.0  # detected non-zeros
            d_nnz = w == 0.0  # detected zeros
            tp = np.sum(r_zer * d_zer)  # true positives
            fp = np.sum(r_nnz * d_zer)  # false positives
            tn = np.sum(r_nnz * d_nnz)  # true negatives
            fn = np.sum(r_zer * d_nnz)  # false negatives
            tpr = tp / np.sum(r_zer) if np.sum(r_zer) else 1.0  # true positive rate
            fpr = fp / np.sum(r_zer) if np.sum(r_zer) else 1.0  # false positive rate
            tnr = tn / np.sum(r_nnz) if np.sum(r_nnz) else 1.0  # true negative rate
            fnr = fn / np.sum(r_nnz) if np.sum(r_nnz) else 1.0  # false negative rate
            f1s = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0  # f1-score
            statistics = dict(
                w_snr=w_snr, 
                Xw_snr=Xw_snr, 
                tpr=tpr, 
                fpr=fpr, 
                tnr=tnr, 
                fnr=fnr, 
                f1s=f1s,
            )
        return statistics

    def compute(self, result):

        solve_time = result[0]
        w = np.array(result[1:]).reshape(self.X.shape[1])
        
        value = 0.5 * np.linalg.norm(self.y - self.X @ w, 2)**2
        n_nnz = np.linalg.norm(w, 0)
        w_true_statistics = self.get_w_true_statistics(w)

        return dict(
            value = value,
            n_nnz = n_nnz,
            solve_time = solve_time,
            **w_true_statistics,
        )

    def get_objective(self):
        return dict(
            X = self.X,
            y = self.y,
            fit_intercept = self.fit_intercept,
        )
