from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metrics import (
        snr,
        fpr,
        fnr,
        tpr,
        tnr,
        f1score,
        auc,
        dist_to_supp,
    )


class Objective(BaseObjective):
    """This benchmark compares the quality of different methods aiming to find
    a sparse vector solving the system y=Xw. It plots different statistics wrt
    a given amount of sparsity targeted in w.
    """

    name = "Sparse support recovery"

    def set_data(self, X, y, w_true=None):
        self.X = X
        self.y = y
        self.w_true = w_true

    def get_one_result(self):
        return dict(w=np.zeros(self.X.shape[1]))

    def evaluate_result(self, w):
        metrics = {}

        r = self.y - self.X @ w
        metrics["value"] = ((r @ r) / r.size)
        metrics["n_nnz"] = np.sum(w != 0)
        metrics["snr_y"] = snr(self.y, self.X @ w)
        metrics["snr_y_dB"] = snr(self.y, self.X @ w, dB=True)

        if self.w_true is not None:
            metrics["snr_y_true"] = snr(self.y, self.X @ self.w_true)
            metrics["snr_y_true_dB"] = snr(
                self.y, self.X @ self.w_true, dB=True
            )
            metrics["snr_w"] = snr(self.w_true, w)
            metrics["snr_w_dB"] = snr(self.w_true, w, dB=True)
            metrics["tpr"] = tpr(self.w_true, w)
            metrics["fpr"] = fpr(self.w_true, w)
            metrics["tnr"] = tnr(self.w_true, w)
            metrics["fnr"] = fnr(self.w_true, w)
            metrics["f1score"] = f1score(self.w_true, w)
            metrics["auc"] = auc(self.w_true, w)
            metrics["dist_to_supp"] = dist_to_supp(self.w_true, w)

        return metrics

    def get_objective(self):
        return dict(X=self.X, y=self.y)
