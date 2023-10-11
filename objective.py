from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metrics import snr, fpr, fnr


class Objective(BaseObjective):
    """This benchmark compares the quality of different methods aiming to find
    a sparse vector solving the system y=Xw. It plots different statistics wrt
    a given amount of sparsity targeted in w.
    """

    name = "Sparse support recovery"
    parameters = {}

    def set_data(self, X, y, w_true=None, w_l0pb={}):
        """A dataset must provide the data `X` and `y`. It may also give the
        ground truth value `w_true` and the solutions of the problem

            min 0.5 * ||y - Xw||_2^2
            st  ||w||_0 <= k

        for different values of k, stored in the dictionary `w_l0pb` with k as
        key."""
        self.X = X
        self.y = y
        self.w_true = w_true
        self.w_l0pb = w_l0pb

    def get_one_result(self):
        return np.zeros(self.X.shape[1])

    def evaluate_result(self, k, w, solve_time):
        """The `run` method in the solvers must return the the sparsity
        targeted `k`, the solution constructed `w` and the solution time.
        Depending on whether `w_true` and `w_l0pb` are available in the
        dataset, different metrics are computed."""

        metrics = {}

        # Solver metrics
        metrics["value"] = 0.5 * np.linalg.norm(self.y - self.X @ w, 2) ** 2
        metrics["n_nnz"] = np.sum(w != 0)
        metrics["snr_y"] = snr(self.y, self.X @ w)
        metrics["solve_time"] = solve_time

        # Metrics with respect to the L0-problem solution
        if k in self.w_l0pb.keys():
            current_w_l0pb = self.w_l0pb[k]
            metrics["snr_y_l0pb"] = snr(self.y, self.X @ current_w_l0pb)
            metrics["snr_w_l0pb"] = snr(current_w_l0pb, w)
            metrics["fpr_w_l0pb"] = fpr(current_w_l0pb, w)
            metrics["fnr_w_l0pb"] = fnr(current_w_l0pb, w)

        # Metrics with respect to the ground truth solution (if available)
        if self.w_true is not None:
            metrics["snr_y_true"] = snr(self.y, self.X @ self.w_true)
            metrics["snr_w_true"] = snr(self.w_true, w)
            metrics["fpr_w_true"] = fpr(self.w_true, w)
            metrics["fnr_w_true"] = fnr(self.w_true, w)

        return metrics

    def get_objective(self):
        return dict(X=self.X, y=self.y)
