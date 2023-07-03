from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize as sci_opt


class Solver(BaseSolver):
    name = 'ista'
    # stopping_strategy = "tolerance"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="tolerance"
    )
    parameters = {"reg_mode": ["1", "2"], "acceleration": [True, False], "debiasing": [True, False]}

    def set_objective(self, X, y, M, lmbd, fit_intercept):
        self.X, self.y, self.M, self.lmbd, self.fit_intercept = X, y, M, lmbd, fit_intercept

    def run(self, tolerance):

        beta = np.zeros(self.X.shape[1])
        z = np.copy(beta)
        L = np.linalg.norm(self.X, ord=2)**2
        old_obj = np.inf

        it = 0
        while True:
            it += 1
            beta_old = np.copy(beta)
            r = self.y - self.X @ z
            beta = z + (self.X.T @ r)/L
            beta = beta*np.maximum(1. - self.lmbd/L/(abs(beta) + 1e-10), 0.)
            beta = np.clip(beta,-self.M,self.M)
        
            if self.acceleration:
                t = (it - 1) / (it+5)
                z = beta + t * (beta - beta_old)
            else:
                z = beta
            
            if self.reg_mode == "1":
                obj = 0.5 * r.dot(r) + self.lmbd * np.linalg.norm(beta, 1)
            elif self.reg_mode == "2":
                obj = 0.5 * r.dot(r) + np.sqrt(2. * self.lmbd) * np.linalg.norm(beta, 1)
            else:
                raise NotImplementedError

            if np.abs(old_obj - obj) < tolerance:
                break
            old_obj = obj

        if self.debiasing:
            S = (beta != 0.)
            if np.any(S):
                res = sci_opt.lsq_linear(self.X[:, S], self.y, (-self.M, self.M))
                beta[S] = res.x

        self.w=beta                  

    def get_result(self):
        return self.w