from pathlib import Path
from benchopt import safe_import_context
from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()

JULIA_SOLVER_FILE = Path(__file__).with_suffix('.jl')

class Solver(JuliaSolver):

    # Config of the solver
    name = 'el0ps'
    stopping_strategy = 'time'
    references = [
        'Guyard, T., Herzet, C., & Elvira, C. (2022, May). Node-Screening Tests'
        'For The L0-Penalized Least-Squares Problem. In ICASSP 2022-2022 IEEE'
        'International Conference on Acoustics, Speech and Signal Processing'
        '(ICASSP) (pp. 5448-5452). IEEE.'
    ]

    julia_requirements = [
        'El0ps::https://github.com/TheoGuyard/El0ps.jl/#master',
        'PyCall',
    ]

    def set_objective(self, X, y, M, lmbd, fit_intercept=False):
        self.X, self.y = X, y
        self.M = M
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_el0ps = jl.solve_el0ps

    def run(self, tolerance):
        self.beta = self.solve_el0ps(self.X, self.y, self.M, self.lmbd, tolerance)

    def get_result(self):
        return self.beta.ravel()