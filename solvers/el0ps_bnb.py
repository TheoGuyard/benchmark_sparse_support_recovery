from pathlib import Path
from benchopt import safe_import_context
from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()

JULIA_SOLVER_FILE = Path(__file__).with_suffix(".jl")


class Solver(JuliaSolver):

    # Config of the solver
    name = "el0ps_bnb"
    stopping_strategy = "tolerance"
    references = ["To appear"]

    julia_requirements = [
        "El0ps::https://github.com/TheoGuyard/El0ps.jl#master",
        "PyCall",
    ]

    parameters = {"acceleration": [False, True]}

    def set_objective(self, X, y, M, lmbd, fit_intercept=False):
        self.X, self.y = X, y
        self.M = M
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_el0ps_bnb = jl.solve_el0ps_bnb

    def run(self, tolerance):
        self.beta, self.relative_gap = self.solve_el0ps_bnb(
            self.X, self.y, self.M, self.lmbd, tolerance, self.acceleration
        )

    def get_result(self):
        return {
            'x': self.beta.ravel(), 
            'relative_gap': self.relative_gap,
        }
