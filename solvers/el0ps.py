from pathlib import Path
from benchopt import safe_import_context
from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()

JULIA_SOLVER_FILE = Path(__file__).with_suffix(".jl")


class Solver(JuliaSolver):

    name = "el0ps"
    stopping_strategy = "tolerance"
    references = ["To appear"]
    julia_requirements = [
        "El0ps::https://github.com/TheoGuyard/El0ps.jl#master",
        "PyCall",
    ]
    parameters = {}

    def set_objective(self, X, y, M, lmbd, L):
        self.X, self.y, self.M, self.lmbd, self.L = X, y, M, lmbd, L
        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.run_el0ps = jl.run_el0ps

    def run(self, tolerance):
        self.w = self.run_el0ps(self.X, self.y, self.M, self.lmbd, tolerance)

    def get_result(self):
        return self.w.ravel()
