from benchopt import safe_import_context
from benchopt.stopping_criterion import StoppingCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class RunOnGridCriterion(StoppingCriterion):
    def __init__(
        self,
        grid=np.linspace(0, 1, 10),
        strategy="iteration",
        key_to_monitor="objective_value",
        **kwargs,
    ):
        super().__init__(strategy=strategy, key_to_monitor=key_to_monitor)
        self.grid = grid
        self.grid_idx = 0

    def get_runner_instance(
        self, max_runs=1, timeout=None, output=None, solver=None
    ):
        self.kwargs["grid"] = self.grid
        self.kwargs["grid_idx"] = self.grid_idx
        return super().get_runner_instance(max_runs, timeout, output, solver)

    def init_stop_val(self):
        return self.grid[self.grid_idx]

    def check_convergence(self, _):
        stop = self.grid_idx >= len(self.grid) - 1
        progress = (self.grid_idx + 1) / len(self.grid)
        return stop, progress

    def get_next_stop_val(self, _):
        self.grid_idx += 1
        return self.grid[self.grid_idx]
