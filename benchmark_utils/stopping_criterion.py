from benchopt import safe_import_context
from benchopt.stopping_criterion import StoppingCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class RunOnGridCriterion(StoppingCriterion):
    def __init__(
        self,
        grid=np.linspace(0, 0.1, 10),
        strategy="iteration",
        key_to_monitor="objective_value",
    ):
        # TODO: Here, the grid is somewhow cached and the default argument
        # grid=np.linspace(0, 1, 10) is always used. If in the solvers, we set
        # stopping_criterion = RunOnGridCriterion(grid=something_else), then
        # something_else argument will not be taken into account. How to fix
        # that ?
        super().__init__(strategy=strategy, key_to_monitor=key_to_monitor)
        self.grid = grid
        self.grid_idx = 0

    def init_stop_val(self):
        return self.grid[self.grid_idx]

    def check_convergence(self, _):
        stop = self.grid_idx >= len(self.grid) - 1
        progress = (self.grid_idx + 1) / len(self.grid)
        return stop, progress

    def get_next_stop_val(self, _):
        self.grid_idx += 1
        return self.grid[self.grid_idx]
