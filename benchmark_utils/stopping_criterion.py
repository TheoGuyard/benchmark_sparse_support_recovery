from benchopt.stopping_criterion import StoppingCriterion


class RunOnGridCriterion(StoppingCriterion):
    def __init__(
        self,
        grid=range(10),
        strategy="iteration",
        key_to_monitor="objective_value",
    ):
        super().__init__(strategy=strategy, key_to_monitor=key_to_monitor)
        self.grid = grid
        self.grid_idx = 0

    def reset_grid(self, grid):
        self.grid = grid
        self.grid_idx = 0

    def check_convergence(self, _):
        stop = self.grid_idx >= len(self.grid) - 1
        progress = (self.grid_idx + 1) / len(self.grid)
        return stop, progress

    def get_next_stop_val(self, _):
        self.grid_idx += 1
        return self.grid[self.grid_idx]
