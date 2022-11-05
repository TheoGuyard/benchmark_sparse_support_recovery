import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {"n_samples, n_features": [(100, 300), (300, 100)]}

    def __init__(self, n_samples=10, n_features=50, random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        return dict(X=X, y=y)
