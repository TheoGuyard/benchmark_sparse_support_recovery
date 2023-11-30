import numpy as np


def generate_sources(n, k, random_state=None):
    rng = np.random.RandomState(random_state)
    w_true = rng.randn(n)
    w_true[rng.choice(n, n - k, replace=False)] = 0.0

    return w_true
