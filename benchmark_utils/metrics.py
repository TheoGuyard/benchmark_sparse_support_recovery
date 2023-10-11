from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


def snr(w_true, w):
    if np.linalg.norm(w_true - w, 2) == 0.0:
        return np.inf
    return np.linalg.norm(w_true, 2) / np.linalg.norm(w_true - w, 2)


def fpr(w_true, w):
    if np.sum(w_true != 0.0) == 0.0:
        return 1.0
    return np.sum((w_true == 0.0) * (w != 0.0)) / np.sum(w_true != 0.0)


def fnr(w_true, w):
    if np.sum(w_true == 0.0) == 0.0:
        return 1.0
    return np.sum((w_true != 0.0) * (w == 0.0)) / np.sum(w_true == 0.0)
