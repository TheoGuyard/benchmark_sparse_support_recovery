import numpy as np


def snr(w_true, w):
    return np.linalg.norm(w_true, 2) / np.linalg.norm(w_true - w, 2)


def fpr(w_true, w):
    return np.sum((w_true == 0.0) * (w != 0.0)) / np.sum(w_true != 0.0)


def fnr(w_true, w):
    return np.sum((w_true != 0.0) * (w == 0.0)) / np.sum(w_true == 0.0)
