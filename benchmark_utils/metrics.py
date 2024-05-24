from benchopt import safe_import_context
from sklearn.metrics import f1_score

with safe_import_context() as import_ctx:
    import numpy as np


def snr(w_true, w, dB=False):
    if np.linalg.norm(w_true - w, 2) == 0.0:
        return np.inf
    snr = np.linalg.norm(w_true, 2) / np.linalg.norm(w_true - w, 2)
    return 10. * np.log10(snr) if dB else snr


def fpr(w_true, w):
    if np.sum(w_true == 0.0) == 0.0:
        return 1.0
    return np.sum((w_true == 0.0) * (w != 0.0)) / np.sum(w_true == 0.0)


def fnr(w_true, w):
    if np.sum(w_true != 0.0) == 0.0:
        return 1.0
    return np.sum((w_true != 0.0) * (w == 0.0)) / np.sum(w_true != 0.0)


def tpr(w_true, w):
    if np.sum(w_true != 0.0) == 0.0:
        return 1.0
    return np.sum((w_true != 0.0) * (w != 0.0)) / np.sum(w_true != 0.0)


def tnr(w_true, w):
    if np.sum(w_true == 0.0) == 0.0:
        return 1.0
    return np.sum((w_true == 0.0) * (w == 0.0)) / np.sum(w_true == 0.0)


def f1score(w_true, w):
    return f1_score(w_true != 0, w != 0)


def auc(w_true, w, num_rounds=10_000):
    """AUC metric for regression data. See https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb."""  # noqa: E501

    def _yield_pairs(w_true, num_rounds):
        if num_rounds == "exact":
            for i in range(len(w_true)):
                for j in np.where(
                    (w_true != w_true[i]) & (np.arange(len(w_true)) > i)
                )[0]:
                    yield i, j
        else:
            for _ in range(num_rounds):
                i = np.random.choice(range(len(w_true)))
                j = np.random.choice(np.where(w_true != w_true[i])[0])
                yield i, j

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(w_true, num_rounds):
        diff_true = w_true[i] - w_true[j]
        diff_score = w[i] - w[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += 0.5
        num_pairs += 1

    return num_same_sign / num_pairs


def dist_to_supp(w_true, w):
    r"""Given $w^{\dagger}$ and $w$, returns the average minimum distance
    between a non-zero entry of $w$ and a non-zero entry in $w^{\dagger}$, that
    is$\frac{1}{n}\sum_{i,w_i \neq 0}\min_{j,w^{\dagger}_j \neq 0} |j - i|$.
    """
    s = np.flatnonzero(w)
    s_true = np.flatnonzero(w_true)
    if s.size == 0:
        return 1.0
    d = 0
    for i in s:
        d += np.min(np.abs(s_true - i)) / len(w)
    return d / len(s)
