import numpy as np
import scipy.io

from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):
    name = "deconvolution"

    parameters = {
        "n_nnz, isnr": [
            (5, 10),
        ],
    }

    def __init__(
        self,
        n_nnz=2,
        isnr=10,
        random_state=27,
    ):
        self.n_nnz = n_nnz
        self.isnr = isnr
        self.random_state = random_state

    def get_data(self):
        X = scipy.io.loadmat('datasets/deconvolution.mat')['H'].T
        w_true = np.zeros(X.shape[1])
        s_true = np.array(np.floor(np.linspace(0, X.shape[1] - 1, num=self.n_nnz)), dtype=int)
        w_true[s_true] = np.random.randn(self.n_nnz) 
        y = X @ w_true 
        sigma = np.mean(y**2)*10**(-self.isnr/10)
        noise = np.random.randn(y.shape[0]) * np.sqrt(sigma)
        y += noise
        return dict(X=X, y=y, w_true=w_true)
