import numpy as np
import scipy.io

from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):
    name = "Bourguignon"

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
        H = scipy.io.loadmat('datasets/Hmatrix_Bourguignon.mat')
        H = H['H']
        H = H.T
        n_features = H.shape[1]
        X, y, w_true = make_correlated_data(
            n_samples = n_features,
            n_features= n_features,
            rho=0,
            snr=self.isnr,
            density=self.n_nnz / n_features,
            random_state=self.random_state,
        )
        y = H@w_true 
        sigma0 = np.mean(y**2)*10**(-self.isnr/10)
        noise = np.random.randn(y.shape[0]) * np.sqrt(sigma0)
        y = y + noise
       
        
        M = np.linalg.norm(w_true, np.inf)
        return dict(X=H, y=y, w_true=w_true, M=M)
