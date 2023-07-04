from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from libsvmdata import fetch_libsvm
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': ["bodyfat", "housing", "triazines"],
    }

    install_cmd = 'conda'
    requirements = ['pip:git+https://github.com/mathurinm/libsvmdata@main']
    references = [
        "C. Chang and CJ. Lin, "
        "'ACM transactions on intelligent systems and technology (TIST)', "
        "Acm New York, USA vol 2 (2011)."
    ]

    def __init__(self, dataset):
        self.dataset = dataset

    def get_data(self):

        scaler = StandardScaler()
        X, y = fetch_libsvm(self.dataset)
        X = scaler.fit_transform(X)
        w_true = None
        M = 1.5 * np.linalg.norm(X.T @ y, np.inf)
        return dict(X=X, y=y, w_true=w_true, M=M)