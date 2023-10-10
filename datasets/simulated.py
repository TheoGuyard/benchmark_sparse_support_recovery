from benchopt import BaseDataset, safe_import_context
from benchopt.datasets.simulated import make_correlated_data

with safe_import_context() as import_ctx:
    import uuid


class Dataset(BaseDataset):
    name = "simulated"
    parameters = {
        "n_samples, n_features, density, rho, snr, random_state": [
            (50, 100, 0.05, 0.9, 10.0, None),
        ],
    }

    def __init__(self, **params):
        self.params = params

    def get_data(self):
        instance_uuid = str(uuid.uuid1())
        X, y, w_true = make_correlated_data(**self.params)
        return dict(instance_uuid=instance_uuid, X=X, y=y, w_true=w_true)
