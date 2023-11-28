from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "libsvm"
    parameters = {'dataset': ["abalon", "bodyfat", "eunite2001", "housing"]}
    install_cmd = 'conda'
    requirements = ['pip:git+https://github.com/mathurinm/libsvmdata@main']

    def get_data(self):
        X, y = fetch_libsvm(self.dataset)
        return dict(X=X, y=y, w_true=None, w_l0pb=None)
