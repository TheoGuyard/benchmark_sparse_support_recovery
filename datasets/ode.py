from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import math
    import numpy as np
    import pysindy as ps
    from scipy.integrate import solve_ivp
    from scipy.linalg import block_diag


integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


def get_system_dim(system_name):
    if system_name == "Lorenz":
        return 3
    elif system_name == "VanderPol":
        return 2
    elif system_name == "Duffing":
        return 4
    elif system_name == "Lotka":
        return 2
    elif system_name == "Hopf":
        return 2
    elif system_name == "Rossler":
        return 3
    elif system_name == "Meanfield":
        return 3
    elif system_name == "AtmosphericOscillator":
        return 3
    elif system_name == "MHD":
        return 6
    elif system_name == "Meanfield":
        return 3
    raise ValueError(f"Unknown system name: {system_name}")


def get_polylib_size(order, dim, include_bias=True):
    size = 1 if include_bias else 0
    for o in range(1, order + 1):
        size += math.comb(dim + o - 1, o)
    return size


class System:
    def __init__(self):
        self.name = ""
        self.is_pde = False
        self.initial_condition = (0, 0)
        self.forward_fn = None
        self.true_coefs = []

    def simulate(self, duration=10, dt=0.01, x0=None):
        x0 = self.initial_condition if x0 is None else x0
        t_train = np.arange(0, duration, dt)
        t_train_span = (t_train[0], t_train[-1])
        x_train = solve_ivp(
            self.forward_fn,
            t_train_span,
            x0,
            t_eval=t_train,
            **integrator_keywords,
        ).y.T
        return t_train, x_train

    def correct_invalid_initial_conditions(self, x0s):
        return x0s

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=10
    ):
        if seed is not None:
            np.random.seed(seed)
        _, canonical_trajectory = self.simulate(duration)
        std = np.std(canonical_trajectory, axis=0)
        ct = np.arange(len(canonical_trajectory))
        starting_ixs = np.random.choice(ct, size=n, replace=False)
        starting_points = canonical_trajectory[starting_ixs, :]
        x0s = np.random.normal(starting_points, std / closeness)
        return self.correct_invalid_initial_conditions(x0s)


class Lorenz(System):
    def __init__(self, p=(10, 8 / 3, 28), library_size=56):
        super(Lorenz).__init__()
        self.name = "Lorenz"
        self.initial_condition = (-8, 8, 27)
        sigma, beta, rho = p
        self.forward_fn = lambda t, x: [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 2]] = [-sigma, sigma]
        true_coefs[1, [1, 2, 6]] = [rho, -1, -1]
        true_coefs[2, [3, 5]] = [-beta, 1]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=10
    ):
        if seed is not None:
            np.random.seed(seed)
        x = np.random.uniform(-5, 5, size=n)
        y = np.random.uniform(-5, 5, size=n)
        z = np.random.uniform(10, 40, size=n)
        return np.column_stack([x, y, z])


class VanderPol(System):
    def __init__(self, p=0.5, library_size=21):
        super(VanderPol).__init__()
        self.name = "Van der Pol"
        self.initial_condition = (1, 0)
        self.p = p
        self.forward_fn = lambda t, x: [
            x[1],
            p * (1 - x[0] ** 2) * x[1] - x[0],
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, 2] = 1
        true_coefs[1, [1, 2, 7]] = [-1, p, -p]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=5
    ):
        if seed is not None:
            np.random.seed(seed)
        p = self.p
        x = np.random.uniform(-1, 1, size=n)
        y = np.random.uniform(-p, p, size=n)
        return np.column_stack([x, y])


class Duffing(System):
    def __init__(self, p=(-1, 1), library_size=10):
        super(Duffing).__init__()
        self.name = "Duffing"
        self.initial_condition = (2.1, 1.1, 2.1, 1.5)
        omega, alpha = p
        self.forward_fn = lambda t, x: [
            x[2],
            x[3],
            -omega * x[0] - alpha * (x[0] ** 3 + x[0] * x[1] ** 2),
            -omega * x[1] - alpha * (x[0] ** 2 * x[1] + x[1] ** 3),
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 6, 8]] = [-omega, -alpha, -alpha]
        true_coefs[1, [2, 7, 9]] = [-omega, -alpha, -alpha]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=5
    ):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-math.pi, math.pi, size=(n, 4))


class Lotka(System):
    def __init__(self, p=(1, 10), library_size=21):
        super(Lotka).__init__()
        self.name = "Lotka"
        self.initial_condition = (0.8, 0.4)
        self.forward_fn = lambda t, x: [
            p[0] * x[0] - p[1] * x[0] * x[1],
            p[1] * x[0] * x[1] - 2 * p[0] * x[1],
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 4]] = [p[0], -p[1]]
        true_coefs[1, [2, 4]] = [-2 * p[0], p[1]]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=5
    ):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(0, 1, size=(n, 2))


class Hopf(System):
    def __init__(self, p=(-0.05, 1, 1), library_size=21):
        super(Hopf).__init__()
        self.name = "Hopf"
        self.initial_condition = (1, 0)
        mu, omega, A = p
        self.forward_fn = lambda t, x: [
            mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
            omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 2, 6, 8]] = [mu, -omega, -A, -A]
        true_coefs[1, [1, 2, 7, 9]] = [omega, mu, -A, -A]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=5
    ):
        if seed is not None:
            np.random.seed(seed)
        theta = np.random.rand(n) * 2 * math.pi
        x0 = np.column_stack([np.cos(theta), np.sin(theta)])
        return (x0.T * np.random.uniform(0.75, 1.25, n)).T


class Rossler(System):
    def __init__(self, p=(0.2, 0.2, 5.7), library_size=56):
        super(Rossler).__init__()
        self.name = "Rossler"
        self.initial_condition = (5, 3, 0)
        self.forward_fn = lambda t, x: [
            -x[1] - x[2],
            x[0] + p[0] * x[1],
            p[1] + (x[0] - p[2]) * x[2],
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [2, 3]] = [-1, -1]
        true_coefs[1, [1, 2]] = [1, p[0]]
        true_coefs[2, [0, 3, 6]] = [p[1], -p[2], 1]
        self.true_coefs = true_coefs

    def correct_invalid_initial_conditions(self, x0s):
        x0s[:, -1] = np.abs(x0s[:, -1])
        return x0s


class Meanfield(System):
    def __init__(self, p=(0.1, 1, -1, 1), library_size=56):
        super(Meanfield).__init__()
        self.name = "Meanfield"
        mu, omega, A, lambd = p
        self.initial_condition = (mu, mu, 0)
        self.forward_fn = lambda t, x: [
            mu * x[0] - omega * x[1] + A * x[0] * x[2],
            omega * x[0] + mu * x[1] + A * x[1] * x[2],
            lambd * (-x[2] + x[0] ** 2 + x[1] ** 2),
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 2, 6]] = [mu, -omega, A]
        true_coefs[1, [1, 2, 8]] = [omega, mu, A]
        true_coefs[2, [3, 4, 7]] = [-lambd, lambd, lambd]
        self.true_coefs = true_coefs


class AtmosphericOscillator(System):
    def __init__(self, p=(0.05, -0.01, 3.0, -2.0, -5.0, 1.1), library_size=56):
        super(AtmosphericOscillator).__init__()
        self.name = "Atmospheric Oscillator"
        self.initial_condition = (0.2, 0.1, 0.4)
        mu1, mu2, omega, alpha, beta, sigma = p
        self.forward_fn = lambda t, x: [
            mu1 * x[0] + sigma * x[0] * x[1],
            mu2 * x[1]
            + (omega + alpha * x[1] + beta * x[2]) * x[2]
            - sigma * x[0] ** 2,
            mu2 * x[2] - (omega + alpha * x[1] + beta * x[2]) * x[1],
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 5]] = [mu1, sigma]
        true_coefs[1, [3, 4, 5, 8, 9]] = [omega, -sigma, -mu2, alpha, beta]
        true_coefs[2, [2, 6, 7, 8]] = [-omega, -mu2, -alpha, -beta]
        self.true_coefs = true_coefs


class MHD(System):
    def __init__(self, p=(0, 0), library_size=84):
        super(MHD).__init__()
        self.name = "MHD"
        self.initial_condition = (1, -1, 0.5, -0.5, -1, 1)
        nu, mu = p
        self.forward_fn = lambda t, x: [
            -2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
            -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
            -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
            -2 * mu * x[3] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
            -5 * mu * x[4] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
            -9 * mu * x[5] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
        ]
        true_coefs = np.zeros((6, library_size))
        true_coefs[0, [1, 14, 26]] = [-2 * nu, 4, -4]
        true_coefs[1, [2, 9, 24]] = [-5 * nu, -7, 7]
        true_coefs[2, [3, 8, 23]] = [-9 * nu, 3, -3]
        true_coefs[3, [4, 17, 20]] = [-2 * mu, 2, -2]
        true_coefs[4, [5, 12, 19]] = [-5 * mu, -5, 5]
        true_coefs[5, [6, 11, 15]] = [-9 * mu, 9, -9]
        self.true_coefs = true_coefs

    def sample_initial_conditions(
        self, n=10, seed=None, duration=10, closeness=5
    ):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-1.5, 1.5, size=(n, 6))


class Dataset(BaseDataset):
    """Credits to the repository https://github.com/wesg52/sindy_mio_paper and
    the PySindy package."""

    name = "ode"
    parameters = {
        "system": [
            "Lorenz"
        ],  # ["VanderPol", "Duffing", "Lotka", "Hopf", "Rossler"],
        "degree": [3],
        "duration": [1],
        "noise_ratio": [0.001],
        "dt": [0.01],
        "seed": [None],
    }
    install_cmd = "pip"
    requirements = ["pysindy"]

    def get_data(self):
        dim = get_system_dim(self.system)
        system_type = eval(self.system)
        system = system_type(library_size=get_polylib_size(self.degree, dim))
        poly_lib = ps.PolynomialLibrary(degree=self.degree)
        diff_method = ps.differentiation.SmoothedFiniteDifference()

        x0 = system.sample_initial_conditions(n=1, seed=self.seed)[0]
        t, x = system.simulate(self.duration, self.dt, x0=x0)
        v = diff_method._differentiate(x, t)
        e = np.random.normal(0, np.linalg.norm(x) * self.noise_ratio, x.shape)
        x_poly = poly_lib.fit_transform(x + e)

        X = block_diag(*(x_poly for _ in range(dim)))
        y = v.T.flatten()
        w_true = system.true_coefs.flatten()

        return dict(X=X, y=y, w_true=w_true, w_l0pb=None)
