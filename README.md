# Benchopt benchmark for sparse support recovery problems

[![Tests](https://github.com/TheoGuyard/benchmark_sparse_support_recovery/actions/workflows/main.yml/badge.svg)](https://github.com/TheoGuyard/benchmark_sparse_support_recovery/actions/workflows/main.yml)
[![Python 3.9](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)

[Benchopt](https://benchopt.github.io) is a package to simplify and make more transparent and
reproducible the benchmarking of optimization algorithms.
This benchmark is dedicated to **sparse support recovery problems** where one considers a noisy linear model

$$y = Xw^{\dagger} + \epsilon$$

and tries to recover the position of the non-zeros in $w^{\dagger}$ from $y$ and $X$. 

## Install

This benchmark can be run using the following commands:

```{shell}
$ pip install -U benchopt
$ git clone https://github.com/TheoGuyard/benchmark_sparse_support_recovery
$ benchopt run benchmark_sparse_support_recovery
``````

Apart from the problem, options can be specified to restrict the benchmarks to some solvers or datasets, e.g.:

```{shell}
$ benchopt run benchmark_sparse_support_recovery -s <solver> -d <dataset> --max-runs 10 --n-repetitions 10
```

Visit [Benchopt documentation](https://benchopt.github.io/api.html) for more details.

## Datasets

A dataset must provide the data $y$ and $X$.
From these data and the solutions provided by the solvers to the sparse support recovery problem, the benchmark computes various performance metrics.
Optionally, the ground truth solution $w^{\dagger}$ can be provided by the dataset which allows for evaluating extra performance metrics.
The following datasets are currently available:

* **deconvolution:** This dataset corresponds to sparse deconvolution problems linked to signal processing applications. The linear operator $X$ is a discrete convolution matrix corresponding to a 21-sample of a sinus-cardinal impulse response. The ground-truth is constructed with non-zero entries at random positions and with an amplitude sampled from a normal distribution. The observation is set as $y = Xw^{\dagger} + \epsilon$ where $\epsilon$ is a centered Gaussian noise whose variance is tuned to meet a given signal-to-noise ratio.
* **lattice:** This dataset is linked to the construction of sparse predictive lattice models for atomic ordering analysis. It is composed of an operator $X$ corresponding to the correlation of atomic structures and an observation $y$ corresponding to energy levels predicted by the density-functional theory. No ground truth are available.
* **libsvm:** This dataset contains various machine-learning sparse regression datasets drawn from the `libsvm` database. Each dataset provides a feature matrix $X$ and a target vector $y$ intended to be linked through a linear model. No ground truth are provided.
* **meg:** This dataset is linked to MEG (Magneto-encephalography) data from an auditory stimulation experiment using 305 sensors. The linear operator $X$ corresponds to the MEG operator. We either generate synthetically a ground truth $w^{\dagger}$ and an observation $y$ as in the **simulated** dataset or provide an observation $y$ corresponding to a real-world experiment, but for which the ground truth is unavailable.
* **ode:** This dataset aims at recovering the parametrization of dynamical systems from a finite number of observations of their trajectory, assuming that they are expressed within a dictionary of basis functions (polynomials, trigonometric, ...). Here, $y$ corresponds to the observations of the trajectory, $X$ concatenates the basis functions evaluated at the observation times and $w^{\dagger}$ is the true parametrization of the dynamical system.
* **portfolio:** This dataset contains five different couples $(X,y)$ representing the mean return and the correlation of some assets and corresponding to portfolio optimization problems. These data are provided by the OR library. No ground truth vectors are available.
* **simulated:** This dataset generates synthetic data for the problem. The linear operator $y$ is built from an auto-regressive model, the ground-truth $w^{\dagger}$ is constructed with non-zero entries at random positions and with an amplitude sampled from a normal distribution and the observation is set as $y = Xw^{\dagger} + \epsilon$ where $\epsilon$ is a centered Gaussian noise whose variance is tuned to meet a given signal-to-noise ratio. Different generation parameters such as the problem dimensionality, the auto-regressive model correlation and the signal-to-noise ratio can be controlled. 

## Solvers

In this benchmark, the solvers are given the tuple $(y,X)$ and a target sparsity amount $\rho \in [0,1]$ and output some solution to the sparse recovery problem with at most $k = ⌊\rho n⌋$ non-zero elements, where $n$ is the size of $w$.
Our benchmark currently includes the following solvers.

* **iht:** Approximate resolution of the $\ell_0$-constrained least-squares problem via the Iterative Hard Thresholding algorithm. This algorithm amounts to applying a projected gradient algorithm on an $\ell_0$-constrained least-squares problem. The thresholding step only keeps the $k$-largest entries in absolute value. 
* **l0constraint:** Exact resolution of the $\ell_0$-constrained least-squares problem using the MIP  solver `gurobi`. The problem is formulated into the MIP formalism via a Big-M constraint where the Big-M value is set as $M = 10 \times \|\|X^{\dagger}y\|\|_{\infty}$, where $X^{\dagger}$ denotes the pseudo-inverse of $X$.
* **l0learn:** Approximate $\ell_0$-penalized least-squares problem solver from [l0learn](https://github.com/hazimehh/L0Learn). The solver fits a regularization path, *i.e.*, it progressively decreases the $\ell_0$-penalty weight and returns the last solution with $k$ non-zero elements in the regularization path.
* **lars:** Lars algorithm from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html).
* **omp:** Orthogonal Matching Pursuit algorithm from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit).
* **skglm:** Lasso, Elastic-Net and MCP problem solver from [skglm](https://contrib.scikit-learn.org/skglm/). The solver fits a regularization path, *i.e.*, it progressively decreases the Lasso, Elastic-Net or MCP penalty weight, and returns the last solution with $k$ non-zero elements in the regularization path.

> The grid of parameters $d$ can be handeled in the solvers using the `RunOnGridCriterion` available in `benchmark_utils/stopping_criterion.py`. If you want to contribute and add a new solver, you can refer to any existing solver for an example of implementation. 
