# Benchopt benchmark for sparse support recovery problems

[![Build status](https://github.com/TheoGuyard/benchmark_sparse_support_recovery/workflows/Tests/badge.svg)](https://github.com/TheoGuyard/benchmark_sparse_support_recovery/actions)
[![Python 3.9](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)

[Benchopt](https://benchopt.github.io) is a package to simplify and make more transparent and
reproducible the benchmarking of optimization algorithms.
This benchmark is dedicated to **sparse support recovery problems** where one considers a noisy model

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
Optionnally, it can give the ground truth solution $w^{\dagger}$ and the solution of the L0-constrained least-squares problem $w^{\ell_0}$ where the sparsity amount targeted is the one of the ground truth.
Some of the performance metrics will only be evaluated if $w^{\dagger}$ and/or $w^{\ell_0}$ are provided.

Currently, the following datasets are available:

* **Simulated:** The data is generated via the `make_correlated_data` function available in [benchopt](https://benchopt.github.io). The size of the matrix $X$, its correlation amount, the noise level in $y$ and the sparsity density in the ground truth can be specified. The ground truth $w^{\dagger}$ is available but $w^{\ell_0}$ is not available yet.

## Solvers

In this bechmark, the `run` method of [benchopt](https://benchopt.github.io) is to be called over a grid of parameters $d \in [0,1]$ where $d$ specifies the target proportion of sparsity.
This can be done using the `RunOnGridStoppingCriterion` available in the `benchmark_utils.stopping_criterion` module. See for instance the IHT solver implementation in `solvers.iht` to implement new solvers.

Currently, the following solvers are available:

* **IHT:** Iterative Hard Thresholding algorithm that approximately solves L0-constrained least-squares problems. See [this paper](https://ieeexplore.ieee.org/abstract/document/1660731?casa_token=fTzhzl62-TMAAAAA:qGzh7V1ewWv81eFlTbepaiaO5yBO80H_6oHN4ovyeO-6dMB8A6PuccoB3-zfE8zz2yt16dSSIQ) for more details.
* **L0-constraint:** Exact solution method for L0-constrained least-squares problems using a Mixed-Integer-Programming solver. The problem is formulated via Big-M constriants with $M = 10  \|x^{\dagger}\|_{\infty}$. See [this paper](https://ieeexplore.ieee.org/abstract/document/7313004?casa_token=vVs9O6nUNwUAAAAA:LNxXd1jMGr6EE-N0Gx4YaM8ZdaWzPWkZzbMsTQJaTjnY9b4U4n23JalnjBZvWGPpyL7U9U2V6Q) for more details.
* **L0Learn:** Uses [l0learn](https://github.com/hazimehh/L0Learn) package that approximately construct a path for L0-penalized least-squares problems and extract the best $k$-sparse solution. See [this paper](https://arxiv.org/abs/2202.04820) for more details.
* **Lars:** `Lars` algorithm from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html).
* **OMP:** `OrthogonalMatchingPursuit` algorithm from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit).
