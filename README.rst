Benchopt benchmark for L0-penalized Least-squares problem with a Big-M constraint
=================================================================================

|Build Status| |Python 3.9+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to the L0-penalized Least-squares problem with a Big-M constraint which reads :

$$\\min_{w} \\Big\\{ \\tfrac{1}{2}\\|y-Aw\\|_2^2 + \\lambda \\|w\\|_0 \\ \\ \\textrm{st.} \\ \\ \\|w\\|_{\\infty} \\leq M \\Big\\} $$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$A \\in \\mathrm{R}^{n \\times p}, \\ y \\in \\mathrm{R}^{n}, \\ \\lambda > 0, \\ M > 0$$

are data of the problem.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/TheoGuyard/benchmark_l0pen_ols_bigm
   $ benchopt run benchmark_l0pen_ols_bigm

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::
   
   $ benchopt run benchmark_l0pen_ols_bigm -s <solver> -d <dataset> --max-runs 10 --n-repetitions 10

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/TheoGuyard/benchmark_l0pen_ols_bigm/workflows/Tests/badge.svg
   :target: https://github.com/TheoGuyard/benchmark_l0pen_ols_bigm/actions
.. |Python 3.9+| image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/release/python-390/
