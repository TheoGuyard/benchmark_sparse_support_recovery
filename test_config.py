import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name == "omp":
        pytest.skip("Bad test")
