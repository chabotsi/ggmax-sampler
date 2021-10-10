#!/usr/bin/env python3
# coding: utf-8

from typing import Callable

from numpy import sqrt, array, ndarray, log10

# Version format is YYYY.MM.DD (https://calver.org/)
__version__ = "2021.10.10"


def get_best_points_for_ggmax_sampling(
    ggmax_func: Callable[[float], float], xmin: float, xmax: float, N: int
) -> ndarray:
    """
    This is a helper to use the `autosampler_N` with ggmax curves.

    Because the ggmax curves are exponential, we sample them in log-scale, and
    then rescale the sampling points to their original scale.

    Find the `N` best points to sample the function `ggmax_func`, between `xmin`
    and `xmax`

    Parameters
    ----------

    ggmax_func:
        The ggmax function to sample
    xmin:
        Start sampling from this point
    xmax:
        Stop sampling at this point
    N:
        The number of sampling points to return between xmin and xmax


    Return
    ------

    res:
        the `N` sampling points
    """

    x0 = log10(xmin)
    xf = log10(xmax)

    return 10 ** autosampler_N(lambda x: ggmax_func(10 ** x), x0, xf, N).flatten()


def autosampler_N(
    func: Callable[[float], float], x0: float, xf: float, N: int
) -> ndarray:
    """Find the `N` best points to sample the function `func`, between `x0` and
    `xf`

    Parameters
    ----------

    func:
        The function to sample
    x0:
        Start sampling from this point
    xf:
        Stop sampling at this point
    N:
        The number of sampling points to return between x0 and xf


    Return
    ------

    res:
        the `N` sampling points
    """
    error_inf = 0.0
    error_sup = 1.0

    while True:
        error = 0.5 * (error_inf + error_sup)
        points = autosampler_error(func, x0, xf, error)

        if len(points) > N:  # reduce the error
            error_inf = error
        elif len(points) < N:
            error_sup = error
        else:
            return points


def autosampler_error(
    func: Callable[[float], float], x0: float, xf: float, error: float
) -> ndarray:
    """Find the minimum number of points to sample the function `func`, between
    `x0` and `xf` and satisfying an L2-error lower than `error`.

    Parameters
    ----------

    func:
        The function to sample
    x0:
        Start sampling from this point
    xf:
        Stop sampling at this point
    error:
        The error criterion


    Return
    ------

    res:
        the sampling points
    """
    points = [
        x0,
    ]

    t_curr = x0
    t_prev = x0 - 1e-8
    C = error * sqrt(120)
    p = 8
    TOL = 1e-8
    while points[-1] < xf:
        t_next = t_curr + (t_curr - t_prev)
        diff_f = func(t_next) - func(t_curr)

        while True:
            t_next = t_curr + (C * (t_next - t_curr) ** (p - 1) / abs(diff_f)) ** (
                1 / p
            )
            diff_f = func(t_next) - func(t_curr)

            if abs((t_next - t_curr) * abs(diff_f) - C) < TOL:
                break

        t_prev = t_curr
        t_curr = t_next
        points.append(t_next)

    points[-1] = xf
    return array(points)
