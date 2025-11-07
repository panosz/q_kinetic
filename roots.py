from collections import namedtuple

import numpy as np
from scipy.optimize import root_scalar

Zero_Cross_Intervals = namedtuple(
    "Zero_Cross_Intervals", ["x_before", "x_after", "y_before", "y_after"]
)


Zero_Cross_Intervals_Only_X = namedtuple(
    "Zero_Cross_Intervals",
    [
        "x_before",
        "x_after",
    ],
)


def zero_cross_index(y):
    sign_diff = np.diff(np.sign(y))
    sign_diff[np.isnan(sign_diff)] = 0  # drops false positives for NaN values

    return np.nonzero(sign_diff)


def zero_cross_elems(x, y):
    """
    find  the values of x for which y crosses zero. The values returned are the
    ones before or exactly at
    the zero crossing. If the last value of x is a root, it does not get
    returned
    :param x: numpy array (sorted)
    :param y: numpy array must have the same length with x
    :return: namedtuple Zero_Cross_Result (xbefore,xafter,ybefore,yafter) .
    """
    assert x.size == y.size, "x and y must have equal size!"
    indices = zero_cross_index(y)[0]
    next_indices = indices + 1

    return Zero_Cross_Intervals(
        x[indices], x[next_indices], y[indices], y[next_indices]
    )


def zero_cross_brackets(x, y, transform=None):
    if transform is not None:
        y = transform(y)

    zc = zero_cross_elems(x, y)

    return [
        ((xb, xa), (yb, ya))
        for xb, xa, yb, ya in zip(
            zc.x_before, zc.x_after, zc.y_before, zc.y_after
        )
    ]


def sample_fun(f, window, n_samples):
    """
    samples a function uniformly for x in the range [window[0], window[1]].
    :param f: callable with a single float argument
    :param window: a tuple specifying the sampling range
    :param n_samples: int, the number of samples
    :return: a tuple of np.arrays x,y
    """
    x0, x1 = window
    x = np.linspace(x0, x1, n_samples)
    return x, f(x)


def roots(f, window, n_samples=100, samples=None, condition=None):
    """
    Find the roots of a scalar function.

    If `condition` is None, find the roots of `f(x)`.

    If `condition` is not None, find the roots of
    `condition(f(x))` instead. See below.


    Parameters
    ----------
    f: callable
        The function.

    window: (xmin, xmax),
        The domain in which to look for points.

    n_samples: int, optional
        The number of samples.
        Ignored of `samples` is not None.
        The samples will be distributed
        uniformly in the `window` domain. The algorithm will track
        down roots, when it detects a change of sing between adjacent
        samples. This means that roots that are close by may be missed
        if the `n_samples` is too small.

        Alternatively, a sequence of pre calculated samples can be
        passed through the `samples` variable.
        Default is 100.

    samples: (x_s, y_s), optional
        If not None, skip the sampling step and use the
        samples passed `here` instead.
        Default is None

    condition: callable, optional.
        If `condition` is specified, then the roots of
        `condition(f(x))` are found instead.

        Use it in your code to distinguish between some meaningful
        quantity `f(x)` and the `condition` it is required to satisfy.

        This is especially useful, when `f(x)` is relatively expensive
        and/or the roots of more than one `condition`s need to be located.

        Disentangling the `condition` from the function `f` makes it
        easy to reuse the same `sample`s with many different
        `condition`s.

    """
    if samples is not None:
        x = np.asarray(samples[0])
        y = np.asarray(samples[1])

    else:
        x, y = sample_fun(f, window, n_samples)

    brackets = zero_cross_brackets(x, y, condition)

    if condition:

        def f_solve(x):
            return condition(f(x))

    else:
        f_solve = f

    for x_bracket, _ in brackets:
        sol = root_scalar(f_solve, bracket=x_bracket)
        yield sol.root, sol.converged
