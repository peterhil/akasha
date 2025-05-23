"""
DSP functions
"""

import numpy as np


@np.vectorize
def unit_step(x):
    """
    Discrete Heaviside unit step function.
    """
    return 0 if x < 0 else 1


@np.vectorize
def unit_impulse(x):
    """
    Kronecker delta function.
    """
    return 1 if x == 0 else 0
