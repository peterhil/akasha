#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
DSP functions
"""

import numpy as np


# Step functions


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
