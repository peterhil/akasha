#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Beta envelopes
"""

import numpy as np

from scipy import special
from scipy import stats

from akasha.utils.python import _super


class Beta:
    """Envelope curves using beta distribution's cumulative
    distribution functions, that are stretched to some time scale.

    https://en.wikipedia.org/wiki/Beta_distribution

    https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    """

    def __init__(self, time=1.0, a=1.0, b=5.0, amp=1.0):
        self.a = float(a)
        self.b = float(b)
        self.amp = np.clip(amp, a_min=0, a_max=1)
        assert time != 0, "Scale can not be zero!"
        self.time = float(time)

    def at(self, times):
        """
        Sample beta cdf at times.
        """
        times = times / self.time
        # Fix NaNs for negative time values
        times = np.where(times < 0, 0, times)

        beta = special.beta(self.a, self.b) * stats.beta.cdf(
            times, self.a, self.b
        )
        clipped = np.clip(self.amp * beta, a_min=0, a_max=1)

        return clipped


class InverseBeta(Beta):
    """
    Inverse of Beta. That is: 1.0 - Beta.at(t).
    """
    def at(self, times):
        return 1.0 - _super(self).at(times)
