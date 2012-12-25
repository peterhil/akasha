#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for generators
"""

import numpy as np

from fractions import Fraction
from numpy.testing.utils import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal_nulp as assert_nulp_diff

from akasha.audio.generators import Generator
from akasha.audio.oscillator import Osc
from akasha.timing import sampler


class LinearGenerator(Generator):
    """Simple generator for testing."""

    def __init__(self, rate=1):
        super(self.__class__, self).__init__()
        self.ratio = Fraction.from_float(rate).limit_denominator(sampler.rate)

    def sample(self, iterable):
        return np.array(iterable) * float(self.ratio)


class TestGenerator(object):
    """Test generator"""

    def test_class(self):
        assert issubclass(Generator, object)

    def test_getitem_sample(self):
        pass

    def test_getitem_with_slice(self):
        pass

    def test_getitem_with_len(self):
        pass


# TODO: Check slicing for differences, when given slices or other sampleable object:

# In [37]: r = LinearGenerator(0.5)

# In [40]: r[0:6]
# Out[40]: array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5])

# In [41]: r[-6:6]
# Out[41]: array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5])

# In [42]: r[-1]
# Out[42]: -0.5

# In [43]: r[-6]
# Out[43]: -3.0

# In [44]: r[np.arange(-6,6)]
# Out[44]: array([-3. , -2.5, -2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5])

# TODO: Enable real based slicing? === Sampling with intervals!
# Works when sample (at) uses a continuous function...

# In [53]: o[np.arange(10.115, 12.3, 1, dtype=np.int64)]
# Out[53]: array([ 0.5727+0.4148j,  0.5471+0.4512j,  0.5225+0.489j ])

# In [54]: [np.arange(10.115, 12.3, 1, dtype=np.int64)]
# Out[54]: [array([10, 11, 12])] <-- SHOULD be array(11, 12)!!!


class TestPeriodicGenerator(object):
    """Test periodic generator"""

    @classmethod
    def setup_class(cls):
        cls.o = Osc.from_ratio(1, 6)
        cls.p = Osc.from_ratio(3, 8)

    def test_getitem_with_list(self):
        assert self.o[-1] == self.o[5] == self.o[11]
        assert_nulp_diff(self.o[0, 6, 12], self.o[0], 1)

    def test_getitem_with_slice(self):
        assert_array_equal(
            self.o.sample,
            self.o[::]
        )
        assert_array_equal(
            Osc.from_ratio(1, 3).sample,
            self.o[:3:2]
        )
        assert_array_equal(
            self.p[7, 2, 5, 0, 3, 6, 1, 4, 7],
            self.p[-1:8:3]
        )

    def test_sample_period_is_accurate(self):
        o = Osc(1)
        s = sampler.rate
        assert_array_equal(o[0 * s: 1 * s], o[1 * s: 2 * s])
        assert_array_equal(o[0 * s: 1 * s], o[2 * s: 3 * s])
