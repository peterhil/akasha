#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for oscillator.py
"""

from oscillator import *
from fractions import Fraction
import unittest

class OscInitTest(unittest.TestCase):
    def setUp(self):
        self.o = Osc(Fraction(1,8))
    
    def testInit(self):
        """Test initialization"""
        assert(isinstance(self.o, Osc))
    
    def testSize(self):
        """Test size of roots"""
        self.assertEqual(8, self.o.roots.size)

if __name__ == "__main__":
    unittest.main()